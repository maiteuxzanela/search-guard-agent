"""Matriz determinística do TICKET-SGA-10 (refinamentos pós-auditoria Diana).

Cobre, sem mocks de regra de negócio e com subprocessos/arquivos reais:

1. ``TreeSitterGuard`` executa UMA única vez no fluxo JS/TS (Camada 4 embutida
   em ``scan_js_ast``, reaproveitada pelo servidor MCP);
2. simetria de ``inconclusive_reasons`` no ramo JS/TS (degrada para
   ``INCONCLUSIVE`` com ``passed=False``);
3. ``_run_go_suite`` exige ``ok_packages > 0`` para ``VERIFIED_PASS``
   (projeto só com ``[no test files]`` vira ``INCONCLUSIVE``);
4. ``total_files_scanned`` do fail-fast da Camada 1 Python reflete
   ``ast_report["files_scanned"]``;
5. fallback lexical determinístico sem tree-sitter instalado mantém o contrato
   do job de resiliência do CI sem reprovar workspace limpo;
6. resolução de caminhos simétrica (symlinks, como ``/var`` → ``/private/var``
   no macOS) na bridge nativa.

Os shims de toolchain usados aqui são executáveis reais em disco (transport
recorder), nunca ``unittest.mock``/``MagicMock``.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from agent_quality_guard.core.schemas import VerdictState
from agent_quality_guard.mcp_server import (
    LAYER_TREE_SITTER,
    _layer2_from_js_scan,
    _quality_guard_core,
    _run_layer2_meta_rules,
)
from agent_quality_guard.memory.needle_ast_guard import NeedleASTGuard
from agent_quality_guard.orchestrator.multi_stack_runner import MultiStackRunner


def _write(root: Path, rel: str, content: str) -> Path:
    """Materializa um arquivo real no workspace."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _node_workspace(root: Path, *, suite_exit: int = 0, prod: dict[str, str] | None = None) -> Path:
    """Workspace Node real com suíte ``tests/run_all.mjs`` controlável."""
    _write(root, "package.json", json.dumps({"name": "sga10-app", "type": "module"}))
    _write(
        root,
        "tests/run_all.mjs",
        "import assert from 'node:assert';\n"
        "assert.ok(true);\n"
        "console.log('SUÍTES: 1/1 passaram');\n"
        + ("process.exit(1);\n" if suite_exit else ""),
    )
    for rel, content in (prod or {"src/ok.js": "export function ok(){ return 1; }\n"}).items():
        _write(root, rel, content)
    return root


def _python_workspace(root: Path, *, failing: bool = False) -> Path:
    """Workspace Python real com uma suíte pytest mínima."""
    _write(root, "pyproject.toml", '[project]\nname = "sga10-py"\nversion = "0.0.1"\n')
    _write(root, "src/calc.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write(
        root,
        "tests/test_calc.py",
        "from src.calc import add\n\n\n"
        "def test_add() -> None:\n"
        f"    assert add(2, 2) == {5 if failing else 4}\n",
    )
    return root


# ──────────────────────────────────────────────────────────────────────────────
# 1. Camada 4 executa uma única vez no fluxo JS/TS
# ──────────────────────────────────────────────────────────────────────────────

def test_js_stack_runs_tree_sitter_exactly_once(tmp_path: Path, monkeypatch):
    """``scan_tree_sitter`` é chamado 1x por homologação JS (instrumentação real)."""
    _node_workspace(tmp_path)

    calls: list[int] = []
    original = MultiStackRunner.scan_tree_sitter

    def recording_scan(cls, *args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        MultiStackRunner, "scan_tree_sitter", classmethod(recording_scan),
    )

    response = asyncio.run(
        _quality_guard_core(workspace=str(tmp_path), audit_mode="full")
    )

    assert response["passed"] is True, response["message"]
    assert len(calls) == 1, f"TreeSitterGuard executou {len(calls)}x no fluxo JS/TS"
    assert LAYER_TREE_SITTER in response["layers_executed"]
    assert LAYER_TREE_SITTER not in response["layers_skipped"]


def test_js_stack_does_not_rerun_layer4_workspace_scan(tmp_path: Path, monkeypatch):
    """A regressão original não pode reaparecer via ``scan_workspace_report``.

    ``test_js_stack_runs_tree_sitter_exactly_once`` instrumenta
    ``MultiStackRunner.scan_tree_sitter``; a regressão do TICKET-SGA-10
    reintroduzia a dupla execução chamando ``_run_layer2_meta_rules``, que
    percorre o workspace por ``TreeSitterGuard.scan_workspace_report`` — outro
    seam. No fluxo JS, esse agregador NÃO pode ser chamado: o resultado já
    executado dentro de ``scan_js_ast`` é o único permitido.
    """
    from agent_quality_guard.memory.tree_sitter_guard import TreeSitterGuard

    _node_workspace(tmp_path)

    calls: list[int] = []
    original = TreeSitterGuard.scan_workspace_report

    def recording_report(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        TreeSitterGuard, "scan_workspace_report", classmethod(recording_report),
    )

    response = asyncio.run(
        _quality_guard_core(workspace=str(tmp_path), audit_mode="full")
    )

    assert response["passed"] is True, response["message"]
    assert calls == [], f"Camada 4 reexecutada {len(calls)}x via scan_workspace_report"


def test_layer2_from_js_scan_reuses_embedded_tree_sitter_result(tmp_path: Path):
    """O adaptador não reexecuta o guard: lê o payload embutido em ``scan_js_ast``."""
    _node_workspace(tmp_path, prod={
        "src/leak.js": "export function f(){\n  setInterval(()=>go(), 10);\n}\n",
    })
    scan = MultiStackRunner.scan_js_ast(tmp_path, "NODE_JS_GENERIC", audit_mode="full")

    assert isinstance(scan.get("tree_sitter_scan"), dict)
    layer2 = _layer2_from_js_scan(scan)

    assert layer2["available"] is True
    assert any(v["rule_id"] == "RESOURCE_LEAK" for v in layer2["warnings"])
    assert layer2["layer_status"] == scan["tree_sitter_scan"]["layer_status"]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Inconclusivo no ramo JS degrada o veredito (paridade com Python)
# ──────────────────────────────────────────────────────────────────────────────

def test_js_inconclusive_reasons_have_strict_non_approval_precedence():
    """Paridade com Python: inconclusivo na Camada 3/4 nunca vira VERIFIED_PASS."""
    from agent_quality_guard.mcp_server import _static_verdict_state

    assert _static_verdict_state([], 0, []) is None
    assert _static_verdict_state([], 0, ["gramática parcial"]) == (
        VerdictState.INCONCLUSIVE.value
    )
    assert _static_verdict_state([], 1, ["gramática parcial"]) == (
        VerdictState.VERIFIED_FAIL.value
    )
    assert _static_verdict_state(["tsc"], 1, []) == VerdictState.INCONCLUSIVE.value


def test_layer2_from_js_scan_propagates_inconclusive_reasons():
    """O adaptador preserva motivos inconclusivos da Camada 4 embutida."""
    scan = {
        "tree_sitter_scan": {
            "blocking": [],
            "warnings": [],
            "total_files": 2,
            "layer_status": "inconclusive",
            "layers_executed": [],
            "layers_skipped": [LAYER_TREE_SITTER],
            "skip_reasons": {},
            "inconclusive_reasons": ["1 arquivo(s) falharam na Camada 4 (Tree-Sitter)"],
            "engine": "tree_sitter_partial",
        },
    }

    layer2 = _layer2_from_js_scan(scan)

    assert layer2["layer_status"] == "inconclusive"
    assert layer2["inconclusive_reasons"] == [
        "1 arquivo(s) falharam na Camada 4 (Tree-Sitter)"
    ]


# ──────────────────────────────────────────────────────────────────────────────
# 3. Go: `[no test files]` não é aprovação
# ──────────────────────────────────────────────────────────────────────────────

def _go_shim(directory: Path, body: str) -> Path:
    """Cria um executável ``go`` real (subprocesso real, sem mock de injeção)."""
    directory.mkdir(parents=True, exist_ok=True)
    shim = directory / "go"
    shim.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return shim


def _go_workspace(root: Path) -> Path:
    _write(root, "go.mod", "module example.com/sga10\n\ngo 1.21\n")
    _write(root, "pkg/pkg.go", "package pkg\n\nfunc Add(a, b int) int { return a + b }\n")
    return root


def test_go_suite_without_test_files_is_never_verified_pass(tmp_path: Path, monkeypatch):
    _go_workspace(tmp_path)
    shim_dir = tmp_path / "_shim_no_tests"
    _go_shim(shim_dir, 'echo "?   example.com/sga10/pkg [no test files]"\nexit 0\n')
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    result = MultiStackRunner.run_tests(tmp_path, "GO_MODULE", tmp_path / "_tmp")

    assert result["total_passed"] == 0
    assert result["passed"] is False
    assert result["verdict_state"] == VerdictState.INCONCLUSIVE.value
    assert result["reason"] == "go_sem_pacotes_ok"


def test_go_suite_with_ok_packages_is_verified_pass(tmp_path: Path, monkeypatch):
    _go_workspace(tmp_path)
    shim_dir = tmp_path / "_shim_ok"
    _go_shim(shim_dir, 'echo "ok  example.com/sga10/pkg  0.003s"\nexit 0\n')
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    result = MultiStackRunner.run_tests(tmp_path, "GO_MODULE", tmp_path / "_tmp")

    assert result["total_passed"] == 1
    assert result["passed"] is True
    assert result["verdict_state"] == VerdictState.VERIFIED_PASS.value


# ──────────────────────────────────────────────────────────────────────────────
# 4. total_files_scanned no fail-fast da Camada 1 Python
# ──────────────────────────────────────────────────────────────────────────────

def test_python_layer1_fail_fast_reports_real_files_scanned(tmp_path: Path):
    """A métrica é o número de arquivos varridos, não o número de violações."""
    _python_workspace(tmp_path)
    _write(tmp_path, "src/danger.py", 'def run(value: str):\n    return eval(value)\n')

    ast_report = NeedleASTGuard.scan_workspace_report(str(tmp_path), audit_mode="full")
    expected_files = int(ast_report["files_scanned"])
    assert expected_files >= 2, ast_report

    response = asyncio.run(
        _quality_guard_core(workspace=str(tmp_path), audit_mode="full")
    )

    assert response["passed"] is False
    assert response["verdict_state"] == VerdictState.VERIFIED_FAIL.value
    assert response["blocking_ast_violations"], "violação de eval não detectada"
    assert response["total_files_scanned"] == expected_files
    assert response["total_files_scanned"] != len(response["blocking_ast_violations"])
    # Fail-fast não mediu linhas: honestidade acima de zero literal.
    assert response["total_lines_scanned"] is None


# ──────────────────────────────────────────────────────────────────────────────
# 5. Fallback lexical determinístico sem tree-sitter (job de resiliência do CI)
# ──────────────────────────────────────────────────────────────────────────────

def test_layer2_without_tree_sitter_declares_degraded_execution(tmp_path: Path):
    """Sem gramáticas instaladas, o fallback lexical executa e se declara degradado."""
    import agent_quality_guard.memory.tree_sitter_guard as ts_guard

    _write(tmp_path, "src/clean.py", "def f(a: int) -> int:\n    return a + 1\n")
    original_available = ts_guard._TREE_SITTER_AVAILABLE
    original_pack = ts_guard._language_pack
    try:
        ts_guard._TREE_SITTER_AVAILABLE = False
        ts_guard._language_pack = None
        result = _run_layer2_meta_rules(tmp_path, "full")
    finally:
        ts_guard._TREE_SITTER_AVAILABLE = original_available
        ts_guard._language_pack = original_pack

    assert result["available"] is True
    assert result["layer_status"] == "executed"
    assert result["degraded"] is True
    assert result["inconclusive_reasons"] == []
    assert "grammar_unavailable" in result["skip_reasons"]


def test_clean_python_workspace_still_approves_without_tree_sitter(
    tmp_path: Path, monkeypatch
):
    """Contrato do job ``test-without-tree-sitter``: workspace limpo continua verde."""
    import agent_quality_guard.memory.tree_sitter_guard as ts_guard

    _python_workspace(tmp_path)
    monkeypatch.setattr(ts_guard, "_TREE_SITTER_AVAILABLE", False)
    monkeypatch.setattr(ts_guard, "_language_pack", None)

    response = asyncio.run(
        _quality_guard_core(workspace=str(tmp_path), audit_mode="full")
    )

    assert response["passed"] is True, response["message"]
    assert response["verdict_state"] == VerdictState.VERIFIED_PASS.value
    assert LAYER_TREE_SITTER in response["layers_executed"]


# ──────────────────────────────────────────────────────────────────────────────
# 6. Resolução de caminhos simétrica (symlinks tipo macOS /var -> /private/var)
# ──────────────────────────────────────────────────────────────────────────────

def test_workspace_relative_tolerates_symlinked_workspace(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    target = _write(real, "src/a.js", "export const a = 1;\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    assert MultiStackRunner._workspace_relative(alias, target) == "src/a.js"


def test_native_bridge_does_not_crash_on_symlinked_workspace(tmp_path: Path):
    """Reproduz a falha de native_bridge em macos-latest (workspace /var symlink)."""
    real = tmp_path / "real"
    real.mkdir()
    source = _write(real, "src/a.js", "export const a = 1;\n")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)

    result = MultiStackRunner.run_native_bridge(
        alias, "NODE_TS_BUN", audit_mode="full", source_files=[source],
    )

    tool_names = {tool["tool"] for tool in result["tools"]}
    assert "native_bridge" not in tool_names, result
    assert not result["critical_failures"], result["critical_failures"]
