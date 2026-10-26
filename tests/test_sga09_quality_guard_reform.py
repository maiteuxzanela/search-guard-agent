"""Matriz determinística de conformidade do TICKET-SGA-09.

Reforma do contrato e da execução do ``agent_quality_guard``:

1. veredito explícito com precedência estrita (``passed=True`` só em
   ``VERIFIED_PASS`` ou ``EMERGENCY_RELEASE``; nenhum caminho aprova sem
   evidência real executada);
2. detecção de stack 100% por capability, sem nomes de repositórios pessoais;
3. expurgo de contadores forjados;
4. timeout de execução de 600s;
5. fail-fast honesto (``layers_executed``/``layers_skipped``/``skip_reasons``/
   ``layer_status``) sem fingir "0 violações" de camadas que não rodaram;
6. suítes poliglota reais: Python puro, monorepo Node/TS com guarda de raiz e
   repositório poliglota.

Sem mocks: todos os cenários materializam workspaces reais em disco e executam
subprocessos reais (pytest/node/npm) ou o runner real da stack. Nenhum
``unittest.mock``/``MagicMock`` é usado para simular comportamento de negócio.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pytest

from agent_quality_guard.core.schemas import (
    NeedleVerifyDeliveryResponse,
    VerdictState,
)
from agent_quality_guard.mcp_server import (
    LAYER_AST_GUARD,
    LAYER_NATIVE,
    LAYER_SCREENER,
    LAYER_TESTS,
    LAYER_TREE_SITTER,
    _ALL_LAYERS,
    _layer_coverage,
    _quality_guard_core,
)
from agent_quality_guard.orchestrator.multi_stack_runner import (
    MultiStackRunner,
    _DEFAULT_NATIVE_BUDGET_SECONDS,
    _MAX_NATIVE_BUDGET_SECONDS,
    _MIN_NATIVE_BUDGET_SECONDS,
    _TEST_COMMAND_TIMEOUT_SECONDS,
)
from agent_quality_guard.orchestrator.pytest_chunk_runner import PytestChunkRunner

REPO_ROOT = Path(__file__).resolve().parents[1]
_PASSING_VERDICTS = {VerdictState.VERIFIED_PASS.value, VerdictState.EMERGENCY_RELEASE.value}
_FAILING_VERDICTS = {
    VerdictState.VERIFIED_FAIL.value,
    VerdictState.NOT_APPLICABLE.value,
    VerdictState.INCONCLUSIVE.value,
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers reais (sem mocks)
# ──────────────────────────────────────────────────────────────────────────────

def _write(root: Path, rel: str, content: str) -> Path:
    """Materializa um arquivo real no workspace temporário."""
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _python_clean_workspace(root: Path) -> Path:
    """Workspace Python puro com código de produção e teste real passando."""
    _write(root, "pyproject.toml", '[project]\nname = "calc-demo"\nversion = "0.0.1"\n')
    _write(
        root,
        "src/calc.py",
        'def add(a: int, b: int) -> int:\n'
        '    """Soma dois inteiros."""\n'
        "    return a + b\n",
    )
    _write(
        root,
        "tests/test_calc.py",
        "from src.calc import add\n\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
    )
    return root


@contextmanager
def _path_without_toolchain(empty_dir: Path) -> Iterator[None]:
    """Remove o toolchain do PATH durante o bloco (controle de ambiente real)."""
    empty_dir.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("PATH", "")
    os.environ["PATH"] = str(empty_dir)
    try:
        yield
    finally:
        os.environ["PATH"] = previous


def _assert_verdict_precedence(response: Dict[str, object]) -> None:
    """Invariante global: ``passed`` e ``verdict_state`` nunca se contradizem."""
    passed = bool(response["passed"])
    verdict = str(response["verdict_state"])
    assert verdict in {state.value for state in VerdictState}
    if passed:
        assert verdict in _PASSING_VERDICTS, (
            f"passed=True com veredito não-aprovador: {verdict}"
        )
    else:
        assert verdict in _FAILING_VERDICTS, (
            f"passed=False com veredito aprovador: {verdict}"
        )
    if verdict == VerdictState.VERIFIED_PASS.value:
        assert response["evidence_executed"] is True
        assert passed is True


# ──────────────────────────────────────────────────────────────────────────────
# 1. Contrato do veredito explícito + precedência
# ──────────────────────────────────────────────────────────────────────────────

def test_verdict_state_enum_is_complete_and_honest():
    assert {state.value for state in VerdictState} == {
        "VERIFIED_PASS",
        "VERIFIED_FAIL",
        "NOT_APPLICABLE",
        "INCONCLUSIVE",
        "EMERGENCY_RELEASE",
    }
    default = NeedleVerifyDeliveryResponse(
        passed=False,
        status="REJECTED",
        attempt_number=1,
        pytest_success=False,
        pytest_chunks=[],
        blocking_ast_violations=[],
        structural_ast_warnings=[],
        total_files_scanned=0,
        total_lines_scanned=0,
        message="sem evidência",
    )
    assert default.verdict_state == VerdictState.INCONCLUSIVE.value
    assert default.evidence_executed is False
    assert default.layers_executed == []
    assert default.layers_skipped == []
    assert default.skip_reasons == {}
    assert default.layer_status == {}
    assert default.inconclusive_reasons == []


@pytest.mark.parametrize(
    ("passed", "evidence", "expected_passed", "expected_verdict"),
    [
        (True, True, True, "VERIFIED_PASS"),
        (False, True, False, "VERIFIED_FAIL"),
        (True, False, False, "INCONCLUSIVE"),
        (False, False, False, "INCONCLUSIVE"),
    ],
)
def test_build_test_result_never_approves_without_evidence(
    passed: bool, evidence: bool, expected_passed: bool, expected_verdict: str,
):
    """Invariante `_build_test_result`: passed=True exige evidence=True."""
    result = MultiStackRunner._build_test_result(
        passed=passed,
        total_passed=2 if evidence else 0,
        total_failed=0,
        summary="cenário",
        failure_details=[],
        start=time.time(),
        evidence=evidence,
    )
    assert result["passed"] is expected_passed
    assert result["verdict_state"] == expected_verdict
    assert result["evidence"] is evidence


def test_build_test_result_degrades_forged_verified_pass():
    """Pedir VERIFIED_PASS sem evidência é degradado para INCONCLUSIVE."""
    result = MultiStackRunner._build_test_result(
        passed=True,
        total_passed=0,
        total_failed=0,
        summary="forjado",
        failure_details=[],
        start=time.time(),
        evidence=False,
        verdict_state="VERIFIED_PASS",
    )
    assert result["passed"] is False
    assert result["verdict_state"] == VerdictState.INCONCLUSIVE.value


def test_clean_python_repo_is_verified_pass_with_real_evidence(tmp_path: Path):
    """Repositório Python puro: suíte real verde => VERIFIED_PASS com evidência."""
    _python_clean_workspace(tmp_path)

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["passed"] is True, response["message"]
    assert response["status"] == "APPROVED"
    assert response["verdict_state"] == VerdictState.VERIFIED_PASS.value
    assert response["evidence_executed"] is True
    assert response["pytest_success"] is True
    assert LAYER_TESTS in response["layers_executed"]
    assert LAYER_TESTS not in response["layers_skipped"]
    assert response["layer_status"][LAYER_TESTS] == "executed"
    assert response["pytest_chunks"] and response["pytest_chunks"][0]["total_passed"] >= 1
    _assert_verdict_precedence(response)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Repositórios que devem ser reprovados (sem evidência / testes falhando)
# ──────────────────────────────────────────────────────────────────────────────

def test_python_repo_without_tests_is_rejected_verified_fail(tmp_path: Path):
    """Workspace Python sem testes: VERIFIED_FAIL, nunca passed=True com 0 testes."""
    _write(tmp_path, "pyproject.toml", '[project]\nname = "sem-testes"\n')
    _write(tmp_path, "src/calc.py", "def add(a: int, b: int) -> int:\n    return a + b\n")

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["passed"] is False
    assert response["status"] == "REJECTED"
    assert response["verdict_state"] == VerdictState.VERIFIED_FAIL.value
    assert response["evidence_executed"] is False
    assert response["pytest_success"] is False
    assert response["layers_executed"] == [LAYER_TESTS]
    assert response["layer_status"][LAYER_TESTS] == "failed"
    assert set(response["layers_skipped"]) == set(_ALL_LAYERS) - {LAYER_TESTS}
    for layer in response["layers_skipped"]:
        assert response["skip_reasons"].get(layer), f"camada {layer} sem motivo de skip"
    _assert_verdict_precedence(response)


def test_failing_python_suite_fails_fast_with_honest_coverage(tmp_path: Path):
    """Suíte Python real falhando: VERIFIED_FAIL + cobertura explícita de fail-fast.

    A cobertura não pode fingir que as camadas de AST rodaram: elas precisam
    aparecer em ``layers_skipped`` com motivo, mesmo que a lista de violações
    esteja vazia por fail-fast.
    """
    _write(tmp_path, "pyproject.toml", '[project]\nname = "suite-vermelha"\n')
    _write(tmp_path, "src/calc.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write(
        tmp_path,
        "tests/test_calc.py",
        "from src.calc import add\n\n\n"
        "def test_add_quebrado() -> None:\n"
        "    assert add(2, 2) == 5\n",
    )

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["passed"] is False
    assert response["status"] == "REJECTED"
    assert response["verdict_state"] == VerdictState.VERIFIED_FAIL.value
    assert response["pytest_success"] is False
    assert response["evidence_executed"] is True
    assert response["pytest_chunks"], "chunks reais do pytest devem ser reportados"
    assert response["pytest_chunks"][0]["total_failed"] >= 1

    assert response["layers_executed"] == [LAYER_TESTS]
    assert response["layer_status"][LAYER_TESTS] == "failed"
    assert set(response["layers_skipped"]) == set(_ALL_LAYERS) - {LAYER_TESTS}
    for layer in response["layers_skipped"]:
        assert response["skip_reasons"].get(layer), f"camada {layer} sem motivo de skip"
    assert f"{LAYER_AST_GUARD} [NOT_EXECUTED (fail-fast)]" in response["message"]
    assert response["blocking_ast_violations"] == []
    _assert_verdict_precedence(response)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Stack desconhecida e toolchain ausente => INCONCLUSIVE
# ──────────────────────────────────────────────────────────────────────────────

def test_unknown_repository_is_inconclusive_never_approved(tmp_path: Path):
    """Workspace sem capability de stack: INCONCLUSIVE com todas as camadas puladas."""
    _write(tmp_path, "notes.txt", "sem manifesto de linguagem\n")

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["passed"] is False
    assert response["status"] == "REJECTED"
    assert response["verdict_state"] == VerdictState.INCONCLUSIVE.value
    assert response["evidence_executed"] is False
    assert response["layers_executed"] == []
    # TICKET-SGA-11 F3: camadas não aplicáveis não são omissões de cobertura.
    assert response["layers_skipped"] == []
    assert set(response["layers_not_applicable"]) == set(_ALL_LAYERS)
    assert all(state == "not_applicable" for state in response["layer_status"].values())
    assert response["inconclusive_reasons"]
    _assert_verdict_precedence(response)


def test_missing_toolchain_is_inconclusive_and_does_not_fake_execution(tmp_path: Path):
    """DART_FLUTTER sem SDK: INCONCLUSIVE; camada de testes não pode constar executada."""
    _write(tmp_path, "pubspec.yaml", "name: demo_app\nversion: 0.0.1\n")
    empty_path = tmp_path / "_empty_path"

    with _path_without_toolchain(empty_path):
        assert shutil.which("dart") is None
        assert shutil.which("flutter") is None
        result = MultiStackRunner.run_tests(
            tmp_path, "DART_FLUTTER", tmp_path / "_needle_tmp"
        )
        response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert result["passed"] is False
    assert result["verdict_state"] == VerdictState.INCONCLUSIVE.value
    assert result["evidence"] is False
    assert result["reason"] == "dart_ausente"

    assert response["passed"] is False
    assert response["verdict_state"] == VerdictState.INCONCLUSIVE.value
    assert response["evidence_executed"] is False
    # Sem evidência executada, a camada de testes NÃO pode ser anunciada como executada.
    assert LAYER_TESTS not in response["layers_executed"]
    assert LAYER_TESTS in response["layers_skipped"]
    _assert_verdict_precedence(response)


def test_circuit_breaker_emergency_release_is_the_only_no_evidence_pass(tmp_path: Path):
    """TICKET-SGA-11 F4: circuit breaker NUNCA aprova sem evidência executada."""
    _python_clean_workspace(tmp_path)
    from agent_quality_guard.core.score_gate import history_set

    history_set(str(tmp_path), {"passed": False, "attempts": 4})

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["passed"] is False
    assert response["status"] == "EMERGENCY_RELEASE"
    assert response["verdict_state"] == VerdictState.INCONCLUSIVE.value
    assert response["approved_with_caveat"] is False
    assert response["evidence_executed"] is False
    assert response["layers_executed"] == []
    assert set(response["layers_skipped"]) == set(_ALL_LAYERS)
    assert all(state == "emergency_release" for state in response["layer_status"].values())
    assert "Laudo de Inviabilidade" in response["message"]
    _assert_verdict_precedence(response)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Detecção por capability (zero nomes pessoais) + poliglota
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    ("files", "expected_stack"),
    [
        (("pyproject.toml",), "PYTHON_GENERIC"),
        (("requirements.txt",), "PYTHON_GENERIC"),
        (("setup.py",), "PYTHON_GENERIC"),
        (("Pipfile",), "PYTHON_GENERIC"),
        (("Cargo.toml",), "RUST_CARGO"),
        (("go.mod",), "GO_MODULE"),
        (("pubspec.yaml",), "DART_FLUTTER"),
    ],
)
def test_detect_stack_by_capability_manifest(
    tmp_path: Path, files: Tuple[str, ...], expected_stack: str,
):
    for name in files:
        _write(tmp_path, name, "")
    assert MultiStackRunner.detect_stack(tmp_path) == expected_stack


def test_detect_stack_node_variants_by_capability(tmp_path: Path):
    _write(tmp_path, "package.json", json.dumps({"name": "plain"}))
    assert MultiStackRunner.detect_stack(tmp_path) == "NODE_JS_GENERIC"

    _write(
        tmp_path,
        "package.json",
        json.dumps({"name": "rn", "dependencies": {"react-native": "0.76.0"}}),
    )
    assert MultiStackRunner.detect_stack(tmp_path) == "REACT_NATIVE_EXPO"

    _write(tmp_path, "package.json", json.dumps({"name": "modern"}))
    _write(tmp_path, "bun.lock", "")
    assert MultiStackRunner.detect_stack(tmp_path) == "NODE_TS_BUN"


def test_detect_stack_empty_is_unknown(tmp_path: Path):
    assert MultiStackRunner.detect_stack(tmp_path) == "UNKNOWN"


def test_polyglot_repository_prefers_python_and_runs_python_suite(tmp_path: Path):
    """Poliglota (pyproject + package.json + Cargo.toml): Python tem precedência.

    O script Node declara `exit 1`; se o runner escolhesse Node, o resultado
    seria VERIFIED_FAIL. A suíte Python real passa => prova de precedência real.
    """
    _write(tmp_path, "pyproject.toml", '[project]\nname = "poly"\n')
    _write(
        tmp_path,
        "package.json",
        json.dumps({"name": "poly", "scripts": {"test": "exit 1"}}),
    )
    _write(tmp_path, "Cargo.toml", '[package]\nname = "poly"\nversion = "0.1.0"\n')
    _write(tmp_path, "src/calc.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    _write(
        tmp_path,
        "tests/test_calc.py",
        "from src.calc import add\n\n\n"
        "def test_add() -> None:\n"
        "    assert add(2, 3) == 5\n",
    )
    tmp_dir = tmp_path / "_needle_tmp"
    tmp_dir.mkdir()

    assert MultiStackRunner.detect_stack(tmp_path) == "PYTHON_GENERIC"
    result = MultiStackRunner.run_tests(tmp_path, "PYTHON_GENERIC", tmp_dir)
    assert result["passed"] is True, result["summary"]
    assert result["verdict_state"] == VerdictState.VERIFIED_PASS.value
    assert result["runner"] == "pytest"
    assert result["total_passed"] >= 1
    assert result["evidence"] is True


def test_monorepo_root_guard_runs_each_package_and_never_root(tmp_path: Path):
    """Monorepo Bun/TS: guarda 'do not run tests from root' => suíte por pacote.

    O script de teste da raiz grava um marcador e sai com erro. Se o runner
    executasse `npm test` na raiz, o marcador existiria e o resultado seria
    VERIFIED_FAIL. O esperado é somar os testes reais dos dois pacotes.
    """
    _write(
        tmp_path,
        "package.json",
        json.dumps(
            {
                "name": "monorepo-root",
                "private": True,
                "workspaces": ["packages/*"],
                "scripts": {
                    "test": (
                        "sh -c 'echo RAN > root_test_ran.txt; "
                        "echo do not run tests from root; exit 1'"
                    )
                },
            }
        ),
    )
    for name, suites in (("alpha", 1), ("beta", 2)):
        package_dir = tmp_path / "packages" / name
        _write(
            package_dir,
            "package.json",
            json.dumps(
                {
                    "name": name,
                    "type": "module",
                    "scripts": {"test": "node test.mjs"},
                }
            ),
        )
        _write(
            package_dir,
            "test.mjs",
            "import assert from 'node:assert';\n"
            "assert.strictEqual(1 + 1, 2);\n"
            f"console.log('SUÍTES: {suites}/{suites} passaram');\n",
        )

    tmp_dir = tmp_path / "_needle_tmp"
    tmp_dir.mkdir()

    assert MultiStackRunner.detect_stack(tmp_path) == "NODE_TS_BUN"
    result = MultiStackRunner.run_tests(tmp_path, "NODE_TS_BUN", tmp_dir)

    assert result["passed"] is True, result["summary"]
    assert result["verdict_state"] == VerdictState.VERIFIED_PASS.value
    assert result["runner"] == "workspaces"
    assert result["total_passed"] == 3
    assert result["total_failed"] == 0
    assert result["evidence"] is True
    assert not (tmp_path / "root_test_ran.txt").exists(), (
        "o script da raiz foi executado apesar da guarda 'do not run tests from root'"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Timeout de 600s e expurgo de contadores forjados / nomes pessoais
# ──────────────────────────────────────────────────────────────────────────────

def test_execution_timeouts_are_600_seconds():
    assert _TEST_COMMAND_TIMEOUT_SECONDS == 600.0
    assert _DEFAULT_NATIVE_BUDGET_SECONDS == 600.0
    assert _MAX_NATIVE_BUDGET_SECONDS == 600.0
    assert _MIN_NATIVE_BUDGET_SECONDS == 0.1

    run_all = inspect.signature(PytestChunkRunner.run_all_chunks)
    assert run_all.parameters["timeout_per_chunk"].default == 600


def test_native_budget_env_is_clamped_to_600():
    previous = os.environ.get("JEV_NATIVE_BRIDGE_BUDGET_SECONDS")
    try:
        os.environ["JEV_NATIVE_BRIDGE_BUDGET_SECONDS"] = "9999"
        assert MultiStackRunner._native_budget_seconds() == 600.0
        os.environ["JEV_NATIVE_BRIDGE_BUDGET_SECONDS"] = "0"
        assert MultiStackRunner._native_budget_seconds() == 0.1
        os.environ["JEV_NATIVE_BRIDGE_BUDGET_SECONDS"] = "invalido"
        assert MultiStackRunner._native_budget_seconds() == 600.0
        os.environ.pop("JEV_NATIVE_BRIDGE_BUDGET_SECONDS", None)
        assert MultiStackRunner._native_budget_seconds() == 600.0
    finally:
        if previous is None:
            os.environ.pop("JEV_NATIVE_BRIDGE_BUDGET_SECONDS", None)
        else:
            os.environ["JEV_NATIVE_BRIDGE_BUDGET_SECONDS"] = previous


def test_detection_and_runner_have_no_personal_repo_fingerprints():
    """A detecção/runner não pode citar repositórios pessoais (capability apenas)."""
    forbidden = (
        "PanelaAndroid",
        "predictera",
        "atelie",
        "sync_kanban",
        "PYTHON_JEV",
        "PREDICTERA",
        "FATURAMENTO",
    )
    sources = (
        REPO_ROOT / "agent_quality_guard" / "orchestrator" / "multi_stack_runner.py",
        REPO_ROOT / "agent_quality_guard" / "orchestrator" / "pytest_chunk_runner.py",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{name!r} ainda aparece em {source.name}"


def test_runner_has_no_forged_counters():
    """Contadores inventados (336/514/+=5) não podem existir no runner."""
    text = (REPO_ROOT / "agent_quality_guard" / "orchestrator" / "multi_stack_runner.py").read_text(
        encoding="utf-8"
    )
    assert "336" not in text
    assert "514" not in text
    assert "+= 5" not in text
    assert "_run_faturamento_suite" not in text


def test_coverage_helper_never_leaves_a_layer_implicit():
    coverage = _layer_coverage([LAYER_TESTS])
    assert coverage["layers_executed"] == [LAYER_TESTS]
    assert set(coverage["layers_skipped"]) == set(_ALL_LAYERS) - {LAYER_TESTS}
    for layer in coverage["layers_skipped"]:
        assert coverage["skip_reasons"][layer]
    assert coverage["layer_status"][LAYER_TESTS] == "executed"
    assert coverage["layer_status"][LAYER_AST_GUARD] == "not_executed"


def test_pytest_runner_discovers_nested_and_root_tests(tmp_path: Path):
    """Descoberta poliglota/tolerante: tests/unit, subpacotes e raiz."""
    _write(tmp_path, "tests/unit/test_nested.py", "def test_nested() -> None:\n    assert True\n")
    _write(tmp_path, "test_root_level.py", "def test_root() -> None:\n    assert True\n")
    _write(tmp_path, "tests/verify_all.py", "raise SystemExit(1)\n")

    chunks = PytestChunkRunner.discover_and_partition_tests(str(tmp_path), audit_mode="full")
    discovered = [path for _, files in chunks for path in files]

    assert "tests/unit/test_nested.py" in discovered
    assert "test_root_level.py" in discovered
    assert all("verify_all.py" not in path for path in discovered)
