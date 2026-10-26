"""Matriz determinística do TICKET-SGA-11 (cobertura e honestidade de diagnóstico).

Cobre os cinco defeitos agrupados do ``agent_quality_guard`` v0.2.0:

F1. ``failure_details``/``summary`` atribuem a falha ao teste real (Bun, node
    ``--test``/TAP e pytest), com asserção e anexo bruto;
F2. modo explícito de diagnóstico estático roda as camadas estáticas com a
    Camada 0 vermelha e NUNCA aprova;
F3. ``layers_not_applicable`` separado de ``layers_skipped``;
F4. ``EMERGENCY_RELEASE`` nunca retorna ``passed=True`` sem evidência executada;
F5. zeros literais substituídos por ``None`` (não medido) ou contagem real.

Sem mocks de regra de negócio: as saídas dos runners usadas como fixture são
capturas textuais reais, e o restante materializa workspaces/subprocessos reais.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_quality_guard.core.schemas import VerdictState
from agent_quality_guard.mcp_server import (
    LAYER_AST_GUARD,
    LAYER_SCREENER,
    LAYER_TESTS,
    _ALL_LAYERS,
    _layer_coverage,
    _quality_guard_core,
)
from agent_quality_guard.orchestrator.multi_stack_runner import MultiStackRunner


def _write(root: Path, rel: str, content: str) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def _python_workspace(root: Path, *, failing: bool = False, danger: bool = False) -> Path:
    _write(root, "pyproject.toml", '[project]\nname = "sga11-py"\nversion = "0.0.1"\n')
    _write(root, "src/calc.py", "def add(a: int, b: int) -> int:\n    return a + b\n")
    if danger:
        _write(root, "src/danger.py", 'def run(value: str):\n    return eval(value)\n')
    _write(
        root,
        "tests/test_calc.py",
        "from src.calc import add\n\n\n"
        "def test_add() -> None:\n"
        f"    assert add(2, 2) == {5 if failing else 4}\n",
    )
    return root


def _node_workspace(root: Path, *, suite_exit: int = 0, prod: dict[str, str] | None = None) -> Path:
    _write(root, "package.json", json.dumps({"name": "sga11-app", "type": "module"}))
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


# ──────────────────────────────────────────────────────────────────────────────
# F1 — atribuição real de falhas (fixtures = saídas reais capturadas)
# ──────────────────────────────────────────────────────────────────────────────

BUN_FAILURE_OUTPUT = """\
src/hardening.test.mjs:
(pass) DREAMS-01 · permissões [0.12ms]
(fail) DREAMS-07 · hardening POSIX · UMASK ADVERSARIAL (0o777) [1.20ms]
  Expected: 384
  Received: 511
(pass) DREAMS-08 · ruído final do reporter [0.03ms]

 3 pass
 1 fail
"""

NODE_TAP_FAILURE_OUTPUT = """\
not ok 1 - reconciliação parseável
  ---
  error: 'JSON inválido'
not ok 2 - segunda falha
# pass 10
# fail 2
"""

PYTEST_FAILURE_OUTPUT = """\
FAILED tests/test_reconcile.py::test_json_valido - AssertionError: assert None == 'ok'
FAILED tests/test_other.py::test_x - assert 1 == 2
1 passed, 2 failed
"""


def test_failure_details_attribute_bun_failure_not_tail_noise():
    details = MultiStackRunner._failure_details_for(BUN_FAILURE_OUTPUT, "bun")

    assert details[0].startswith("Testes com falha:")
    assert "UMASK ADVERSARIAL" in details[0]
    assert any("Expected" in line or "Received" in line for line in details[1:])
    assert "Trecho bruto da saída (anexo)" in details[-1]
    # O ruído final não pode ser a única informação.
    assert "ruído final" not in details[0]


def test_failure_summary_lists_up_to_three_real_test_names():
    names = MultiStackRunner._extract_failed_test_names(BUN_FAILURE_OUTPUT)
    assert names == ["DREAMS-07 · hardening POSIX · UMASK ADVERSARIAL (0o777)"]

    summary = MultiStackRunner._summary_with_failures("bun: 1 falha", BUN_FAILURE_OUTPUT)
    assert "testes com falha:" in summary
    assert "DREAMS-07" in summary


def test_failure_extraction_supports_node_tap_and_pytest():
    tap_names = MultiStackRunner._extract_failed_test_names(NODE_TAP_FAILURE_OUTPUT)
    assert tap_names == ["reconciliação parseável", "segunda falha"]

    pytest_names = MultiStackRunner._extract_failed_test_names(PYTEST_FAILURE_OUTPUT)
    assert pytest_names == [
        "tests/test_reconcile.py::test_json_valido",
        "tests/test_other.py::test_x",
    ]


def test_failure_details_handle_multiple_and_unattributed_failures():
    details = MultiStackRunner._failure_details_for(PYTEST_FAILURE_OUTPUT, "pytest")
    assert "tests/test_reconcile.py::test_json_valido" in details[0]
    assert "tests/test_other.py::test_x" in details[0]
    assert "AssertionError" in details[0] or any(
        "AssertionError" in line for line in details[1:]
    )

    unattributed = MultiStackRunner._failure_details_for("comando terminou com rc=1", "tool")
    assert "sem nome de teste identificável" in unattributed[0]


def test_node_command_failure_carries_attribution_end_to_end(tmp_path: Path):
    """Execução real de subprocesso: contagem parseada + atribuição no payload."""
    script = (
        "printf '%s\\n' "
        "'(fail) soma quebrada [0.1ms]' "
        "'  Expected: 5' "
        "'  Received: 4' "
        "' 1 pass' "
        "' 1 fail'; exit 1"
    )

    result = MultiStackRunner._run_node_command(
        ["sh", "-c", script], tmp_path, {}, 0.0, "bun",
    )

    assert result["passed"] is False
    assert result["total_passed"] == 1
    assert result["total_failed"] == 1
    assert "soma quebrada" in result["summary"]
    assert result["failure_details"][0].startswith("Testes com falha:")
    assert any("Expected" in line for line in result["failure_details"])


# ──────────────────────────────────────────────────────────────────────────────
# F2 — diagnóstico estático explícito (nunca aprova)
# ──────────────────────────────────────────────────────────────────────────────

def test_default_mode_keeps_fail_fast_with_zero_static_layers(tmp_path: Path):
    _python_workspace(tmp_path, failing=True)

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path), audit_mode="full"))

    assert response["passed"] is False
    assert response["layers_executed"] == [LAYER_TESTS]
    assert LAYER_AST_GUARD not in response["layers_executed"]
    assert f"{LAYER_AST_GUARD} [NOT_EXECUTED (fail-fast)]" in response["message"]


def test_static_diagnostic_runs_static_layers_but_never_approves(tmp_path: Path):
    _python_workspace(tmp_path, failing=True)

    response = asyncio.run(
        _quality_guard_core(
            workspace=str(tmp_path), audit_mode="full", static_diagnostic=True,
        )
    )

    assert response["passed"] is False
    assert response["status"] == "REJECTED"
    assert response["verdict_state"] == VerdictState.VERIFIED_FAIL.value
    assert response["evidence_executed"] is True
    assert LAYER_TESTS in response["layers_executed"]
    assert LAYER_AST_GUARD in response["layers_executed"]
    assert LAYER_AST_GUARD not in response["layers_skipped"]
    assert "STATIC DIAGNOSTIC" in response["message"]


def test_static_diagnostic_attaches_blocking_findings(tmp_path: Path):
    _python_workspace(tmp_path, failing=True, danger=True)

    response = asyncio.run(
        _quality_guard_core(
            workspace=str(tmp_path), audit_mode="full", static_diagnostic=True,
        )
    )

    assert response["passed"] is False
    assert response["blocking_ast_violations"], "achado estático bloqueante não anexado"
    assert response["verdict_state"] == VerdictState.VERIFIED_FAIL.value


def test_static_diagnostic_js_attaches_layer_findings(tmp_path: Path):
    _node_workspace(
        tmp_path, suite_exit=1,
        prod={"src/leak.js": "export function f(){\n  setInterval(()=>go(), 10);\n}\n"},
    )

    response = asyncio.run(
        _quality_guard_core(
            workspace=str(tmp_path), audit_mode="full", static_diagnostic=True,
        )
    )

    assert response["passed"] is False
    assert response["verdict_state"] in {
        VerdictState.VERIFIED_FAIL.value, VerdictState.INCONCLUSIVE.value,
    }
    assert any(v["rule_id"] == "RESOURCE_LEAK" for v in response["layer2_meta_rule_violations"])
    assert "STATIC DIAGNOSTIC" in response["message"]


def test_static_diagnostic_via_audit_mode_value(tmp_path: Path):
    _python_workspace(tmp_path, failing=True)

    response = asyncio.run(
        _quality_guard_core(workspace=str(tmp_path), audit_mode="static_diagnostic")
    )

    assert response["passed"] is False
    assert LAYER_AST_GUARD in response["layers_executed"]


# ──────────────────────────────────────────────────────────────────────────────
# F3 — layers_not_applicable separado de layers_skipped
# ──────────────────────────────────────────────────────────────────────────────

def test_layer_coverage_separates_not_applicable_from_skipped():
    coverage = _layer_coverage(
        [LAYER_TESTS],
        statuses={LAYER_TESTS: "executed", LAYER_AST_GUARD: "not_applicable"},
        skip_reasons={LAYER_AST_GUARD: "Python-only"},
    )

    assert coverage["layers_executed"] == [LAYER_TESTS]
    assert coverage["layers_not_applicable"] == [LAYER_AST_GUARD]
    assert LAYER_AST_GUARD not in coverage["layers_skipped"]
    assert LAYER_AST_GUARD not in coverage["skip_reasons"]
    assert coverage["not_applicable_reasons"][LAYER_AST_GUARD] == "Python-only"
    assert LAYER_SCREENER in coverage["layers_skipped"]


def test_js_run_reports_python_layers_as_not_applicable(tmp_path: Path):
    _node_workspace(tmp_path)

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path), audit_mode="full"))

    assert response["passed"] is True, response["message"]
    assert set(response["layers_not_applicable"]) == {LAYER_AST_GUARD, LAYER_SCREENER}
    assert LAYER_AST_GUARD not in response["layers_skipped"]
    assert LAYER_SCREENER not in response["layers_skipped"]
    assert set(response["not_applicable_reasons"]) == {LAYER_AST_GUARD, LAYER_SCREENER}


def test_unknown_stack_marks_all_layers_not_applicable(tmp_path: Path):
    _write(tmp_path, "notes.txt", "sem manifesto\n")

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["passed"] is False
    assert set(response["layers_not_applicable"]) == set(_ALL_LAYERS)
    assert response["layers_skipped"] == []


# ──────────────────────────────────────────────────────────────────────────────
# F4 — circuito terminal nunca aprova sem evidência
# ──────────────────────────────────────────────────────────────────────────────

def test_circuit_breaker_never_approves_without_evidence(tmp_path: Path):
    _python_workspace(tmp_path, failing=True)

    responses = [
        asyncio.run(_quality_guard_core(workspace=str(tmp_path)))
        for _ in range(5)
    ]

    assert all(r["passed"] is False for r in responses), [
        r["status"] for r in responses
    ]
    terminal = responses[-1]
    assert terminal["status"] == "EMERGENCY_RELEASE"
    assert terminal["evidence_executed"] is False
    assert terminal["approved_with_caveat"] is False
    assert terminal["verdict_state"] == VerdictState.INCONCLUSIVE.value
    assert "Laudo de Inviabilidade" in terminal["message"]


# ──────────────────────────────────────────────────────────────────────────────
# F5 — zeros literais só quando medido
# ──────────────────────────────────────────────────────────────────────────────

def test_fail_fast_totals_are_none_not_zero(tmp_path: Path):
    _python_workspace(tmp_path, failing=True)

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["passed"] is False
    assert response["total_files_scanned"] is None
    assert response["total_lines_scanned"] is None


def test_js_approved_reports_real_scanned_totals(tmp_path: Path):
    _node_workspace(
        tmp_path,
        prod={
            "src/a.js": "export const a = 1;\n",
            "src/b.js": "export const b = 2;\n\nexport const c = 3;\n",
        },
    )

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path), audit_mode="full"))

    assert response["passed"] is True, response["message"]
    assert response["total_files_scanned"] >= 2
    assert response["total_lines_scanned"] >= 3


def test_unknown_stack_totals_are_none(tmp_path: Path):
    _write(tmp_path, "notes.txt", "sem manifesto\n")

    response = asyncio.run(_quality_guard_core(workspace=str(tmp_path)))

    assert response["total_files_scanned"] is None
    assert response["total_lines_scanned"] is None
