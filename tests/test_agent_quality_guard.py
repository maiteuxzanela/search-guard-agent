"""Suíte determinística da tool canônica ``agent_quality_guard`` (frente A1 — Túlio).

TICKET-HARNESS-43: rebatismo do quality gate para ``agent_quality_guard`` com o
alias retrocompatível ``needle_verify_delivery`` (MESMA implementação), pipeline
100% determinístico (Laya/Noul fora do caminho crítico), circuit breaker de 5
tentativas com Saída de Emergência e sem qualquer chamada pesada de GPU.

Sem mocks: as camadas são exercitadas sobre arquivos reais em diretórios
temporários e o núcleo de homologação roda a suíte de testes real do workspace.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from agent_quality_guard.mcp_server import (
    _coerce_violations,
    _quality_guard_core,
    _run_layer1_native_bridge,
    _run_layer2_meta_rules,
    agent_quality_guard,
    app,
    needle_verify_delivery,
)
from agent_quality_guard.core.schemas import (
    AgentQualityGuardRequest,
    AgentQualityGuardResponse,
    ASTViolation,
    LinterDiagnostic,
    MetaRuleViolation,
    META_RULE_BLOCKING_CATEGORIES,
    NATIVE_TOOLCHAIN_BLOCKING_CATEGORIES,
    NeedleVerifyDeliveryRequest,
    NeedleVerifyDeliveryResponse,
    ToolchainToolStatus,
)


# ──────────────────────────────────────────────────────────────────────────────
# Registro canônico + alias (mesma implementação e mesma assinatura)
# ──────────────────────────────────────────────────────────────────────────────

def test_canonical_tool_and_alias_share_same_implementation():
    assert agent_quality_guard is needle_verify_delivery
    tools = app._tool_manager._tools
    assert "agent_quality_guard" in tools
    assert "needle_verify_delivery" in tools
    assert tools["agent_quality_guard"].fn is tools["needle_verify_delivery"].fn


def test_canonical_and_alias_input_schemas_are_identical():
    tools = app._tool_manager._tools
    canonical = tools["agent_quality_guard"]
    alias = tools["needle_verify_delivery"]
    assert canonical.parameters == alias.parameters
    assert set(canonical.parameters["properties"]) == {
        "task_type", "workspace", "attempt_number", "pedido", "resultado", "engine",
        "audit_mode", "static_diagnostic",
    }
    assert "SEM ML" in canonical.description or "determinística" in canonical.description
    assert "Alias legado" in alias.description


def test_tool_catalog_contains_canonical_and_alias():
    from agent_quality_guard.core.tools import get_tool_definition

    canonical = get_tool_definition("agent_quality_guard")
    alias = get_tool_definition("needle_verify_delivery")
    assert canonical is not None and alias is not None
    assert canonical["function"]["name"] == "agent_quality_guard"
    assert alias["function"]["name"] == "needle_verify_delivery"
    assert canonical["function"]["parameters"] == alias["function"]["parameters"]


# ──────────────────────────────────────────────────────────────────────────────
# Contratos de schema
# ──────────────────────────────────────────────────────────────────────────────

def test_request_schema_is_alias_and_rejects_attempt_above_five():
    from pydantic import ValidationError

    assert issubclass(AgentQualityGuardRequest, NeedleVerifyDeliveryRequest)
    assert AgentQualityGuardRequest(attempt_number=5).attempt_number == 5
    with pytest.raises(ValidationError):
        AgentQualityGuardRequest(attempt_number=6)


_LEGACY_REQUIRED = {
    "passed": True,
    "status": "APPROVED",
    "attempt_number": 1,
    "pytest_success": True,
    "pytest_chunks": [],
    "blocking_ast_violations": [],
    "structural_ast_warnings": [],
    "total_files_scanned": 0,
    "total_lines_scanned": 0,
    "message": "ok",
}


def test_response_superset_retrocompat_and_defaults():
    assert issubclass(AgentQualityGuardResponse, NeedleVerifyDeliveryResponse)

    response = AgentQualityGuardResponse(**_LEGACY_REQUIRED)
    assert response.tool_name == "agent_quality_guard"
    assert response.layer1_toolchain_diagnostics == []
    assert response.toolchain_status == []
    assert response.layer2_meta_rule_violations == []
    assert response.confirmed_defects == []
    assert response.dismissed_warnings == []

    payload = response.model_dump()
    # chaves legadas continuam presentes para clientes antigos
    for legacy_key in ("passed", "status", "attempt_number", "pytest_success", "message"):
        assert legacy_key in payload
    for new_key in (
        "tool_name", "layer1_toolchain_diagnostics", "toolchain_status", "layer2_meta_rule_violations",
    ):
        assert new_key in payload


def test_response_accepts_legacy_payload_without_new_fields():
    """Payload legado (só campos de NeedleVerifyDeliveryResponse) é aceito; extras têm default."""
    response = AgentQualityGuardResponse(**_LEGACY_REQUIRED)
    assert response.passed is True
    assert response.tool_name == "agent_quality_guard"
    assert response.layer2_meta_rule_violations == []


def test_new_violation_models_are_typed_ast_violations():
    assert issubclass(LinterDiagnostic, ASTViolation)
    assert issubclass(MetaRuleViolation, ASTViolation)
    linter = LinterDiagnostic(
        rule_id="RUFF_F401", category="lint_error", file_path="src/a.py",
        line_number=1, code_snippet="import os", message="unused", tool="ruff", severity="error",
    )
    assert linter.tool == "ruff" and linter.severity == "error"
    meta = MetaRuleViolation(
        rule_id="DANGEROUS_EVAL", category="security", file_path="src/a.js",
        line_number=2, code_snippet="eval(x)", message="eval", severity="blocking", language="javascript",
    )
    assert meta.severity == "blocking" and meta.language == "javascript"
    status = ToolchainToolStatus(name="ruff", available=True)
    assert status.name == "ruff" and status.skipped is False and status.duration_ms == 0.0


def test_blocking_category_constants():
    assert NATIVE_TOOLCHAIN_BLOCKING_CATEGORIES == ["type_error", "lint_error"]
    assert META_RULE_BLOCKING_CATEGORIES == ["anti_mock_producao", "security"]


# ──────────────────────────────────────────────────────────────────────────────
# Coerção fail-soft das violações externas
# ──────────────────────────────────────────────────────────────────────────────

def test_coerce_violations_passthrough_dicts_and_ignores_bad_shapes():
    already = ASTViolation(
        rule_id="SECURITY_EVAL", category="security", file_path="a.js",
        line_number=1, code_snippet="eval(x)", message="m",
    )
    coerced = _coerce_violations(
        [
            already,
            {
                "rule_id": "RUFF_F401", "category": "lint_error", "file_path": "a.py",
                "line_number": 1, "code_snippet": "import os", "message": "m",
            },
            {"campo": "irrelevante"},
            "string-invalida",
            None,
        ],
        LinterDiagnostic,
    )
    assert len(coerced) == 2
    assert coerced[0] is already
    assert coerced[1].rule_id == "RUFF_F401"


# ──────────────────────────────────────────────────────────────────────────────
# Camada 2 no gate (meta-regras Tree-Sitter)
# ──────────────────────────────────────────────────────────────────────────────

def _workspace_with_meta_rules(root: str) -> None:
    src = Path(root) / "src"
    src.mkdir(parents=True, exist_ok=True)
    (src / "danger.js").write_text("export function f(x){ return eval(x); }\n", encoding="utf-8")
    (src / "leak.js").write_text("export function g(){ setInterval(()=>h(), 10); }\n", encoding="utf-8")


def test_layer2_meta_rules_classify_blocking_and_warning(tmp_path: Path):
    _workspace_with_meta_rules(str(tmp_path))
    started = time.monotonic()
    result = _run_layer2_meta_rules(tmp_path, "full")
    elapsed = time.monotonic() - started

    assert result["available"] is True
    assert {v["rule_id"] for v in result["blocking"]} == {"DANGEROUS_EVAL"}
    assert {v["rule_id"] for v in result["warnings"]} == {"RESOURCE_LEAK"}
    # Runners de CI compartilhados são 2-5x mais lentos que a máquina local:
    # o contrato de subsegundo vale localmente, com teto relaxado no CI.
    max_elapsed = 3.0 if os.environ.get("CI") else 1.0
    assert elapsed < max_elapsed, f"Camada 2 levou {elapsed:.3f}s (contrato < {max_elapsed}s)"


def test_layer2_meta_rules_missing_workspace_reports_coverage(tmp_path: Path):
    result = _run_layer2_meta_rules(tmp_path / "nao-existe", "full")
    assert {"available", "blocking", "warnings", "total_files"} <= set(result)
    assert result["blocking"] == [] and result["warnings"] == []
    # Cobertura honesta: workspace ausente nunca vira "0 violações" silencioso.
    if result["available"]:
        assert result["layer_status"] == "not_applicable"
        assert result["layers_skipped"] == ["layer4_tree_sitter"]
        assert "workspace inexistente" in result["skip_reasons"]["layer4_tree_sitter"]
    else:
        assert result["layer_status"] in {"not_executed", "not_applicable"}
        assert result["layers_skipped"] == ["layer4_tree_sitter"]
        assert result["skip_reasons"]["layer4_tree_sitter"]


def test_layer2_meta_rules_executed_exposes_coverage(tmp_path: Path):
    _workspace_with_meta_rules(str(tmp_path))
    result = _run_layer2_meta_rules(tmp_path, "full")
    if not result["available"]:
        pytest.skip("tree-sitter ausente neste ambiente (fallback coberto no CI)")
    assert result["layer_status"] == "executed"
    assert result["layers_executed"] == ["layer4_tree_sitter"]
    assert result["layers_skipped"] == []
    assert result["files_scanned"] == result["total_files"]


def test_layer1_and_layer2_helpers_are_subsecond(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("JEV_NATIVE_BRIDGE_BUDGET_SECONDS", "0.5")
    started = time.monotonic()
    layer1 = _run_layer1_native_bridge(tmp_path, "PYTHON_GENERIC", "full")
    layer2 = _run_layer2_meta_rules(tmp_path, "full")
    elapsed = time.monotonic() - started

    assert layer1["available"] is True
    assert layer1["blocking"] == []
    assert layer2["available"] is True
    assert elapsed < 1.0, f"Camadas 1+2 levariam {elapsed:.3f}s"


# ──────────────────────────────────────────────────────────────────────────────
# Núcleo do gate: determinismo, fail-fast, engine ignorado
# ──────────────────────────────────────────────────────────────────────────────

def _build_python_workspace(root: str) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@searchguard.local"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=root, check=True)
    (Path(root) / "README.md").write_text("# Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=root, check=True)

    src = Path(root) / "src"
    src.mkdir()
    (src / "math_ops.py").write_text(
        'def multiply(a: int, b: int) -> int:\n    """Multiplica."""\n    return a * b\n',
        encoding="utf-8",
    )
    tests = Path(root) / "tests"
    tests.mkdir()
    (tests / "test_math_ops.py").write_text(
        "from src.math_ops import multiply\n\n\ndef test_multiply():\n    assert multiply(3, 4) == 12\n",
        encoding="utf-8",
    )


def test_quality_guard_core_approves_clean_workspace_with_canonical_tool_name():
    with tempfile.TemporaryDirectory() as tmp:
        _build_python_workspace(tmp)
        response = asyncio.run(_quality_guard_core(workspace=tmp))

        assert response["passed"] is True
        assert response["status"] == "APPROVED"
        assert response["tool_name"] == "agent_quality_guard"
        assert response["pytest_success"] is True
        assert response["blocking_ast_violations"] == []
        assert response["confirmed_defects"] == []
        assert response["dismissed_warnings"] == []
        assert "layer2_meta_rule_violations" in response
        assert "toolchain_status" in response


def test_quality_guard_core_ignores_engine_argument():
    """O argumento ``engine`` é aceito por retrocompat e NUNCA introspectado (sem ML)."""
    sentinel = object()
    with tempfile.TemporaryDirectory() as tmp:
        _build_python_workspace(tmp)
        with_sentinel = asyncio.run(_quality_guard_core(workspace=tmp, engine=sentinel, attempt_number=1))
        assert with_sentinel["status"] == "APPROVED"
        assert with_sentinel["confirmed_defects"] == []

    source = inspect.getsource(_quality_guard_core)
    assert "del engine" in source
    assert "evaluate_noul" not in source
    assert "_get_engine_async" not in source


def test_quality_guard_core_circuit_breaker_emergency_release():
    with tempfile.TemporaryDirectory() as tmp:
        for expected in range(1, 5):
            response = asyncio.run(_quality_guard_core(workspace=tmp))
            assert response["status"] == "REJECTED"
            assert response["attempt_number"] == expected

        emergency = asyncio.run(_quality_guard_core(workspace=tmp))
        assert emergency["status"] == "EMERGENCY_RELEASE"
        assert emergency["attempt_number"] == 5
        # TICKET-SGA-11 F4: sem evidência executada NUNCA há aprovação.
        assert emergency["passed"] is False
        assert emergency["approved_with_caveat"] is False
        assert emergency["evidence_executed"] is False
        assert emergency["pytest_success"] is False
        assert "STATUS TERMINAL" in emergency["message"]


def test_canonical_and_alias_yield_identical_verdicts():
    """Chamar a tool canônica e o alias em workspaces idênticos dá o mesmo veredito."""
    with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
        _build_python_workspace(tmp_a)
        _build_python_workspace(tmp_b)
        canonical = asyncio.run(agent_quality_guard(workspace=tmp_a))
        alias = asyncio.run(needle_verify_delivery(workspace=tmp_b))

        assert canonical["status"] == alias["status"] == "APPROVED"
        assert canonical["tool_name"] == alias["tool_name"] == "agent_quality_guard"
        assert set(canonical) == set(alias)


# ──────────────────────────────────────────────────────────────────────────────
# Integração end-to-end: Camada 2 alimentando o gate JS
# ──────────────────────────────────────────────────────────────────────────────

def _build_node_workspace(root: str, prod_files: dict[str, str]) -> None:
    workspace = Path(root)
    (workspace / "package.json").write_text('{"name": "app", "type": "module"}', encoding="utf-8")
    (workspace / "tests").mkdir(exist_ok=True)
    (workspace / "tests" / "run_all.mjs").write_text(
        "import assert from 'node:assert';\nassert.ok(true);\nconsole.log('SUÍTES: 1/1 passaram');\n",
        encoding="utf-8",
    )
    for rel, content in prod_files.items():
        target = workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def test_quality_guard_core_surfaces_layer2_warning_as_telemetry():
    with tempfile.TemporaryDirectory() as tmp:
        _build_node_workspace(tmp, {"src/leak.js": "export function f(){\n  setInterval(()=>go(), 10);\n}\n"})
        response = asyncio.run(_quality_guard_core(workspace=tmp, audit_mode="full"))

        assert response["status"] == "APPROVED", response["message"]
        assert any(v["rule_id"] == "RESOURCE_LEAK" for v in response["layer2_meta_rule_violations"])
        assert any(v["rule_id"] == "RESOURCE_LEAK" for v in response["structural_ast_warnings"])
        # telemetria estrutural não bloqueia
        assert response["blocking_ast_violations"] == []
        assert response["confirmed_defects"] == []


def test_quality_guard_core_rejects_blocking_violation_from_layer2_or_guard():
    with tempfile.TemporaryDirectory() as tmp:
        _build_node_workspace(tmp, {"src/danger.js": "export function f(x){ return eval(x); }\n"})
        response = asyncio.run(_quality_guard_core(workspace=tmp))

        assert response["status"] == "REJECTED", response["message"]
        assert response["passed"] is False
        assert response["blocking_ast_violations"], "violação bloqueante ausente"

    # ── Ausência de qualquer dependência pesada de GPU/Laya no caminho crítico ──
    module_path = Path(__file__).resolve().parents[1] / "agent_quality_guard" / "mcp_server.py"
    source = module_path.read_text(encoding="utf-8")
    guard_block = source[source.index("def _quality_guard_core"):source.index("# Registro das tools")]
    for forbidden in ("evaluate_noul", "_get_engine_async", "torch", "cuda", "laya"):
        assert forbidden not in guard_block, f"caminho crítico do gate referencia {forbidden}"


# ──────────────────────────────────────────────────────────────────────────────
# Gate 2: auto-aprovação do repositório real em audit_mode="full"
# ──────────────────────────────────────────────────────────────────────────────

#: Sentinel de ambiente que quebra a recursão do auto-teste do gate.
#: Definido por ``test_quality_guard_full_audit_self_approves_repo_root``
#: ANTES da chamada interna; o run interno de pytest herda o env via
#: ``PytestChunkRunner`` (``os.environ.copy()``) e este teste se auto-pula,
#: limitando a profundidade de aninhamento a exatamente 2 níveis.
_NESTED_RUN_ENV = "JEV_AQG_NESTED_RUN"


def test_quality_guard_full_audit_self_approves_repo_root():
    """Gate 2 real: ``agent_quality_guard(workspace=".", audit_mode="full")`` aprova.

    Verifica os três contratos do estado limpo do pipeline completo sobre o
    PRÓPRIO repositório (pytest em chunks + NeedleASTGuard + Screener +
    bridge nativa + meta-regras Tree-Sitter):

    * ``passed is True`` e ``status == "APPROVED"``;
    * ``blocking_ast_violations == []`` (zero bloqueios);
    * ``pytest_success`` com todos os chunks verdes (suíte real, sem mocks).

    Isolamentos determinísticos (nenhum mock):
      * ``JEV_STATE_DIR`` → diretório temporário: o histórico do circuit
        breaker deste workspace (estado real) não é lido nem contaminado;
        a tentativa server-side nasce em 1.
      * ``JEV_AQG_NESTED_RUN=1`` → o run interno de pytest do gate herda o
        env e este teste se auto-pula, quebrando a recursão.
    """
    if os.environ.get(_NESTED_RUN_ENV) == "1":
        pytest.skip("execução aninhada do quality gate: auto-pulado para quebrar recursão")

    root = Path(__file__).resolve().parents[1]
    prev_cwd = os.getcwd()
    prev_state = os.environ.get("JEV_STATE_DIR")
    prev_nested = os.environ.get(_NESTED_RUN_ENV)

    try:
        if Path(prev_cwd).resolve() != root:
            os.chdir(root)
        with tempfile.TemporaryDirectory(prefix="aqg_self_state_") as state_dir:
            os.environ["JEV_STATE_DIR"] = state_dir
            os.environ[_NESTED_RUN_ENV] = "1"
            try:
                resp = asyncio.run(agent_quality_guard(workspace=".", audit_mode="full"))
            finally:
                os.environ.pop(_NESTED_RUN_ENV, None)
    finally:
        if prev_state is None:
            os.environ.pop("JEV_STATE_DIR", None)
        else:
            os.environ["JEV_STATE_DIR"] = prev_state
        if prev_nested is None:
            os.environ.pop(_NESTED_RUN_ENV, None)
        else:
            os.environ[_NESTED_RUN_ENV] = prev_nested
        if Path(os.getcwd()).resolve() != Path(prev_cwd).resolve():
            os.chdir(prev_cwd)

    assert resp["passed"] is True, resp["message"]
    assert resp["status"] == "APPROVED", resp["message"]
    assert resp["blocking_ast_violations"] == [], (
        f"{len(resp['blocking_ast_violations'])} bloqueio(s): "
        f"{[v['rule_id'] + '@' + v['file_path'] for v in resp['blocking_ast_violations']]}"
    )
    assert resp["pytest_success"] is True, "suíte real do repositório falhou no pipeline"
    assert resp["attempt_number"] == 1, (
        f"estado do circuit breaker não isolado: tentativa {resp['attempt_number']} (esperado 1)"
    )
    chunks = resp["pytest_chunks"]
    assert chunks, "pipeline não executou nenhum chunk de pytest"
    failed = [c["chunk_id"] for c in chunks if not c["passed"]]
    assert not failed, f"chunks vermelhos: {failed}"
