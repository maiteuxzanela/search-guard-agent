"""Suíte de testes de regressão determinísticos para a ferramenta MCP needle_verify_delivery.

Garante 100% de aprovação factual sem o uso de mocks em 3 cenários principais:
1. Workspace limpo com testes de unidade passando -> APPROVED (passed=True, pytest_success=True)
2. Workspace com violações de AST sob src/ (ex: import unittest.mock) -> REJECTED (passed=False)
3. 5ª tentativa de homologação -> Saída de Emergência EMERGENCY_RELEASE (passed=True)
"""

import os
import subprocess
import tempfile
from pathlib import Path
import pytest

from agent_quality_guard.mcp_server import needle_verify_delivery


@pytest.mark.asyncio
async def test_needle_verify_delivery_clean_workspace_approved():
    """Teste 1: Invocação em workspace limpo com testes passando.
    
    Deve retornar passed: True, status: "APPROVED", pytest_success: True.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Inicializa repositório Git
        subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@searchguard.local"], cwd=tmp_dir, check=True)
        subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=tmp_dir, check=True)

        # Cria código de produção limpo
        src_dir = Path(tmp_dir) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "math_ops.py").write_text(
            "def multiply(a: int, b: int) -> int:\n"
            "    \"\"\"Multiplica dois inteiros.\"\"\"\n"
            "    return a * b\n",
            encoding="utf-8",
        )

        # Cria teste de unidade passando
        tests_dir = Path(tmp_dir) / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_math_ops.py").write_text(
            "from src.math_ops import multiply\n\n"
            "def test_multiply():\n"
            "    assert multiply(3, 4) == 12\n",
            encoding="utf-8",
        )

        # Adiciona e faz commit inicial
        subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
        subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_dir, check=True)

        # Executa homologação determinística
        resp = await needle_verify_delivery(
            workspace=tmp_dir,
            attempt_number=1,
        )

        assert resp["passed"] is True
        assert resp["status"] == "APPROVED"
        assert resp["pytest_success"] is True
        assert resp["blocking_ast_violations"] == []


@pytest.mark.asyncio
async def test_needle_verify_delivery_ast_violation_rejected():
    """Teste 2: Invocação em workspace com violação determinística de AST.
    
    Import de unittest.mock em arquivo sob src/ deve retornar passed: False,
    status: "REJECTED", contendo blocking_ast_violations.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Inicializa repositório Git
        subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@searchguard.local"], cwd=tmp_dir, check=True)
        subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=tmp_dir, check=True)

        # Commit inicial para ter baseline
        readme = Path(tmp_dir) / "README.md"
        readme.write_text("# Test Project\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
        subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_dir, check=True)

        # Cria arquivo com import proibido sob src/
        src_dir = Path(tmp_dir) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "mock_service.py").write_text(
            "import unittest.mock\n\n"
            "def get_mock():\n"
            "    return unittest.mock.MagicMock()\n",
            encoding="utf-8",
        )

        # Cria teste passando
        tests_dir = Path(tmp_dir) / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)
        (tests_dir / "test_dummy.py").write_text(
            "def test_dummy():\n"
            "    assert True\n",
            encoding="utf-8",
        )

        # Executa homologação determinística
        resp = await needle_verify_delivery(
            workspace=tmp_dir,
            attempt_number=1,
            pedido="Integrar serviço externo",
            resultado="mock_service.py implementado com mock",
        )

        assert resp["passed"] is False
        assert resp["status"] == "REJECTED"
        assert resp["pedido_audit"] == "Integrar serviço externo"
        assert resp["resultado_audit"] == "mock_service.py implementado com mock"
        assert len(resp["blocking_ast_violations"]) >= 1
        violation_rules = [v["rule_id"] for v in resp["blocking_ast_violations"]]
        assert "mock_in_production" in violation_rules


@pytest.mark.asyncio
async def test_needle_verify_delivery_circuit_breaker_emergency_release():
    """Teste 3: 5ª tentativa real (contagem server-side) -> Saída de Emergência.

    O contador vem exclusivamente do histórico persistido por workspace: as 4
    primeiras chamadas reprovam (workspace sem suíte) e avançam a contagem.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        for expected_attempt in range(1, 5):
            resp = await needle_verify_delivery(workspace=tmp_dir)
            assert resp["passed"] is False
            assert resp["status"] == "REJECTED"
            assert resp["attempt_number"] == expected_attempt
            assert resp["approved_with_caveat"] is False

        resp = await needle_verify_delivery(workspace=tmp_dir)
        # TICKET-SGA-11 F4: sem evidência executada, o circuito terminal NÃO aprova.
        assert resp["passed"] is False
        assert resp["status"] == "EMERGENCY_RELEASE"
        assert resp["attempt_number"] == 5
        assert resp["pytest_success"] is False, (
            "Saída de Emergência não executa pytest e não pode reportá-lo como sucesso"
        )
        assert resp["pytest_chunks"] == []
        assert resp["approved_with_caveat"] is False
        assert resp["evidence_executed"] is False
        assert "STATUS TERMINAL" in resp["message"]
        assert "Laudo de Inviabilidade" in resp["message"]

        # Chamada repetida no mesmo estado mantém o status terminal sem aprovar
        resp_repeat = await needle_verify_delivery(workspace=tmp_dir)
        assert resp_repeat["status"] == "EMERGENCY_RELEASE"
        assert resp_repeat["attempt_number"] == 5
        assert resp_repeat["passed"] is False
        assert "STATUS TERMINAL" in resp_repeat["message"]
        assert "NÃO re-execute needle_verify_delivery" in resp_repeat["message"]


@pytest.mark.asyncio
async def test_client_cannot_forge_emergency_on_first_attempt():
    """BLOQUEANTE corrigido (auditoria 09/2026): attempt_number do chamador não destrava emergência.

    Antes da correção, `attempt_number=5` já na 1ª chamada devolvia
    EMERGENCY_RELEASE com pytest_success=True sem executar pytest, Guard nem
    Screener. A tentativa é resolvida server-side pelo histórico.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        resp = await needle_verify_delivery(workspace=tmp_dir, attempt_number=5)

        assert resp["status"] != "EMERGENCY_RELEASE"
        assert resp["attempt_number"] == 1
        assert resp["approved_with_caveat"] is False


@pytest.mark.asyncio
async def test_workspace_without_tests_is_rejected():
    """BLOQUEANTE corrigido: workspace sem suíte real não pode ser homologado.

    O runner devolvia `pytest_success=True, chunks=[]` para diretório sem
    `tests/` — aprovação sem executar um único teste.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        resp = await needle_verify_delivery(workspace=tmp_dir)

        assert resp["passed"] is False
        assert resp["status"] == "REJECTED"
        assert resp["pytest_success"] is False
        assert resp["pytest_chunks"] == []
        assert "Nenhum teste coletado" in resp["message"]


@pytest.mark.asyncio
async def test_needle_verify_delivery_resets_attempt_count_on_new_git_commit():
    """Garante que novo commit no workspace reinicia o ciclo de tentativas para 1."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Inicializa repo git
        subprocess.run(["git", "init"], cwd=tmp_dir, capture_output=True, check=True)
        subprocess.run(["git", "config", "user.name", "Tester"], cwd=tmp_dir, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_dir, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "commit 1"], cwd=tmp_dir, capture_output=True, check=True)

        # 1ª e 2ª chamadas falham (sem testes)
        resp1 = await needle_verify_delivery(workspace=tmp_dir)
        assert resp1["attempt_number"] == 1
        assert resp1["status"] == "REJECTED"

        resp2 = await needle_verify_delivery(workspace=tmp_dir)
        assert resp2["attempt_number"] == 2
        assert resp2["status"] == "REJECTED"

        # Novo commit efetuado (ex: entrega comitada)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "commit 2"], cwd=tmp_dir, capture_output=True, check=True)

        # Nova chamada deve reiniciar na tentativa 1
        resp3 = await needle_verify_delivery(workspace=tmp_dir)
        assert resp3["attempt_number"] == 1
        assert resp3["status"] == "REJECTED"


def test_history_clear_is_scoped_and_state_dir_is_isolated():
    """BLOQUEANTE corrigido: a suíte não pode apagar o estado do workspace real.

    `history_clear()` global limpava `/tmp/openjev_state` inteiro (registro de
    homologação + contador de tentativas de todos os workspaces). Com
    `JEV_STATE_DIR` (tests/conftest.py) a suíte opera em diretório próprio.
    """
    import json

    from agent_quality_guard.core.score_gate import _get_state_path, history_clear, history_set

    real_state_dir = Path(tempfile.gettempdir()) / "openjev_state"
    sentinel = real_state_dir / "score_state_deadbeef.json"
    real_state_dir.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({"history": {"passed": False, "attempts": 4}}), encoding="utf-8")

    try:
        ws = "/workspace/estado_isolado"
        history_set(ws, {"passed": False, "attempts": 3})

        state_path = _get_state_path(ws)
        assert str(state_path).startswith(os.environ["JEV_STATE_DIR"]), (
            f"estado do teste vazou para fora do diretório isolado: {state_path}"
        )

        history_clear()

        assert sentinel.exists(), (
            "history_clear() da suíte apagou estado de OUTRO workspace — "
            "o circuit breaker e o registro de homologação ficariam sem persistência"
        )
        assert not state_path.exists(), "reset do próprio estado do teste falhou"
    finally:
        if sentinel.exists():
            sentinel.unlink()


class DeterministicNoulEngine:
    """Stand-in real e determinístico da interface do motor (apenas para provar
    que o argumento ``engine`` é IGNORADO pelo pipeline 100% determinístico do
    quality guard — o ML/Laya foi removido do caminho crítico no TICKET-43)."""

    def __init__(self, probability: float = 0.90, is_ready: bool = True):
        self.probability = probability
        self.is_ready = is_ready

    async def evaluate_noul(self, req):
        from agent_quality_guard.core.schemas import NoulResponse
        passed = self.probability >= req.threshold
        return NoulResponse(
            passed=passed,
            probability=self.probability,
            confidence=round(abs(self.probability - 0.5) * 2.0, 4),
            threshold=req.threshold,
            latency_ms=10.0,
        )


def _build_untyped_workspace(tmp_dir: str) -> None:
    """Materializa um workspace git com função sem tipagem/docstring (alerta estrutural).

    Produz alerta de qualidade estrutural do Screener (tipagem/docstring) mas
    NENHUMA violação determinística bloqueante.
    """
    subprocess.run(["git", "init"], cwd=tmp_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@searchguard.local"], cwd=tmp_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=tmp_dir, check=True)

    (Path(tmp_dir) / "README.md").write_text("# Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=tmp_dir, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=tmp_dir, check=True)

    src_dir = Path(tmp_dir) / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "untyped.py").write_text(
        "def calculate_total(subtotal, tax):\n"
        "    return subtotal + tax\n",
        encoding="utf-8",
    )

    tests_dir = Path(tmp_dir) / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_untyped.py").write_text(
        "from src.untyped import calculate_total\n\n"
        "def test_calculate_total():\n"
        "    assert calculate_total(100, 10) == 110\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_needle_verify_delivery_structural_warnings_are_telemetry():
    """TICKET-43: alerta estrutural vira TELEMETRIA — não bloqueia (sem Laya/Noul).

    Antes, a arbitragem Laya/Noul confirmava o defeito e reprovava. Com o
    pipeline 100% determinístico, ausência de violação bloqueante (Guard +
    Camada 1 toolchain + Camada 2 meta-regras) => APPROVED, com os alertas
    estruturais reportados como telemetria.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _build_untyped_workspace(tmp_dir)

        resp = await needle_verify_delivery(workspace=tmp_dir, attempt_number=1)

        assert resp["passed"] is True
        assert resp["status"] == "APPROVED"
        assert resp["confirmed_defects"] == []
        assert resp["dismissed_warnings"] == []
        assert len(resp["structural_ast_warnings"]) >= 1
        assert resp["tool_name"] == "agent_quality_guard"
        assert "meta-regras universais de AST" in resp["message"]


@pytest.mark.asyncio
async def test_needle_verify_delivery_engine_argument_is_ignored():
    """TICKET-43: retrocompat de assinatura — `engine` é aceito mas IGNORADO.

    O veredito determinístico deve ser idêntico com ou sem engine injetado,
    provando o desacoplamento total de Laya/Noul no caminho crítico.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        _build_untyped_workspace(tmp_dir)

        engine = DeterministicNoulEngine(probability=0.99, is_ready=True)

        resp = await needle_verify_delivery(
            workspace=tmp_dir,
            attempt_number=1,
            engine=engine,
            pedido="Adicionar calculate_total com tipagem",
            resultado="calculate_total implementado em src/untyped.py com teste",
        )

        assert resp["passed"] is True
        assert resp["status"] == "APPROVED"
        assert resp["pedido_audit"] == "Adicionar calculate_total com tipagem"
        assert resp["resultado_audit"] == "calculate_total implementado em src/untyped.py com teste"
        assert resp["confirmed_defects"] == []
        assert resp["dismissed_warnings"] == []


def test_agent_quality_guard_canonical_alias_and_registration():
    """`agent_quality_guard` (canônico) e `needle_verify_delivery` (alias) = MESMA impl.

    Ambos devem estar registradas no servidor MCP e resolver para a mesma função.
    """
    from agent_quality_guard.mcp_server import agent_quality_guard, app

    assert agent_quality_guard is needle_verify_delivery

    registered = set(app._tool_manager._tools.keys())
    assert "agent_quality_guard" in registered, "tool canônica ausente do catálogo MCP"
    assert "needle_verify_delivery" in registered, "alias legado ausente do catálogo MCP"


@pytest.mark.asyncio
async def test_needle_verify_delivery_multistack_chunk_id_and_audit():
    """Valida tipagem string de chunk_id e propagação de pedido_audit em multi-stack."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / "package.json").write_text('{"name": "test-app"}', encoding="utf-8")
        (Path(tmp_dir) / "app.json").write_text('{"expo": {}}', encoding="utf-8")
        src_dir = Path(tmp_dir) / "src"
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / "index.js").write_text("console.log('app');\n", encoding="utf-8")

        resp = await needle_verify_delivery(
            workspace=tmp_dir,
            pedido="Refatorar componente mobile",
            resultado="index.js ajustado",
        )

        assert resp["status"] == "REJECTED"
        assert resp["pedido_audit"] == "Refatorar componente mobile"
        assert resp["resultado_audit"] == "index.js ajustado"
        assert len(resp["pytest_chunks"]) == 1
        assert isinstance(resp["pytest_chunks"][0]["chunk_id"], str)
        assert resp["pytest_chunks"][0]["chunk_id"].startswith("chunk_multistack_")


def test_needle_verify_delivery_request_rejects_attempt_above_five():
    """Caminho de erro real: attempt_number > 5 estoura no schema (circuit breaker)."""
    from pydantic import ValidationError

    from agent_quality_guard.core.schemas import NeedleVerifyDeliveryRequest

    with pytest.raises(ValidationError):
        NeedleVerifyDeliveryRequest(attempt_number=6)


def _build_node_ts_bun_workspace(
    tmp_dir: str,
    run_all_body: str,
    prod_files: dict | None = None,
) -> Path:
    """Materializa um workspace NODE_TS_BUN real (package.json + run_all.mjs + src/)."""
    ws = Path(tmp_dir)
    (ws / "package.json").write_text('{"name": "app", "type": "module"}', encoding="utf-8")
    (ws / "tests").mkdir(exist_ok=True)
    (ws / "tests" / "run_all.mjs").write_text(run_all_body, encoding="utf-8")
    for rel, content in (prod_files or {}).items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return ws


@pytest.mark.asyncio
async def test_needle_verify_delivery_node_ts_bun_approved():
    """Gate NODE_TS_BUN end-to-end: runner real node + AST limpa => APPROVED."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _build_node_ts_bun_workspace(
            tmp_dir,
            "import assert from 'node:assert';\n"
            "assert.strictEqual(1 + 1, 2);\n"
            "console.log('SUÍTES: 1/1 passaram');\n",
            prod_files={"src/math.mjs": "export function add(a, b) { return a + b; }\n"},
        )

        resp = await needle_verify_delivery(workspace=tmp_dir)

        assert resp["status"] == "APPROVED", resp["message"]
        assert resp["passed"] is True
        assert resp["pytest_success"] is True
        assert len(resp["pytest_chunks"]) == 1
        chunk = resp["pytest_chunks"][0]
        assert chunk["chunk_id"] == "chunk_multistack_node_ts_bun"
        assert chunk["passed"] is True
        assert chunk["total_passed"] >= 1
        assert resp["blocking_ast_violations"] == []


@pytest.mark.asyncio
async def test_needle_verify_delivery_node_ts_bun_test_failure_rejected_failfast():
    """Suíte Node real falhando => REJECTED Fail-Fast com pytest_chunks_typed preenchido."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _build_node_ts_bun_workspace(
            tmp_dir,
            "import assert from 'node:assert';\n"
            "assert.strictEqual(1, 2, 'falha real esperada');\n",
        )

        resp = await needle_verify_delivery(workspace=tmp_dir, pedido="p", resultado="r")

        assert resp["status"] == "REJECTED"
        assert resp["passed"] is False
        assert resp["pytest_success"] is False
        assert len(resp["pytest_chunks"]) == 1
        chunk = resp["pytest_chunks"][0]
        assert chunk["chunk_id"] == "chunk_multistack_node_ts_bun"
        assert chunk["passed"] is False
        assert "NODE_TS_BUN Fail-Fast" in resp["message"]
        assert resp["pedido_audit"] == "p"
        assert resp["resultado_audit"] == "r"


@pytest.mark.asyncio
async def test_needle_verify_delivery_node_ts_bun_ast_violation_rejected():
    """Gate AST NODE_TS_BUN: eval() em `.mjs` de produção bloqueia a entrega."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        _build_node_ts_bun_workspace(
            tmp_dir,
            "import assert from 'node:assert';\n"
            "assert.ok(true);\n"
            "console.log('SUÍTES: 1/1 passaram');\n",
            prod_files={"src/danger.mjs": "export function run(code) { return eval(code); }\n"},
        )

        resp = await needle_verify_delivery(workspace=tmp_dir)

        assert resp["status"] == "REJECTED", resp["message"]
        assert resp["passed"] is False
        assert resp["pytest_success"] is True
        rule_ids = [v["rule_id"] for v in resp["blocking_ast_violations"]]
        assert "SECURITY_EVAL" in rule_ids


@pytest.mark.asyncio
async def test_needle_verify_delivery_node_ts_bun_without_tests_is_rejected():
    """INVARIANTE §5: workspace NODE_TS_BUN sem suíte executável NÃO pode ser homologado.

    Um repositório TS (src/*.ts) sem `tests/run_all.mjs`, sem `scripts.test` e sem
    lock/bunfig do Bun não possui runner algum. O caminho Python já reprova com
    "Nenhum teste coletado"; o caminho NODE_TS_BUN deve rejeitar de forma idêntica
    (falha-fechada), nunca aprovar com zero testes executados.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        ws = Path(tmp_dir)
        (ws / "package.json").write_text('{"name": "app", "type": "module"}', encoding="utf-8")
        (ws / "src").mkdir()
        (ws / "src" / "index.ts").write_text("export const x: number = 1;\n", encoding="utf-8")

        resp = await needle_verify_delivery(workspace=tmp_dir)

        assert resp["status"] == "REJECTED", (
            "workspace NODE_TS_BUN sem testes foi homologado: " + resp["message"]
        )
        assert resp["passed"] is False
        assert resp["pytest_success"] is False
