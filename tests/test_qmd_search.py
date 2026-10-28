"""Testes reais do motor de busca híbrido QMD (Ripgrep + Needle 3 + RRF).

Validação estrita SEM MOCKS utilizando sistema de arquivos e subprocessos reais.
"""

import asyncio
import inspect
import os
import tempfile
import time
from pathlib import Path

import pytest

from agent_quality_guard.memory.qmd_search import QmdHybridSearchEngine, find_rg_binary
from agent_quality_guard.mcp_server import qmd_search


@pytest.fixture
def sample_workspace():
    """Cria workspace real temporário com múltiplos arquivos para teste de busca híbrida."""
    with tempfile.TemporaryDirectory(prefix="test_qmd_ws_") as tmp_dir:
        ws = Path(tmp_dir)

        # Arquivo 1: Módulo Python com funções e regras
        py_file1 = ws / "billing_service.py"
        py_file1.write_text(
            "import os\n\n"
            "def calculate_total_billing(patient_id: str, days: int) -> float:\n"
            "    # RN_FAT_009: regra de valoracao de diarias de UTI\n"
            "    daily_rate = 1500.0\n"
            "    multiplier = 1.0 if days <= 2 else 0.5\n"
            "    return days * daily_rate * multiplier\n\n"
            "def validate_eligibility(card_number: str) -> bool:\n"
            "    return len(card_number) == 16\n",
            encoding="utf-8",
        )

        # Arquivo 2: Arquivo de texto/config com literais sem AST
        cfg_file = ws / "settings.conf"
        cfg_file.write_text(
            "ENVIRONMENT=production\n"
            "SECRET_OPME_HASH=ANVISA_10332340023_STENT_KEY\n"
            "MAX_CONNECTIONS=100\n",
            encoding="utf-8",
        )

        yield ws


def test_find_rg_binary_available():
    """Verifica se o binário do ripgrep ou fallback foi localizado."""
    rg = find_rg_binary()
    assert rg is not None
    assert os.path.exists(rg)


@pytest.mark.asyncio
async def test_qmd_search_hybrid_intersection(sample_workspace: Path):
    """Testa busca híbrida onde Grep e Needle 3 convergem no mesmo arquivo e nó AST."""
    engine = QmdHybridSearchEngine(workspace_root=str(sample_workspace))
    res = await engine.search(
        query="calculate_total_billing RN_FAT_009",
        max_tokens=800,
    )

    assert res["status"] == "success"
    assert res["snippets_count"] >= 1
    assert res["files_found"] >= 1

    first_snip = res["snippets"][0]
    assert first_snip["file"] == "billing_service.py"
    assert first_snip["source"] == "hybrid"
    assert first_snip["rrf_score"] > 0.0
    assert any("type:function" in r for r in first_snip["relations"])
    assert any("RN_FAT_009" in m["text"] for m in first_snip["exact_matches"])


@pytest.mark.asyncio
async def test_qmd_search_fallback_grep_literal(sample_workspace: Path):
    """Refinamento 1: Fallback assimétrico para literais não indexados em AST Python."""
    engine = QmdHybridSearchEngine(workspace_root=str(sample_workspace))
    res = await engine.search(
        query="ANVISA_10332340023_STENT_KEY",
        max_tokens=800,
    )

    assert res["status"] == "success"
    assert res["snippets_count"] >= 1
    snip = res["snippets"][0]
    assert snip["file"] == "settings.conf"
    assert snip["source"] == "grep_literal"
    assert any("SECRET_OPME_HASH" in m["text"] for m in snip["exact_matches"])


@pytest.mark.asyncio
async def test_qmd_search_mcp_tool(sample_workspace: Path):
    """Testa invocação da tool MCP qmd_search diretamente."""
    res = await qmd_search(
        query="validate_eligibility",
        workspace=str(sample_workspace),
        max_tokens=600,
    )

    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert res["snippets_count"] >= 1
    assert res["snippets"][0]["file"] == "billing_service.py"
    assert "validate_eligibility" in res["snippets"][0]["content"]


def test_qmd_search_estimate_tokens_type_error():
    """Valida tratamento de exceção com pytest.raises para tipo inválido em _estimate_tokens."""
    engine = QmdHybridSearchEngine()
    with pytest.raises(TypeError):
        engine._estimate_tokens(None)  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Timeout global: cancelamento gracioso das lanes pendentes
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_global_timeout_leaves_no_pending_lane_tasks(sample_workspace: Path):
    """Comportamental: ``status="timeout"`` é determinístico e não sobra lane órfã.

    Regressão do runner macOS (Python 3.10): o timer do ``asyncio.wait_for`` tem
    granularidade de milissegundos (kqueue), então com cota de 100 µs as duas
    lanes podiam concluir ANTES do tick do timer — o ``gather`` retornava
    normalmente, o motor via ``degraded_lanes=['ripgrep_lexical']`` e devolvia
    ``'partial'`` mesmo com o tempo total já estourado. A re-checagem de
    deadline pós-``gather`` (``_build_timeout_response``) fecha essa janela em
    qualquer SO, sem depender de sleep nem de ordenação de threads.

    ``grep_task`` (ripgrep em thread) e ``needle_task`` (AST) são criados antes
    do ``asyncio.wait_for``; a espera abaixo é ativa (estado + deadline duro de
    5 s), sem sleep cego de sincronização.
    """
    engine = QmdHybridSearchEngine(workspace_root=str(sample_workspace))
    res = await engine.search(query="calculate_total_billing", timeout=0.0001)

    assert res["status"] == "timeout"
    assert res["timeout_s"] == 0.0001
    assert res["degraded_lanes"] == ["global_timeout"]
    assert res["snippets_count"] == 0
    # A cota foi de fato consumida: 100 µs é ordens de grandeza abaixo do custo
    # de spawnar o ripgrep, então a latência medida nunca fica abaixo da cota.
    assert res["latency_ms"] >= 0.1

    deadline = time.monotonic() + 5.0
    while True:
        pending = [
            t for t in asyncio.all_tasks()
            if t is not asyncio.current_task() and not t.done()
        ]
        if not pending or time.monotonic() >= deadline:
            break
        await asyncio.sleep(0.01)

    assert not pending, (
        f"task(s) de lane ainda pendente(s) após timeout global: "
        f"{[t.get_name() for t in pending]}"
    )


def test_build_timeout_response_contract():
    """Contrato da resposta de timeout: payload estrito, sem snippets parciais."""
    res = QmdHybridSearchEngine()._build_timeout_response(0.25, 42.5)

    assert res["status"] == "timeout"
    assert res["timeout_s"] == 0.25
    assert res["latency_ms"] == 42.5
    assert res["degraded_lanes"] == ["global_timeout"]
    assert res["snippets"] == []
    assert res["snippets_count"] == 0
    assert res["files_found"] == 0
    assert res["total_tokens"] == 0
    assert res["token_savings_pct"] == 0.0
    assert res["warnings"] and "timeout global de 0.25s" in res["warnings"][0]


def test_global_deadline_rechecked_after_gather():
    """Contrato estrutural: o deadline é re-checado DEPOIS do gather das lanes.

    O handler de ``asyncio.TimeoutError`` sozinho não basta — no macOS o
    ``wait_for`` pode não disparar com as lanes concluídas após a cota. O teste
    trava a ordem no código-fonte: sem a re-checagem pós-``gather``, o motor
    volta a responder ``'partial'`` no cenário de race condition.
    """
    src = inspect.getsource(QmdHybridSearchEngine.search)
    gather_idx = src.find("await asyncio.gather(grep_task, needle_task)")
    assert gather_idx != -1, "gather das lanes ausente no search() híbrido"

    post_gather = src[gather_idx:]
    assert "_build_timeout_response" in post_gather, (
        "re-checagem de deadline ausente após o gather: race condition do "
        "timer do asyncio no macOS volta a devolver status='partial'"
    )
    assert "time.perf_counter() - t0" in post_gather, (
        "a re-checagem pós-gather precisa medir o tempo total desde t0"
    )


def test_global_timeout_handler_cancels_lanes_explicitly():
    """Contrato estrutural: o handler de ``TimeoutError`` cancela as duas lanes.

    Garantia dupla sobre o cancelamento do ``gather`` interno do
    ``asyncio.wait_for`` — sem ``grep_task.cancel()``/``needle_task.cancel()``
    explícitos no handler, o veredito ``status="timeout"`` pode ser devolvido
    deixando tarefas agendadas no loop. O comportamento efetivo de interrupção
    da thread do ripgrep não é observável de forma determinística (threads não
    são canceláveis), por isso o lock estrutural sobre o corpo do handler —
    remover qualquer uma das duas chamadas derruba este teste.
    """
    src = inspect.getsource(QmdHybridSearchEngine.search)
    idx = src.find("except asyncio.TimeoutError")
    assert idx != -1, "handler de asyncio.TimeoutError ausente no search() híbrido"

    handler = src[idx:idx + 900]
    assert "grep_task.cancel()" in handler, (
        "handler de timeout não cancela explicitamente grep_task (lane ripgrep)"
    )
    assert "needle_task.cancel()" in handler, (
        "handler de timeout não cancela explicitamente needle_task (lane AST)"
    )
