#!/usr/bin/env python3
"""MCP Server `cirurgic-search-mcp` (Projeto JEV).

Disponibiliza estritamente as ferramentas essenciais e ativas:

* busca cirúrgica — ``cirurgic_search`` (canônica) + alias legado ``qmd_search``;
* homologação determinística — ``agent_quality_guard`` + alias legado
  ``needle_verify_delivery``.

TICKET-HARNESS-47: o motor Laya RLCD / ``OpenJevEngine`` (PyTorch, VRAM, socket
IPC legado e lock de GPU) foi desativado. Este processo não importa
torch/transformers nem aloca VRAM, portanto o boot é quase instantâneo.
"""

import hashlib
import logging
import os
import signal
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Redireciona todos os logs para stderr (stdout exclusivo do protocolo MCP stdio)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] [JEV MCP] %(message)s",
)

# Adiciona raiz do projeto JEV ao sys.path
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR))

from mcp.server import MCPServer
from mcp.server.mcpserver import Context
from agent_quality_guard.core.schemas import (
    DEFECT_DESCRIPTIONS,
    DEFECT_MEMBERS,
    DiagnosisResult,
    AgentQualityGuardResponse,
    LinterDiagnostic,
    MetaRuleViolation,
    ToolchainToolStatus,
    META_RULE_BLOCKING_CATEGORIES,
    PytestChunkResult,
    ASTViolation,
    VerdictState,
)
from agent_quality_guard.memory.ast_diff_extractor import extract_workspace_ast_symbols
from agent_quality_guard.memory.needle_defect_screener import (
    screen_workspace_report,
)
from agent_quality_guard.memory.qmd_search import QmdHybridSearchEngine
from agent_quality_guard.memory.needle_ast_guard import NeedleASTGuard
from agent_quality_guard.orchestrator.pytest_chunk_runner import PytestChunkRunner
from agent_quality_guard.orchestrator.multi_stack_runner import MultiStackRunner

logger = logging.getLogger("agent_quality_guard.mcp")

app = MCPServer("cirurgic-search-mcp")

# TICKET-HARNESS-47: motor Laya/OpenJevEngine desativado neste processo.
# Nenhum peso, PyTorch, VRAM, socket IPC legado ou lock de GPU é carregado no
# boot do MCP. O símbolo permanece apenas como tombstone de
# import para módulos legados (ex.: src/decision/daemon.py), que devem falhar
# alto caso insistam no caminho ML em vez de usar as tools determinísticas.
def get_engine() -> Any:
    """Tombstone: o motor Laya/OpenJevEngine foi removido no TICKET-HARNESS-47."""
    raise RuntimeError(
        "OpenJevEngine/Laya foi desativado no TICKET-HARNESS-47. "
        "Use cirurgic_search (busca cirúrgica) e agent_quality_guard "
        "(homologação determinística), que não usam ML, PyTorch, VRAM nem IPC."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Busca cirúrgica: nome canônico `cirurgic_search` + alias legado `qmd_search`
# (MESMA implementação, mesmos schemas de input/output, retrocompatibilidade
# total com clientes antigos e chamadas em cache).
# ──────────────────────────────────────────────────────────────────────────────
_CIRURGIC_SEARCH_DESCRIPTION = (
    "Busca cirurgica (cirurgic-search-mcp): hibrido AST + Ripgrep + RRF em CPU. "
    "Ferramenta primaria obrigatoria para explorar codigo, funcoes, regras de negocio "
    "e investigacao de bugs. Combina busca lexica exata com analise semantica/AST e "
    "ranqueamento Reciprocal Rank Fusion. Nome canonico atual; `qmd_search` permanece "
    "disponivel como alias retrocompativel da MESMA implementacao. "
    "`depth` (opcional, padrao None = ilimitado) corta a profundidade da arvore de "
    "diretorios da varredura e do grafo relacional AST quando aplicavel; "
    "`exclude_patterns` soma exclusoes pontuais as universais padrao e ao arquivo "
    "`.cirurgicignore` da raiz do workspace (se existir)."
)

_QMD_SEARCH_LEGACY_ALIAS_DESCRIPTION = (
    "DEPRECADO — use `cirurgic_search` (nome canonico). Alias legado de "
    "`cirurgic_search` (MESMA implementacao e assinatura), mantido para "
    "compatibilidade retroativa total com clientes e caches que ainda chamam `qmd_search`. "
    "Busca hibrida cirurgica (Ripgrep + Needle 3 + RRF em CPU). Ferramenta primaria "
    "obrigatoria para explorar codigo, funcoes, regras de negocio e investigacao de bugs. "
    "Combina busca lexica exata com analise semantica/AST e ranqueamento Reciprocal Rank Fusion. "
    "Novos clientes devem migrar para `cirurgic_search`: o alias permanece registrado e em "
    "paridade total, mas sem novas funcionalidades exclusivas."
)


async def _cirurgic_search_core(
    query: str,
    workspace: str = "",
    scope: str = "",
    max_tokens: int = 1200,
    depth: Optional[int] = None,
    exclude_patterns: Optional[List[str]] = None,
    timeout: Optional[float] = None,
) -> dict:
    """Executa busca híbrida de código via QMD (Ripgrep + Needle 3 + RRF).

    Núcleo único compartilhado por `cirurgic_search` (canônico) e `qmd_search`
    (alias legado), garantindo contratos idênticos de entrada e saída.

    Contrato de saída (aditivo e retrocompatível — TICKET-SGA-03): o dict
    retornado sempre inclui as chaves ``status`` (``success`` | ``partial`` |
    ``error``), ``warnings`` (avisos operacionais) e ``degraded_lanes``
    (lanes de busca degradadas). Validação de referência: schema
    ``CirurgicSearchResponse`` em ``agent_quality_guard.core.schemas``.

    ``timeout`` (TICKET-SGA-04): cota global por requisição em segundos —
    ``None`` usa o padrão de 15 s; valores ``<= 0`` desativam a cota. Parâmetro
    opcional, portanto retrocompatível com chamadores antigos.

    Contrato TICKET-SGA-05:
    * ``depth: Optional[int] = None`` — profundidade MÁXIMA de busca: níveis da
      árvore de diretórios abaixo de cada alvo de escopo e profundidade do grafo
      relacional AST quando aplicável. ``None`` (padrão) = profundidade
      ILIMITADA (comportamento idêntico ao histórico para quem omite o
      parâmetro). Clientes MCP que já enviam ``depth`` são 100% retrocompatíveis.
    * ``exclude_patterns: Optional[List[str]] = None`` — globs de exclusão
      somados às exclusões universais padrão e ao ``.cirurgicignore`` da raiz
      do workspace (se existir; linhas ``#``/em branco ignoradas). Mesma
      semântica do arquivo: nome puro casa em qualquer nível, padrão com ``/``
      é ancorado na raiz, ``/`` final marca diretório, negação ``!`` não é
      suportada.

    Violações de sandbox/allowlist do workspace são capturadas (``ValueError``
    explícito do motor) e convertidas em ``status="error"`` estruturado — o
    servidor MCP nunca derruba o transporte stdio por caminho inválido.

    Exemplo:
        >>> import asyncio
        >>> res = asyncio.run(cirurgic_search(query="agent_quality_guard"))
        >>> isinstance(res, dict)
        True
    """
    target_workspace = workspace.strip() if workspace and workspace.strip() else os.getcwd()
    scope_list = [s.strip() for s in scope.split(",") if s.strip()] if scope else None

    try:
        engine = QmdHybridSearchEngine(workspace_root=target_workspace)
        result = await engine.search(
            query=query,
            scope=scope_list,
            max_tokens=max_tokens,
            depth=depth,
            exclude_patterns=list(exclude_patterns) if exclude_patterns else None,
            timeout=timeout,
        )
    except ValueError as sandbox_err:
        # Allowlist/sandbox do motor: workspace/scope fora das raízes
        # autorizadas (CIRURGIC_ALLOWED_ROOTS). Nunca vaza exceção crua
        # para o cliente MCP — erro estruturado, contrato CirurgicSearchResponse.
        logger.warning(
            "cirurgic_search: violação de sandbox rejeitada (workspace=%r, scope=%r): %s",
            target_workspace,
            scope_list,
            sandbox_err,
        )
        return {
            "status": "error",
            "error": "sandbox_violation",
            "message": str(sandbox_err),
            "degraded_lanes": ["sandbox"],
            "warnings": [f"Sandbox rejeitou a busca: {sandbox_err}"],
            "latency_ms": 0.0,
            "total_tokens": 0,
            "token_savings_pct": 0.0,
            "snippets_count": 0,
            "files_found": 0,
            "snippets": [],
        }

    # Normalização aditiva do contrato (TICKET-SGA-03): garante as chaves
    # status/warnings/degraded_lanes mesmo quando o motor ainda não as emite
    # (no-op assim que o motor passar a emitir — setdefault nunca sobrescreve).
    result.setdefault("status", "success")
    result.setdefault("warnings", [])
    result.setdefault("degraded_lanes", [])
    return result


cirurgic_search = app.tool(
    name="cirurgic_search", description=_CIRURGIC_SEARCH_DESCRIPTION
)(_cirurgic_search_core)
qmd_search = app.tool(
    name="qmd_search", description=_QMD_SEARCH_LEGACY_ALIAS_DESCRIPTION
)(_cirurgic_search_core)


# NOTA (TICKET-HARNESS-47): as primitivas probabilísticas legadas do motor
# Laya RLCD foram removidas deste servidor. A superfície MCP fica restrita a
# `cirurgic_search` (+ alias `qmd_search`) e `agent_quality_guard` (+ alias
# `needle_verify_delivery`), ambas determinísticas e sem ML/GPU.


from agent_quality_guard.core.score_gate import (
    MAX_ATTEMPTS as _SCORE_GATE_MAX_ATTEMPTS,
    CAVEAT_BAND as _CAVEAT_BAND,
    get_workspace_git_signature as _score_gate_git_signature,
    history_get as _history_get,
    history_increment_attempts as _history_increment_attempts,
    history_set as _history_set,
    resolve_attempt_number as _resolve_attempt_number,
    stages_cache_set as _stages_cache_set,
)

MAX_ATTEMPTS = _SCORE_GATE_MAX_ATTEMPTS


def _get_workspace_git_signature(workspace_dir: str) -> str:
    """Assinatura do workspace — delega a score_gate (HEAD + diff + untracked)."""
    return _score_gate_git_signature(workspace_dir)


def _emergency_exit_response(
    threshold: float, attempts: int, last_score: float, reason: str
) -> dict:
    """Resposta-padrão da Saída de Emergência (breaker universal ou anti-gaming)."""
    return {
        "score": last_score,
        "passed": False,
        "threshold": threshold,
        "attempts": attempts,
        "max_attempts": MAX_ATTEMPTS,
        "emergency_exit_unlocked": True,
        "diagnosis": None,
        "message": (
            f"🚨 [SAÍDA DE EMERGÊNCIA DESBLOQUEADA] (Tentativa {attempts}/{MAX_ATTEMPTS}): "
            f"{reason} "
            "O agente está autorizado a emitir Laudo de Inviabilidade Técnica."
        ),
    }


def _no_change_response(attempts: int) -> dict:
    """Recusa anti-premise gaming: avaliação repetida sem mudança no disco."""
    return {
        "error": (
            f"🚨 AVALIAÇÃO RECUSADA (Tentativa {attempts}/{MAX_ATTEMPTS}): Nenhuma alteração detectada "
            "no repositório desde a última reprovação. Refatore o código no disco antes de reavaliar."
        ),
        "attempts": attempts,
        "max_attempts": MAX_ATTEMPTS,
        "emergency_exit_unlocked": False,
    }


def _evaluation_gate(
    target_workspace: str,
    hist: Optional[Dict[str, Any]],
    current_git_state: str,
    threshold: float,
) -> Optional[dict]:
    """Camada 0: circuit breaker universal + anti-premise gaming, sem aninhamento.

    Condições checadas em passos curtos com retorno antecipado (evita predicado
    denso e cadeia de if de nível 3). Devolve a resposta-padrão de bloqueio ou
    ``None`` quando a avaliação pode prosseguir.
    """
    if not hist or hist.get("passed", True):
        return None
    attempts = int(hist.get("attempts", 1))
    last_score = hist.get("last_score", 0.0)
    if attempts >= MAX_ATTEMPTS:
        logger.warning(
            f"🚨 [Circuit Breaker Universal]: Limite de {MAX_ATTEMPTS} tentativas atingido para {target_workspace}. "
            "Desbloqueando Saída de Emergência."
        )
        return _emergency_exit_response(
            threshold,
            attempts,
            last_score,
            f"Limite de {MAX_ATTEMPTS} tentativas consecutivas atingido.",
        )
    prev_state = hist.get("git_state", "")
    if current_git_state != prev_state or current_git_state == "":
        return None
    new_attempts = _history_increment_attempts(target_workspace)
    if new_attempts >= MAX_ATTEMPTS:
        return _emergency_exit_response(
            threshold,
            new_attempts,
            last_score,
            f"Limite de {MAX_ATTEMPTS} tentativas atingido sem alteração no repositório.",
        )
    return _no_change_response(new_attempts)


def _flag_suspect_failed_stages(
    stages: List[Any], refuted_suspicions: List[Dict[str, Any]]
) -> None:
    """Na reprovação por suspeita não-resolvida, marca as etapas afetadas."""
    suspect_keys = {
        f"{r['file_path']}::{r['symbol_name']}" for r in refuted_suspicions
    }
    for st in stages:
        if not st.cached and st.stage_id in suspect_keys:
            st.passed = False


def _cache_passed_stages(
    stages: List[Any], symbols_map: Dict[str, Any], target_workspace: str
) -> None:
    """Persiste estágios aprovados no cache do workspace."""
    for st in stages:
        if st.passed:
            sym_obj = symbols_map.get(st.stage_id)
            s_h = hashlib.sha256(sym_obj.content.encode("utf-8")).hexdigest() if sym_obj else ""
            _stages_cache_set(target_workspace, st.stage_id, {
                "score": st.score,
                "passed": True,
                "hash": s_h,
            })


def _is_test_symbol_path(file_path: str) -> bool:
    """True quando o símbolo vem de um diretório de testes (tests/, test/)."""
    return any(part in ("tests", "test") for part in file_path.split("/"))


def _truncate_at_line(text: str, limit: int) -> str:
    """Trunca `text` em no máximo `limit` caracteres, SEMPRE em fronteira de linha.

    O corte cego de caracteres (ex.: `text[:350]`) podia fatiar uma declaração no
    meio (ex.: `out: ` de `out: list[int] = []`), apresentando ao Laya código
    sintaticamente quebrado e derrubando a nota mesmo com entrega impecável.
    """
    if len(text) <= limit:
        return text
    window = text[:limit]
    cut = window.rfind("\n")
    if cut < 60:
        return window
    return window[:cut]


def _build_code_context(premise: str, symbols: List[Any], headroom: int = 1480) -> str:
    """Seleciona os snippets de código que enriquecem a premissa de score.

    - Ordem estável alternando símbolos de produção e de teste (round-robin),
      preenchendo o orçamento de caracteres em vez de um cap cego de 3 símbolos;
    - garante pelo menos UM snippet de teste quando o diff contém testes, para
      que o critério "cobertura de testes reais" da rubrica codigo_producao
      tenha evidência de código (não só prosa na premissa);
    - nunca excede o teto da premissa combinada (o score_gate recusa > 1500).
    """
    marker = "\n\nSímbolos alterados:\n"
    if "\nSímbolos alterados:\n" in premise:
        return ""
    budget = headroom - len(premise) - len(marker)
    if budget < 60:
        return ""
    src_syms = [s for s in symbols if not _is_test_symbol_path(s.file_path)]
    test_syms = [s for s in symbols if _is_test_symbol_path(s.file_path)]
    ordered: List[Any] = []
    for i in range(max(len(src_syms), len(test_syms))):
        if i < len(src_syms):
            ordered.append(src_syms[i])
        if i < len(test_syms):
            ordered.append(test_syms[i])
    parts: List[str] = []
    used = 0
    for s in ordered:
        # full_context inclui o cabeçalho de imports: sem ele, trechos de teste
        # exibem chamadas sem definição (ex.: add(...)) e leem como código quebrado.
        piece = (
            f"Arquivo {s.file_path} ({s.symbol_type} {s.symbol_name}):\n"
            f"{_truncate_at_line(s.full_context, 350)}"
        )
        cost = len(piece) + (5 if parts else 0)
        if used + cost > budget:
            break
        parts.append(piece)
        used += cost
    return "\n---\n".join(parts)


def _build_failure_diagnosis(
    confirmed_defects: List[Dict[str, Any]],
    refuted_suspicions: List[Dict[str, Any]],
    delivery_score: float,
    threshold: float,
    latency_ms: float,
    quality_confirmed: Optional[List[Dict[str, Any]]] = None,
) -> tuple:
    """Monta ``(diagnosis, message)`` da reprovação por prioridade:
    (1) defeito duro confirmado pelo Noul, (2) suspeita dura não-resolvida de
    maior probabilidade (investigar mesmo com Noul negativo), (3) categoria de
    análise confirmada (tier qualidade — só quando não há suspeita dura)."""
    if confirmed_defects:
        confirmed_defects.sort(key=lambda d: d["probability"], reverse=True)
        top_defect = confirmed_defects[0]
        diag = DiagnosisResult(
            selected=top_defect["category"],
            confidence=top_defect["probability"],
            probabilities={top_defect["category"]: top_defect["probability"]},
            description=top_defect["description"],
            candidate_defects=top_defect["candidate_defects"],
            diagnosis_latency_ms=latency_ms,
        )
        msg = (
            f"Defeito '{top_defect['category']}' confirmado pelo Noul (P={top_defect['probability']:.1%}) "
            f"em {top_defect['file_path']}::{top_defect['symbol_name']} (linhas {top_defect['start_line']}-{top_defect['end_line']}): "
            f"{top_defect['evidence']}"
        )
        return diag, msg
    if refuted_suspicions:
        top_susp = max(refuted_suspicions, key=lambda r: r["probability"])
        diag = DiagnosisResult(
            selected=top_susp["category"],
            confidence=top_susp["probability"],
            probabilities={top_susp["category"]: top_susp["probability"]},
            description=DEFECT_DESCRIPTIONS.get(top_susp["category"], ""),
            candidate_defects=list(DEFECT_MEMBERS.get(top_susp["category"], [])),
            diagnosis_latency_ms=latency_ms,
        )
        msg = (
            f"Score insuficiente ({delivery_score:.2f}/{threshold:.2f}) com suspeita estrutural "
            f"do Needle 3 em {top_susp['file_path']}::{top_susp['symbol_name']} "
            f"(linhas {top_susp['start_line']}-{top_susp['end_line']}, categoria "
            f"'{top_susp['category']}'): {top_susp['evidence']}. O Noul refutou a hipótese "
            f"(P={top_susp['probability']:.1%} < 0.78), mas a nota está abaixo do corte: "
            "investigar o símbolo no disco antes de reavaliar."
        )
        return diag, msg
    if quality_confirmed:
        quality_confirmed.sort(key=lambda d: d["probability"], reverse=True)
        top_quality = quality_confirmed[0]
        diag = DiagnosisResult(
            selected=top_quality["category"],
            confidence=top_quality["probability"],
            probabilities={top_quality["category"]: top_quality["probability"]},
            description=top_quality["description"],
            candidate_defects=top_quality["candidate_defects"],
            diagnosis_latency_ms=latency_ms,
        )
        msg = (
            f"Categoria de análise '{top_quality['category']}' confirmada pelo Noul "
            f"(P={top_quality['probability']:.1%}) em {top_quality['file_path']}::"
            f"{top_quality['symbol_name']} (linhas {top_quality['start_line']}-{top_quality['end_line']}): "
            f"{top_quality['evidence']} Score ({delivery_score:.2f}/{threshold:.2f}) abaixo do corte: "
            "investigar e refinar o símbolo no disco antes de reavaliar."
        )
        return diag, msg
    raise ValueError(
        "Nenhuma suspeita disponível para diagnosis: chamador deve barrar "
        "reprovação sem suspeita (caminho da banda de indiferença)."
    )


def _caveat_message(delivery_score: float, threshold: float, zero_suspicions: bool) -> Optional[str]:
    """Mensagem da aprovação com ressalva na banda de indiferença (None se não é ressalva)."""
    floor = threshold - _CAVEAT_BAND
    if delivery_score >= threshold:
        return None
    if zero_suspicions:
        return (
            f"Entrega aprovada com ressalva: score {delivery_score:.2f} caiu na banda de "
            f"indiferença [{floor:.2f}, {threshold:.2f}) abaixo do corte de {threshold:.2f} e o "
            "Needle 3 não apontou nenhuma suspeita estrutural a investigar."
        )
    return (
        f"Entrega aprovada com ressalva: score {delivery_score:.2f} caiu na banda de "
        f"indiferença [{floor:.2f}, {threshold:.2f}) e o workspace permanece idêntico ao "
        "aprovado anteriormente (cache de estágios)."
    )




# NOTA: a tool `openjev_score` foi expurgada (set/2026) — todo o fluxo de
# score (validação de premissa, catálogo de rubricas, anti-premise gaming,
# circuit breaker e rubric_source) vive exclusivamente em
# `openjev_score_and_diagnose`, logo abaixo.

# Circuit breaker (MAX_ATTEMPTS) vem de score_gate — fonte única.

# NOTA (out/2026, TICKET-HARNESS-43): o pipeline de homologação de entrega
# (`agent_quality_guard`) é 100% DETERMINÍSTICO — Laya/Noul foram removidos do
# caminho crítico. A antiga arbitragem probabilística (Etapa 3) operava sobre as
# 10 categorias do Screener via as constantes `_CATEGORY_PRIORITY` e
# `MAX_ARBITRATIONS`, agora descontinuadas. O Screener permanece apenas como
# telemetria estrutural; o bloqueio determinístico vem das Camadas 1 (toolchain
# nativa) e 2 (meta-regras Tree-Sitter). Ver DEFECT_OPTIONS em src/core/schemas.py.


# ──────────────────────────────────────────────────────────────────────────────
# Agent Quality Guard — pipeline 100% DETERMINÍSTICO de homologação de entrega.
# Camada 0: testes reais | Camada 1: Guard + toolchain nativa | Camada 2: meta-regras.
# Laya RLCD / JEV Noul NÃO participam do caminho crítico (sem pesos de ML/GPU).
# ──────────────────────────────────────────────────────────────────────────────

_QUALITY_GUARD_DESCRIPTION = (
    "Validação de entrega de código 100% determinística e SEM ML (Pytest Real + "
    "Camada 1 compiladores/linters/guard + Camada 2 meta-regras Tree-Sitter). "
    "Executa em ordem fail-fast: "
    "1º testes reais (pytest em 3-4 chunks sequenciais / runner multi-stack por capability); "
    "2º NeedleASTGuard (anti-mock em produção, SQL Injection, eval perigoso, segredos hardcoded); "
    "3º bridge nativa de compiladores/linters do projeto (tsc, eslint/biome, ruff/mypy, clippy, go vet; timeout 600s); "
    "4º meta-regras universais de AST via Tree-Sitter (floating promises, silent catch, leaks). "
    "Veredito explícito: VERIFIED_PASS, VERIFIED_FAIL, NOT_APPLICABLE, INCONCLUSIVE ou EMERGENCY_RELEASE; "
    "nenhuma aprovação sem evidência real executada. Cobertura exposta em layers_executed/layers_skipped/"
    "layers_not_applicable/skip_reasons. static_diagnostic=true executa as camadas estáticas mesmo com "
    "testes vermelhos, sem nunca aprovar. "
    "Circuit breaker universal de 5 tentativas: na 5ª tentativa sem aprovação emite Laudo de "
    "Inviabilidade Técnica (EMERGENCY_RELEASE com passed=false; nunca aprova sem evidência)."
)

_LEGACY_ALIAS_DESCRIPTION = (
    "Alias legado de agent_quality_guard (MESMA implementação e assinatura). "
    "Mantido para compatibilidade retroativa total com clientes antigos. "
    "Pipeline 100% determinístico, sem Laya/Noul no caminho crítico. "
    "Circuit breaker universal de 5 tentativas com Laudo de Inviabilidade (EMERGENCY_RELEASE, "
    "nunca passed=true sem evidência executada)."
)

# Stacks cujo runner multi-stack executa a suíte real de testes.
_MULTISTACK_STACKS = (
    "REACT_NATIVE_EXPO",
    "WEB_JS_FIREBASE",
    "NODE_TS_BUN",
    "NODE_JS_GENERIC",
    "RUST_CARGO",
    "GO_MODULE",
    "DART_FLUTTER",
)
# Subconjunto JS/TS que usa o scan AST dedicado (scan_js_ast) na camada de análise.
_JS_MULTISTACK_STACKS = ("REACT_NATIVE_EXPO", "WEB_JS_FIREBASE", "NODE_TS_BUN", "NODE_JS_GENERIC")

# ── Identificadores canônicos de camada (cobertura explícita — TICKET-SGA-09) ──
LAYER_TESTS = "layer0_tests"
LAYER_AST_GUARD = "layer1_ast_guard"
LAYER_SCREENER = "layer2_screener"
LAYER_NATIVE = "layer3_native_toolchain"
LAYER_TREE_SITTER = "layer4_tree_sitter"
_ALL_LAYERS = (
    LAYER_TESTS,
    LAYER_AST_GUARD,
    LAYER_SCREENER,
    LAYER_NATIVE,
    LAYER_TREE_SITTER,
)

_NATIVE_RULE_PREFIXES = (
    "TSC_", "ESLINT_", "BIOME_", "OXLINT_", "RUFF_", "MYPY_",
    "GOVET_", "CLIPPY_", "DART_",
)


def _is_native_rule(rule_id: Any) -> bool:
    """Indica se um ``rule_id`` foi emitido por compilador/linter nativo (Camada 3)."""
    text = str(rule_id or "")
    return text.startswith(_NATIVE_RULE_PREFIXES)


def _layer_coverage(
    executed: List[str],
    statuses: Optional[Dict[str, str]] = None,
    skip_reasons: Optional[Dict[str, str]] = None,
    default_skip_reason: str = "não executada",
) -> Dict[str, Any]:
    """Monta cobertura explícita de camadas (executadas, puladas e não aplicáveis).

    TICKET-SGA-11 F3: camadas com status ``not_applicable`` saem de
    ``layers_skipped`` e vão para ``layers_not_applicable`` — deixam de simular
    omissão de cobertura (uma stack JS não tem "camadas puladas" por ser JS).
    Nenhuma camada fica implícita: as realmente omitidas seguem em
    ``layers_skipped`` com motivo em ``skip_reasons``.

    Exemplo:
        >>> cov = _layer_coverage([LAYER_TESTS])
        >>> LAYER_AST_GUARD in cov["layers_skipped"]
        True
        >>> cov_js = _layer_coverage([LAYER_TESTS], statuses={LAYER_AST_GUARD: "not_applicable"})
        >>> cov_js["layers_not_applicable"]
        ['layer1_ast_guard']
    """
    executed_set = {layer for layer in executed if layer in _ALL_LAYERS}
    status_map = {
        layer: ("executed" if layer in executed_set else "not_executed")
        for layer in _ALL_LAYERS
    }
    for layer, state in (statuses or {}).items():
        if layer in _ALL_LAYERS:
            status_map[layer] = state
    ordered_executed = [layer for layer in _ALL_LAYERS if layer in executed_set]
    ordered_not_applicable = [
        layer for layer in _ALL_LAYERS
        if layer not in executed_set and status_map.get(layer) == "not_applicable"
    ]
    ordered_skipped = [
        layer for layer in _ALL_LAYERS
        if layer not in executed_set and layer not in ordered_not_applicable
    ]
    reasons = dict(skip_reasons or {})
    not_applicable_reasons: Dict[str, str] = {}
    for layer in ordered_not_applicable:
        declared = reasons.pop(layer, None)
        if declared:
            not_applicable_reasons[layer] = declared
    for layer in ordered_skipped:
        reasons.setdefault(layer, default_skip_reason)
    for layer in ordered_executed:
        reasons.pop(layer, None)
    return {
        "layers_executed": ordered_executed,
        "layers_skipped": ordered_skipped,
        "layers_not_applicable": ordered_not_applicable,
        "skip_reasons": reasons,
        "not_applicable_reasons": not_applicable_reasons,
        "layer_status": status_map,
    }


def _fail_fast_notice(skipped: List[str]) -> str:
    """Renderiza o aviso textual de camadas não executadas por fail-fast."""
    if not skipped:
        return ""
    return " | ".join(f"{layer} [NOT_EXECUTED (fail-fast)]" for layer in skipped)


def _coerce_violation(raw: Any, model: type) -> Optional[Any]:
    """Normaliza um item bruto (dict ou ASTViolation) para o modelo tipado alvo.

    Fail-soft: item com shape inesperado é ignorado com log de depuração — uma
    ferramenta externa nunca derruba o quality gate por formato de saída.
    """
    if isinstance(raw, ASTViolation):
        return raw
    if isinstance(raw, dict):
        try:
            return model(**raw)
        except Exception as exc:  # noqa: BLE001 — shape externo imprevisível
            logger.debug("Violação ignorada (shape inválido para %s): %s", model.__name__, exc)
            return None
    return None


def _coerce_violations(items: Any, model: type) -> List[Any]:
    """Converte iterável de itens brutos em lista de modelos tipados válidos."""
    coerced: List[Any] = []
    for raw in items or []:
        item = _coerce_violation(raw, model)
        if item is not None:
            coerced.append(item)
    return coerced


def _dedup_typed(items: List[Any]) -> List[Any]:
    """Remove duplicatas tipadas por (rule_id, file_path, line_number), preservando ordem."""
    seen: set = set()
    unique: List[Any] = []
    for item in items:
        key = (
            getattr(item, "rule_id", ""),
            getattr(item, "file_path", ""),
            getattr(item, "line_number", 0),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _run_layer1_native_bridge(target_path: Path, stack: str, audit_mode: str) -> Dict[str, Any]:
    """Camada 3 — bridge nativa de compiladores/linters (determinística e honesta).

    Delega a ``MultiStackRunner.run_native_bridge``. Indisponibilidade do helper
    nunca é reportada como "0 erros": o retorno declara ``layer_status`` e
    ``critical_failures`` para que o chamador degrade para ``INCONCLUSIVE``.
    """
    empty: Dict[str, Any] = {
        "available": False, "blocking": [], "warnings": [], "tools": [], "total_files": 0,
        "layer_status": "skipped", "critical_failures": [],
        "inconclusive_reasons": ["bridge nativa indisponível (fail-open removido)"],
        "skipped_tools": {},
    }
    bridge = getattr(MultiStackRunner, "run_native_bridge", None)
    if not callable(bridge):
        logger.debug("Camada 3: MultiStackRunner.run_native_bridge indisponível.")
        return empty
    try:
        res = bridge(target_path, stack, audit_mode=audit_mode) or {}
    except Exception as exc:  # noqa: BLE001 — falha operacional vira inconclusivo
        logger.warning("Camada 3 (native bridge) falhou: %s", exc)
        empty["inconclusive_reasons"] = [f"bridge nativa falhou: {exc}"]
        return empty
    if not isinstance(res, dict):
        logger.warning("Camada 3 (native bridge) retornou %s.", type(res).__name__)
        empty["inconclusive_reasons"] = ["bridge nativa retornou payload inválido"]
        return empty
    return {
        "available": True,
        "blocking": res.get("blocking") or [],
        "warnings": res.get("warnings") or [],
        "tools": res.get("tools") or [],
        "total_files": int(res.get("total_files", 0) or 0),
        "layer_status": res.get("layer_status", "skipped"),
        "critical_failures": res.get("critical_failures") or [],
        "inconclusive_reasons": res.get("inconclusive_reasons") or [],
        "skipped_tools": res.get("skipped_tools") or {},
    }


def _layer2_unavailable(reason: str, status: str = "not_executed") -> Dict[str, Any]:
    """Cobertura explícita para a Camada 4 quando ela não pôde executar.

    Nunca devolve "0 violações" implícito: a camada fica em ``layers_skipped``
    com o motivo e o status real (``not_executed``, ``not_applicable``,
    ``error``...).
    """
    return {
        "available": False,
        "blocking": [],
        "warnings": [],
        "total_files": 0,
        "layer_status": status,
        "layers_executed": [],
        "layers_skipped": [LAYER_TREE_SITTER],
        "skip_reasons": {LAYER_TREE_SITTER: reason},
        "inconclusive_reasons": [] if status in ("not_executed", "not_applicable") else [reason],
        "engine": "unavailable",
        "languages": [],
        "languages_skipped": [],
        "files_scanned": 0,
    }


def _run_layer2_meta_rules(target_path: Path, audit_mode: str) -> Dict[str, Any]:
    """Camada 4 — meta-regras universais de AST via Tree-Sitter (cobertura explícita).

    Prefere ``TreeSitterGuard.scan_workspace_report`` (agregado já classificado
    e com cobertura); recorre a ``scan_workspace`` (iterável) classificando por
    ``severity``/``category``. Módulo ausente NÃO vira "0 violações implícitas":
    ``available`` fica False e a camada aparece em ``layers_skipped`` com motivo.
    """
    try:
        from agent_quality_guard.memory.tree_sitter_guard import TreeSitterGuard  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.debug("Camada 4: TreeSitterGuard indisponível: %s", exc)
        return _layer2_unavailable(
            f"TreeSitterGuard indisponível no ambiente: {exc}", status="not_executed"
        )

    try:
        report = getattr(TreeSitterGuard, "scan_workspace_report", None)
        if callable(report):
            res = report(target_path, audit_mode=audit_mode) or {}
            if not isinstance(res, dict):
                return _layer2_unavailable(
                    "scan_workspace_report devolveu formato inesperado", status="error"
                )
            status = str(res.get("layer_status") or "executed")
            executed = list(res.get("layers_executed") or [])
            skipped = list(res.get("layers_skipped") or [])
            engine = str(res.get("engine") or "tree_sitter")
            inconclusive = [str(reason) for reason in (res.get("inconclusive_reasons") or [])]
            files_scanned = int(res.get("files_scanned", 0) or 0)
            # TICKET-SGA-10 item 5: sem o pacote opcional tree-sitter o fallback
            # lexical determinístico cobre TODOS os arquivos elegíveis. Isso é
            # cobertura degradada declarada (engine + grammar_unavailable), não
            # ausência de evidência: não pode reprovar um workspace limpo no job
            # de resiliência do CI. Arquivos ilegíveis/excedidos continuam
            # tornando a camada genuinamente inconclusiva.
            degraded_lexical = (
                status == "inconclusive"
                and engine == "tree_sitter_lexical_fallback"
                and files_scanned > 0
                and len(inconclusive) <= 1
                and all(
                    "gramática" in reason or "fallback lexical" in reason
                    for reason in inconclusive
                )
            )
            if degraded_lexical:
                status = "executed"
                inconclusive = []
            if status == "executed" and LAYER_TREE_SITTER not in executed:
                executed = [LAYER_TREE_SITTER]
            if status != "executed" and LAYER_TREE_SITTER not in skipped:
                skipped = [LAYER_TREE_SITTER]
            return {
                "available": True,
                "blocking": res.get("blocking") or [],
                "warnings": res.get("warnings") or [],
                "total_files": int(res.get("total_files", 0) or 0),
                "layer_status": status,
                "layers_executed": executed,
                "layers_skipped": skipped,
                "skip_reasons": dict(res.get("skip_reasons") or {}),
                "inconclusive_reasons": inconclusive,
                "engine": engine,
                "degraded": bool(res.get("degraded") or degraded_lexical),
                "languages": list(res.get("languages") or []),
                "languages_skipped": list(res.get("languages_skipped") or []),
                "files_scanned": files_scanned,
            }
        scan = getattr(TreeSitterGuard, "scan_workspace", None)
        if not callable(scan):
            return _layer2_unavailable(
                "TreeSitterGuard não expõe scan_workspace_report nem scan_workspace",
                status="error",
            )
        items = scan(target_path, audit_mode=audit_mode) or []
        blocking: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        for item in items:
            data = item.to_dict() if hasattr(item, "to_dict") else (item if isinstance(item, dict) else {})
            severity = str(data.get("severity", "warning")).lower()
            if severity == "blocking" or data.get("category") in META_RULE_BLOCKING_CATEGORIES:
                blocking.append(data)
            else:
                warnings.append(data)
        return {
            "available": True,
            "blocking": blocking,
            "warnings": warnings,
            "total_files": 0,
            "layer_status": "executed",
            "layers_executed": [LAYER_TREE_SITTER],
            "layers_skipped": [],
            "skip_reasons": {},
            "inconclusive_reasons": [],
            "engine": "tree_sitter_legacy_scan",
            "languages": [],
            "languages_skipped": [],
            "files_scanned": 0,
        }
    except Exception as exc:  # noqa: BLE001 — falha reportada, nunca fail-open silencioso
        logger.warning("Camada 4 (tree-sitter) falhou — reportando INCONCLUSIVE: %s", exc)
        return _layer2_unavailable(
            f"falha ao executar a Camada 4 (Tree-Sitter): {exc}", status="error"
        )


def _layer2_from_js_scan(ast_scan: Dict[str, Any]) -> Dict[str, Any]:
    """Reaproveita a Camada 4 já executada dentro de ``scan_js_ast``.

    TICKET-SGA-10 item 1: ``MultiStackRunner.scan_js_ast`` já roda o
    ``TreeSitterGuard`` sobre os arquivos JS/TS coletados. Chamar
    ``_run_layer2_meta_rules`` depois disso varria o workspace uma segunda vez.
    Este adaptador normaliza o sub-resultado bruto (``tree_sitter_scan``) no
    mesmo contrato de ``_run_layer2_meta_rules`` sem reexecutar nada.
    """
    raw = ast_scan.get("tree_sitter_scan")
    if not isinstance(raw, dict):
        state = str(ast_scan.get("tree_sitter_layer_status") or "not_executed")
        return {
            "available": bool(ast_scan.get("tree_sitter_layer_status")),
            "blocking": [],
            "warnings": [],
            "total_files": int(ast_scan.get("tree_sitter_files", 0) or 0),
            "layer_status": state,
            "layers_executed": [LAYER_TREE_SITTER] if state == "executed" else [],
            "layers_skipped": [] if state == "executed" else [LAYER_TREE_SITTER],
            "skip_reasons": dict(ast_scan.get("tree_sitter_skip_reasons") or {}),
            "inconclusive_reasons": list(ast_scan.get("tree_sitter_inconclusive_reasons") or []),
            "engine": str(ast_scan.get("tree_sitter_engine") or "unavailable"),
            "degraded": False,
            "languages": [],
            "languages_skipped": [],
            "files_scanned": int(ast_scan.get("tree_sitter_files", 0) or 0),
        }
    state = str(raw.get("layer_status") or "executed")
    return {
        "available": True,
        "blocking": list(raw.get("blocking") or []),
        "warnings": list(raw.get("warnings") or []),
        "total_files": int(raw.get("total_files", 0) or 0),
        "layer_status": state,
        "layers_executed": list(raw.get("layers_executed") or []),
        "layers_skipped": list(raw.get("layers_skipped") or []),
        "skip_reasons": dict(raw.get("skip_reasons") or {}),
        "inconclusive_reasons": list(raw.get("inconclusive_reasons") or []),
        "engine": str(raw.get("engine") or "tree_sitter"),
        "degraded": bool(raw.get("degraded")),
        "languages": list(raw.get("languages") or []),
        "languages_skipped": list(raw.get("languages_skipped") or []),
        "files_scanned": int(raw.get("files_scanned", raw.get("total_files", 0)) or 0),
    }


def _inherited_non_approval_verdict(test_verdict: str) -> str:
    """Veredito não aprovador herdado da Camada 0 (modo diagnóstico estático)."""
    if test_verdict == VerdictState.INCONCLUSIVE.value:
        return VerdictState.INCONCLUSIVE.value
    if test_verdict == VerdictState.NOT_APPLICABLE.value:
        return VerdictState.NOT_APPLICABLE.value
    return VerdictState.VERIFIED_FAIL.value


def _static_verdict_state(
    critical_failures: List[str],
    blocking_count: int,
    inconclusive_reasons: List[str],
) -> Optional[str]:
    """Precedência estrita das camadas estáticas (TICKET-SGA-10 item 2).

    Devolve ``None`` quando a entrega pode ser aprovada; caso contrário o
    veredito não aprovador. Simetria garantida entre o ramo JS/TS e o Python:
    ferramenta crítica > bloqueio > cobertura inconclusiva.

    Exemplo:
        >>> _static_verdict_state([], 0, ["gramática ausente"]) == VerdictState.INCONCLUSIVE.value
        True
        >>> _static_verdict_state([], 0, []) is None
        True
    """
    if critical_failures:
        return VerdictState.INCONCLUSIVE.value
    if blocking_count:
        return VerdictState.VERIFIED_FAIL.value
    if inconclusive_reasons:
        return VerdictState.INCONCLUSIVE.value
    return None


async def _quality_guard_core(
    task_type: str = "codigo_producao",
    workspace: str = "",
    attempt_number: int = 1,
    pedido: str = "",
    resultado: str = "",
    engine: Any = None,
    audit_mode: str = "delta",
    static_diagnostic: bool = False,
    ctx: Optional[Context] = None,
) -> dict:
    """Homologa uma entrega com pipeline 100% determinístico (sem ML/Laya).

    Camada 0: testes reais; Camada 1: NeedleASTGuard + bridge nativa de
    compiladores/linters; Camada 2: meta-regras universais de AST via Tree-Sitter.
    A contagem de tentativas é exclusivamente server-side
    (``score_gate.resolve_attempt_number``, histórico persistido por workspace);
    ``attempt_number`` é apenas informativo e NÃO participa da decisão do
    circuit breaker. Não existe parâmetro de reset de histórico.

    ``engine`` é aceito apenas por retrocompatibilidade de assinatura e é
    IGNORADO: a validação de entrega não usa Laya RLCD/Noul nem inferência na GPU.

    ``static_diagnostic`` (TICKET-SGA-11 F2): executa as camadas estáticas mesmo
    quando a Camada 0 falha, para permitir diagnóstico da entrega com a suíte
    vermelha. NUNCA aprova: o veredito herda o resultado não-aprovador da
    Camada 0 (ou vira ``INCONCLUSIVE`` se uma ferramenta crítica falhar). O
    comportamento padrão continua sendo fail-fast.

    Exemplo:
        >>> import asyncio, tempfile
        >>> with tempfile.TemporaryDirectory() as ws:
        ...     out = asyncio.run(_quality_guard_core(workspace=ws))
        >>> out["status"] in {"APPROVED", "REJECTED", "EMERGENCY_RELEASE"}
        True
    """
    del engine  # retrocompat de assinatura: ML/Laya removido do caminho crítico
    static_diagnostic = bool(static_diagnostic) or audit_mode == "static_diagnostic"
    target_workspace = workspace.strip() if workspace and workspace.strip() else os.getcwd()

    hist = _history_get(target_workspace)
    current_attempt = _resolve_attempt_number(hist)
    if hist and hist.get("emergency"):
        # Laudo terminal já emitido: não consumir novas tentativas nem sair do
        # teto do circuit breaker (re-execução no mesmo commit é status terminal).
        current_attempt = min(
            int(hist.get("attempts", MAX_ATTEMPTS) or MAX_ATTEMPTS), MAX_ATTEMPTS,
        )
    if attempt_number != current_attempt:
        logger.info(
            "attempt_number do chamador (%s) é informativo; tentativa real server-side = %s.",
            attempt_number,
            current_attempt,
        )

    # ── 1. CIRCUIT BREAKER UNIVERSAL (limite de 5 tentativas -> laudo de inviabilidade) ──
    if current_attempt >= MAX_ATTEMPTS:
        already_emergency = bool(hist and hist.get("emergency"))
        if already_emergency:
            logger.warning(
                f"🚨 [Circuit Breaker - STATUS TERMINAL] Laudo de inviabilidade já emitido para "
                f"{target_workspace}. Re-execução sem novo commit bloqueada."
            )
            msg = (
                f"🚨 [Circuit Breaker - STATUS TERMINAL/INVIÁVEL] Limite de tentativas já consumido "
                f"({current_attempt}/{MAX_ATTEMPTS}) para este commit. NÃO HÁ APROVAÇÃO: nenhuma evidência "
                "foi executada nesta chamada. NÃO re-execute needle_verify_delivery para este mesmo commit: "
                "comite as alterações (git commit) para iniciar novo ciclo ou trate as falhas apontadas "
                "no Laudo de Inviabilidade Técnica."
            )
        else:
            logger.warning(
                f"🚨 [Circuit Breaker] {MAX_ATTEMPTS}ª tentativa atingida em {target_workspace}; "
                "nenhuma evidência executada — aprovação recusada."
            )
            msg = (
                f"🚨 [Circuit Breaker - STATUS TERMINAL/INVIÁVEL] {MAX_ATTEMPTS}ª tentativa sem aprovação. "
                "Laudo de Inviabilidade Técnica: pytest/Guard/Screener NÃO executados nesta chamada e "
                "não há evidência real que sustente uma liberação. NÃO APROVADO (passed=false). "
                "Comite as alterações para iniciar novo ciclo ou corrija as falhas registradas."
            )

        emergency_reason = (
            "circuit breaker: 5ª tentativa sem aprovação; nenhuma evidência executada "
            "(Laudo de Inviabilidade Técnica — não aprovado)"
        )
        emergency_coverage = _layer_coverage(
            [],
            statuses={layer: "emergency_release" for layer in _ALL_LAYERS},
            skip_reasons={layer: emergency_reason for layer in _ALL_LAYERS},
        )
        # TICKET-SGA-11 F4: EMERGENCY_RELEASE nunca aprova sem evidência executada.
        _history_set(target_workspace, {
            "passed": False,
            "attempts": current_attempt,
            "emergency": True
        })
        return AgentQualityGuardResponse(
            passed=False,
            status="EMERGENCY_RELEASE",
            attempt_number=current_attempt,
            approved_with_caveat=False,
            verdict_state=VerdictState.INCONCLUSIVE.value,
            evidence_executed=False,
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            pytest_success=False,
            pytest_chunks=[],
            blocking_ast_violations=[],
            structural_ast_warnings=[],
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=None,
            total_lines_scanned=None,
            inconclusive_reasons=[emergency_reason],
            **emergency_coverage,
            message=msg + " | " + _fail_fast_notice(emergency_coverage["layers_skipped"]),
        ).model_dump()

    target_path = Path(target_workspace)
    stack = MultiStackRunner.detect_stack(target_path)
    tmp_dir = Path(tempfile.gettempdir()) / "needle_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if stack in _MULTISTACK_STACKS:
        # Execução Multi-Stack especializada (suite real da linguagem detectada)
        test_res = MultiStackRunner.run_tests(target_path, stack, tmp_dir)
        test_verdict = str(test_res.get("verdict_state") or VerdictState.INCONCLUSIVE.value)
        test_evidence = bool(test_res.get("evidence", False))
        test_success = bool(test_res.get("passed", False)) and test_verdict == VerdictState.VERIFIED_PASS.value
        pytest_chunks_typed = [PytestChunkResult(
            chunk_id=f"chunk_multistack_{stack.lower()}",
            passed=test_success,
            total_passed=test_res.get("total_passed", 0),
            total_failed=test_res.get("total_failed", 0),
            execution_time_seconds=test_res.get("duration", 0.0),
            failure_details=test_res.get("failure_details", []),
        )]

        static_test_failure: Optional[Dict[str, Any]] = None
        if not test_success:
            verdict_state = (
                test_verdict
                if test_verdict in (
                    VerdictState.VERIFIED_FAIL.value,
                    VerdictState.INCONCLUSIVE.value,
                    VerdictState.NOT_APPLICABLE.value,
                )
                else VerdictState.VERIFIED_FAIL.value
            )
            if not static_diagnostic:
                _history_set(target_workspace, {
                    "passed": False,
                    "attempts": current_attempt
                })
                executed_status = "failed" if verdict_state == VerdictState.VERIFIED_FAIL.value else "inconclusive"
                reason = (
                    f"fail-fast: layer0_tests não comprovou evidência executada "
                    f"(verdict_state={verdict_state}, reason={test_res.get('reason')})"
                )
                # Honestidade de cobertura: sem evidência real a camada de testes NÃO
                # é anunciada como executada (contradiria evidence_executed=False).
                test_layers = [LAYER_TESTS] if test_evidence else []
                coverage = _layer_coverage(
                    test_layers,
                    statuses={LAYER_TESTS: executed_status},
                    skip_reasons={layer: reason for layer in _ALL_LAYERS if layer not in test_layers},
                )
                return AgentQualityGuardResponse(
                    passed=False,
                    status="REJECTED",
                    attempt_number=current_attempt,
                    pytest_success=False,
                    pytest_chunks=pytest_chunks_typed,
                    blocking_ast_violations=[],
                    structural_ast_warnings=[],
                    confirmed_defects=[],
                    dismissed_warnings=[],
                    total_files_scanned=None,
                    total_lines_scanned=None,
                    verdict_state=verdict_state,
                    evidence_executed=test_evidence,
                    inconclusive_reasons=(
                        [str(test_res.get("summary", ""))]
                        if verdict_state == VerdictState.INCONCLUSIVE.value else []
                    ),
                    pedido_audit=pedido or None,
                    resultado_audit=resultado or None,
                    **coverage,
                    message=(
                        f"🚨 [{stack} Fail-Fast] Suíte de testes não aprovada: {test_res.get('summary', '')}. "
                        f"verdict_state={verdict_state}. " + _fail_fast_notice(coverage["layers_skipped"])
                    )
                ).model_dump()
            # Modo diagnóstico estático (TICKET-SGA-11 F2): segue para as camadas
            # estáticas, mas o veredito final nunca aprova.
            static_test_failure = {
                "verdict_state": verdict_state,
                "evidence": test_evidence,
                "reason": test_res.get("reason"),
                "summary": str(test_res.get("summary", "")),
            }

        if stack in _JS_MULTISTACK_STACKS:
            ast_scan = MultiStackRunner.scan_js_ast(target_path, stack, audit_mode=audit_mode)
            # TICKET-SGA-10 item 1: a Camada 4 já executou dentro de scan_js_ast.
            # Reaproveita o resultado bruto em vez de rodar o TreeSitterGuard 2x.
            layer2 = _layer2_from_js_scan(ast_scan)

            blocking_violations = _dedup_typed(_coerce_violations(ast_scan.get("blocking", []), ASTViolation))
            structural_warnings = _dedup_typed(
                _coerce_violations(ast_scan.get("warnings", []), ASTViolation)
                + _coerce_violations(layer2["warnings"], MetaRuleViolation)
            )
            layer2_viols = (
                _coerce_violations(layer2["blocking"], MetaRuleViolation)
                + _coerce_violations(layer2["warnings"], MetaRuleViolation)
            )
            layer1_diags = [
                _coerce_violation(v, LinterDiagnostic)
                for v in ast_scan.get("blocking", []) + ast_scan.get("warnings", [])
                if _is_native_rule(v.get("rule_id"))
            ]
            layer1_diags = [item for item in layer1_diags if item is not None]
            toolchain_status = _coerce_violations(ast_scan.get("native_tools", []), ToolchainToolStatus)

            scan_executed = ([LAYER_TESTS] if test_evidence else []) + [
                layer for layer in (LAYER_NATIVE, LAYER_TREE_SITTER)
                if layer in (ast_scan.get("layers_executed") or [])
            ]
            reasons = dict(ast_scan.get("skip_reasons") or {})
            reasons.setdefault(
                LAYER_AST_GUARD,
                "AST Guard Python não aplicável à stack JS (varredura semântica JS executada na Camada 3)",
            )
            reasons.setdefault(
                LAYER_SCREENER,
                "Screener Python não aplicável à stack JS",
            )
            native_state = (ast_scan.get("layer_status") or {}).get(LAYER_NATIVE, "executed")
            ts_state = (ast_scan.get("layer_status") or {}).get(LAYER_TREE_SITTER, "executed")
            if static_test_failure is not None:
                tests_state = (
                    "failed" if static_test_failure["evidence"] else "inconclusive"
                )
            else:
                tests_state = "executed"
            coverage = _layer_coverage(
                scan_executed,
                statuses={
                    LAYER_TESTS: tests_state,
                    LAYER_NATIVE: native_state,
                    LAYER_TREE_SITTER: ts_state,
                    LAYER_AST_GUARD: "not_applicable",
                    LAYER_SCREENER: "not_applicable",
                },
                skip_reasons=reasons,
            )

            critical_failures = ast_scan.get("critical_failures") or []
            # Simetria com o ramo Python (TICKET-SGA-10 item 2): motivos
            # inconclusivos da Camada 3/4 impedem VERIFIED_PASS.
            inconclusive = list(dict.fromkeys(
                list(ast_scan.get("inconclusive_reasons") or [])
                + list(layer2.get("inconclusive_reasons") or [])
            ))
            static_verdict = _static_verdict_state(
                list(critical_failures), len(blocking_violations), inconclusive,
            )
            static_layers_ran = any(
                layer in scan_executed for layer in (LAYER_NATIVE, LAYER_TREE_SITTER)
            )
            if static_test_failure is not None:
                passed = False
                status_label = "REJECTED"
                verdict_state = _inherited_non_approval_verdict(
                    str(static_test_failure["verdict_state"])
                )
                if static_verdict is not None:
                    verdict_state = static_verdict
                message = (
                    f"🔎 [{stack} STATIC DIAGNOSTIC] Camada 0 não aprovada "
                    f"({static_test_failure['summary']}; verdict_state={verdict_state}). "
                    f"Achados estáticos: {len(blocking_violations)} bloqueio(s), "
                    f"{len(structural_warnings)} alerta(s). NÃO APROVADO."
                )
            elif static_verdict == VerdictState.INCONCLUSIVE.value and critical_failures:
                passed = False
                verdict_state = VerdictState.INCONCLUSIVE.value
                status_label = "REJECTED"
                message = (
                    f"🚨 [{stack}] Camada 3 INCONCLUSIVE: ferramenta crítica falhou/timeout "
                    f"({', '.join(critical_failures)}). Nenhum veredito silencioso de '0 erros'. "
                    + _fail_fast_notice([layer for layer in coverage["layers_skipped"]])
                )
            elif static_verdict == VerdictState.VERIFIED_FAIL.value:
                passed = False
                verdict_state = VerdictState.VERIFIED_FAIL.value
                status_label = "REJECTED"
                message = (
                    f"🚨 [{stack}] {test_res.get('summary', '')} | "
                    f"{len(blocking_violations)} bloqueio(s); {len(structural_warnings)} alerta(s)."
                )
            elif static_verdict == VerdictState.INCONCLUSIVE.value:
                passed = False
                verdict_state = VerdictState.INCONCLUSIVE.value
                status_label = "REJECTED"
                message = (
                    f"🚨 [{stack}] Camada(s) estática(s) INCONCLUSIVE: "
                    f"{'; '.join(inconclusive[:3])}. "
                    + _fail_fast_notice([layer for layer in coverage["layers_skipped"]])
                )
            else:
                passed = True
                verdict_state = VerdictState.VERIFIED_PASS.value
                status_label = "APPROVED"
                message = (
                    f"✅ [{stack}] {test_res.get('summary', '')} | "
                    f"Camada 3 ({native_state}) sem bloqueios; "
                    f"Camada 4 ({ts_state}) com {len(layer2_viols)} achado(s) de meta-regra."
                )

            _history_set(target_workspace, {
                "passed": passed,
                "attempts": current_attempt
            })
            return AgentQualityGuardResponse(
                passed=passed,
                status=status_label,
                attempt_number=current_attempt,
                pytest_success=test_success,
                pytest_chunks=pytest_chunks_typed,
                blocking_ast_violations=blocking_violations,
                structural_ast_warnings=structural_warnings,
                confirmed_defects=[],
                dismissed_warnings=[],
                total_files_scanned=int(ast_scan.get("total_files", 0) or 0),
                total_lines_scanned=int(ast_scan.get("total_lines", 0) or 0),
                verdict_state=verdict_state,
                evidence_executed=bool(test_evidence or static_layers_ran),
                inconclusive_reasons=inconclusive,
                pedido_audit=pedido or None,
                resultado_audit=resultado or None,
                layer1_toolchain_diagnostics=layer1_diags,
                toolchain_status=toolchain_status,
                layer2_meta_rule_violations=layer2_viols,
                **coverage,
                message=message,
            ).model_dump()

        # Stacks não-JS (Rust/Go/Dart): testes reais passaram; Camada 3 nativa é aplicável.
        layer1 = _run_layer1_native_bridge(target_path, stack, audit_mode)
        layer1_blocking = _coerce_violations(layer1["blocking"], LinterDiagnostic)
        layer1_warnings = _coerce_violations(layer1["warnings"], LinterDiagnostic)
        toolchain_status = _coerce_violations(layer1["tools"], ToolchainToolStatus)
        critical_failures = layer1.get("critical_failures") or []
        inconclusive = layer1.get("inconclusive_reasons") or []
        non_js_skips = {
            LAYER_AST_GUARD: "AST Guard Python não aplicável à stack da linguagem",
            LAYER_SCREENER: "Screener Python não aplicável à stack da linguagem",
            LAYER_TREE_SITTER: "meta-regras universais não aplicáveis à stack da linguagem",
        }
        if static_test_failure is not None:
            tests_state = "failed" if static_test_failure["evidence"] else "inconclusive"
        else:
            tests_state = "executed"
        native_executed = [LAYER_TESTS, LAYER_NATIVE]
        if static_test_failure is not None and not static_test_failure["evidence"]:
            native_executed = [LAYER_NATIVE]
        coverage = _layer_coverage(
            native_executed,
            statuses={
                LAYER_TESTS: tests_state,
                LAYER_NATIVE: layer1.get("layer_status", "skipped"),
                LAYER_AST_GUARD: "not_applicable",
                LAYER_SCREENER: "not_applicable",
                LAYER_TREE_SITTER: "not_applicable",
            },
            skip_reasons=non_js_skips,
        )
        if static_test_failure is not None:
            passed = False
            status_label = "REJECTED"
            verdict_state = _inherited_non_approval_verdict(
                str(static_test_failure["verdict_state"])
            )
            if critical_failures or inconclusive:
                verdict_state = VerdictState.INCONCLUSIVE.value
            message = (
                f"🔎 [{stack} STATIC DIAGNOSTIC] Camada 0 não aprovada "
                f"({static_test_failure['summary']}; verdict_state={verdict_state}). "
                f"Camada 3 nativa: {len(layer1_blocking)} bloqueio(s), "
                f"{len(layer1_warnings)} alerta(s). NÃO APROVADO."
            )
        elif critical_failures:
            passed = False
            verdict_state = VerdictState.INCONCLUSIVE.value
            status_label = "REJECTED"
            message = (
                f"🚨 [{stack}] Camada 3 INCONCLUSIVE: ferramenta crítica falhou/timeout "
                f"({', '.join(critical_failures)})."
            )
        elif layer1_blocking:
            passed = False
            verdict_state = VerdictState.VERIFIED_FAIL.value
            status_label = "REJECTED"
            message = f"🚨 [{stack}] {len(layer1_blocking)} bloqueio(s) na Camada 3 nativa."
        else:
            passed = True
            verdict_state = VerdictState.VERIFIED_PASS.value
            status_label = "APPROVED"
            message = f"✅ [{stack}] {test_res.get('summary', '')} | Camada 3 nativa sem bloqueios."
        _history_set(target_workspace, {"passed": passed, "attempts": current_attempt})
        return AgentQualityGuardResponse(
            passed=passed,
            status=status_label,
            attempt_number=current_attempt,
            pytest_success=test_success,
            pytest_chunks=pytest_chunks_typed,
            blocking_ast_violations=layer1_blocking,
            structural_ast_warnings=layer1_warnings,
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=int(layer1.get("total_files", 0) or 0),
            total_lines_scanned=None,
            verdict_state=verdict_state,
            evidence_executed=bool(
                (static_test_failure or {}).get("evidence", test_evidence)
                or layer1.get("layer_status") == "executed"
            ),
            inconclusive_reasons=inconclusive,
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            layer1_toolchain_diagnostics=layer1_blocking + layer1_warnings,
            toolchain_status=toolchain_status,
            **coverage,
            message=message,
        ).model_dump()

    if not stack.startswith("PYTHON"):
        # Stack sem runner mapeado: NUNCA exigir 'tests/test_*.py' de repositório
        # não-Python. Veredito honesto: inconclusivo, passed=False.
        reason = f"stack '{stack}' sem runner de testes mapeado; nenhuma camada executada"
        coverage = _layer_coverage(
            [],
            statuses={layer: "not_applicable" for layer in _ALL_LAYERS},
            skip_reasons={layer: reason for layer in _ALL_LAYERS},
        )
        _history_set(target_workspace, {"passed": False, "attempts": current_attempt})
        return AgentQualityGuardResponse(
            passed=False,
            status="REJECTED",
            attempt_number=current_attempt,
            pytest_success=False,
            pytest_chunks=[],
            blocking_ast_violations=[],
            structural_ast_warnings=[],
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=None,
            total_lines_scanned=None,
            verdict_state=VerdictState.INCONCLUSIVE.value,
            evidence_executed=False,
            inconclusive_reasons=[reason],
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            **coverage,
            message=(
                f"🚨 [Stack {stack} INCONCLUSIVE] Nenhum runner determinístico mapeado. "
                "Nenhum teste coletado/executado — a entrega não pode ser aprovada sem "
                "evidência real executada. "
                + _fail_fast_notice(coverage["layers_skipped"])
            ),
        ).model_dump()

    # ── Etapa 0: Execução Sequencial de Testes Reais em Chunks (Pytest 1º Lugar) ──
    if ctx:
        try:
            await ctx.report_progress(1, 4)
        except Exception as ctx_err:
            logger.debug(f"Progress report etapa 0 falhou: {ctx_err}")

    try:
        import importlib
        import agent_quality_guard.orchestrator.pytest_chunk_runner as pcr_mod
        pcr_mod = importlib.reload(pcr_mod)
        runner_cls = getattr(pcr_mod, "PytestChunkRunner", PytestChunkRunner)
    except Exception as reload_err:
        logger.debug(f"Recorrendo ao PytestChunkRunner em memória: {reload_err}")
        runner_cls = PytestChunkRunner

    pytest_out = runner_cls.run_all_chunks(
        workspace_path=target_workspace,
        audit_mode=audit_mode,
        timeout_per_chunk=600,
        max_total_timeout=None,
    )
    raw_chunks = pytest_out.get("chunks", [])
    pytest_success = bool(pytest_out.get("pytest_success", False))
    chunks_evidence = all(
        bool(c.get("evidence", bool(c.get("total_passed") or c.get("total_failed"))))
        for c in raw_chunks
    )
    if not pytest_success or not chunks_evidence:
        pytest_success = False

    pytest_chunks_typed = []
    for c in raw_chunks:
        pytest_chunks_typed.append(PytestChunkResult(
            chunk_id=c["chunk_id"],
            passed=bool(c["passed"]) and bool(
                c.get("evidence", bool(c.get("total_passed") or c.get("total_failed")))
            ),
            total_passed=c["total_passed"],
            total_failed=c["total_failed"],
            execution_time_seconds=c["execution_time_seconds"],
            failure_details=c.get("failure_details", []),
        ))

    python_static_failure: Optional[Dict[str, Any]] = None
    if not pytest_chunks_typed:
        no_tests_reason = "fail-fast: nenhum teste Python real coletado (tests/test_*.py ausente)"
        if not static_diagnostic:
            _history_set(target_workspace, {
                "passed": False,
                "attempts": current_attempt
            })
            coverage = _layer_coverage(
                [LAYER_TESTS],
                statuses={LAYER_TESTS: "failed"},
                skip_reasons={layer: no_tests_reason for layer in _ALL_LAYERS if layer != LAYER_TESTS},
            )
            return AgentQualityGuardResponse(
                passed=False,
                status="REJECTED",
                attempt_number=current_attempt,
                pytest_success=False,
                pytest_chunks=[],
                blocking_ast_violations=[],
                structural_ast_warnings=[],
                confirmed_defects=[],
                dismissed_warnings=[],
                total_files_scanned=None,
                total_lines_scanned=None,
                verdict_state=VerdictState.VERIFIED_FAIL.value,
                evidence_executed=False,
                inconclusive_reasons=[no_tests_reason],
                pedido_audit=pedido or None,
                resultado_audit=resultado or None,
                **coverage,
                message=(
                    "🚨 [Pytest Fail-Fast: TESTES OBRIGATÓRIOS NÃO ENCONTRADOS]\n"
                    f"Nenhum teste coletado — workspace Python '{target_workspace}' sem "
                    "suíte real não pode ser homologado (aprovação sem executar teste é proibida).\n"
                    "👉 AÇÃO OBRIGATÓRIA DO AGENTE: Crie o arquivo de teste 'tests/test_<funcionalidade>.py' "
                    "com testes automatizados reais (sem mocks em src/) cobrindo os caminhos críticos e submeta novamente. "
                    + _fail_fast_notice(coverage["layers_skipped"])
                )
            ).model_dump()
        python_static_failure = {
            "verdict_state": VerdictState.VERIFIED_FAIL.value,
            "evidence": False,
            "summary": no_tests_reason,
        }

    if not pytest_success and not static_diagnostic:
        _history_set(target_workspace, {
            "passed": False,
            "attempts": current_attempt
        })
        failed_evidence = bool(pytest_chunks_typed) and any(
            bool(c.total_passed or c.total_failed) for c in pytest_chunks_typed
        )
        verdict_state = (
            VerdictState.VERIFIED_FAIL.value if chunks_evidence
            else VerdictState.INCONCLUSIVE.value
        )
        reason = "fail-fast: layer0_tests falhou ou não comprovou evidência executada"
        coverage = _layer_coverage(
            [LAYER_TESTS],
            statuses={LAYER_TESTS: "failed" if chunks_evidence else "inconclusive"},
            skip_reasons={layer: reason for layer in _ALL_LAYERS if layer != LAYER_TESTS},
        )
        return AgentQualityGuardResponse(
            passed=False,
            status="REJECTED",
            attempt_number=current_attempt,
            pytest_success=False,
            pytest_chunks=pytest_chunks_typed,
            blocking_ast_violations=[],
            structural_ast_warnings=[],
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=None,
            total_lines_scanned=None,
            verdict_state=verdict_state,
            evidence_executed=failed_evidence,
            inconclusive_reasons=(
                [pytest_out.get("message", "suíte de testes não aprovada")]
                if verdict_state == VerdictState.INCONCLUSIVE.value else []
            ),
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            **coverage,
            message=(
                "🚨 [Pytest Fail-Fast] Suíte de testes reais falhou em um ou mais chunks. "
                f"verdict_state={verdict_state}. Resolva as falhas antes de prosseguir. "
                + _fail_fast_notice(coverage["layers_skipped"])
            )
        ).model_dump()

    if not pytest_success and python_static_failure is None:
        # MODO DIAGNÓSTICO ESTÁTICO (TICKET-SGA-11 F2): segue para as camadas
        # estáticas, mas o veredito final permanece não-aprovador.
        failed_evidence = bool(pytest_chunks_typed) and any(
            bool(c.total_passed or c.total_failed) for c in pytest_chunks_typed
        )
        python_static_failure = {
            "verdict_state": (
                VerdictState.VERIFIED_FAIL.value if chunks_evidence
                else VerdictState.INCONCLUSIVE.value
            ),
            "evidence": failed_evidence,
            "summary": "suíte de testes reais não aprovada",
        }

    # ── Etapa 1: NeedleASTGuard (Camada 1: Bloqueios de Segurança/Integridade) ──
    if ctx:
        try:
            await ctx.report_progress(2, 4)
        except Exception as ctx_err:
            logger.debug(f"Progress report etapa 1 falhou: {ctx_err}")

    ast_report = NeedleASTGuard.scan_workspace_report(target_workspace, audit_mode=audit_mode)
    ast_violations = list(ast_report["violations"])
    blocking_violations = []
    for v in ast_violations:
        blocking_violations.append(ASTViolation(
            rule_id=v.rule_id,
            category=v.category,
            file_path=v.file_path,
            line_number=v.line_number,
            code_snippet=v.snippet,
            message=v.message,
            suggestion=v.suggestion,
        ))

    if blocking_violations:
        _history_set(target_workspace, {
            "passed": False,
            "attempts": current_attempt
        })
        reason = "fail-fast: layer1_ast_guard encontrou violação bloqueante"
        if python_static_failure is not None:
            tests_evidence = bool(python_static_failure.get("evidence"))
            guard_executed = ([LAYER_TESTS] if tests_evidence else []) + [LAYER_AST_GUARD]
            guard_tests_state = "failed" if tests_evidence else "inconclusive"
        else:
            guard_executed = [LAYER_TESTS, LAYER_AST_GUARD]
            guard_tests_state = "executed"
        coverage = _layer_coverage(
            guard_executed,
            statuses={LAYER_TESTS: guard_tests_state, LAYER_AST_GUARD: "failed"},
            skip_reasons={
                layer: reason for layer in (LAYER_SCREENER, LAYER_NATIVE, LAYER_TREE_SITTER)
            },
        )
        return AgentQualityGuardResponse(
            passed=False,
            status="REJECTED",
            attempt_number=current_attempt,
            pytest_success=pytest_success,
            pytest_chunks=pytest_chunks_typed,
            blocking_ast_violations=blocking_violations,
            structural_ast_warnings=[],
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=int(ast_report.get("files_scanned", 0)),
            total_lines_scanned=None,
            verdict_state=VerdictState.VERIFIED_FAIL.value,
            evidence_executed=True,
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            **coverage,
            message=(
                f"🚨 [Needle 3 AST Guardrail] {len(blocking_violations)} violação(ões) de "
                "segurança/integridade bloqueante(s) detectada(s). "
                + _fail_fast_notice(coverage["layers_skipped"])
            )
        ).model_dump()

    # ── Etapa 2: NeedleASTScreener (Camada 2: Alertas de Qualidade Estrutural) ──
    if ctx:
        try:
            await ctx.report_progress(3, 4)
        except Exception as ctx_err:
            logger.debug(f"Progress report etapa 2 falhou: {ctx_err}")

    symbols = extract_workspace_ast_symbols(target_workspace)
    screener_report = screen_workspace_report(symbols, workspace_root=target_workspace)
    suspicions_map = screener_report["suspicions"]
    structural_warnings = []

    suspicions_pairs = [
        (sym_key, s)
        for sym_key, suspicions_list in suspicions_map.items()
        for s in suspicions_list
    ]
    for sym_key, s in suspicions_pairs:
        file_path = sym_key.split("::")[0] if "::" in sym_key else target_workspace
        warning = ASTViolation(
            rule_id=f"screener_{s.category}",
            category=s.category,
            file_path=s.file_path or file_path,
            line_number=s.line_number,
            code_snippet=s.snippet or s.trigger_evidence or "",
            message=s.trigger_evidence or f"Alerta de qualidade estrutural: {s.category}",
            suggestion=f"Refatorar símbolo em {sym_key} para resolver alerta de {s.category}."
        )
        structural_warnings.append(warning)

    # ── Etapa 3: Camada 3 — bridge nativa de compiladores/linters (determinística) ──
    layer1 = _run_layer1_native_bridge(target_path, stack, audit_mode)
    layer1_blocking = _coerce_violations(layer1["blocking"], LinterDiagnostic)
    layer1_warnings = _coerce_violations(layer1["warnings"], LinterDiagnostic)
    toolchain_status = _coerce_violations(layer1["tools"], ToolchainToolStatus)

    # ── Etapa 4: Camada 4 — meta-regras universais de AST via Tree-Sitter (determinística) ──
    layer2 = _run_layer2_meta_rules(target_path, audit_mode)
    layer2_blocking = _coerce_violations(layer2["blocking"], MetaRuleViolation)
    layer2_warnings = _coerce_violations(layer2["warnings"], MetaRuleViolation)

    # Telemetria estrutural: Screener legado + alertas das Camadas 3/4 (não bloqueiam).
    structural_warnings = structural_warnings + layer1_warnings + layer2_warnings
    layer1_toolchain_diagnostics = layer1_blocking + layer1_warnings
    layer2_meta_rule_violations = layer2_blocking + layer2_warnings

    skip_reasons = {
        f"tool:{tool_name}": str(reason)
        for tool_name, reason in (layer1.get("skipped_tools") or {}).items()
    }
    if not layer2.get("available", False):
        skip_reasons[LAYER_TREE_SITTER] = "TreeSitterGuard indisponível no ambiente"
    skip_reasons.update(ast_report.get("skip_reasons") or {})
    skip_reasons.update(screener_report.get("skip_reasons") or {})
    skip_reasons.update(layer2.get("skip_reasons") or {})

    native_state = layer1.get("layer_status", "skipped")
    ts_state = str(layer2.get("layer_status") or ("executed" if layer2.get("available") else "not_executed"))
    ast_state = str(ast_report.get("layer_status") or "not_executed")
    screener_state = str(screener_report.get("layer_status") or "not_executed")
    if python_static_failure is not None:
        tests_state = (
            "failed" if python_static_failure.get("evidence") else "inconclusive"
        )
    else:
        tests_state = "executed"
    layer_states = {
        LAYER_TESTS: tests_state,
        LAYER_AST_GUARD: ast_state,
        LAYER_SCREENER: screener_state,
        LAYER_NATIVE: native_state,
        LAYER_TREE_SITTER: ts_state,
    }
    executed_layers = [layer for layer in _ALL_LAYERS if layer_states[layer] == "executed"]
    if (
        python_static_failure is not None
        and python_static_failure.get("evidence")
        and LAYER_TESTS not in executed_layers
    ):
        # Os testes EXECUTARAM (com falha): a camada entra como executada com
        # status ``failed`` para não esconder evidência real coletada.
        executed_layers = [LAYER_TESTS] + executed_layers
    coverage = _layer_coverage(
        executed_layers,
        statuses=layer_states,
        skip_reasons=skip_reasons,
    )
    critical_failures = layer1.get("critical_failures") or []
    inconclusive_reasons = list(layer1.get("inconclusive_reasons") or [])
    for layer_id, state in layer_states.items():
        if state in ("inconclusive", "error", "timeout"):
            source = (
                ast_report if layer_id == LAYER_AST_GUARD
                else screener_report if layer_id == LAYER_SCREENER
                else layer2
            )
            layer_reasons = [str(reason) for reason in (source.get("inconclusive_reasons") or [])]
            if not layer_reasons:
                layer_reasons = [f"{layer_id} terminou com status={state}"]
            inconclusive_reasons.extend(layer_reasons)

    # Bloqueio determinístico combinado (NeedleASTGuard já barrou acima).
    combined_blocking = layer1_blocking + layer2_blocking
    if critical_failures:
        _history_set(target_workspace, {"passed": False, "attempts": current_attempt})
        return AgentQualityGuardResponse(
            passed=False,
            status="REJECTED",
            attempt_number=current_attempt,
            pytest_success=pytest_success,
            pytest_chunks=pytest_chunks_typed,
            blocking_ast_violations=combined_blocking,
            structural_ast_warnings=structural_warnings,
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=int(ast_report.get("files_scanned", 0)),
            total_lines_scanned=sum(sym.end_line - sym.start_line + 1 for sym in symbols),
            verdict_state=VerdictState.INCONCLUSIVE.value,
            evidence_executed=True,
            inconclusive_reasons=inconclusive_reasons,
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            layer1_toolchain_diagnostics=layer1_toolchain_diagnostics,
            toolchain_status=toolchain_status,
            layer2_meta_rule_violations=layer2_meta_rule_violations,
            **coverage,
            message=(
                "🚨 [Quality Guard INCONCLUSIVE] Camada 3: ferramenta crítica falhou/timeout "
                f"({', '.join(critical_failures)}). Nenhum veredito silencioso de '0 erros'. "
                + _fail_fast_notice(coverage["layers_skipped"])
            )
        ).model_dump()

    if combined_blocking:
        _history_set(target_workspace, {
            "passed": False,
            "attempts": current_attempt
        })
        return AgentQualityGuardResponse(
            passed=False,
            status="REJECTED",
            attempt_number=current_attempt,
            pytest_success=pytest_success,
            pytest_chunks=pytest_chunks_typed,
            blocking_ast_violations=combined_blocking,
            structural_ast_warnings=structural_warnings,
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=int(ast_report.get("files_scanned", 0)),
            total_lines_scanned=sum(sym.end_line - sym.start_line + 1 for sym in symbols),
            verdict_state=VerdictState.VERIFIED_FAIL.value,
            evidence_executed=True,
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            layer1_toolchain_diagnostics=layer1_toolchain_diagnostics,
            toolchain_status=toolchain_status,
            layer2_meta_rule_violations=layer2_meta_rule_violations,
            **coverage,
            message=(
                f"🚨 [Quality Guard Determinístico] {len(layer1_blocking)} erro(s) de toolchain "
                f"(Camada 3) e {len(layer2_blocking)} violação(ões) de meta-regra bloqueante(s) "
                f"(Camada 4). Alertas estruturais registrados: {len(structural_warnings)}."
            )
        ).model_dump()

    if inconclusive_reasons:
        _history_set(target_workspace, {"passed": False, "attempts": current_attempt})
        return AgentQualityGuardResponse(
            passed=False,
            status="REJECTED",
            attempt_number=current_attempt,
            pytest_success=pytest_success,
            pytest_chunks=pytest_chunks_typed,
            blocking_ast_violations=combined_blocking,
            structural_ast_warnings=structural_warnings,
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=int(ast_report.get("files_scanned", 0)),
            total_lines_scanned=sum(sym.end_line - sym.start_line + 1 for sym in symbols),
            verdict_state=VerdictState.INCONCLUSIVE.value,
            evidence_executed=True,
            inconclusive_reasons=inconclusive_reasons,
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            layer1_toolchain_diagnostics=layer1_toolchain_diagnostics,
            toolchain_status=toolchain_status,
            layer2_meta_rule_violations=layer2_meta_rule_violations,
            **coverage,
            message=(
                "🚨 [Quality Guard INCONCLUSIVE] Camada(s) de AST não comprovaram cobertura "
                f"completa: {'; '.join(inconclusive_reasons[:3])}. "
                + _fail_fast_notice(coverage["layers_skipped"])
            ),
        ).model_dump()

    if python_static_failure is not None:
        _history_set(target_workspace, {"passed": False, "attempts": current_attempt})
        inherited = _inherited_non_approval_verdict(
            str(python_static_failure.get("verdict_state"))
        )
        static_reason = (
            "diagnóstico estático: Camada 0 não aprovada "
            f"({python_static_failure.get('summary')})"
        )
        if inherited == VerdictState.INCONCLUSIVE.value and not inconclusive_reasons:
            inconclusive_reasons = [static_reason]
        return AgentQualityGuardResponse(
            passed=False,
            status="REJECTED",
            attempt_number=current_attempt,
            pytest_success=pytest_success,
            pytest_chunks=pytest_chunks_typed,
            blocking_ast_violations=combined_blocking,
            structural_ast_warnings=structural_warnings,
            confirmed_defects=[],
            dismissed_warnings=[],
            total_files_scanned=int(ast_report.get("files_scanned", 0)),
            total_lines_scanned=sum(sym.end_line - sym.start_line + 1 for sym in symbols),
            verdict_state=inherited,
            evidence_executed=bool(coverage["layers_executed"]),
            inconclusive_reasons=inconclusive_reasons,
            pedido_audit=pedido or None,
            resultado_audit=resultado or None,
            layer1_toolchain_diagnostics=layer1_toolchain_diagnostics,
            toolchain_status=toolchain_status,
            layer2_meta_rule_violations=layer2_meta_rule_violations,
            **coverage,
            message=(
                f"🔎 [Quality Guard STATIC DIAGNOSTIC] Camada 0 não aprovada "
                f"({python_static_failure.get('summary')}; verdict_state={inherited}). "
                f"Achados estáticos: {len(combined_blocking)} bloqueio(s), "
                f"{len(structural_warnings)} alerta(s). NÃO APROVADO."
            ),
        ).model_dump()

    # ── Aprovação Determinística (sem arbitragem ML/Laya) ──
    if ctx:
        try:
            await ctx.report_progress(4, 4)
        except Exception as ctx_err:
            logger.debug(f"Progress report final falhou: {ctx_err}")

    _history_set(target_workspace, {
        "passed": True,
        "attempts": current_attempt
    })

    approval_msg = (
        "Entrega 100% aprovada deterministicamente por testes reais (pytest), Needle 3 AST Guard (Camada 1) "
        "e meta-regras universais de AST (Camada 4). "
        f"Toolchain nativa sem bloqueios; {len(structural_warnings)} alerta(s) estrutural(is) como telemetria."
    )

    return AgentQualityGuardResponse(
        passed=True,
        status="APPROVED",
        attempt_number=current_attempt,
        pytest_success=True,
        pytest_chunks=pytest_chunks_typed,
        blocking_ast_violations=[],
        structural_ast_warnings=structural_warnings,
        confirmed_defects=[],
        dismissed_warnings=[],
        total_files_scanned=max(
            int(ast_report.get("files_scanned", 0) or 0),
            int(layer1.get("total_files", 0) or 0),
            int(layer2.get("files_scanned", 0) or 0),
        ),
        total_lines_scanned=sum(sym.end_line - sym.start_line + 1 for sym in symbols),
        verdict_state=VerdictState.VERIFIED_PASS.value,
        evidence_executed=True,
        pedido_audit=pedido or None,
        resultado_audit=resultado or None,
        layer1_toolchain_diagnostics=layer1_toolchain_diagnostics,
        toolchain_status=toolchain_status,
        layer2_meta_rule_violations=layer2_meta_rule_violations,
        **coverage,
        message=approval_msg,
    ).model_dump()


# ──────────────────────────────────────────────────────────────────────────────
# Registro das tools: nome canônico `agent_quality_guard` + alias legado
# `needle_verify_delivery` (MESMA implementação, retrocompatibilidade total).
# ──────────────────────────────────────────────────────────────────────────────
agent_quality_guard = app.tool(
    name="agent_quality_guard", description=_QUALITY_GUARD_DESCRIPTION
)(_quality_guard_core)
needle_verify_delivery = app.tool(
    name="needle_verify_delivery", description=_LEGACY_ALIAS_DESCRIPTION
)(_quality_guard_core)


def main() -> None:
    import ctypes

    # ── Proteção contra processos órfãos (Linux PR_SET_PDEATHSIG) ───────────
    # Garante que o Kernel envie SIGTERM no microssegundo em que a IDE / Language Server pai morrer
    try:
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6")
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception as e:
        logger.debug("prctl(PR_SET_PDEATHSIG) não suportado ou indisponível: %s", e)

    def _graceful_exit(signum, frame) -> None:
        logger.info("Sinal %s recebido no servidor MCP, encerrando de forma limpa...", signum)
        os._exit(0)

    try:
        signal.signal(signal.SIGTERM, _graceful_exit)
        signal.signal(signal.SIGINT, _graceful_exit)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _graceful_exit)
    except Exception as e:
        logger.debug("Sinais de terminação não puderam ser registrados: %s", e)

    # TICKET-HARNESS-47: sem pré-load de motor, sem alocação de VRAM, sem lock
    # de GPU e sem socket IPC. O boot fica restrito ao registro das tools
    # determinísticas (cirurgic_search/agent_quality_guard e aliases).
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
