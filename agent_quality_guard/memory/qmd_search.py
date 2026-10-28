"""QMD Search: Motor de Busca Híbrido Cirúrgico (Ripgrep + Needle 3 + RRF).

Combina busca léxica de altíssima velocidade (Grep/Ripgrep) com recuperação semântica
e relacional de AST (Needle 3) através de Reciprocal Rank Fusion (RRF) em CPU.

Refinamentos de Produção:
1. Fallback Assimétrico: degradação suave quando uma das fontes não encontra resultados.
2. Fusão Macro-Micro: une o ranqueamento de arquivos (macro) aos blocos AST e linhas exatas (micro).
3. Atualização em Tempo Real: o Grep ao vivo captura edições recentes imediatamente.
"""

import asyncio
import itertools
import logging
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..core.schemas import CHARS_PER_TOKEN, Needle3Query, Needle3Response, Needle3Snippet
from .ast_cache import file_size_allowed
from .local_needle import (
    LocalNeedleEngine,
    SandboxViolationError,
    UNIVERSAL_EXCLUDED_DIRS,
    ensure_not_system_restricted,
    ensure_within_allowed_roots,
    exclude_patterns_to_rg_globs,
    merge_exclude_patterns,
    path_is_excluded,
    resolve_allowed_roots,
    substantive_terms,
)

logger = logging.getLogger("agent_quality_guard.qmd")

# TICKET-SGA-03 — timeout rígido da lane léxica (ripgrep). Estourou => a lane
# degrada (status='partial' + degraded_lanes=['ripgrep_lexical']), nunca um
# sucesso vazio silencioso.
RIPGREP_TIMEOUT_SECONDS = 5.0

# Preview literal é fragmento de contexto, nunca dump de arquivo: precisa
# caber no mesmo teto de caracteres do chunker adaptativo do Needle.
MAX_LITERAL_PREVIEW_CHARS = 1800

# TICKET-SGA-04 — Guardrails de I/O da busca híbrida:
# * Timeout GLOBAL configurável por requisição (padrão 15 s). `timeout <= 0`
#   desativa a cota — semântica alinhada a SpawnSubagentRequest.timeout.
# * Subprocesso do ripgrep tem cota própria (5 s), recortada pelo timeout global.
DEFAULT_SEARCH_TIMEOUT_S = 15.0
DEFAULT_RIPGREP_TIMEOUT_S = 5.0

# Arquivos de código executável são o alvo usual da busca cirúrgica; docs e
# dados continuam indexados, mas não devem vencer o código que implementa o
# comportamento consultado.
SOURCE_CODE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts",
        ".cts", ".rs", ".go", ".dart", ".kt", ".kts", ".java", ".cs", ".rb",
        ".php", ".swift", ".c", ".h", ".cc", ".cpp", ".hpp", ".sh", ".bash",
    }
)
SOURCE_CODE_BOOST = 1.25
DOC_SCOPE_MARKERS = frozenset({"docs", "doc", "personas"})
DOC_SCOPE_EXTENSIONS = frozenset({".md", ".json"})

# TICKET-SGA-05 — Denylist UNIVERSAL (segura em qualquer repositório): caches,
# VCS e build artifacts. Regras particulares de ambiente local (ex.:
# ``panel/*.js``, ``.agent-teams``, ``screenshots``) foram DESACOPLADAS daqui:
# entram via ``.cirurgicignore`` na raiz do workspace ou pelo parâmetro
# opcional ``exclude_patterns`` das tools de busca. Os diretórios vêm da fonte
# única ``UNIVERSAL_EXCLUDED_DIRS`` (memory/local_needle.py), garantindo
# paridade exata com o walk da rota AST.
IGNORED_GLOBS = tuple(
    f"!**/{directory}/**" for directory in sorted(UNIVERSAL_EXCLUDED_DIRS)
) + (
    # Artefatos derivados universais (duplicam a semântica do código-fonte com
    # conteúdo minificado e poluem o recall léxico).
    "!**/*.bundle.js",
    "!**/*.min.js",
    "!**/*.min.mjs",
    "!**/*.map",
)

GREP_EXCLUDED_DIRS = tuple(sorted(UNIVERSAL_EXCLUDED_DIRS))

# Arquivos densos repetem um identificador muitas vezes; truncar em 5 esconde
# a evidência conjuntiva (cobertura de vários termos distintos da query).
DEFAULT_MAX_MATCHES_PER_FILE = 20


def _escape_literal(text: str) -> str:
    """Escapa metacaracteres de regex sem escapar espaços.

    O ``re.escape`` também escapa o espaço, o que o motor Rust do ripgrep
    rejeita como sequência de escape inválida.
    """
    return re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", text)


def _scope_is_documentation(scope: Optional[List[str]]) -> bool:
    """Retorna True quando o chamador escopou explicitamente docs/dados."""
    if not scope:
        return False
    for item in scope:
        lowered = str(item).lower()
        if Path(lowered).suffix in DOC_SCOPE_EXTENSIONS:
            return True
        if {part for part in Path(lowered).parts} & DOC_SCOPE_MARKERS:
            return True
    return False


def _type_multiplier(file_path: str, documentation_scope: bool) -> float:
    """Multiplicador tipológico do RRF para um arquivo."""
    if documentation_scope:
        return 1.0
    if Path(file_path).suffix.lower() in SOURCE_CODE_EXTENSIONS:
        return SOURCE_CODE_BOOST
    return 1.0

# Candidatos para o binário do ripgrep (rg)
KNOWN_RG_PATHS = [
    str(Path.home() / ".local/node/lib/node_modules/@deepseek-ai/dsh/node_modules/@vscode/ripgrep-linux-x64/bin/rg"),
    str(Path.home() / ".local/share/antigravity-ide/resources/app/node_modules/@vscode/ripgrep/bin/rg"),
]


def find_rg_binary() -> Optional[str]:
    """Localiza o binário do ripgrep disponível no sistema.

    Exemplo:
        >>> p = find_rg_binary()
        >>> p is None or isinstance(p, str)
        True
    """
    for p in KNOWN_RG_PATHS:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    system_rg = shutil.which("rg")
    if system_rg and os.access(system_rg, os.X_OK):
        return system_rg
    return None


def _parse_grep_line(line: str, workspace_root: Path) -> Optional[Tuple[str, int, str]]:
    """Extrai arquivo relativo, linha e texto de uma linha de saída do grep/ripgrep.

    Exemplo:
        >>> res = _parse_grep_line("src/main.py:10:def foo():", Path("."))
        >>> res is not None and res[1] == 10
        True
    """
    parts = line.split(":", 2)
    if len(parts) < 3:
        return None
    raw_path, line_no_str, matched_text = parts[0], parts[1], parts[2]
    try:
        line_no = int(line_no_str)
    except ValueError as parse_err:
        logger.debug("Linha invalida na saida do grep: %s", parse_err)
        return None
    try:
        abs_p = Path(raw_path).resolve()
        rel_p = str(abs_p.relative_to(workspace_root))
    except (ValueError, RuntimeError) as path_err:
        logger.debug("Falha ao resolver caminho relativo: %s", path_err)
        rel_p = raw_path
    return rel_p, line_no, matched_text.strip()


def _within_depth(
    abs_candidate: Path,
    targets: Sequence[Path],
    max_depth: Optional[int],
) -> bool:
    """True quando o caminho respeita o corte de profundidade dos alvos.

    Semântica (paridade com o ``--max-depth`` do ripgrep): a profundidade é o
    número de níveis abaixo de cada alvo de escopo. Um arquivo direto no alvo
    tem profundidade 1; um alvo explícito de arquivo vale profundidade 0 e
    sempre é aceito. ``max_depth=None`` (padrão) desliga o corte.
    """
    if max_depth is None:
        return True
    for target in targets:
        if target.is_file():
            if abs_candidate == target:
                return True
            continue
        try:
            rel = abs_candidate.relative_to(target)
        except ValueError:
            continue
        if len(rel.parts) <= max_depth:
            return True
    return False


class QmdHybridSearchEngine:
    """Motor Híbrido de Busca Cirúrgica (Needle 3 + Ripgrep + RRF)."""

    def __init__(self, workspace_root: Optional[str] = None):
        # TICKET-SGA-03: workspace canônico validado pela sandbox (mesma
        # autoridade do Needle Local) + raízes autorizadas resolvidas.
        self.workspace_root = self._resolve_workspace(workspace_root)
        self.allowed_roots: List[Path] = resolve_allowed_roots(self.workspace_root)
        self.needle_engine = LocalNeedleEngine(workspace_root=str(self.workspace_root))
        self.rg_binary = find_rg_binary()

    def _resolve_workspace(self, workspace_root: Optional[str] = None) -> Path:
        """Resolve a raiz do workspace de forma canônica e validada pela sandbox.

        Política TICKET-SGA-06: a denylist de sistema é sempre aplicada; sem
        ``CIRURGIC_ALLOWED_ROOTS`` e sem ``CIRURGIC_SANDBOX_STRICT=1`` qualquer
        workspace de projeto/worktree é aceito (fluxo sem atrito). Com allowlist
        explícita ou modo estrito, o workspace precisa caber no perímetro
        autorizado; caso contrário :class:`SandboxViolationError`.

        Exemplo:
            >>> QmdHybridSearchEngine()._resolve_workspace(None) == Path.cwd().resolve()
            True
        """
        raw = Path(workspace_root or os.getcwd()).expanduser()
        # Valida o caminho ORIGINAL (antes de resolver symlinks) para que
        # aliases como ``/lib`` não escapem virando ``/usr/lib``.
        resolved = ensure_not_system_restricted(raw, context="workspace_root")
        ensure_within_allowed_roots(
            resolved, resolve_allowed_roots(resolved), context="workspace_root"
        )
        return resolved

    def _estimate_tokens(self, text: str) -> int:
        """Estima a contagem de tokens a partir do comprimento do texto.

        Usa ``CHARS_PER_TOKEN = 3.8`` (heurística de alta fidelidade para
        orçamento rápido de snippets; não é garantia criptográfica estrita).

        Exemplo:
            >>> QmdHybridSearchEngine()._estimate_tokens("exemplo de codigo") > 0
            True
        """
        return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))

    def _collect_needle_snippets_for_file(
        self,
        file_path: str,
        file_rrf: float,
        is_hybrid: bool,
        grep_matches: List[Dict[str, Any]],
        needle_snips: List[Needle3Snippet],
    ) -> List[Dict[str, Any]]:
        """Empacota snippets AST Needle enriquecidos com matches exatos do grep.

        Exemplo:
            >>> QmdHybridSearchEngine()._collect_needle_snippets_for_file("a.py", 0.5, False, [], [])
            []
        """
        results: List[Dict[str, Any]] = []
        for snip in needle_snips:
            in_range_matches = [
                m for m in grep_matches
                if snip.start_line <= m["line"] <= snip.end_line
            ]
            results.append({
                "file": file_path,
                "lines": f"{snip.start_line}-{snip.end_line}",
                "rrf_score": file_rrf,
                "relations": snip.relations,
                "exact_matches": in_range_matches or list(grep_matches[:2]),
                "content": snip.content,
                "source": "hybrid" if is_hybrid else "needle_semantic",
                "tokens": self._estimate_tokens(snip.content),
            })
        return results

    def _collect_grep_fallback_snippet(
        self,
        file_path: str,
        abs_path: Path,
        file_rrf: float,
        grep_matches: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Cria snippet de fallback para literais capturados pelo grep sem nó AST.

        Exemplo:
            >>> QmdHybridSearchEngine()._collect_grep_fallback_snippet("a.txt", Path("a.txt"), 0.5, [])
            None
        """
        if not grep_matches:
            return None
        lines_preview: List[str] = []
        first_line = grep_matches[0]["line"]
        start_l = max(1, first_line - 3)
        end_l = first_line + 5
        try:
            # TICKET-SGA-04 — Guardrails de I/O do preview:
            # * arquivo acima de 2 MB nunca é lido (só os matches do grep);
            # * leitura LIMITADA às linhas [start_l, end_l] via islice — o arquivo
            #   inteiro nunca é materializado em memória.
            if abs_path.is_file() and file_size_allowed(abs_path, purpose="preview literal"):
                with abs_path.open("r", encoding="utf-8", errors="replace") as fh:
                    lines_preview = [
                        line.rstrip("\r\n")
                        for line in itertools.islice(fh, start_l - 1, end_l)
                    ]
        except Exception as read_err:
            logger.debug("Falha ao ler preview de %s: %s", abs_path, read_err)

        content_preview = "\n".join(lines_preview) or "\n".join(m["text"] for m in grep_matches)
        # Teto rígido de caracteres: preview literal nunca vira dump de arquivo.
        content_preview = content_preview[:MAX_LITERAL_PREVIEW_CHARS]
        return {
            "file": file_path,
            "lines": f"{start_l}-{end_l}",
            "rrf_score": file_rrf,
            "relations": [f"literal_matches:{len(grep_matches)}"],
            "exact_matches": grep_matches,
            "content": content_preview,
            "source": "grep_literal",
            "tokens": self._estimate_tokens(content_preview),
        }

    def _append_file_snippets(
        self,
        final_snippets: List[Dict[str, Any]],
        file_items: List[Dict[str, Any]],
        max_tokens: int,
        total_tokens: int,
    ) -> Tuple[int, bool]:
        """Adiciona itens de um arquivo à lista final respeitando limite de tokens.

        Exemplo:
            >>> out = []
            >>> t, stop = QmdHybridSearchEngine()._append_file_snippets(out, [], 100, 0)
            >>> t == 0 and not stop
            True
        """
        curr_tokens = total_tokens
        for item in file_items:
            snip_tokens = item.pop("tokens", 0)
            if curr_tokens + snip_tokens > max_tokens and final_snippets:
                return curr_tokens, True
            final_snippets.append(item)
            curr_tokens += snip_tokens
        return curr_tokens, False

    def _resolve_scope(
        self, scope: Optional[List[str]]
    ) -> Tuple[List[str], List[str]]:
        """Resolve ``scope`` em alvos absolutos existentes + filtros basename.

        Entradas relativas são resolvidas contra ``self.workspace_root``,
        comportando-se de forma consistente com o motor de AST. Entradas
        *path-like* inexistentes retornam ``([], [])`` — a rota léxica devolve
        nada em vez de ampliar silenciosamente para o workspace inteiro.

        Compatibilidade: um basename puro sem separador de caminho (ex.:
        ``scope="alpha.py"``) continua sendo um filtro de nome de arquivo,
        aplicado como glob relativo ao workspace (nunca varredura de disco).

        TICKET-SGA-03 — sandbox: todo alvo é resolvido canonicamente e precisa
        caber em ``self.allowed_roots``; ``..``/symlink/absoluto fora da
        sandbox viram :class:`SandboxViolationError` explícita.
        """
        if not scope:
            return [str(self.workspace_root)], []
        targets: List[str] = []
        globs: List[str] = []
        for item in scope:
            raw = item.strip()
            if not raw:
                continue
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = self.workspace_root / candidate
            # Contenção canônica antes de qualquer teste de existência: um
            # caminho que escapa da sandbox é violação mesmo que não exista.
            resolved = ensure_within_allowed_roots(
                candidate, self.allowed_roots, context=f"scope '{raw}'"
            )
            if resolved.exists():
                targets.append(str(resolved))
                continue
            is_basename = "/" not in raw and "\\" not in raw
            if is_basename and Path(raw).suffix:
                globs.append(f"*{raw}*")
                continue
            # Escopo path-like inexistente: nada a buscar.
            return [], []
        if not targets and globs:
            targets.append(str(self.workspace_root))
        return targets, globs

    def _build_pattern(self, query: str) -> Tuple[str, List[str]]:
        """Constrói o regex OR único do ripgrep e os termos substantivos.

        Stopwords bilíngues são removidas para que o pattern seja guiado pelos
        termos substantivos da pergunta. Fallback seguro: se a query inteira for
        stopwords, os tokens originais são preservados e o pattern nunca fica
        vazio.
        """
        clean_q = query.strip()
        raw_words = [w for w in re.findall(r"[\w_]+", clean_q) if len(w) >= 2]
        terms = substantive_terms(raw_words, min_length=2)

        if not terms:
            return _escape_literal(clean_q), terms
        escaped_terms = [_escape_literal(term) for term in terms]
        if len(terms) == 1 and len(raw_words) == 1:
            return _escape_literal(clean_q), terms
        if len(terms) == 1:
            return escaped_terms[0], terms
        return "|".join([_escape_literal(clean_q)] + escaped_terms[:6]), terms

    def _run_ripgrep(
        self,
        query: str,
        scope: Optional[List[str]] = None,
        max_results_per_file: int = DEFAULT_MAX_MATCHES_PER_FILE,
        max_files: int = 40,
        timeout: float = DEFAULT_RIPGREP_TIMEOUT_S,
        exclude_patterns: Optional[Sequence[str]] = None,
        max_depth: Optional[int] = None,
        detailed: bool = False,
    ) -> Any:
        """Fachada legada: só o dicionário por arquivo (compat testes/CLI).

        ``timeout`` (TICKET-SGA-04) limita o subprocesso em segundos; ao expirar,
        a rota léxica degrada para vazio e a rota AST segue viva (fallback
        assimétrico), nunca derrubando a requisição.

        ``exclude_patterns`` (já mesclados com o ``.cirurgicignore`` da raiz)
        e ``max_depth`` (None = ilimitado) valem para as duas rotas léxicas:
        nomes puros de exclusão são empurrados ao ripgrep como negations de
        glob, e o filtro determinístico em Python cobre padrões ancorados e o
        corte de profundidade com a mesma semântica da rota AST.

        ``detailed=True`` (TICKET-SGA-03): retorna a tupla completa
        ``(results_by_file, warnings, degraded_lanes)`` consumida internamente
        pela busca híbrida.

        Exemplo:
            >>> res = QmdHybridSearchEngine()._run_ripgrep("needle")
            >>> isinstance(res, dict)
            True
        """
        results, warnings, degraded = self._run_ripgrep_detailed(
            query,
            scope,
            max_results_per_file,
            max_files,
            timeout=timeout,
            exclude_patterns=exclude_patterns,
            max_depth=max_depth,
        )
        if detailed:
            return results, warnings, degraded
        return results

    def _run_ripgrep_detailed(
        self,
        query: str,
        scope: Optional[List[str]] = None,
        max_results_per_file: int = DEFAULT_MAX_MATCHES_PER_FILE,
        max_files: int = 40,
        timeout: float = DEFAULT_RIPGREP_TIMEOUT_S,
        exclude_patterns: Optional[Sequence[str]] = None,
        max_depth: Optional[int] = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[str], List[str]]:
        """Busca léxica via ripgrep com diagnóstico de resiliência (TICKET-SGA-03).

        Retorna ``(resultados_por_arquivo, warnings, degraded_lanes)``.
        Timeout do subprocesso (``subprocess.TimeoutExpired``) e falhas de
        I/O/execução (``OSError``) degradam a lane ``ripgrep_lexical`` com
        warning explícito — nunca um sucesso vazio silencioso.
        """
        results_by_file: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []
        degraded_lanes: List[str] = []
        if not query.strip():
            return results_by_file, warnings, degraded_lanes

        pattern, terms = self._build_pattern(query)
        targets, scope_globs = self._resolve_scope(scope)
        if not targets:
            return results_by_file, warnings, degraded_lanes

        extra_patterns = merge_exclude_patterns(exclude_patterns, self.workspace_root)

        cmd = [self.rg_binary or "grep", "-n", "-i"]
        if self.rg_binary:
            cmd.extend([
                "--no-messages",
                # -H: alvo explícito de arquivo único também prefixa o path,
                # preservando o contrato path:line:text do parser.
                "-H",
                "-e", pattern,
                "--max-count", str(max_results_per_file),
            ])
            if max_depth is not None:
                # Profundidade máxima de varredura (níveis abaixo de cada alvo);
                # paridade com o walk da rota AST (Needle 3).
                cmd.extend(["--max-depth", str(max_depth)])
            for glob in IGNORED_GLOBS:
                cmd.extend(["--glob", glob])
            # Otimização de latência: só nomes puros (semântica idêntica ao
            # matcher em Python); padrões ancorados ficam no pós-filtro.
            for glob in exclude_patterns_to_rg_globs(extra_patterns):
                cmd.extend(["--glob", glob])
            for glob in scope_globs:
                cmd.extend(["-g", glob])
            cmd.append("--")
            cmd.extend(targets)
        else:
            # Fallback para grep padrão se rg não estiver presente
            cmd.extend(["-E", "-r", "-I"])
            for directory in GREP_EXCLUDED_DIRS:
                cmd.extend(["--exclude-dir", directory])
            for pattern_name in extra_patterns:
                core = pattern_name.rstrip("/")
                if core and "/" not in core:
                    cmd.extend(["--exclude-dir", core])
            for glob in scope_globs:
                cmd.extend(["--include", glob])
            cmd.extend(["-e", pattern])
            cmd.extend(targets)

        all_texts: Dict[str, List[str]] = {}
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # TICKET-SGA-03/SGA-04: timeout não é sucesso vazio — degrada a lane.
            msg = f"ripgrep timed out after {timeout:.1f}s"
            logger.warning("QMD: %s (query=%r)", msg, query)
            warnings.append(msg)
            degraded_lanes.append("ripgrep_lexical")
            return results_by_file, warnings, degraded_lanes
        except OSError as io_err:
            # Falha de I/O/execução do binário (ex.: binário ausente, permissão).
            msg = f"lexical lane I/O failure: {io_err}"
            logger.warning("QMD: %s (query=%r)", msg, query)
            warnings.append(msg)
            degraded_lanes.append("ripgrep_lexical")
            return results_by_file, warnings, degraded_lanes

        resolved_targets = [Path(t) for t in targets]
        for line in res.stdout.splitlines():
            parsed = _parse_grep_line(line, self.workspace_root)
            if not parsed:
                continue
            rel_p, line_no, matched_text = parsed
            abs_candidate = self.workspace_root / rel_p
            if path_is_excluded(rel_p, extra_patterns):
                continue
            if not _within_depth(abs_candidate, resolved_targets, max_depth):
                continue
            if rel_p not in results_by_file and len(results_by_file) >= max_files:
                break
            entry = results_by_file.setdefault(
                rel_p,
                {
                    "file": rel_p,
                    "match_count": 0,
                    "matches": [],
                    "distinct_terms": 0,
                    "term_coverage": 0.0,
                    "matched_terms": [],
                },
            )
            entry["match_count"] += 1
            all_texts.setdefault(rel_p, []).append(matched_text)
            if len(entry["matches"]) < max_results_per_file:
                entry["matches"].append({
                    "line": line_no,
                    "text": matched_text,
                })

        # Cobertura conjuntiva: quantos termos DISTINTOS da query o arquivo
        # realmente contém. Um arquivo que repete um único keyword quente
        # perde para outro que cobre os dois termos da pergunta.
        for rel_p, entry in results_by_file.items():
            haystack = "\n".join(all_texts.get(rel_p, [])).lower()
            matched = [term for term in terms if term in haystack]
            entry["matched_terms"] = matched
            entry["distinct_terms"] = len(matched)
            entry["term_coverage"] = round(len(matched) / max(1, len(terms)), 4)

        return results_by_file, warnings, degraded_lanes

    def _build_timeout_response(
        self, effective_timeout: float, elapsed_ms: float
    ) -> Dict[str, Any]:
        """Monta a resposta canônica e imutável de timeout global da busca híbrida.

        Ponto único de verdade para o veredito ``status="timeout"``: consumido
        tanto pelo handler de :class:`asyncio.TimeoutError` quanto pela
        re-checagem de deadline executada depois do ``gather`` das lanes.
        A segunda chamada existe porque o timer do ``asyncio.wait_for`` tem
        granularidade de milissegundos (kqueue no macOS): as lanes podem
        concluir antes do tick do timer mesmo com o tempo total já estourado,
        e o motor devolveria ``status="partial"`` (ripgrep degradado) sem nunca
        sinalizar o estouro da cota global.

        Exemplo:
            >>> res = QmdHybridSearchEngine()._build_timeout_response(0.5, 12.0)
            >>> res["status"], res["timeout_s"], res["degraded_lanes"]
            ('timeout', 0.5, ['global_timeout'])
            >>> res["snippets_count"], res["latency_ms"]
            (0, 12.0)
        """
        logger.warning(
            "cirurgic_search: timeout global de %.2fs atingido "
            "(latência até aqui: %.1f ms) — refine `scope` ou aumente "
            "`timeout`.",
            effective_timeout,
            elapsed_ms,
        )
        return {
            "status": "timeout",
            "latency_ms": elapsed_ms,
            "total_tokens": 0,
            "token_savings_pct": 0.0,
            "snippets_count": 0,
            "files_found": 0,
            "snippets": [],
            "timeout_s": effective_timeout,
            "message": (
                f"Busca híbrida excedeu {effective_timeout}s. "
                "Restrinja `scope` ou aumente `timeout` (<=0 desativa)."
            ),
            "warnings": [
                f"Busca híbrida excedeu o timeout global de {effective_timeout}s."
            ],
            "degraded_lanes": ["global_timeout"],
        }

    async def search(
        self,
        query: str,
        scope: Optional[List[str]] = None,
        max_tokens: int = 1200,
        depth: Optional[int] = None,
        k: float = 60.0,
        weight_grep: float = 1.0,
        weight_needle: float = 1.2,
        exclude_patterns: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Executa a busca híbrida unificada com fusão RRF em CPU.

        TICKET-SGA-03/SGA-04/SGA-05 — contrato de resiliência e guardrails:
        * ``status`` é ``'success'``, ``'partial'`` ou ``'timeout'``;
          ``degraded_lanes`` lista lanes que falharam (ex.:
          ``['ripgrep_lexical']``, ``['needle_ast']``, ``['global_timeout']``) e
          ``warnings`` carrega o motivo (timeout, I/O, teto de tokens).
        * ``timeout`` é a cota global por requisição em segundos (padrão 15 s;
          <= 0 desativa). A cota é verificada duas vezes: pelo
          ``asyncio.wait_for`` e por uma re-checagem de deadline depois do
          ``gather`` (o timer do event loop tem granularidade de milissegundos,
          então as lanes podem concluir antes do tick sem que o estouro da cota
          seja observado); em ambos os casos o veredito é ``'timeout'``.
        * ``depth`` (padrão ``None`` = profundidade ILIMITADA) corta a árvore de
          diretórios em ``depth`` níveis abaixo de cada alvo de escopo e sobrepõe
          ``relational_depth`` do Needle 3 quando enviado — clientes que já
          enviam ``depth`` permanecem 100% retrocompatíveis.
        * ``exclude_patterns`` soma exclusões pontuais às universais padrão e ao
          ``.cirurgicignore`` da raiz do workspace (se existir), aplicadas de
          forma idêntica nas rotas léxica (ripgrep/grep) e AST (walk).

        Exemplo:
            >>> import asyncio
            >>> res = asyncio.run(QmdHybridSearchEngine().search(query="needle_engine"))
            >>> res["status"] in ("success", "partial")
            True
        """
        t0 = time.perf_counter()

        effective_timeout = (
            DEFAULT_SEARCH_TIMEOUT_S if timeout is None else float(timeout)
        )
        # Cota do subprocesso rg: nunca maior que a cota global da requisição.
        rg_timeout = (
            min(DEFAULT_RIPGREP_TIMEOUT_S, effective_timeout)
            if effective_timeout > 0
            else DEFAULT_RIPGREP_TIMEOUT_S
        )

        # 1. Executa busca léxica via Ripgrep em thread (com diagnóstico, exclusions, depth e timeout)
        grep_task = asyncio.create_task(
            asyncio.to_thread(
                self._run_ripgrep,
                query,
                scope,
                timeout=rg_timeout,
                exclude_patterns=exclude_patterns,
                max_depth=depth,
                detailed=True,
            )
        )

        # 2. Executa busca semântica/AST via Needle 3 (com diagnóstico)
        needle_q = Needle3Query(
            query=query,
            scope=scope,
            max_tokens=max_tokens * 2,  # Captura candidatos amplos para rankear
            # `depth=None` mantém o padrão canônico do schema (2) no grafo
            # relacional AST e libera a árvore de diretórios (ilimitada);
            # quando enviado, o valor sobrepõe ambos sem quebra retroativa.
            relational_depth=depth if depth is not None else 2,
            max_depth=depth,
            exclude_patterns=list(exclude_patterns) if exclude_patterns else None,
            preserve_relational_integrity=True,
        )
        needle_task = asyncio.create_task(
            self.needle_engine.search_with_diagnostics(needle_q)
        )

        if effective_timeout > 0:
            try:
                (
                    (grep_results, grep_warnings, grep_degraded),
                    (needle_resp, needle_warnings, needle_degraded),
                ) = await asyncio.wait_for(
                    asyncio.gather(grep_task, needle_task),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # Cancelamento gracioso das lanes pendentes: `asyncio.wait_for`
                # já cancela o `gather` interno, mas garantimos explicitamente
                # para não deixar tarefas órfãs rodando após o retorno.
                grep_task.cancel()
                needle_task.cancel()
                elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                return self._build_timeout_response(effective_timeout, elapsed_ms)
        else:
            (
                (grep_results, grep_warnings, grep_degraded),
                (needle_resp, needle_warnings, needle_degraded),
            ) = await asyncio.gather(grep_task, needle_task)

        # Checagem de deadline PÓS-gather: garante o contrato de cota global
        # mesmo quando o `wait_for` não dispara. Nesse caminho as duas lanes já
        # concluíram (nada órfão), mas o tempo total pode ter estourado a cota
        # — devolver `partial` mascararia o timeout.
        elapsed_s = time.perf_counter() - t0
        if effective_timeout > 0 and elapsed_s >= effective_timeout:
            return self._build_timeout_response(
                effective_timeout, round(elapsed_s * 1000.0, 2)
            )

        warnings: List[str] = list(grep_warnings) + list(needle_warnings)
        degraded_lanes: List[str] = list(grep_degraded) + list(needle_degraded)

        # 3. Ranqueamento de Arquivos
        # Grep: ordem conjuntiva — arquivos que cobrem mais termos DISTINTOS da
        # query vencem os que apenas repetem um keyword quente muitas vezes.
        query_terms = substantive_terms(re.findall(r"[\w_]+", query), min_length=2)
        multi_term_query = len(query_terms) > 1

        def lexical_sort_key(file_path: str) -> Tuple[int, int, float]:
            entry = grep_results[file_path]
            return (
                int(entry.get("distinct_terms", 1)),
                int(entry.get("match_count", 0)),
                float(entry.get("term_coverage", 0.0)),
            )

        sorted_grep_files = sorted(
            grep_results.keys(),
            key=lexical_sort_key,
            reverse=True,
        )
        grep_rank: Dict[str, int] = {f: idx + 1 for idx, f in enumerate(sorted_grep_files)}

        # Needle: agrupa snippets por arquivo e extrai maior score por arquivo
        needle_snippets_by_file: Dict[str, List[Needle3Snippet]] = {}
        for snip in needle_resp.snippets:
            needle_snippets_by_file.setdefault(snip.file_path, []).append(snip)

        sorted_needle_files = sorted(
            needle_snippets_by_file.keys(),
            key=lambda f: max(s.relevance_score for s in needle_snippets_by_file[f]),
            reverse=True,
        )
        needle_rank: Dict[str, int] = {f: idx + 1 for idx, f in enumerate(sorted_needle_files)}

        # 4. Fusão RRF (Reciprocal Rank Fusion) com multiplicador tipológico.
        # Código-fonte recebe +25% sobre docs/dados; a neutralidade de escopo
        # preserva 1.0x quando o chamador apontou explicitamente para docs.
        all_candidate_files = set(grep_rank.keys()) | set(needle_rank.keys())
        documentation_scope = _scope_is_documentation(scope)
        rrf_scores: Dict[str, float] = {}

        for f in all_candidate_files:
            score = 0.0
            if f in grep_rank:
                lexical_weight = weight_grep
                entry = grep_results.get(f, {})
                # Match isolado de termo único é evidência fraca quando a
                # query contém múltiplos termos distintos.
                if multi_term_query and int(entry.get("distinct_terms", 1)) <= 1:
                    lexical_weight *= 0.5
                score += lexical_weight / (k + grep_rank[f])
            if f in needle_rank:
                score += weight_needle / (k + needle_rank[f])
            score *= _type_multiplier(f, documentation_scope)
            rrf_scores[f] = score

        # Refinamento 1: Fallback Assimétrico
        ranked_files = sorted(
            all_candidate_files,
            key=lambda f: rrf_scores[f],
            reverse=True,
        )

        # 5. Refinamento 2: Fusão Macro-Micro e Empacotamento de Snippets
        final_snippets: List[Dict[str, Any]] = []
        total_tokens = 0
        raw_bytes_evaluated = 0

        for f in ranked_files:
            file_rrf = round(rrf_scores[f], 5)
            grep_matches = grep_results.get(f, {}).get("matches", [])
            needle_snips = needle_snippets_by_file.get(f, [])

            abs_path = self.workspace_root / f
            if abs_path.is_file():
                try:
                    raw_bytes_evaluated += abs_path.stat().st_size
                except Exception as stat_err:
                    logger.debug("Falha ao ler tamanho de %s: %s", abs_path, stat_err)

            if needle_snips:
                file_items = self._collect_needle_snippets_for_file(
                    file_path=f,
                    file_rrf=file_rrf,
                    is_hybrid=(f in grep_rank),
                    grep_matches=grep_matches,
                    needle_snips=needle_snips,
                )
            else:
                fallback_item = self._collect_grep_fallback_snippet(
                    file_path=f,
                    abs_path=abs_path,
                    file_rrf=file_rrf,
                    grep_matches=grep_matches,
                )
                file_items = [fallback_item] if fallback_item else []

            total_tokens, should_stop = self._append_file_snippets(
                final_snippets=final_snippets,
                file_items=file_items,
                max_tokens=max_tokens,
                total_tokens=total_tokens,
            )
            if should_stop or total_tokens >= max_tokens:
                break

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        # Heurística CHARS_PER_TOKEN (aproximação de alta fidelidade, não
        # garantia criptográfica estrita) para estimar o custo bruto avaliado.
        est_raw_tokens = max(1, math.ceil(raw_bytes_evaluated / CHARS_PER_TOKEN)) if raw_bytes_evaluated else total_tokens
        savings_pct = max(0.0, round((1.0 - (total_tokens / max(est_raw_tokens, total_tokens, 1))) * 100.0, 2))

        # TICKET-SGA-03 — alerta de orçamento de tokens insuficiente: havia
        # arquivos candidatos, mas o teto rígido de max_tokens não coube
        # nenhum snippet. Warning explícito, nunca sucesso silencioso.
        files_found = len(ranked_files)
        snippets_count = len(final_snippets)
        if files_found > 0 and snippets_count == 0:
            warnings.append(
                f"Token budget max_tokens={max_tokens} was too small to fit any "
                f"snippet; 0 snippets included from {files_found} matching files."
            )

        # TICKET-SGA-03: status parcial quando alguma lane degradou; senão success.
        status = "partial" if degraded_lanes else "success"

        return {
            "status": status,
            "latency_ms": round(total_latency_ms, 2),
            "total_tokens": total_tokens,
            "token_savings_pct": savings_pct,
            "snippets_count": snippets_count,
            "files_found": files_found,
            "snippets": final_snippets,
            "degraded_lanes": degraded_lanes,
            "warnings": warnings,
        }
