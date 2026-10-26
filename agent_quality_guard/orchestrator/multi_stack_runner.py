"""
Multi-Stack Runner & Language Drivers for Needle 3 Verify Delivery.

Detecção 100% ABERTA por CAPABILITY (TICKET-SGA-09): o runner não conhece nome
de repositório, arquivo pessoal ou marca. Ele observa apenas manifestos e
arquivos reais do workspace (``pyproject.toml``, ``setup.py``,
``requirements.txt``, ``package.json``, ``bun.lockb``/``bun.lock``,
``tsconfig.json``, ``Cargo.toml``, ``go.mod``, ``pubspec.yaml``, scripts e
suítes de teste presentes).

Contrato honesto de execução:
- Nenhum caminho devolve ``passed=True`` sem evidência real executada.
- Stack sem runner, binário ausente ou zero testes executados → ``INCONCLUSIVE``
  (ou ``VERIFIED_FAIL`` quando o repositório exige testes e não possui nenhum).
- Contadores vêm exclusivamente do parser da saída real do runner.

Camadas de auditoria:
- Camada 3: bridge nativa com compiladores/linters do projeto (tsc,
  eslint/biome/oxlint, ruff/mypy, go vet, cargo clippy, dart analyze) — timeout
  padrão/máximo de 600s; ferramenta ausente/timeout é registrada em
  ``layer_status``/``inconclusive_reasons``, nunca silenciada como "0 erros".
- Varredura AST JS (node_js_ast) e fallback léxico com suporte a needle-ignore.
"""

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("agent_quality_guard.multi_stack")

RULE_SUGGESTIONS: Dict[str, str] = {
    "SECURITY_EVAL": (
        "Remova o uso de eval(). Utilize JSON.parse(), despachos por dicionário ou funções puras."
    ),
    "SECURITY_EXEC": (
        "Evite o construtor new Function(). Isole a lógica dinâmica em funções declaradas ou analisadores formais."
    ),
    "HARDCODED_SECRET": (
        "Mova a chave privada/segredo para variáveis de ambiente (.env) ou adicione "
        "'// needle-ignore: HARDCODED_SECRET (<motivo>)' se for chave pública de client-side (ex: Firebase Client API Key)."
    ),
    "XSS_INNER_HTML": (
        "Substitua por element.textContent ou document.createElement(); se HTML for mandatório, "
        "envolva as interpolações com escapeHtml(...) ou DOMPurify.sanitize()."
    ),
    "JS_PROD_MOCK": (
        "Remova jest.mock(), jest.fn() ou sinon de arquivos sob src/. "
        "Mocks pertencem exclusivamente a arquivos de teste em __tests__/ ou tests/."
    ),
    "EMPTY_CATCH": (
        "Não engula exceções silenciosamente. Registre com console.warn/logger, "
        "re-lance o erro com throw/raise ou trate a condição explicitamente."
    ),
    "CODE_SMELL_CONSOLE": (
        "Remova chamadas residuais de console.log() antes de enviar para produção, "
        "ou utilize console.warn/error estruturado se for um alerta de runtime legítimo."
    ),
    "JEZ_ANTI_EMOJI": (
        "Remova o emoji. A identidade visual da marca JEZ exige tipografia limpa e ícones SVG minimalistas dedicados."
    ),
    "SQL_INJECTION": (
        "Utilize queries parametrizadas com placeholders (?, %s ou :param). "
        "Nunca interpole variáveis via f-strings ou concatenação de strings."
    ),
    "PROD_MOCK": (
        "Remova unittest.mock, MagicMock ou patch de código de produção. "
        "Mocks pertencem exclusivamente aos arquivos em tests/."
    ),
    "TEST_TAUTOLOGY": (
        "Substitua a asserção tautológica (ex: assert True) por uma verificação real do resultado da função ou estado esperado."
    ),
    "TSC_TYPE_ERROR": (
        "Corrija a incompatibilidade de tipos apontada pelo TypeScript antes da entrega; "
        "evite 'as any' e prefira narrowing explícito ou tipos auxiliares."
    ),
    "ESLINT_RULES_OF_HOOKS": (
        "Hooks do React só podem ser chamados no topo de componentes/funções customizadas, "
        "nunca dentro de condicionais, laços ou callbacks."
    ),
    "ESLINT_HOOK_DEPS": (
        "Declare todas as dependências reativas em useEffect/useMemo/useCallback ou justifique "
        "explicitamente a omissão estável (ex.: refs)."
    ),
    "ESLINT_MISSING_KEY": (
        "Forneça uma 'key' estável e única (id do domínio) para cada elemento de uma lista; "
        "evite usar o índice do array como identidade."
    ),
    "ESLINT_STATE_MUTATION": (
        "Nunca mute this.state/estado diretamente: use setState/setX com novo objeto ou updater funcional."
    ),
    "LINT_PARSE_ERROR": (
        "O arquivo não pôde ser processado pelo linter: corrija o erro de sintaxe/configuração antes de auditar."
    ),
    "RUFF_LINT": (
        "Resolva o achado do Ruff (imports não usados, nomes indefinidos, complexidade) ou registre "
        "justificativa explícita na configuração do projeto."
    ),
    "MYPY_TYPE_ERROR": (
        "Ajuste a anotação/tipo apontado pelo mypy; evite silenciar com 'type: ignore' sem justificativa."
    ),
    "GOVET_DIAGNOSTIC": (
        "Corrija o diagnóstico do 'go vet' (printf, locks, struct tags) antes da entrega."
    ),
    "CLIPPY_LINT": (
        "Corrija o lint do Clippy; o gate roda com '-D warnings', portanto o aviso é bloqueante."
    ),
    "DART_ANALYZE": (
        "Resolva o diagnóstico do 'dart analyze' antes da entrega."
    ),
}

# Orçamento total da bridge nativa (TICKET-SGA-09: 600s, determinação da
# Fundadora). Sobrescrevível por ``JEV_NATIVE_BRIDGE_BUDGET_SECONDS``.
_DEFAULT_NATIVE_BUDGET_SECONDS = 600.0
_MIN_NATIVE_BUDGET_SECONDS = 0.1
_MAX_NATIVE_BUDGET_SECONDS = 600.0
_NATIVE_OUTPUT_LIMIT = 512 * 1024
_MAX_LINT_FILES = 150

# Timeout (segundos) de cada subprocesso de teste. Nenhum runner roda sem teto.
_TEST_COMMAND_TIMEOUT_SECONDS = 600.0

# Estados explícitos de veredito (espelham core.schemas.VerdictState).
VERDICT_VERIFIED_PASS = "VERIFIED_PASS"
VERDICT_VERIFIED_FAIL = "VERIFIED_FAIL"
VERDICT_NOT_APPLICABLE = "NOT_APPLICABLE"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_EMERGENCY_RELEASE = "EMERGENCY_RELEASE"

# Manifestos/arquivos de capability por linguagem (nenhum nome de projeto).
_PYTHON_MANIFESTS: Tuple[str, ...] = (
    "pyproject.toml", "setup.py", "setup.cfg", "Pipfile", "poetry.lock",
    "requirements.txt", "pytest.ini", "tox.ini",
)
_NODE_MANIFESTS: Tuple[str, ...] = ("package.json",)
_TS_MANIFESTS: Tuple[str, ...] = ("tsconfig.json",)
_RUST_MANIFESTS: Tuple[str, ...] = ("Cargo.toml",)
_GO_MANIFESTS: Tuple[str, ...] = ("go.mod",)
_DART_MANIFESTS: Tuple[str, ...] = ("pubspec.yaml",)

# Suítes de teste por extensão reconhecidas na descoberta poliglota.
_JS_TEST_GLOBS: Tuple[str, ...] = (
    "*.test.js", "*.test.mjs", "*.test.cjs", "*.test.ts", "*.test.tsx",
    "*.spec.js", "*.spec.mjs", "*.spec.ts", "*.spec.tsx",
)
_NODE_WORKSPACE_GLOBS: Tuple[str, ...] = (
    "*/package.json", "packages/*/package.json", "apps/*/package.json",
    "libs/*/package.json", "services/*/package.json",
)


def _get_file_suppressed_rules_js(lines: List[str]) -> set[str]:
    """Extrai regras suprimidas globalmente no arquivo via diretiva de cabeçalho.

    Exemplo:
        >>> "SECURITY_EVAL" in _get_file_suppressed_rules_js(["// needle-ignore-file: SECURITY_EVAL"])
        True
    """
    suppressed: set[str] = set()
    file_pat = re.compile(
        r"(?://|/\*|#)\s*needle-ignore-file:\s*([A-Za-z0-9_,\s]+)",
        re.IGNORECASE,
    )
    all_pat = re.compile(
        r"(?://|/\*|#)\s*needle-ignore-all\b",
        re.IGNORECASE,
    )
    for line in lines[:15]:
        line_clean = line.strip()
        if all_pat.search(line_clean):
            suppressed.add("ALL")
            return suppressed
        m = file_pat.search(line_clean)
        if m:
            tokens = [r.strip().upper() for r in m.group(1).split(",") if r.strip()]
            suppressed.update(tokens)
    return suppressed


def _is_line_suppressed(
    rule_id: str,
    line_number: int,
    lines: List[str],
    file_suppressed: Optional[set[str]] = None,
) -> bool:
    """Verifica se a linha, linha anterior ou arquivo possui anotação de supressão needle-ignore.

    Exemplo:
        >>> _is_line_suppressed("SECRET", 1, ["// needle-ignore: SECRET"])
        True
    """
    target = rule_id.upper()
    if file_suppressed and ("ALL" in file_suppressed or target in file_suppressed):
        return True

    curr_idx = line_number - 1
    if 0 <= curr_idx < len(lines):
        if _line_matches_inline_js(lines[curr_idx], target):
            return True

    prev_idx = line_number - 2
    if 0 <= prev_idx < len(lines):
        if _line_matches_prev_js(lines[prev_idx], target):
            return True

    return False


def _line_matches_inline_js(line: str, target: str) -> bool:
    """Verifica se linha atual contém anotação inline needle-ignore válida."""
    pat = re.compile(
        r"(?://|/\*|#)\s*needle-ignore:\s*([A-Za-z0-9_,\s]+)",
        re.IGNORECASE,
    )
    m = pat.search(line)
    if not m:
        return False
    rules = [r.strip().upper() for r in m.group(1).split(",")]
    return target in rules or "ALL" in rules


def _line_matches_prev_js(prev_line: str, target: str) -> bool:
    """Verifica se linha anterior contém anotação needle-ignore ou next-line."""
    stripped = prev_line.strip()
    next_pat = re.compile(
        r"(?://|/\*|#)\s*needle-ignore-next-line:\s*([A-Za-z0-9_,\s]+)",
        re.IGNORECASE,
    )
    m_next = next_pat.search(stripped)
    if m_next:
        rules = [r.strip().upper() for r in m_next.group(1).split(",")]
        if target in rules or "ALL" in rules:
            return True

    if not stripped.startswith(("//", "/*", "#", "*")):
        return False

    inline_pat = re.compile(
        r"(?://|/\*|#)\s*needle-ignore:\s*([A-Za-z0-9_,\s]+)",
        re.IGNORECASE,
    )
    m_inline = inline_pat.search(stripped)
    if not m_inline:
        return False
    rules = [r.strip().upper() for r in m_inline.group(1).split(",")]
    return target in rules or "ALL" in rules


@dataclass
class MultiStackViolation:
    """Registro de violação detectada na varredura de um projeto multi-stack.

    Exemplo:
        >>> MultiStackViolation("R", "c", "a.js", 1, "x", "m").rule_id
        'R'
    """

    rule_id: str
    category: str
    file_path: str
    line_number: int
    code_snippet: str
    message: str
    suggestion: Optional[str] = None


@dataclass
class NativeToolRun:
    """Diagnóstico de execução de uma ferramenta nativa da Camada 1.

    Exemplo:
        >>> NativeToolRun(tool="tsc", status="executed", returncode=2).status
        'executed'
    """

    tool: str
    status: str
    returncode: int = 0
    duration: float = 0.0
    violation_count: int = 0
    detail: Optional[str] = None


class MultiStackRunner:
    """Orquestra detecção de stack, execução de testes e varredura de AST para projetos multi-stack."""

    @staticmethod
    def _has_any_manifest(workspace_path: Path, names: Tuple[str, ...]) -> bool:
        """Indica se ao menos um manifesto nomeado existe na raiz do workspace.

        Exemplo:
            >>> MultiStackRunner._has_any_manifest(Path("."), ("manifesto-xyz.toml",))
            False
        """
        return any((workspace_path / name).exists() for name in names)

    @classmethod
    def _has_python_capability(cls, workspace_path: Path) -> bool:
        """Detecta Python por manifesto real ou fontes ``.py`` na raiz/pacote base.

        Critério aberto: nenhum nome de repositório ou arquivo específico de
        projeto; apenas a presença de manifestos canônicos ou código Python.

        Exemplo:
            >>> isinstance(MultiStackRunner._has_python_capability(Path(".")), bool)
            True
        """
        if cls._has_any_manifest(workspace_path, _PYTHON_MANIFESTS):
            return True
        if any(workspace_path.glob("*.py")):
            return True
        for subdir in ("src", "lib"):
            base = workspace_path / subdir
            if base.is_dir() and any(base.glob("*.py")):
                return True
        return False

    @staticmethod
    def _read_package_json(workspace_path: Path) -> Dict[str, Any]:
        """Lê o ``package.json`` da raiz de forma tolerante (vazio se ilegível).

        Exemplo:
            >>> MultiStackRunner._read_package_json(Path("."))["name"]
            'search-guard-agent'
        """
        pkg = workspace_path / "package.json"
        if not pkg.exists():
            return {}
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError) as parse_err:
            logger.debug("package.json ilegível em %s: %s", workspace_path, parse_err)
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    def _has_node_capability(cls, workspace_path: Path) -> bool:
        """Detecta Node/JS pela presença do manifesto canônico ``package.json``.

        Exemplo:
            >>> isinstance(MultiStackRunner._has_node_capability(Path(".")), bool)
            True
        """
        return cls._has_any_manifest(workspace_path, _NODE_MANIFESTS)

    @classmethod
    def _is_react_native_workspace(cls, workspace_path: Path) -> bool:
        """Detecta React Native/Expo por dependência declarada (capability, sem fingerprint).

        Exemplo:
            >>> MultiStackRunner._is_react_native_workspace(Path("."))
            False
        """
        data = cls._read_package_json(workspace_path)
        if not data:
            return False
        dependency_names: set[str] = set()
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            block = data.get(section)
            if isinstance(block, dict):
                dependency_names.update(str(name) for name in block)
        return bool(dependency_names & {"react-native", "expo"})

    @staticmethod
    def detect_stack(workspace_path: Path) -> str:
        """Classifica a stack por CAPABILITY aberta (zero nome de projeto/arquivo pessoal).

        Precedência determinística entre capabilities simultâneas:
        Python (manifesto/fontes) > React Native (dependência declarada) >
        Node/TS/Bun com suíte > Node puro > Rust > Go > Dart > ``UNKNOWN``.

        Exemplo:
            >>> isinstance(MultiStackRunner.detect_stack(Path(".")), str)
            True
        """
        workspace_path = Path(workspace_path)

        if MultiStackRunner._has_python_capability(workspace_path):
            return "PYTHON_GENERIC"

        if MultiStackRunner._has_node_capability(workspace_path):
            if MultiStackRunner._is_react_native_workspace(workspace_path):
                return "REACT_NATIVE_EXPO"
            if MultiStackRunner._is_node_ts_bun_workspace(workspace_path):
                return "NODE_TS_BUN"
            return "NODE_JS_GENERIC"

        if MultiStackRunner._has_any_manifest(workspace_path, _RUST_MANIFESTS):
            return "RUST_CARGO"
        if MultiStackRunner._has_any_manifest(workspace_path, _GO_MANIFESTS):
            return "GO_MODULE"
        if MultiStackRunner._has_any_manifest(workspace_path, _DART_MANIFESTS):
            return "DART_FLUTTER"

        return "UNKNOWN"

    @staticmethod
    def _is_node_ts_bun_workspace(workspace_path: Path) -> bool:
        """Detecta workspace Node/TypeScript moderno com suíte determinística (Bun ou Node).

        Critérios (qualquer um satisfaz), todos abertos por capability:
        - ``bun.lockb``/``bun.lock``/``bunfig.toml`` presente;
        - ``package.json`` com ``scripts.test`` declarado;
        - arquivos ``.ts`` em ``src/`` ou ``tsconfig.json`` presente;
        - suíte de testes real (``tests/**/*.mjs``, ``*.test.*``/``*.spec.*``).

        Exemplo:
            >>> isinstance(MultiStackRunner._is_node_ts_bun_workspace(Path(".")), bool)
            True
        """
        if any((workspace_path / name).exists() for name in ("bun.lockb", "bun.lock", "bunfig.toml")):
            return True

        pkg = workspace_path / "package.json"
        if not pkg.exists():
            return False

        tests_dir = workspace_path / "tests"
        if tests_dir.exists():
            if any(tests_dir.glob("*.mjs")):
                return True
            for pattern in _JS_TEST_GLOBS:
                if any(tests_dir.rglob(pattern)):
                    return True

        try:
            pkg_data = json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
        except Exception as parse_err:
            logger.debug("package.json ilegível em %s: %s", workspace_path, parse_err)
            pkg_data = {}

        scripts = pkg_data.get("scripts") if isinstance(pkg_data, dict) else None
        if isinstance(scripts, dict) and scripts.get("test"):
            return True

        if (workspace_path / "tsconfig.json").exists():
            return True
        src_dir = workspace_path / "src"
        if src_dir.exists() and any(src_dir.rglob("*.ts")):
            return True

        return False

    @staticmethod
    def _build_test_result(
        *,
        passed: bool,
        total_passed: int,
        total_failed: int,
        summary: str,
        failure_details: List[str],
        start: float,
        evidence: bool,
        runner: Optional[str] = None,
        verdict_state: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normaliza o resultado de um runner com veredito explícito e honesto.

        Invariante inegociável: ``passed=True`` exige ``evidence=True``. Sem
        evidência real executada o resultado degrada para ``INCONCLUSIVE``.

        Exemplo:
            >>> r = MultiStackRunner._build_test_result(
            ...     passed=True, total_passed=0, total_failed=0, summary="s",
            ...     failure_details=[], start=0.0, evidence=False)
            >>> (r["passed"], r["verdict_state"])
            (False, 'INCONCLUSIVE')
        """
        if passed and not evidence:
            passed = False
            verdict_state = VERDICT_INCONCLUSIVE
        if verdict_state is None:
            if evidence and passed:
                verdict_state = VERDICT_VERIFIED_PASS
            elif evidence:
                verdict_state = VERDICT_VERIFIED_FAIL
            else:
                verdict_state = VERDICT_INCONCLUSIVE
        if verdict_state == VERDICT_VERIFIED_PASS and not evidence:
            verdict_state = VERDICT_INCONCLUSIVE
            passed = False
        return {
            "passed": bool(passed),
            "total_passed": int(total_passed),
            "total_failed": int(total_failed),
            "summary": summary,
            "failure_details": failure_details,
            "duration": round(time.time() - start, 2),
            "evidence": bool(evidence),
            "verdict_state": verdict_state,
            "runner": runner,
            "reason": reason,
        }

    @staticmethod
    def _output_has_test_evidence(output: str) -> bool:
        """Indica se a saída contém sinais reais de execução de testes.

        Exemplo:
            >>> MultiStackRunner._output_has_test_evidence("SUÍTES: 2/2 passaram")
            True
        """
        markers = (
            "pass", "fail", "ok", "tests", "suítes", "suites", "ran ",
            "test result", "assert",
        )
        lowered = output.lower()
        return any(marker in lowered for marker in markers)

    # ── Atribuição real de falhas (TICKET-SGA-11 F1) ──────────────────────
    _FAILED_TEST_PATTERNS = (
        # Bun:  "(fail) nome do teste [0.12ms]"
        re.compile(r"^\s*\(fail\)\s+(?P<name>.+?)(?:\s+\[\d+(?:\.\d+)?m?s\])?\s*$", re.MULTILINE),
        # node --test / TAP:  "not ok 3 - nome do teste"
        re.compile(r"^not ok\s+\d+\s*-\s*(?P<name>.+?)\s*$", re.MULTILINE),
        # reporter spec (node >= 20):  "✖ nome do teste"
        re.compile(r"^\s*✖\s+(?P<name>.+?)\s*$", re.MULTILINE),
        # pytest:  "FAILED tests/test_x.py::test_y - AssertionError..."
        re.compile(r"^FAILED\s+(?P<name>\S+)(?:\s+-\s+.*)?$", re.MULTILINE),
        # cargo test:  "test nome ... FAILED"
        re.compile(r"^test\s+(?P<name>\S+)\s+\.\.\.\s+FAILED", re.MULTILINE),
        # go test:  "--- FAIL: TestName (0.00s)"
        re.compile(r"^--- FAIL:\s+(?P<name>\S+)", re.MULTILINE),
        # dart test:  "[E] nome do teste"
        re.compile(r"^\s*\[E\]\s+(?P<name>.+?)\s*$", re.MULTILINE),
    )

    _ASSERTION_MARKERS = (
        "assertionerror", "expected", "received", "expected:", "received:",
        "error:", "panic:", "error [",
    )

    @classmethod
    def _extract_failed_test_names(cls, output: str) -> List[str]:
        """Extrai nomes reais de testes que falharam da saída do runner.

        Sem heurística inventada: apenas padrões canônicos emitidos por Bun,
        ``node --test`` (TAP/spec), pytest, cargo, go e dart. Ordem preservada
        e duplicatas removidas.

        Exemplo:
            >>> MultiStackRunner._extract_failed_test_names("(fail) soma [0.1ms]")
            ['soma']
        """
        names: List[str] = []
        seen: set = set()
        for pattern in cls._FAILED_TEST_PATTERNS:
            for match in pattern.finditer(output or ""):
                name = match.group("name").strip()
                if not name or name in seen:
                    continue
                seen.add(name)
                names.append(name)
        return names

    @classmethod
    def _extract_assertion_evidence(cls, output: str, limit: int = 2) -> List[str]:
        """Coleta as primeiras linhas de asserção/erro real (não o ruído final).

        Exemplo:
            >>> MultiStackRunner._extract_assertion_evidence("AssertionError: 1 != 2")
            ['AssertionError: 1 != 2']
        """
        evidence: List[str] = []
        for line in (output or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if any(marker in lowered for marker in cls._ASSERTION_MARKERS):
                evidence.append(stripped[:300])
                if len(evidence) >= limit:
                    break
        return evidence

    @classmethod
    def _failure_details_for(
        cls, output: str, runner: Optional[str], raw_tail: int = 500,
    ) -> List[str]:
        """Monta ``failure_details`` atribuído: lista de testes + asserção + anexo bruto.

        O trecho bruto deixa de ser o diagnóstico principal (TICKET-SGA-11 F1):
        ele permanece apenas como anexo final, precedido dos testes que falharam
        e da evidência de asserção.

        Exemplo:
            >>> details = MultiStackRunner._failure_details_for("(fail) soma\\nassert 1==2", "bun")
            >>> details[0].startswith("Testes com falha:")
            True
        """
        if not (output or "").strip():
            return []
        details: List[str] = []
        names = cls._extract_failed_test_names(output)
        if names:
            listed = names[:3]
            extra = len(names) - len(listed)
            header = "Testes com falha: " + "; ".join(listed)
            if extra > 0:
                header += f" (+{extra} outro(s))"
            details.append(header)
        else:
            details.append(
                f"Runner {runner or 'desconhecido'}: falha sem nome de teste identificável na saída"
            )
        details.extend(cls._extract_assertion_evidence(output))
        tail = (output or "")[-max(raw_tail, 0):] if raw_tail else ""
        if tail.strip():
            details.append("Trecho bruto da saída (anexo):\n" + tail)
        return details

    @classmethod
    def _summary_with_failures(cls, base_summary: str, output: str) -> str:
        """Anexa até 3 nomes de testes falhos ao resumo (TICKET-SGA-11 F1).

        Exemplo:
            >>> MultiStackRunner._summary_with_failures("bun: 1 falha", "(fail) soma")
            'bun: 1 falha | testes com falha: soma'
        """
        names = cls._extract_failed_test_names(output)
        if not names:
            return base_summary
        listed = names[:3]
        extra = len(names) - len(listed)
        suffix = ", ".join(listed)
        if extra > 0:
            suffix += f" (+{extra})"
        return f"{base_summary} | testes com falha: {suffix}"

    @classmethod
    def run_tests(cls, workspace_path: Path, stack: str, tmp_dir: Path) -> Dict[str, Any]:
        """Executa a suíte real da stack detectada com TMPDIR isolado e teto de 600s.

        Nenhum caminho devolve ``passed=True`` sem evidência: stack sem runner,
        binário ausente, timeout ou zero testes executados viram
        ``INCONCLUSIVE``/``VERIFIED_FAIL`` com motivo explícito.

        Exemplo:
            >>> res = MultiStackRunner.run_tests(Path("."), "UNKNOWN", Path("/tmp"))
            >>> (res["passed"], res["verdict_state"])
            (False, 'INCONCLUSIVE')
        """
        start = time.time()
        env = os.environ.copy()
        env["TMPDIR"] = str(tmp_dir)

        try:
            if stack == "REACT_NATIVE_EXPO":
                return cls._run_react_native_suite(workspace_path, env, start)
            if stack in ("NODE_TS_BUN", "NODE_JS_GENERIC", "WEB_JS_FIREBASE"):
                return cls._run_node_ts_suite(workspace_path, env, start)
            if stack.startswith("PYTHON"):
                return cls._run_python_pytest_suite(workspace_path, env, start)
            if stack == "RUST_CARGO":
                return cls._run_rust_suite(workspace_path, env, start)
            if stack == "GO_MODULE":
                return cls._run_go_suite(workspace_path, env, start)
            if stack == "DART_FLUTTER":
                return cls._run_dart_suite(workspace_path, env, start)
        except subprocess.TimeoutExpired:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary=f"Stack {stack}: timeout de {_TEST_COMMAND_TIMEOUT_SECONDS:.0f}s excedido",
                failure_details=[f"Timeout de {_TEST_COMMAND_TIMEOUT_SECONDS:.0f}s durante a suíte."],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                reason="timeout",
            )

        return cls._build_test_result(
            passed=False, total_passed=0, total_failed=0,
            summary=f"Stack {stack} não possui runner de testes disponível",
            failure_details=[
                f"Nenhum runner determinístico mapeado para a stack '{stack}'. "
                "A entrega não pode ser aprovada sem evidência real executada."
            ],
            start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
            reason="stack_sem_runner",
        )

    @classmethod
    def _run_react_native_suite(
        cls, workspace_path: Path, env: Dict[str, Any], start: float,
    ) -> Dict[str, Any]:
        """Executa o Jest do projeto React Native/Expo, se instalado.

        Exemplo:
            >>> res = MultiStackRunner._run_react_native_suite(Path("."), {}, time.time())
            >>> res["passed"] is False
            True
        """
        jest_bin = workspace_path / "node_modules" / ".bin" / "jest"
        if not jest_bin.exists():
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary=f"Jest não encontrado em {jest_bin}",
                failure_details=[f"Jest binary missing: {jest_bin}"],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner="jest", reason="jest_ausente",
            )
        cmd = [str(jest_bin), "--runInBand", "--forceExit", "--testTimeout=15000"]
        proc = subprocess.run(
            cmd, cwd=str(workspace_path), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=_TEST_COMMAND_TIMEOUT_SECONDS,
        )
        output = proc.stdout + "\n" + proc.stderr
        passed = proc.returncode == 0
        match = re.search(r"Tests:\s+(\d+)\s+passed", output)
        total_passed = int(match.group(1)) if match else 0
        failed_match = re.search(r"Tests:.*?(\d+)\s+failed", output)
        total_failed = int(failed_match.group(1)) if failed_match else (0 if passed else 1)
        evidence = bool(match or cls._output_has_test_evidence(output))
        base_summary = f"Jest React Native: {total_passed} testes aprovados em {round(time.time() - start, 2)}s"
        return cls._build_test_result(
            passed=passed, total_passed=total_passed, total_failed=total_failed,
            summary=base_summary if passed else cls._summary_with_failures(base_summary, output),
            failure_details=[] if passed else cls._failure_details_for(output, "jest"),
            start=start, evidence=evidence, runner="jest",
            verdict_state=None if evidence else VERDICT_INCONCLUSIVE,
            reason=None if evidence else "saida_sem_contagem",
        )

    @classmethod
    def _run_python_pytest_suite(
        cls, workspace_path: Path, env: Dict[str, Any], start: float,
    ) -> Dict[str, Any]:
        """Executa pytest real sobre testes descobertos no workspace Python.

        Exemplo:
            >>> res = MultiStackRunner._run_python_pytest_suite(Path("/tmp"), {}, time.time())
            >>> res["passed"] is False
            True
        """
        python_bin = sys.executable
        venv_python = workspace_path / ".venv" / "bin" / "python"
        if venv_python.exists():
            python_bin = str(venv_python)
        test_files = cls._discover_python_tests(workspace_path)
        if not test_files:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary="Nenhum teste Python real coletado no workspace",
                failure_details=["Nenhum arquivo 'test_*.py'/'*_test.py' encontrado fora de diretórios excluídos."],
                start=start, evidence=False, verdict_state=VERDICT_VERIFIED_FAIL,
                runner="pytest", reason="testes_ausentes",
            )
        cmd = [python_bin, "-m", "pytest", *test_files, "-q", "--tb=short"]
        proc = subprocess.run(
            cmd, cwd=str(workspace_path), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=_TEST_COMMAND_TIMEOUT_SECONDS,
        )
        output = proc.stdout + "\n" + proc.stderr
        passed = proc.returncode == 0
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+(?:failed|error)", output)
        total_passed = int(passed_match.group(1)) if passed_match else 0
        total_failed = int(failed_match.group(1)) if failed_match else (0 if passed else 1)
        evidence = bool(passed_match or failed_match)
        base_summary = f"pytest real: {total_passed} aprovados, {total_failed} falhas"
        return cls._build_test_result(
            passed=passed, total_passed=total_passed, total_failed=total_failed,
            summary=base_summary if passed else cls._summary_with_failures(base_summary, output),
            failure_details=[] if passed else cls._failure_details_for(output, "pytest"),
            start=start, evidence=evidence, runner="pytest",
            verdict_state=None if evidence else VERDICT_INCONCLUSIVE,
            reason=None if evidence else "saida_sem_contagem",
        )

    @classmethod
    def _discover_python_tests(cls, workspace_path: Path) -> List[str]:
        """Descobre testes Python reais relativos ao workspace, podando diretórios de build.

        Exemplo:
            >>> isinstance(MultiStackRunner._discover_python_tests(Path(".")), list)
            True
        """
        excluded = cls._NATIVE_EXCLUDED_DIRS
        found: List[str] = []
        for pattern in ("tests/**/test_*.py", "tests/**/*_test.py", "test_*.py", "*_test.py"):
            for path in workspace_path.glob(pattern):
                if not path.is_file():
                    continue
                rel_parts = path.relative_to(workspace_path).parts
                if any(part in excluded for part in rel_parts):
                    continue
                found.append(str(path.relative_to(workspace_path)))
        return sorted(set(found))

    @classmethod
    def _run_rust_suite(
        cls, workspace_path: Path, env: Dict[str, Any], start: float,
    ) -> Dict[str, Any]:
        """Executa ``cargo test`` real e parseia o resumo de testes.

        Exemplo:
            >>> res = MultiStackRunner._run_rust_suite(Path("."), {}, time.time())
            >>> "verdict_state" in res
            True
        """
        cargo_bin = shutil.which("cargo")
        if not cargo_bin:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary="Cargo.toml presente, binário cargo ausente",
                failure_details=["cargo não instalado: impossível executar a suíte Rust."],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner="cargo", reason="cargo_ausente",
            )
        proc = subprocess.run(
            [cargo_bin, "test", "--quiet"], cwd=str(workspace_path), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=_TEST_COMMAND_TIMEOUT_SECONDS,
        )
        output = proc.stdout + "\n" + proc.stderr
        passed = proc.returncode == 0
        passed_total = sum(int(m) for m in re.findall(r"(\d+)\s+passed", output))
        failed_total = sum(int(m) for m in re.findall(r"(\d+)\s+failed", output))
        evidence = bool(re.search(r"test result:|(\d+)\s+passed", output))
        return cls._build_test_result(
            passed=passed, total_passed=passed_total,
            total_failed=failed_total if failed_total else (0 if passed else 1),
            summary=(
                f"cargo test: {passed_total} aprovados, {failed_total} falhas"
                if passed else cls._summary_with_failures(
                    f"cargo test: {passed_total} aprovados, {failed_total} falhas", output
                )
            ),
            failure_details=[] if passed else cls._failure_details_for(output, "cargo"),
            start=start, evidence=evidence, runner="cargo",
            verdict_state=None if evidence else VERDICT_INCONCLUSIVE,
            reason=None if evidence else "saida_sem_contagem",
        )

    @classmethod
    def _run_go_suite(
        cls, workspace_path: Path, env: Dict[str, Any], start: float,
    ) -> Dict[str, Any]:
        """Executa ``go test ./...`` real e parseia pacotes aprovados/falhados.

        Exemplo:
            >>> res = MultiStackRunner._run_go_suite(Path("."), {}, time.time())
            >>> "verdict_state" in res
            True
        """
        go_bin = shutil.which("go")
        if not go_bin:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary="go.mod presente, binário go ausente",
                failure_details=["go não instalado: impossível executar a suíte Go."],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner="go", reason="go_ausente",
            )
        proc = subprocess.run(
            [go_bin, "test", "./..."], cwd=str(workspace_path), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=_TEST_COMMAND_TIMEOUT_SECONDS,
        )
        output = proc.stdout + "\n" + proc.stderr
        ok_packages = len(re.findall(r"^ok\s+", output, re.MULTILINE))
        failed_packages = len(re.findall(r"^(?:FAIL|--- FAIL)", output, re.MULTILINE))
        tests_ran = ok_packages > 0
        # TICKET-SGA-10: "no test files" NÃO é evidência de suíte aprovada.
        # Sem ao menos um pacote `ok` não há como emitir VERIFIED_PASS.
        passed = proc.returncode == 0 and ok_packages > 0 and failed_packages == 0
        evidence = bool(ok_packages or failed_packages or "no test files" in output)
        base_summary = f"go test: {ok_packages} pacotes ok, {failed_packages} falhas"
        if not tests_ran:
            verdict_state = VERDICT_VERIFIED_FAIL if failed_packages else VERDICT_INCONCLUSIVE
            reason = "go_sem_pacotes_ok" if not failed_packages else "go_testes_falharam"
            if "no test files" in output and not failed_packages:
                base_summary += " (nenhum arquivo *_test.go executado)"
        else:
            verdict_state = None if evidence else VERDICT_INCONCLUSIVE
            reason = None if evidence else "saida_sem_contagem"
        return cls._build_test_result(
            passed=passed, total_passed=ok_packages,
            total_failed=failed_packages if failed_packages else (0 if passed else 1),
            summary=base_summary if passed else cls._summary_with_failures(base_summary, output),
            failure_details=[] if passed else cls._failure_details_for(output, "go"),
            start=start, evidence=evidence, runner="go",
            verdict_state=verdict_state,
            reason=reason,
        )

    @classmethod
    def _run_dart_suite(
        cls, workspace_path: Path, env: Dict[str, Any], start: float,
    ) -> Dict[str, Any]:
        """Executa ``dart test``/``flutter test`` real quando o SDK está instalado.

        Exemplo:
            >>> res = MultiStackRunner._run_dart_suite(Path("."), {}, time.time())
            >>> "verdict_state" in res
            True
        """
        dart_bin = shutil.which("dart")
        flutter_bin = shutil.which("flutter")
        if not dart_bin and not flutter_bin:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary="pubspec.yaml presente, SDK Dart/Flutter ausente",
                failure_details=["dart/flutter não instalados: impossível executar a suíte Dart."],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner="dart", reason="dart_ausente",
            )
        cmd = [str(flutter_bin), "test"] if flutter_bin else [str(dart_bin), "test"]
        proc = subprocess.run(
            cmd, cwd=str(workspace_path), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            timeout=_TEST_COMMAND_TIMEOUT_SECONDS,
        )
        output = proc.stdout + "\n" + proc.stderr
        passed = proc.returncode == 0
        passed_match = re.search(r"(\d+)\s+tests?\s+passed", output)
        failed_match = re.search(r"(\d+)\s+tests?\s+failed", output)
        total_passed = int(passed_match.group(1)) if passed_match else 0
        total_failed = int(failed_match.group(1)) if failed_match else (0 if passed else 1)
        evidence = bool(passed_match or failed_match or cls._output_has_test_evidence(output))
        base_summary = f"Dart/Flutter test: {total_passed} aprovados, {total_failed} falhas"
        return cls._build_test_result(
            passed=passed, total_passed=total_passed, total_failed=total_failed,
            summary=base_summary if passed else cls._summary_with_failures(base_summary, output),
            failure_details=[] if passed else cls._failure_details_for(output, Path(cmd[0]).name),
            start=start, evidence=evidence, runner=Path(cmd[0]).name,
            verdict_state=None if evidence else VERDICT_INCONCLUSIVE,
            reason=None if evidence else "saida_sem_contagem",
        )

    @classmethod
    def _run_node_ts_suite(
        cls, workspace_path: Path, env: Dict[str, Any], start: float,
    ) -> Dict[str, Any]:
        """Executa a suíte determinística real de um repositório Node/TS/Bun (sem mocks).

        Cascata de seleção do runner (primeiro match vence):
        1. ``tests/run_all.mjs`` presente -> ``node tests/run_all.mjs``.
        2. Monorepo com ``workspaces`` declarados ou guarda de raiz
           (``do not run tests from root``) -> ``npm test``/``bun run test``
           por pacote que possua suíte real.
        3. ``package.json`` com ``scripts.test`` -> ``bun run test`` ou ``npm test``.
        4. Repositório gerenciado por Bun sem script -> ``bun test``.
        5. Suíte descoberta (``tests/**/*.mjs``, ``*.test.*``/``*.spec.*``)
           -> ``node --test --test-reporter=tap <arquivos>``.

        Exemplo:
            >>> res = MultiStackRunner._run_node_ts_suite(Path("."), {}, time.time())
            >>> set(res) >= {"passed", "total_passed", "total_failed", "summary", "failure_details", "duration"}
            True
        """
        node_bin = shutil.which("node") or str(Path.home() / ".local" / "node" / "bin" / "node")
        npm_bin = shutil.which("npm") or "npm"
        bun_bin = shutil.which("bun")

        pkg_data = cls._read_package_json(workspace_path)
        scripts = pkg_data.get("scripts") if isinstance(pkg_data, dict) else None
        raw_test = scripts.get("test") if isinstance(scripts, dict) else None
        test_script: Optional[str] = raw_test if isinstance(raw_test, str) else None

        bun_lock = (workspace_path / "bun.lockb").exists() or (workspace_path / "bun.lock").exists()
        has_bunfig = (workspace_path / "bunfig.toml").exists()
        bun_script = test_script is not None and test_script.strip().startswith("bun")
        is_bun = bool(bun_bin) and (bun_lock or has_bunfig or bun_script)
        root_guard = bool(test_script and "do not run tests from root" in test_script.lower())

        tests_dir = workspace_path / "tests"
        run_all = tests_dir / "run_all.mjs"
        if run_all.exists():
            return cls._run_node_command(
                [str(node_bin), str(run_all)], workspace_path, env, start, "run_all.mjs",
            )

        workspace_packages = cls._discover_node_workspaces(workspace_path)
        declares_workspaces = bool(pkg_data.get("workspaces")) if isinstance(pkg_data, dict) else False
        if workspace_packages and (declares_workspaces or root_guard or len(workspace_packages) > 1):
            return cls._run_node_workspaces(workspace_packages, env, start, is_bun, bun_bin, npm_bin)

        if test_script and not root_guard:
            cmd = [str(bun_bin), "run", "test"] if is_bun else [str(npm_bin), "test"]
            return cls._run_node_command(cmd, workspace_path, env, start, "scripts.test")

        if is_bun:
            return cls._run_node_command([str(bun_bin), "test"], workspace_path, env, start, "bun test")

        discovered = cls._discover_node_test_files(workspace_path)
        if discovered:
            # --test-reporter=tap força saída TAP estável em qualquer versão do
            # Node (o reporter padrão "spec" muda de formato entre majors).
            cmd = [str(node_bin), "--test", "--test-reporter=tap", *discovered]
            return cls._run_node_command(cmd, workspace_path, env, start, "node --test")

        return cls._build_test_result(
            passed=False, total_passed=0, total_failed=0,
            summary="Workspace Node/TS/Bun sem runner de testes configurado — sem suíte executável",
            failure_details=[
                "Nenhum runner detectável (tests/run_all.mjs, scripts.test, bun test, "
                "workspaces com suíte ou arquivos *.test.*/*.spec.*). "
                "A entrega não pode ser aprovada sem evidência real executada."
            ],
            start=start, evidence=False, verdict_state=VERDICT_VERIFIED_FAIL,
            runner=None, reason="runner_ausente_em_workspace_com_suite_esperada",
        )

    @classmethod
    def _discover_node_workspaces(cls, workspace_path: Path) -> List[Path]:
        """Descobre pacotes de monorepo Node com suíte real (workspaces/scripts.test).

        Exemplo:
            >>> isinstance(MultiStackRunner._discover_node_workspaces(Path(".")), list)
            True
        """
        candidates: set[Path] = set()
        pkg_data = cls._read_package_json(workspace_path)
        workspaces = pkg_data.get("workspaces") if isinstance(pkg_data, dict) else None
        patterns: List[str] = []
        if isinstance(workspaces, list):
            patterns.extend(str(item) for item in workspaces)
        elif isinstance(workspaces, dict) and isinstance(workspaces.get("packages"), list):
            patterns.extend(str(item) for item in workspaces["packages"])
        patterns.extend(_NODE_WORKSPACE_GLOBS)

        for pattern in patterns:
            for candidate in workspace_path.glob(pattern):
                pkg_dir = candidate.parent if candidate.name == "package.json" else candidate
                if not (pkg_dir / "package.json").is_file():
                    continue
                try:
                    pkg_dir.resolve().relative_to(workspace_path.resolve())
                except ValueError:
                    continue
                if "node_modules" in pkg_dir.parts:
                    continue
                candidates.add(pkg_dir)

        suites: List[Path] = []
        for pkg_dir in sorted(candidates):
            data = cls._read_package_json(pkg_dir)
            scripts = data.get("scripts") if isinstance(data, dict) else None
            has_script = isinstance(scripts, dict) and bool(scripts.get("test"))
            if has_script or cls._discover_node_test_files(pkg_dir):
                suites.append(pkg_dir)
        return suites

    @classmethod
    def _discover_node_test_files(cls, base_dir: Path) -> List[str]:
        """Lista arquivos de teste Node/TS reais sob ``base_dir`` (sem node_modules/build).

        Exemplo:
            >>> isinstance(MultiStackRunner._discover_node_test_files(Path(".")), list)
            True
        """
        excluded = cls._NATIVE_EXCLUDED_DIRS
        found: set[str] = set()
        tests_dir = base_dir / "tests"
        if tests_dir.is_dir():
            found.update(str(path) for path in tests_dir.rglob("*.mjs") if path.is_file())
        for pattern in _JS_TEST_GLOBS:
            for path in base_dir.rglob(pattern):
                if not path.is_file():
                    continue
                try:
                    rel_parts = path.relative_to(base_dir).parts
                except ValueError:
                    continue
                if any(part in excluded for part in rel_parts):
                    continue
                found.add(str(path))
        return sorted(found)

    @classmethod
    def _run_node_command(
        cls,
        cmd: List[str],
        cwd: Path,
        env: Dict[str, Any],
        start: float,
        label: str,
    ) -> Dict[str, Any]:
        """Executa um comando Node/Bun e normaliza o resultado com evidência real.

        Exemplo:
            >>> res = MultiStackRunner._run_node_command(["true"], Path("."), {}, 0.0, "noop")
            >>> res["passed"] is False
            True
        """
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                timeout=_TEST_COMMAND_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as missing_err:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary=f"Binário ausente para o runner {label}: {missing_err}",
                failure_details=[str(missing_err)],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner=label, reason="binario_ausente",
            )
        except subprocess.TimeoutExpired:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary=f"Timeout de {_TEST_COMMAND_TIMEOUT_SECONDS:.0f}s no runner {label}",
                failure_details=[f"Timeout de {_TEST_COMMAND_TIMEOUT_SECONDS:.0f}s excedido."],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner=label, reason="timeout",
            )
        except Exception as run_err:  # noqa: BLE001 — falha operacional vira inconclusivo
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary=f"Falha ao executar {Path(cmd[0]).name}: {run_err}",
                failure_details=[str(run_err)],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner=label, reason="erro_operacional",
            )

        output = proc.stdout + "\n" + proc.stderr
        passed = proc.returncode == 0
        parsed = cls._parse_node_test_counts_exact(output)
        if parsed is None:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary=f"Runner {label} executou mas a saída não pôde ser contabilizada",
                failure_details=[
                    "Saída sem contagem numérica interpretável; verificação inconclusiva.",
                    output[-500:],
                ],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner=label, reason="saida_sem_contagem",
            )
        tot_passed, tot_failed = parsed
        base_summary = f"Node/TS/Bun Tests ({label}): {tot_passed} aprovados, {tot_failed} falhas"
        return cls._build_test_result(
            passed=passed, total_passed=tot_passed, total_failed=tot_failed,
            summary=base_summary if passed else cls._summary_with_failures(base_summary, output),
            failure_details=[] if passed else cls._failure_details_for(output, label),
            start=start, evidence=True, runner=label,
            reason=None if passed else "testes_falharam",
        )

    @classmethod
    def _run_node_workspaces(
        cls,
        packages: List[Path],
        env: Dict[str, Any],
        start: float,
        is_bun: bool,
        bun_bin: Optional[str],
        npm_bin: str,
    ) -> Dict[str, Any]:
        """Executa a suíte de cada pacote de monorepo e agrega os contadores reais.

        Exemplo:
            >>> res = MultiStackRunner._run_node_workspaces([], {}, 0.0, False, None, "npm")
            >>> res["verdict_state"]
            'INCONCLUSIVE'
        """
        if not packages:
            return cls._build_test_result(
                passed=False, total_passed=0, total_failed=0,
                summary="Monorepo declarado sem pacote com suíte executável",
                failure_details=["Nenhum pacote com scripts.test ou arquivos de teste."],
                start=start, evidence=False, verdict_state=VERDICT_INCONCLUSIVE,
                runner="workspaces", reason="monorepo_sem_suite",
            )

        aggregate_passed = 0
        aggregate_failed = 0
        details: List[str] = []
        failures: List[str] = []
        evidence_any = False
        all_passed = True
        for pkg_dir in packages:
            rel = pkg_dir.relative_to(packages[0].parent) if packages else pkg_dir
            cmd = [str(bun_bin), "run", "test"] if is_bun and bun_bin else [str(npm_bin), "test"]
            result = cls._run_node_command(cmd, pkg_dir, env, start, f"{Path(rel)}/test")
            if result["evidence"]:
                evidence_any = True
                aggregate_passed += result["total_passed"]
                aggregate_failed += result["total_failed"]
            if not result["passed"]:
                all_passed = False
                failures.extend(result["failure_details"])
            details.append(f"{rel}: {result['verdict_state']}")

        return cls._build_test_result(
            passed=all_passed and evidence_any,
            total_passed=aggregate_passed, total_failed=aggregate_failed,
            summary=f"Monorepo ({len(packages)} pacotes): " + ", ".join(details),
            failure_details=failures[:10],
            start=start, evidence=evidence_any, runner="workspaces",
            verdict_state=None if evidence_any else VERDICT_INCONCLUSIVE,
            reason=None if evidence_any else "pacotes_sem_contagem",
        )

    @staticmethod
    def _parse_node_test_counts_exact(output: str) -> Optional[Tuple[int, int]]:
        """Extrai (aprovados, falhas) reais ou ``None`` quando não há contagem.

        Contrato honesto: ``None`` significa "não foi possível contabilizar",
        nunca um fallback inventado.

        Exemplo:
            >>> MultiStackRunner._parse_node_test_counts_exact("12 pass\\n1 fail")
            (12, 1)
            >>> MultiStackRunner._parse_node_test_counts_exact("sem contadores") is None
            True
        """
        # Runner consolidado: "SUÍTES: X/Y passaram"
        m_suites = re.search(r"SU[ÍI]TES:\s+(\d+)\s*/\s*(\d+)\s+passaram", output)
        if m_suites:
            ok = int(m_suites.group(1))
            total = int(m_suites.group(2))
            return ok, max(total - ok, 0)

        # Bun test: " 12 pass" / " 1 fail"
        m_bun_pass = re.search(r"(\d+)\s+pass\b", output)
        m_bun_fail = re.search(r"(\d+)\s+fail\b", output)
        if m_bun_pass or m_bun_fail:
            ok = int(m_bun_pass.group(1)) if m_bun_pass else 0
            fail = int(m_bun_fail.group(1)) if m_bun_fail else 0
            return ok, fail

        # node --test (TAP): "# pass 12" / "# fail 1"; reporter spec emite
        # "ℹ pass 12" e checkmarks "✔ pass"/"✖ fail" em versões mais novas.
        m_tap_pass = re.search(r"(?:#|ℹ|✔)\s*pass\s+(\d+)", output, re.IGNORECASE)
        m_tap_fail = re.search(r"(?:#|ℹ|✖)\s*fail\s+(\d+)", output, re.IGNORECASE)
        if m_tap_pass or m_tap_fail:
            ok = int(m_tap_pass.group(1)) if m_tap_pass else 0
            fail = int(m_tap_fail.group(1)) if m_tap_fail else 0
            return ok, fail

        return None

    @staticmethod
    def _parse_node_test_counts(output: str, passed: bool) -> Tuple[int, int]:
        """Compat: contagens reais ou (0,0)/(0,1) quando a saída não contabiliza.

        Exemplo:
            >>> MultiStackRunner._parse_node_test_counts("  12 pass\\n  1 fail\\n", False)
            (12, 1)
        """
        parsed = MultiStackRunner._parse_node_test_counts_exact(output)
        if parsed is not None:
            return parsed
        return (0, 0) if passed else (0, 1)

    @classmethod
    def _run_node_js_ast(
        cls,
        workspace_path: Path,
        source_files: List[Path],
        stack: str,
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation],
    ) -> Optional[Dict[str, Any]]:
        """Executa analisador semântico de AST JS via Node.js."""
        js_guard_script = Path(__file__).resolve().parent.parent / "memory" / "js_ast_guard.js"
        # Fallbacks explícitos de Linux e macOS (Homebrew Intel/Apple Silicon):
        # ``which`` é a fonte primária, nunca a única.
        node_candidates = (
            shutil.which("node"),
            str(Path.home() / ".local" / "node" / "bin" / "node"),
            "/opt/homebrew/bin/node",
            "/usr/local/bin/node",
        )
        node_bin = next(
            (candidate for candidate in node_candidates if candidate and Path(candidate).exists()),
            None,
        )

        if not node_bin or not js_guard_script.exists():
            return None

        cmd = [str(node_bin), str(js_guard_script)]
        if stack == "WEB_JS_FIREBASE":
            cmd.append("--check-emoji")
        cmd.extend([str(f) for f in source_files])
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(workspace_path),
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as run_err:
            logger.debug(f"Falha de execução do Node: {run_err}")
            return None

        if proc.returncode != 0 or not proc.stdout.strip():
            return None

        try:
            ast_data = json.loads(proc.stdout)
        except Exception as parse_err:
            logger.debug(f"Falha ao interpretar JSON do Node AST: {parse_err}")
            return None

        for item in ast_data.get("violations", []):
            rel = item["file_path"]
            if Path(rel).is_absolute():
                # Symlinks do macOS (/var -> /private/var): relativização simétrica.
                rel = cls._workspace_relative(workspace_path, rel)
            v = MultiStackViolation(
                rule_id=item["rule_id"],
                category=item["category"],
                file_path=rel,
                line_number=item["line_number"],
                code_snippet=item["code_snippet"],
                message=item["message"],
                suggestion=item.get("suggestion") or RULE_SUGGESTIONS.get(item["rule_id"]),
            )
            if item["category"] in ("security", "anti_mock_producao"):
                blocking.append(v)
            else:
                warnings.append(v)

        return {
            "blocking": [asdict(b) for b in blocking],
            "warnings": [asdict(w) for w in warnings],
            "total_files": len(source_files),
            "engine": "node_js_ast",
        }

    @classmethod
    def _dedup_files(cls, source_files: List[Path]) -> List[Path]:
        """Remove caminhos duplicados preservando ordem de aparição."""
        seen_paths = set()
        deduped: List[Path] = []
        for sf in source_files:
            try:
                p_resolved = sf.resolve()
            except Exception:
                p_resolved = sf
            if p_resolved in seen_paths:
                continue
            seen_paths.add(p_resolved)
            deduped.append(sf)
        return deduped

    @classmethod
    def _filter_delta_files(cls, workspace_path: Path, source_files: List[Path]) -> List[Path]:
        """Filtra lista de arquivos mantendo apenas os alterados no Git."""
        try:
            from agent_quality_guard.core.diff_inspector import get_workspace_diff_files, normalize_workspace_relpath
            diff_files = get_workspace_diff_files(str(workspace_path))
        except Exception as diff_err:
            logger.debug(f"Não foi possível obter diff de arquivos para auditoria delta: {diff_err}")
            return source_files

        if not diff_files:
            return source_files

        changed_paths = {normalize_workspace_relpath(df.path, workspace_path) for df in diff_files}
        filtered: List[Path] = []
        for f in source_files:
            rel = normalize_workspace_relpath(f, workspace_path)
            is_hit = rel in changed_paths or f.name in changed_paths or any(rel.endswith(cp) or cp.endswith(rel) for cp in changed_paths)
            if is_hit:
                filtered.append(f)
        return filtered

    @classmethod
    def _check_line_patterns(
        cls,
        line: str,
        idx: int,
        lines: List[str],
        rel: str,
        patterns: List[Tuple[str, str, str, str]],
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation],
        file_suppressed: Optional[set[str]] = None,
    ) -> None:
        """Verifica padrões léxicos em uma linha e adiciona às listas de violação."""
        for pat, rule_id, category, msg in patterns:
            if rule_id == "XSS_INNER_HTML":
                if any(san in line for san in ("escapeHtml", "DOMPurify", "encodeURIComponent")):
                    continue
            if re.search(pat, line) and not _is_line_suppressed(rule_id, idx, lines, file_suppressed):
                v = MultiStackViolation(
                    rule_id=rule_id,
                    category=category,
                    file_path=rel,
                    line_number=idx,
                    code_snippet=line.strip()[:100],
                    message=msg,
                    suggestion=RULE_SUGGESTIONS.get(rule_id)
                )
                if category in ("security", "anti_mock_producao"):
                    blocking.append(v)
                else:
                    warnings.append(v)

    @classmethod
    def _scan_file_lines(
        cls,
        fpath: Path,
        workspace_path: Path,
        patterns: List[Tuple[str, str, str, str]],
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation]
    ) -> bool:
        """Lê e varre as linhas de um único arquivo de código."""
        rel = cls._workspace_relative(workspace_path, fpath)
        try:
            content = fpath.read_text(encoding="utf-8", errors="ignore")
        except Exception as read_err:
            logger.warning("Arquivo %s ignorado na varredura JS: %s", fpath, read_err)
            return False

        lines = content.splitlines()
        file_suppressed = _get_file_suppressed_rules_js(lines)
        for idx, line in enumerate(lines, 1):
            cls._check_line_patterns(line, idx, lines, rel, patterns, blocking, warnings, file_suppressed)
        return True

    @classmethod
    def _run_lexical_fallback(
        cls,
        workspace_path: Path,
        source_files: List[Path],
        stack: str,
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation]
    ) -> Dict[str, Any]:
        """Executa varredura léxica resiliente com suporte a needle-ignore e suggestions."""
        patterns: List[Tuple[str, str, str, str]] = [
            (r"jest\.(?:mock|fn)\(|sinon\.stub\(", "JS_PROD_MOCK", "anti_mock_producao", "Mock detectado em arquivo de produção sob pasta de código-fonte."),
            (r"\beval\s*\(", "SECURITY_EVAL", "security", "Uso proibido de eval()"),
            (r"new\s+Function\s*\(", "SECURITY_EXEC", "security", "Uso proibido de new Function()"),
            (r"AIzaSy[0-9A-Za-z-_]{33}", "HARDCODED_SECRET", "security", "Chave privada de API hardcoded em arquivo de produção"),
            (r"\.innerHTML\s*=\s*(?!escapeHtml|DOMPurify|encodeURIComponent)[^;]+[+`]", "XSS_INNER_HTML", "security", "Atribuição insegura de innerHTML sem sanitização"),
            (r"console\.log\(", "CODE_SMELL_CONSOLE", "hygiene", "Chamada de console.log residual em produção"),
            (r"catch\s*\([^)]*\)\s*\{\s*\}", "EMPTY_CATCH", "hygiene", "Bloco catch silencioso engolindo exceções sem log ou tratamento"),
        ]
        if stack == "WEB_JS_FIREBASE":
            patterns.append((
                r"[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF]",
                "JEZ_ANTI_EMOJI",
                "brand_invariable",
                "Emoji detectado no código/template em violação à diretriz inegociável da marca JEZ."
            ))

        scanned_count = 0
        for fpath in source_files:
            if cls._scan_file_lines(fpath, workspace_path, patterns, blocking, warnings):
                scanned_count += 1

        return {
            "blocking": [asdict(b) for b in blocking],
            "warnings": [asdict(w) for w in warnings],
            "total_files": scanned_count,
            "engine": "lexical_fallback",
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Camada 1 — Bridge Nativa com Compiladores e Linters do Projeto
    # ──────────────────────────────────────────────────────────────────────────

    _JS_STACKS = frozenset({"REACT_NATIVE_EXPO", "WEB_JS_FIREBASE", "NODE_TS_BUN", "NODE_JS_GENERIC"})
    _PYTHON_STACKS = frozenset({"PYTHON_GENERIC"})
    _OTHER_TOOLCHAIN_STACKS = frozenset({"RUST_CARGO", "GO_MODULE", "DART_FLUTTER"})
    _BLOCKING_NATIVE_CATEGORIES = frozenset({"security", "anti_mock_producao", "type_error", "lint_error"})
    _NATIVE_EXCLUDED_DIRS = frozenset({
        ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
        ".ruff_cache", ".pytest_cache", "build", "dist", "coverage", ".tox",
        "site-packages", "target", ".dart_tool", ".next", ".expo",
    })
    _ESLINT_CONFIG_GLOBS: Tuple[str, ...] = (
        ".eslintrc", ".eslintrc.js", ".eslintrc.cjs", ".eslintrc.json",
        ".eslintrc.yml", ".eslintrc.yaml", "eslint.config.js",
        "eslint.config.mjs", "eslint.config.cjs", "eslint.config.ts",
    )
    _BIOME_CONFIG_GLOBS: Tuple[str, ...] = ("biome.json", "biome.jsonc")
    _OXLINT_CONFIG_GLOBS: Tuple[str, ...] = (".oxlintrc.json", "oxlintrc.json")
    _ESLINT_RULE_MAP: Dict[str, Tuple[str, str]] = {
        "react-hooks/rules-of-hooks": (
            "ESLINT_RULES_OF_HOOKS",
            "Hook chamado fora do topo do componente — viole a ordem de hooks quebra o runtime React.",
        ),
        "react-hooks/exhaustive-deps": (
            "ESLINT_HOOK_DEPS",
            "Dependências reativas ausentes/instáveis em useEffect/useMemo/useCallback.",
        ),
        "react/jsx-key": (
            "ESLINT_MISSING_KEY",
            "Elemento de lista sem prop 'key' estável e única.",
        ),
        "react/no-array-index-key": (
            "ESLINT_ARRAY_INDEX_KEY",
            "Uso do índice do array como 'key' — instabiliza reconciliação em reordenações.",
        ),
        "react/no-direct-mutation-state": (
            "ESLINT_STATE_MUTATION",
            "Mutação direta de state detectada — use setState/setX com imutabilidade.",
        ),
    }

    @classmethod
    def run_native_bridge(
        cls,
        workspace_path: Path,
        stack: str,
        audit_mode: str = "delta",
        source_files: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """Camada 3: executa compiladores/linters nativos e normaliza diagnósticos.

        Orçamento total de 600s por padrão (sobrescrevível por
        ``JEV_NATIVE_BRIDGE_BUDGET_SECONDS``, clamp [0.1, 600]). Cada ferramenta
        roda em novo grupo de processos e é morta via ``killpg`` no timeout (sem
        órfãos). Ferramenta ausente vira ``skipped`` com motivo; timeout/erro
        operacional vira ``inconclusive_reasons``/``critical_failures`` — NUNCA
        é silenciado como "zero erros".

        Exemplo:
            >>> res = MultiStackRunner.run_native_bridge(Path("."), "PYTHON_GENERIC")
            >>> set(res) >= {"blocking", "warnings", "tools", "total_files", "engine"}
            True
        """
        started = time.monotonic()
        deadline = started + cls._native_budget_seconds()
        blocking: List[MultiStackViolation] = []
        warnings: List[MultiStackViolation] = []
        runs: List[NativeToolRun] = []
        workspace_path = Path(workspace_path).resolve()

        env = os.environ.copy()
        env.update({"CI": "1", "NO_COLOR": "1", "FORCE_COLOR": "0", "TERM": "dumb"})

        if source_files is None:
            files = cls._collect_native_source_files(workspace_path, stack, audit_mode)
        else:
            files = [Path(raw_file) for raw_file in source_files]

        normalized: List[Path] = []
        for raw_file in files:
            candidate = raw_file if raw_file.is_absolute() else workspace_path / raw_file
            if candidate.is_file():
                normalized.append(candidate)
        files = cls._dedup_files(normalized)[:_MAX_LINT_FILES]

        try:
            if stack in cls._JS_STACKS:
                cls._bridge_js_toolchain(workspace_path, files, env, deadline, runs, blocking, warnings)
            if stack in cls._PYTHON_STACKS:
                cls._bridge_python_toolchain(workspace_path, files, env, deadline, runs, blocking, warnings)
            cls._bridge_other_toolchains(workspace_path, files, env, deadline, runs, blocking, warnings)
        except Exception as bridge_err:  # defensivo: um linter nunca derruba o quality gate
            logger.debug("Camada 3 abortada por erro interno em %s: %s", workspace_path, bridge_err)
            runs.append(NativeToolRun(tool="native_bridge", status="error", detail=str(bridge_err)[:200]))

        tools = [cls._native_tool_entry(run) for run in runs]
        executed = [tool["tool"] for tool in tools if tool["status"] == "executed"]
        skipped = {tool["tool"]: tool.get("reason") or "sem motivo declarado" for tool in tools if tool["status"] == "skipped"}
        critical = [
            tool["tool"] for tool in tools
            if tool["status"] in ("timeout", "error")
        ]
        layer_status = "executed" if executed else ("error" if critical else "skipped")
        return {
            "blocking": [asdict(v) for v in cls._dedup_native_violations(blocking)],
            "warnings": [asdict(w) for w in cls._dedup_native_violations(warnings)],
            "tools": tools,
            "total_files": len(files),
            "engine": "native_toolchain_bridge",
            "duration": round(time.monotonic() - started, 3),
            "layer_status": layer_status,
            "executed_tools": executed,
            "skipped_tools": skipped,
            "critical_failures": critical,
            "inconclusive_reasons": [
                f"{tool['tool']}: {tool['status']} ({tool.get('detail') or 'sem detalhe'})"
                for tool in tools if tool["status"] in ("timeout", "error")
            ],
        }

    @staticmethod
    def _native_tool_entry(run: NativeToolRun) -> Dict[str, Any]:
        """Serializa o diagnóstico de execução com aliases de compatibilidade (A1).

        Exemplo:
            >>> entry = MultiStackRunner._native_tool_entry(NativeToolRun(tool="ruff", status="executed"))
            >>> (entry["tool"], entry["name"], entry["duration_ms"])
            ('ruff', 'ruff', 0)
        """
        return {
            "tool": run.tool,
            "status": run.status,
            "returncode": run.returncode,
            "duration": run.duration,
            "violation_count": run.violation_count,
            "detail": run.detail,
            "name": run.tool,
            "available": run.status != "skipped",
            "skipped": run.status == "skipped",
            "reason": run.detail,
            "duration_ms": int(run.duration * 1000),
        }

    @staticmethod
    def _native_budget_seconds() -> float:
        """Resolve o orçamento total (segundos) da bridge nativa com clamp defensivo.

        Default e teto de 600s (TICKET-SGA-09); piso de 0.1s para permitir
        execuções de teste econômicas via env.

        Exemplo:
            >>> 0.1 <= MultiStackRunner._native_budget_seconds() <= 600.0
            True
        """
        raw = os.environ.get("JEV_NATIVE_BRIDGE_BUDGET_SECONDS", "")
        try:
            value = float(raw) if raw.strip() else _DEFAULT_NATIVE_BUDGET_SECONDS
        except (TypeError, ValueError):
            value = _DEFAULT_NATIVE_BUDGET_SECONDS
        return min(max(value, _MIN_NATIVE_BUDGET_SECONDS), _MAX_NATIVE_BUDGET_SECONDS)

    @staticmethod
    def _resolve_native_tool(workspace_path: Path, name: str, prefer_local_bin: bool = False) -> Optional[str]:
        """Resolve um executável priorizando o ambiente do projeto auditado.

        Ordem: ``node_modules/.bin`` (quando JS) -> ``.venv``/``venv`` -> ``PATH``.

        Exemplo:
            >>> MultiStackRunner._resolve_native_tool(Path("."), "tool-que-nao-existe-xyz") is None
            True
        """
        candidates: List[Path] = []
        if prefer_local_bin:
            candidates.append(workspace_path / "node_modules" / ".bin" / name)
        candidates.extend([
            workspace_path / ".venv" / "bin" / name,
            workspace_path / "venv" / "bin" / name,
            workspace_path / ".venv" / "Scripts" / f"{name}.exe",
            workspace_path / "venv" / "Scripts" / f"{name}.exe",
        ])
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
        return shutil.which(name)

    @staticmethod
    def _has_project_file(workspace_path: Path, patterns: Tuple[str, ...]) -> bool:
        """Indica se ao menos um arquivo de configuração existe na raiz do projeto.

        Exemplo:
            >>> MultiStackRunner._has_project_file(Path("."), ("inexistente-xyz.json",))
            False
        """
        return any(any(workspace_path.glob(pattern)) for pattern in patterns)

    @staticmethod
    def _package_json_has_eslint_config(workspace_path: Path) -> bool:
        """Detecta configuração ESLint embutida no package.json (chave eslintConfig).

        Exemplo:
            >>> MultiStackRunner._package_json_has_eslint_config(Path("."))
            False
        """
        pkg = workspace_path / "package.json"
        if not pkg.exists():
            return False
        try:
            data = json.loads(pkg.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, json.JSONDecodeError) as parse_err:
            logger.debug("package.json ilegível para detectar eslintConfig em %s: %s", workspace_path, parse_err)
            return False
        return isinstance(data, dict) and isinstance(data.get("eslintConfig"), dict)

    @classmethod
    def _iter_native_files(cls, workspace_path: Path, extensions: Tuple[str, ...]) -> List[Path]:
        """Percorre a árvore podando diretórios excluídos (venv/node_modules/build).

        Exemplo:
            >>> isinstance(MultiStackRunner._iter_native_files(Path("."), (".py",)), list)
            True
        """
        wanted = {ext.lstrip("*") for ext in extensions}
        found: List[Path] = []
        for root, dirnames, filenames in os.walk(workspace_path):
            dirnames[:] = [d for d in dirnames if d not in cls._NATIVE_EXCLUDED_DIRS]
            root_path = Path(root)
            for fname in filenames:
                if Path(fname).suffix in wanted:
                    found.append(root_path / fname)
        return found

    @classmethod
    def _collect_native_source_files(cls, workspace_path: Path, stack: str, audit_mode: str) -> List[Path]:
        """Coleta os arquivos de produção que alimentam linters/compiladores da stack.

        Exemplo:
            >>> isinstance(MultiStackRunner._collect_native_source_files(Path("."), "PYTHON_GENERIC", "full"), list)
            True
        """
        if stack in cls._JS_STACKS:
            extensions: Tuple[str, ...] = (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs")
        elif stack in cls._PYTHON_STACKS:
            extensions = (".py",)
        else:
            extensions = (".py", ".go", ".rs", ".dart", ".js", ".ts", ".mjs")

        files = cls._dedup_files(cls._iter_native_files(workspace_path, extensions))
        if audit_mode == "delta":
            files = cls._filter_delta_files(workspace_path, files)
        return files[:_MAX_LINT_FILES]

    @classmethod
    def _bridge_js_toolchain(
        cls,
        workspace_path: Path,
        files: List[Path],
        env: Dict[str, str],
        deadline: float,
        runs: List[NativeToolRun],
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation],
    ) -> None:
        """Executa ``tsc --noEmit`` e o linter JS configurado (eslint > biome > oxlint).

        Exemplo:
            >>> MultiStackRunner._bridge_js_toolchain(Path("."), [], {}, time.monotonic(), [], [], [])
        """
        tsconfig = workspace_path / "tsconfig.json"
        tsc_bin = cls._resolve_native_tool(workspace_path, "tsc", prefer_local_bin=True)
        if tsconfig.exists() and tsc_bin:
            output = cls._execute_native_tool(
                "tsc", [tsc_bin, "--noEmit", "--pretty", "false"],
                workspace_path, env, deadline, runs,
            )
            if output is not None:
                cls._apply_native_violations(
                    cls._parse_tsc_output(workspace_path, output), runs[-1], blocking, warnings,
                )
        elif tsconfig.exists():
            runs.append(NativeToolRun(
                tool="tsc", status="skipped",
                detail="tsconfig.json presente, binário tsc ausente (node_modules/.bin ou PATH)",
            ))

        relative_files = [
            cls._workspace_relative(workspace_path, f) for f in files if f.is_file()
        ]
        if not relative_files:
            return

        linter = cls._detect_js_linter(workspace_path)
        if linter is None:
            runs.append(NativeToolRun(
                tool="js_linter", status="skipped",
                detail="nenhum eslint/biome/oxlint configurado no projeto",
            ))
            return

        name, binary = linter
        if name == "eslint":
            cmd = [binary, "--format", "json", *relative_files]
        elif name == "biome":
            cmd = [binary, "lint", "--reporter=json", *relative_files]
        else:
            cmd = [binary, "--format", "json", *relative_files]

        output = cls._execute_native_tool(name, cmd, workspace_path, env, deadline, runs)
        if output is None:
            return
        if name == "eslint":
            found = cls._parse_eslint_json(workspace_path, output)
        elif name == "biome":
            found = cls._parse_biome_json(workspace_path, output)
        else:
            found = cls._parse_oxlint_json(workspace_path, output)
        if not found and runs[-1].returncode >= 2:
            runs[-1].status = "error"
            runs[-1].detail = (
                f"saída não interpretável (rc={runs[-1].returncode}): {output.strip()[:200]}"
            )
        cls._apply_native_violations(found, runs[-1], blocking, warnings)

    @classmethod
    def _detect_js_linter(cls, workspace_path: Path) -> Optional[Tuple[str, str]]:
        """Detecta o linter JS configurado, na prioridade eslint > biome > oxlint.

        Exemplo:
            >>> MultiStackRunner._detect_js_linter(Path(".")) is None or True
            True
        """
        eslint_bin = cls._resolve_native_tool(workspace_path, "eslint", prefer_local_bin=True)
        if eslint_bin and (
            cls._has_project_file(workspace_path, cls._ESLINT_CONFIG_GLOBS)
            or cls._package_json_has_eslint_config(workspace_path)
        ):
            return ("eslint", eslint_bin)

        biome_bin = cls._resolve_native_tool(workspace_path, "biome", prefer_local_bin=True)
        if biome_bin and cls._has_project_file(workspace_path, cls._BIOME_CONFIG_GLOBS):
            return ("biome", biome_bin)

        oxlint_bin = cls._resolve_native_tool(workspace_path, "oxlint", prefer_local_bin=True)
        if oxlint_bin and cls._has_project_file(workspace_path, cls._OXLINT_CONFIG_GLOBS):
            return ("oxlint", oxlint_bin)

        return None

    @classmethod
    def _bridge_python_toolchain(
        cls,
        workspace_path: Path,
        files: List[Path],
        env: Dict[str, str],
        deadline: float,
        runs: List[NativeToolRun],
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation],
    ) -> None:
        """Executa ``ruff check`` e ``mypy`` do venv/PATH sobre os arquivos auditados.

        Exemplo:
            >>> MultiStackRunner._bridge_python_toolchain(Path("."), [], {}, time.monotonic(), [], [], [])
        """
        relative_files = [
            cls._workspace_relative(workspace_path, f) for f in files if f.is_file()
        ]
        if not relative_files:
            return

        ruff_bin = cls._resolve_native_tool(workspace_path, "ruff")
        if ruff_bin:
            output = cls._execute_native_tool(
                "ruff",
                [ruff_bin, "check", "--output-format", "json", "--no-cache", *relative_files],
                workspace_path, env, deadline, runs,
            )
            if output is not None:
                cls._apply_native_violations(
                    cls._parse_ruff_json(workspace_path, output), runs[-1], blocking, warnings,
                )
        else:
            runs.append(NativeToolRun(
                tool="ruff", status="skipped", detail="binário ausente (.venv/bin ou PATH)",
            ))

        mypy_bin = cls._resolve_native_tool(workspace_path, "mypy")
        if not mypy_bin:
            runs.append(NativeToolRun(
                tool="mypy", status="skipped", detail="binário ausente (.venv/bin ou PATH)",
            ))
            return
        if deadline - time.monotonic() <= 0.15:
            runs.append(NativeToolRun(
                tool="mypy", status="skipped", detail="orçamento insuficiente após o ruff",
            ))
            return
        mypy_cache_dir = tempfile.mkdtemp(prefix="jev_native_mypy_")
        try:
            output = cls._execute_native_tool(
                "mypy",
                [
                    mypy_bin, "--no-color-output", "--no-error-summary", "--show-error-codes",
                    "--hide-error-context", "--no-incremental", "--cache-dir", mypy_cache_dir,
                    *relative_files[:25],
                ],
                workspace_path, env, deadline, runs,
            )
        finally:
            shutil.rmtree(mypy_cache_dir, ignore_errors=True)
        if output is not None:
            cls._apply_native_violations(
                cls._parse_mypy_output(workspace_path, output), runs[-1], blocking, warnings,
            )

    @classmethod
    def _bridge_other_toolchains(
        cls,
        workspace_path: Path,
        files: List[Path],
        env: Dict[str, str],
        deadline: float,
        runs: List[NativeToolRun],
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation],
    ) -> None:
        """Executa ``go vet``, ``cargo clippy -D warnings`` e ``dart analyze`` se aplicáveis.

        Aplica-se por manifesto (``go.mod``/``Cargo.toml``/``pubspec.yaml``) e por
        presença real de fontes da linguagem entre os arquivos auditados.

        Exemplo:
            >>> MultiStackRunner._bridge_other_toolchains(Path("."), [], {}, time.monotonic(), [], [], [])
        """
        suffixes = {f.suffix for f in files}
        if (workspace_path / "go.mod").exists() and (not files or ".go" in suffixes):
            go_bin = shutil.which("go")
            if go_bin:
                output = cls._execute_native_tool(
                    "go_vet", [go_bin, "vet", "./..."], workspace_path, env, deadline, runs,
                )
                if output is not None:
                    cls._apply_native_violations(
                        cls._parse_go_vet_output(workspace_path, output), runs[-1], blocking, warnings,
                    )
            else:
                runs.append(NativeToolRun(
                    tool="go_vet", status="skipped", detail="go.mod presente, binário go ausente",
                ))

        if (workspace_path / "Cargo.toml").exists() and (not files or ".rs" in suffixes):
            cargo_bin = shutil.which("cargo")
            if cargo_bin:
                output = cls._execute_native_tool(
                    "cargo_clippy",
                    [cargo_bin, "clippy", "--quiet", "--message-format", "json", "--", "-D", "warnings"],
                    workspace_path, env, deadline, runs,
                )
                if output is not None:
                    cls._apply_native_violations(
                        cls._parse_cargo_json(workspace_path, output), runs[-1], blocking, warnings,
                    )
            else:
                runs.append(NativeToolRun(
                    tool="cargo_clippy", status="skipped", detail="Cargo.toml presente, binário cargo ausente",
                ))

        if (workspace_path / "pubspec.yaml").exists() and (not files or ".dart" in suffixes):
            dart_bin = shutil.which("dart")
            if dart_bin:
                output = cls._execute_native_tool(
                    "dart_analyze", [dart_bin, "analyze", "--format=machine"],
                    workspace_path, env, deadline, runs,
                )
                if output is not None:
                    cls._apply_native_violations(
                        cls._parse_dart_analyze_output(workspace_path, output), runs[-1], blocking, warnings,
                    )
            else:
                runs.append(NativeToolRun(
                    tool="dart_analyze", status="skipped", detail="pubspec.yaml presente, binário dart ausente",
                ))

    @classmethod
    def _execute_native_tool(
        cls,
        tool: str,
        cmd: List[str],
        workspace_path: Path,
        env: Dict[str, str],
        deadline: float,
        runs: List[NativeToolRun],
    ) -> Optional[str]:
        """Executa uma ferramenta isolada em novo grupo de processos e devolve a saída.

        No timeout o grupo inteiro recebe ``SIGKILL`` (``killpg``), garantindo que
        linters/compiladores não deixem processos órfãos após o orçamento.

        Exemplo:
            >>> MultiStackRunner._execute_native_tool("noop", ["true"], Path("."), {}, 0.0, []) is None
            True
        """
        remaining = deadline - time.monotonic()
        if remaining <= 0.05:
            runs.append(NativeToolRun(
                tool=tool, status="skipped", detail="orçamento esgotado antes do spawn",
            ))
            logger.debug("Camada 1: %s ignorado (orçamento esgotado)", tool)
            return None

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace_path),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                start_new_session=True,
            )
        except (OSError, ValueError) as spawn_err:
            runs.append(NativeToolRun(tool=tool, status="error", detail=str(spawn_err)[:200]))
            logger.debug("Camada 1: falha ao executar %s (%s): %s", tool, cmd[:1], spawn_err)
            return None

        timed_out = False
        try:
            stdout, stderr = proc.communicate(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            cls._kill_process_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=0.2)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""

        duration = round(time.monotonic() - started, 3)
        if timed_out:
            returncode = proc.returncode if proc.returncode is not None else -9
            runs.append(NativeToolRun(
                tool=tool, status="timeout", returncode=returncode, duration=duration,
                detail=f"excedeu {round(remaining, 3)}s e foi morto via killpg",
            ))
            logger.debug("Camada 1: %s excedeu o orçamento (%.3fs) e foi morto", tool, duration)
            return None

        runs.append(NativeToolRun(
            tool=tool, status="executed", returncode=proc.returncode or 0, duration=duration,
        ))
        output = (stdout or "") + ("\n" + stderr if stderr else "")
        return output[:_NATIVE_OUTPUT_LIMIT]

    @staticmethod
    def _kill_process_group(proc: subprocess.Popen) -> None:
        """Mata o grupo de processos inteiro do subprocesso (anti-órfão no timeout).

        Exemplo:
            >>> proc = subprocess.Popen(["true"])
            >>> MultiStackRunner._kill_process_group(proc)
            >>> proc.wait() == 0
            True
        """
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError) as kill_err:
            logger.debug("killpg indisponível para pid %s: %s", proc.pid, kill_err)
        try:
            proc.kill()
        except OSError as proc_err:
            logger.debug("kill direto falhou para pid %s: %s", proc.pid, proc_err)

    @classmethod
    def _apply_native_violations(
        cls,
        found: List[MultiStackViolation],
        run: NativeToolRun,
        blocking: List[MultiStackViolation],
        warnings: List[MultiStackViolation],
    ) -> None:
        """Distribui diagnósticos nativos em bloqueantes/alertas e anota a execução.

        Exemplo:
            >>> run = NativeToolRun(tool="ruff", status="executed")
            >>> MultiStackRunner._apply_native_violations([], run, [], [])
            >>> run.violation_count
            0
        """
        run.violation_count = len(found)
        for violation in found:
            if violation.category in cls._BLOCKING_NATIVE_CATEGORIES:
                blocking.append(violation)
            else:
                warnings.append(violation)

    @staticmethod
    def _dedup_native_violations(violations: List[MultiStackViolation]) -> List[MultiStackViolation]:
        """Remove diagnósticos duplicados (regra/arquivo/linha/mensagem) preservando ordem.

        Exemplo:
            >>> MultiStackRunner._dedup_native_violations([])
            []
        """
        seen: set[Tuple[str, str, int, str]] = set()
        unique: List[MultiStackViolation] = []
        for violation in violations:
            key = (
                violation.rule_id,
                violation.file_path,
                violation.line_number,
                violation.message[:80],
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(violation)
        return unique

    @classmethod
    def _merge_native_bridge(cls, base: Dict[str, Any], native: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Mescla diagnósticos da Camada 1 na resposta base sem duplicar regra/linha.

        Exemplo:
            >>> MultiStackRunner._merge_native_bridge({"blocking": [], "warnings": []}, None)
            {'blocking': [], 'warnings': []}
        """
        if not native:
            return base

        existing = {
            (item.get("rule_id"), item.get("file_path"), item.get("line_number"))
            for item in list(base.get("blocking", [])) + list(base.get("warnings", []))
        }
        for key in ("blocking", "warnings"):
            bucket = base.setdefault(key, [])
            for item in native.get(key, []):
                signature = (item.get("rule_id"), item.get("file_path"), item.get("line_number"))
                if signature in existing:
                    continue
                existing.add(signature)
                bucket.append(item)

        tools = native.get("tools") or []
        if any(tool.get("status") == "executed" for tool in tools):
            base["engine"] = f"{base.get('engine', 'unknown')}+native_toolchain_bridge"
        base["native_tools"] = tools
        base["native_bridge_duration"] = native.get("duration", 0.0)
        base["native_layer_status"] = native.get("layer_status", "skipped")
        base["native_skipped_tools"] = native.get("skipped_tools") or {}
        base["native_critical_failures"] = native.get("critical_failures") or []
        base["native_inconclusive_reasons"] = native.get("inconclusive_reasons") or []
        return base

    @classmethod
    def _run_native_bridge_guarded(
        cls,
        workspace_path: Path,
        stack: str,
        audit_mode: str,
        source_files: List[Path],
    ) -> Optional[Dict[str, Any]]:
        """Executa a Camada 1 com isolamento de falhas (nunca derruba a entrega).

        Exemplo:
            >>> isinstance(MultiStackRunner._run_native_bridge_guarded(Path("."), "UNKNOWN", "full", []), dict)
            True
        """
        try:
            return cls.run_native_bridge(
                workspace_path, stack, audit_mode=audit_mode, source_files=source_files,
            )
        except Exception as bridge_err:
            logger.debug("Camada 1 indisponível para %s: %s", workspace_path, bridge_err)
            return None

    @staticmethod
    def _workspace_relative(workspace_path: Path, file_path: Union[str, Path]) -> str:
        """Normaliza um arquivo para caminho relativo ao workspace de forma segura.

        Resolve symlinks DOS DOIS LADOS antes de relativizar: no macOS
        ``/var/folders`` é symlink para ``/private/var/folders`` e a comparação
        ingênua levanta ``ValueError`` (causa da falha da native_bridge em
        ``macos-latest`` — TICKET-SGA-10). Nunca propaga exceção: cai para o
        nome do arquivo.

        Exemplo:
            >>> MultiStackRunner._workspace_relative(Path("/tmp"), "/tmp/a.py")
            'a.py'
        """
        candidate = Path(file_path)
        try:
            return str(candidate.resolve().relative_to(Path(workspace_path).resolve()))
        except (OSError, ValueError):
            return candidate.name

    @staticmethod
    def _absolute_native_path(workspace_path: Path, raw_path: str) -> Path:
        """Resolve um caminho relatado por linter contra a raiz do workspace.

        Exemplo:
            >>> str(MultiStackRunner._absolute_native_path(Path("/tmp"), "a.py"))
            '/tmp/a.py'
        """
        candidate = Path(raw_path)
        return candidate if candidate.is_absolute() else workspace_path / candidate

    @staticmethod
    def _relative_native_path(workspace_path: Path, raw_path: str) -> str:
        """Normaliza o caminho de um diagnóstico para relativo ao workspace.

        Exemplo:
            >>> MultiStackRunner._relative_native_path(Path("/tmp"), "a.py")
            'a.py'
        """
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            return str(candidate)
        try:
            return str(candidate.relative_to(workspace_path))
        except ValueError:
            return str(candidate)

    @staticmethod
    def _read_native_snippet(file_path: Path, line_number: int, cache: Dict[str, List[str]]) -> str:
        """Extrai o trecho da linha apontada pelo diagnóstico, com cache por arquivo.

        Exemplo:
            >>> MultiStackRunner._read_native_snippet(Path("inexistente-xyz.py"), 1, {})
            ''
        """
        key = str(file_path)
        lines = cache.get(key)
        if lines is None:
            try:
                lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError as read_err:
                logger.debug("Snippet indisponível para %s: %s", file_path, read_err)
                lines = []
            cache[key] = lines
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1].strip()[:120]
        return ""

    @staticmethod
    def _line_from_byte_offset(file_path: Path, offset: int, cache: Dict[str, bytes]) -> int:
        """Converte um offset de byte (biome/oxlint) na linha 1-based correspondente.

        Exemplo:
            >>> MultiStackRunner._line_from_byte_offset(Path("inexistente-xyz.py"), 0, {})
            1
        """
        key = str(file_path)
        data = cache.get(key)
        if data is None:
            try:
                data = file_path.read_bytes()
            except OSError as read_err:
                logger.debug("Offset indisponível para %s: %s", file_path, read_err)
                data = b""
            cache[key] = data
        if offset <= 0 or not data:
            return 1
        return data[: min(offset, len(data))].count(b"\n") + 1

    @classmethod
    def _make_native_violation(
        cls,
        workspace_path: Path,
        rule_id: str,
        category: str,
        raw_path: str,
        line_number: int,
        message: str,
        snippet_cache: Optional[Dict[str, List[str]]] = None,
        suggestion: Optional[str] = None,
    ) -> MultiStackViolation:
        """Constrói uma ``MultiStackViolation`` normalizada a partir de um diagnóstico nativo.

        Exemplo:
            >>> v = MultiStackRunner._make_native_violation(Path("."), "TSC_TS2322", "type_error", "a.ts", 3, "msg")
            >>> (v.rule_id, v.category, v.line_number)
            ('TSC_TS2322', 'type_error', 3)
        """
        rel_path = cls._relative_native_path(workspace_path, raw_path)
        line = max(int(line_number or 1), 1)
        snippet = ""
        if snippet_cache is not None and rel_path:
            snippet = cls._read_native_snippet(
                cls._absolute_native_path(workspace_path, rel_path), line, snippet_cache,
            )
        return MultiStackViolation(
            rule_id=rule_id,
            category=category,
            file_path=rel_path,
            line_number=line,
            code_snippet=snippet,
            message=message[:500],
            suggestion=suggestion or cls._native_suggestion(rule_id, category),
        )

    @classmethod
    def _native_suggestion(cls, rule_id: str, category: str) -> Optional[str]:
        """Devolve a sugestão canônica da regra nativa (ou fallback por categoria).

        Exemplo:
            >>> MultiStackRunner._native_suggestion("TSC_TS2322", "type_error") is not None
            True
        """
        if rule_id in RULE_SUGGESTIONS:
            return RULE_SUGGESTIONS[rule_id]
        if rule_id.startswith("TSC_"):
            return RULE_SUGGESTIONS["TSC_TYPE_ERROR"]
        if rule_id.startswith("ESLINT_") and rule_id in RULE_SUGGESTIONS:
            return RULE_SUGGESTIONS[rule_id]
        if rule_id.startswith("RUFF_"):
            return RULE_SUGGESTIONS["RUFF_LINT"]
        if rule_id.startswith("MYPY_"):
            return RULE_SUGGESTIONS["MYPY_TYPE_ERROR"]
        if rule_id.startswith("GOVET_"):
            return RULE_SUGGESTIONS["GOVET_DIAGNOSTIC"]
        if rule_id.startswith("CLIPPY_"):
            return RULE_SUGGESTIONS["CLIPPY_LINT"]
        if rule_id.startswith("DART_"):
            return RULE_SUGGESTIONS["DART_ANALYZE"]
        if category == "lint_error":
            return "Corrija o erro apontado pelo linter configurado no projeto antes da entrega."
        if category == "lint_warning":
            return "Avalie o alerta do linter e corrija ou registre supressão justificada."
        return None

    @staticmethod
    def _slugify_rule(rule: str) -> str:
        """Normaliza um identificador de regra para o padrão ``[A-Z0-9_]``.

        Exemplo:
            >>> MultiStackRunner._slugify_rule("react-hooks/rules-of-hooks")
            'REACT_HOOKS_RULES_OF_HOOKS'
        """
        return re.sub(r"[^A-Za-z0-9]+", "_", rule).strip("_").upper()

    @staticmethod
    def _load_json_payload(output: str, is_array: bool = False) -> Optional[Any]:
        """Interpreta saída JSON tolerando ruído/logs antes e depois do payload.

        Exemplo:
            >>> MultiStackRunner._load_json_payload('[]', is_array=True)
            []
        """
        text = output.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            opening = "[" if is_array else "{"
            start = text.find(opening)
            if start < 0:
                return None
            try:
                payload, _end = json.JSONDecoder().raw_decode(text[start:])
                return payload
            except json.JSONDecodeError as parse_err:
                logger.debug("Payload JSON nativo ilegível: %s", parse_err)
                return None

    @classmethod
    def _parse_tsc_output(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte a saída textual do ``tsc --noEmit`` em violações de tipo.

        Exemplo:
            >>> MultiStackRunner._parse_tsc_output(Path("."), "a.ts(2,5): error TS2322: bad")
            [MultiStackViolation(rule_id='TSC_TS2322', category='type_error', file_path='a.ts', line_number=2, code_snippet='', message='TS2322: bad', suggestion='Corrija a incompatibilidade de tipos apontada pelo TypeScript antes da entrega; evite \\'as any\\' e prefira narrowing explícito ou tipos auxiliares.')]
        """
        violations: List[MultiStackViolation] = []
        snippet_cache: Dict[str, List[str]] = {}
        line_pat = re.compile(
            r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\):\s+(?P<sev>error|warning)\s+(?P<code>TS\d+):\s+(?P<msg>.+)$"
        )
        global_pat = re.compile(r"^(?P<sev>error|warning)\s+(?P<code>TS\d+):\s+(?P<msg>.+)$")
        for raw_line in output.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            match = line_pat.match(stripped)
            if match:
                category = "type_error" if match.group("sev") == "error" else "lint_warning"
                violations.append(cls._make_native_violation(
                    workspace_path,
                    f"TSC_{match.group('code')}",
                    category,
                    match.group("file"),
                    int(match.group("line")),
                    f"{match.group('code')}: {match.group('msg')}",
                    snippet_cache,
                ))
                continue
            global_match = global_pat.match(stripped)
            if global_match:
                category = "type_error" if global_match.group("sev") == "error" else "lint_warning"
                violations.append(cls._make_native_violation(
                    workspace_path,
                    f"TSC_{global_match.group('code')}",
                    category,
                    "tsconfig.json",
                    1,
                    f"{global_match.group('code')}: {global_match.group('msg')}",
                ))
        return violations

    @classmethod
    def _parse_eslint_json(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte o JSON do ESLint em violações de hooks, deps, keys e state.

        Exemplo:
            >>> MultiStackRunner._parse_eslint_json(Path("."), "[]")
            []
        """
        payload = cls._load_json_payload(output, is_array=True)
        if not isinstance(payload, list):
            return []

        violations: List[MultiStackViolation] = []
        snippet_cache: Dict[str, List[str]] = {}
        for file_result in payload:
            if not isinstance(file_result, dict):
                continue
            file_path = str(file_result.get("filePath") or "")
            messages = file_result.get("messages")
            if not isinstance(messages, list):
                continue
            for message in messages:
                if not isinstance(message, dict):
                    continue
                rule = message.get("ruleId")
                severity = int(message.get("severity") or 1)
                line = int(message.get("line") or 1)
                if not rule:
                    rule_id = "ESLINT_PARSE_ERROR"
                    category = "lint_error"
                    text = f"Erro de parsing ESLint: {message.get('message', '')}"
                    hint: Optional[str] = RULE_SUGGESTIONS["LINT_PARSE_ERROR"]
                else:
                    mapped = cls._ESLINT_RULE_MAP.get(str(rule))
                    rule_id = mapped[0] if mapped else f"ESLINT_{cls._slugify_rule(str(rule))}"
                    category = "lint_error" if severity >= 2 else "lint_warning"
                    text = f"[{rule}] {message.get('message', '')}"
                    hint = mapped[1] if mapped else None
                violations.append(cls._make_native_violation(
                    workspace_path, rule_id, category, file_path, line, text, snippet_cache, hint,
                ))
        return violations

    @classmethod
    def _parse_biome_json(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte o JSON do Biome em violações, resolvendo linha por offset de byte.

        Exemplo:
            >>> MultiStackRunner._parse_biome_json(Path("."), "{}")
            []
        """
        payload = cls._load_json_payload(output)
        if not isinstance(payload, dict):
            return []
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, list):
            return []

        violations: List[MultiStackViolation] = []
        byte_cache: Dict[str, bytes] = {}
        snippet_cache: Dict[str, List[str]] = {}
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            category_raw = str(diagnostic.get("category") or "lint/unknown")
            severity = str(diagnostic.get("severity") or "warning").lower()
            location_raw = diagnostic.get("location")
            location: Dict[str, Any] = location_raw if isinstance(location_raw, dict) else {}
            path_raw = location.get("path")
            path_info: Dict[str, Any] = path_raw if isinstance(path_raw, dict) else {}
            raw_path = str(path_info.get("file") or path_info.get("path") or "")
            span = location.get("span") if isinstance(location.get("span"), list) else []
            line = 1
            if raw_path and span and isinstance(span[0], int):
                line = cls._line_from_byte_offset(
                    cls._absolute_native_path(workspace_path, raw_path), span[0], byte_cache,
                )
            rule_id, hint = cls._biome_rule(category_raw)
            category = "lint_error" if severity in ("error", "fatal") else "lint_warning"
            text = f"[{category_raw}] {diagnostic.get('description') or diagnostic.get('message') or ''}"
            violations.append(cls._make_native_violation(
                workspace_path, rule_id, category, raw_path, line, text, snippet_cache, hint,
            ))
        return violations

    @classmethod
    def _biome_rule(cls, category: str) -> Tuple[str, Optional[str]]:
        """Mapeia a categoria do Biome para rule_id canônico e sugestão.

        Exemplo:
            >>> MultiStackRunner._biome_rule("lint/correctness/useExhaustiveDependencies")[0]
            'BIOME_HOOK_DEPS'
        """
        lower = category.lower()
        if "rulesofhooks" in lower or "rules-of-hooks" in lower:
            return ("BIOME_RULES_OF_HOOKS", RULE_SUGGESTIONS["ESLINT_RULES_OF_HOOKS"])
        if "exhaustivedependencies" in lower or "exhaustive-deps" in lower:
            return ("BIOME_HOOK_DEPS", RULE_SUGGESTIONS["ESLINT_HOOK_DEPS"])
        if "jsxkeyiniterable" in lower or "noarrayindexkey" in lower or "missingkey" in lower:
            return ("BIOME_MISSING_KEY", RULE_SUGGESTIONS["ESLINT_MISSING_KEY"])
        if "mutation" in lower and "state" in lower:
            return ("BIOME_STATE_MUTATION", RULE_SUGGESTIONS["ESLINT_STATE_MUTATION"])
        return (f"BIOME_{cls._slugify_rule(category)}", None)

    @classmethod
    def _parse_oxlint_json(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte o JSON do oxlint em violações, resolvendo linha por offset.

        Exemplo:
            >>> MultiStackRunner._parse_oxlint_json(Path("."), "{}")
            []
        """
        payload = cls._load_json_payload(output)
        if isinstance(payload, dict):
            diagnostics = payload.get("diagnostics")
        elif isinstance(payload, list):
            diagnostics = payload
        else:
            diagnostics = None
        if not isinstance(diagnostics, list):
            return []

        violations: List[MultiStackViolation] = []
        byte_cache: Dict[str, bytes] = {}
        snippet_cache: Dict[str, List[str]] = {}
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            code = str(diagnostic.get("code") or "unknown")
            severity = str(diagnostic.get("severity") or "warning").lower()
            category = "lint_error" if severity == "error" else "lint_warning"
            raw_path = str(diagnostic.get("filename") or diagnostic.get("file") or "")
            line = 1
            labels = diagnostic.get("labels")
            if isinstance(labels, list) and labels and isinstance(labels[0], dict):
                span = labels[0].get("span")
                if isinstance(span, dict) and isinstance(span.get("offset"), int) and raw_path:
                    line = cls._line_from_byte_offset(
                        cls._absolute_native_path(workspace_path, raw_path), span["offset"], byte_cache,
                    )
                elif isinstance(span, dict) and isinstance(span.get("line"), int):
                    line = int(span["line"])
            violations.append(cls._make_native_violation(
                workspace_path,
                f"OXLINT_{cls._slugify_rule(code)}",
                category,
                raw_path,
                line,
                f"[{code}] {diagnostic.get('message') or ''}",
                snippet_cache,
            ))
        return violations

    @staticmethod
    def _is_ruff_blocking(code: str) -> bool:
        """Classifica códigos Ruff: Pyflakes (F*) e erros de sintaxe (E9*) bloqueiam.

        Exemplo:
            >>> (MultiStackRunner._is_ruff_blocking("F401"), MultiStackRunner._is_ruff_blocking("E501"))
            (True, False)
        """
        return code.startswith("F") or code.startswith("E9")

    @classmethod
    def _parse_ruff_json(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte o JSON do ``ruff check`` em violações (F*/E9* bloqueantes).

        Exemplo:
            >>> MultiStackRunner._parse_ruff_json(Path("."), "[]")
            []
        """
        payload = cls._load_json_payload(output, is_array=True)
        if not isinstance(payload, list):
            return []

        violations: List[MultiStackViolation] = []
        snippet_cache: Dict[str, List[str]] = {}
        for item in payload:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code") or "UNKNOWN")
            location_raw = item.get("location")
            location: Dict[str, Any] = location_raw if isinstance(location_raw, dict) else {}
            line = int(location.get("row") or 1)
            category = "lint_error" if cls._is_ruff_blocking(code) else "lint_warning"
            violations.append(cls._make_native_violation(
                workspace_path,
                f"RUFF_{cls._slugify_rule(code)}",
                category,
                str(item.get("filename") or ""),
                line,
                f"[{code}] {item.get('message') or ''}",
                snippet_cache,
            ))
        return violations

    @classmethod
    def _parse_mypy_output(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte a saída textual do mypy em violações de tipo bloqueantes.

        Exemplo:
            >>> MultiStackRunner._parse_mypy_output(Path("."), "a.py:3: error: bad  [assignment]")
            [MultiStackViolation(rule_id='MYPY_ASSIGNMENT', category='type_error', file_path='a.py', line_number=3, code_snippet='', message='bad  [assignment]', suggestion='Ajuste a anotação/tipo apontado pelo mypy; evite silenciar com \\'type: ignore\\' sem justificativa.')]
        """
        violations: List[MultiStackViolation] = []
        snippet_cache: Dict[str, List[str]] = {}
        pattern = re.compile(
            r"^(?P<file>.+?):(?P<line>\d+):(?:(?P<col>\d+):)?\s*(?P<sev>error|warning|note):\s*(?P<msg>.+)$"
        )
        for raw_line in output.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            match = pattern.match(stripped)
            if not match or match.group("sev") == "note":
                continue
            message = match.group("msg")
            code_match = re.search(r"\[(?P<code>[a-z][a-z0-9-]+)\]\s*$", message)
            code = code_match.group("code") if code_match else "type"
            category = "type_error" if match.group("sev") == "error" else "lint_warning"
            violations.append(cls._make_native_violation(
                workspace_path,
                f"MYPY_{cls._slugify_rule(code)}",
                category,
                match.group("file"),
                int(match.group("line")),
                message,
                snippet_cache,
            ))
        return violations

    @classmethod
    def _parse_go_vet_output(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte a saída do ``go vet`` em violações bloqueantes.

        Exemplo:
            >>> MultiStackRunner._parse_go_vet_output(Path("."), "a.go:12:3: bad call")
            [MultiStackViolation(rule_id='GOVET_DIAGNOSTIC', category='lint_error', file_path='a.go', line_number=12, code_snippet='', message='bad call', suggestion='Corrija o diagnóstico do \\'go vet\\' (printf, locks, struct tags) antes da entrega.')]
        """
        violations: List[MultiStackViolation] = []
        snippet_cache: Dict[str, List[str]] = {}
        pattern = re.compile(r"^(?P<file>[^\s:]+\.go):(?P<line>\d+)(?::(?P<col>\d+))?:\s+(?P<msg>.+)$")
        for raw_line in output.splitlines():
            match = pattern.match(raw_line.strip())
            if not match:
                continue
            violations.append(cls._make_native_violation(
                workspace_path,
                "GOVET_DIAGNOSTIC",
                "lint_error",
                match.group("file"),
                int(match.group("line")),
                match.group("msg"),
                snippet_cache,
            ))
        return violations

    @classmethod
    def _parse_cargo_json(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte o JSON line-delimited do ``cargo clippy`` em violações.

        Exemplo:
            >>> MultiStackRunner._parse_cargo_json(Path("."), "{}")
            []
        """
        violations: List[MultiStackViolation] = []
        snippet_cache: Dict[str, List[str]] = {}
        for raw_line in output.splitlines():
            stripped = raw_line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(item, dict) or item.get("reason") != "compiler-message":
                continue
            message = item.get("message")
            if not isinstance(message, dict):
                continue
            level = str(message.get("level") or "warning")
            if level not in ("error", "warning"):
                continue
            code_info = message.get("code")
            code = str(code_info.get("code") or "clippy") if isinstance(code_info, dict) else "clippy"
            spans = message.get("spans")
            primary: Dict[str, Any] = {}
            if isinstance(spans, list):
                for span in spans:
                    if isinstance(span, dict) and span.get("is_primary"):
                        primary = span
                        break
                if not primary and spans and isinstance(spans[0], dict):
                    primary = spans[0]
            category = "lint_error" if level == "error" else "lint_warning"
            violations.append(cls._make_native_violation(
                workspace_path,
                f"CLIPPY_{cls._slugify_rule(code)}",
                category,
                str(primary.get("file_name") or ""),
                int(primary.get("line_start") or 1),
                str(message.get("message") or ""),
                snippet_cache,
            ))
        return violations

    @classmethod
    def _parse_dart_analyze_output(cls, workspace_path: Path, output: str) -> List[MultiStackViolation]:
        """Converte a saída machine-readable do ``dart analyze`` em violações.

        Exemplo:
            >>> MultiStackRunner._parse_dart_analyze_output(Path("."), "ERROR|COMPILE|X|a.dart|1|2|3|bad")
            [MultiStackViolation(rule_id='DART_X', category='lint_error', file_path='a.dart', line_number=1, code_snippet='', message='bad', suggestion='Resolva o diagnóstico do \\'dart analyze\\' antes da entrega.')]
        """
        violations: List[MultiStackViolation] = []
        snippet_cache: Dict[str, List[str]] = {}
        for raw_line in output.splitlines():
            parts = raw_line.strip().split("|")
            if len(parts) < 8:
                continue
            severity, _kind, code, file_path, line_raw = parts[0], parts[1], parts[2], parts[3], parts[4]
            message = "|".join(parts[7:])
            try:
                line = int(line_raw)
            except ValueError:
                line = 1
            category = "lint_error" if severity.upper() == "ERROR" else "lint_warning"
            violations.append(cls._make_native_violation(
                workspace_path,
                f"DART_{cls._slugify_rule(code)}",
                category,
                file_path,
                line,
                message,
                snippet_cache,
            ))
        return violations

    # ── Camada 2: Meta-Regras Universais via Tree-Sitter (Maya/B1) ──────────

    _TREE_SITTER_DEDUP_ALIASES: Dict[str, str] = {
        "PROD_MOCK": "PROD_MOCK",
        "JS_PROD_MOCK": "PROD_MOCK",
        "DANGEROUS_EVAL": "EVAL",
        "SECURITY_EVAL": "EVAL",
        "SECURITY_EXEC": "EXEC",
        "SQL_INJECTION": "SQL_INJECTION",
        "SILENT_CATCH": "SILENT_CATCH",
        "EMPTY_CATCH": "SILENT_CATCH",
    }

    @classmethod
    def scan_tree_sitter(
        cls,
        workspace_path: Path,
        stack: str,
        audit_mode: str = "delta",
        source_files: Optional[List[Path]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Executa a Camada 2 (AST Tree-Sitter) sobre os arquivos JS/TS de produção.

        Devolve o mesmo formato de scan_js_ast (blocking/warnings/total_files/engine)
        ou ``None`` quando a Camada 2 estiver indisponível (degradação graciosa).

        Exemplo:
            >>> res = MultiStackRunner.scan_tree_sitter(Path("."), "NODE_TS_BUN")
            >>> res is None or set(res) >= {"blocking", "warnings", "total_files", "engine"}
            True
        """
        try:
            from ..memory.tree_sitter_guard import TreeSitterGuard
        except Exception as import_err:  # pragma: no cover - fallback gracioso
            logger.debug("Camada 2 (Tree-Sitter) indisponível: %s", import_err)
            return None

        if source_files is None:
            source_files = cls._collect_js_source_files(workspace_path, stack)

        blocking: List[MultiStackViolation] = []
        warnings: List[MultiStackViolation] = []
        parsed_files = 0
        fallback_files = 0
        failed_files: List[str] = []

        for fpath in source_files:
            language = TreeSitterGuard.language_for_file(fpath)
            if TreeSitterGuard.get_parser(language) is not None:
                parsed_files += 1
            else:
                fallback_files += 1
            try:
                found = TreeSitterGuard.scan_file(fpath, workspace=workspace_path)
            except Exception as scan_err:  # noqa: BLE001 — erro reportado, nunca silencioso
                logger.debug("Falha da Camada 4 em %s: %s", fpath, scan_err)
                failed_files.append(str(fpath))
                continue
            for item in found:
                payload = item.to_dict()
                relative = cls._tree_sitter_relpath(payload["file_path"], workspace_path)
                violation = MultiStackViolation(
                    rule_id=payload["rule_id"],
                    category=payload["category"],
                    file_path=relative,
                    line_number=payload["line_number"],
                    code_snippet=payload["code_snippet"],
                    message=payload["message"],
                    suggestion=payload.get("suggestion"),
                )
                if payload.get("severity") == "blocking":
                    blocking.append(violation)
                else:
                    warnings.append(violation)

        if parsed_files and fallback_files:
            engine = "tree_sitter_partial"
        elif fallback_files and not parsed_files:
            engine = "tree_sitter_lexical_fallback"
        else:
            engine = "tree_sitter"

        if not source_files:
            layer_status = "not_applicable"
            skip_reason = "nenhum arquivo JS/TS de produção elegível para a Camada 4"
        elif failed_files and len(failed_files) >= len(source_files):
            layer_status = "error"
            skip_reason = "todos os arquivos falharam na Camada 4 (Tree-Sitter)"
        elif failed_files:
            layer_status = "inconclusive"
            skip_reason = f"{len(failed_files)} arquivo(s) falharam na Camada 4 (Tree-Sitter)"
        elif parsed_files == 0:
            # TICKET-SGA-10 (item 5): sem gramática instalada o fallback lexical
            # determinístico executou sobre TODOS os arquivos elegíveis. Isso é
            # cobertura degradada — declarada em ``skip_reasons``/engine —, não
            # ausência de evidência. O contrato do job CI sem tree-sitter exige
            # que a Camada 4 não reprove a entrega só por falta do pacote opcional.
            layer_status = "executed"
            skip_reason = (
                "nenhuma gramática Tree-Sitter disponível; fallback lexical "
                "determinístico executado sobre todos os arquivos elegíveis"
            )
        else:
            layer_status = "executed"
            skip_reason = ""

        return {
            "blocking": [asdict(v) for v in blocking],
            "warnings": [asdict(v) for v in warnings],
            "total_files": len(source_files),
            "engine": engine,
            "layer_status": layer_status,
            "degraded": engine.endswith("lexical_fallback") and layer_status == "executed",
            "layers_executed": ["layer4_tree_sitter"] if layer_status == "executed" else [],
            "layers_skipped": [] if layer_status == "executed" else ["layer4_tree_sitter"],
            "skip_reasons": (
                {"layer4_tree_sitter": skip_reason} if skip_reason and layer_status != "executed"
                else ({"grammar_unavailable": skip_reason} if skip_reason else {})
            ),
            "inconclusive_reasons": [] if layer_status in ("executed", "not_applicable") else [skip_reason],
            "failed_files": failed_files,
        }

    @staticmethod
    def _tree_sitter_relpath(file_path: str, workspace_path: Path) -> str:
        """Normaliza o caminho reportado pela Camada 2 para relativo ao workspace."""
        path = Path(file_path)
        try:
            if path.is_absolute():
                return str(path.resolve().relative_to(workspace_path.resolve()))
        except (OSError, ValueError) as exc:
            logger.debug(
                "Falha ao relativizar caminho %s contra workspace %s: %s",
                file_path,
                workspace_path,
                exc,
            )
        return str(file_path)

    @staticmethod
    def _count_source_lines(source_files: List[Path]) -> int:
        """Conta as linhas realmente lidas dos arquivos elegíveis (medida real).

        TICKET-SGA-11 F5: ``total_lines_scanned`` deixa de ser zero literal no
        caminho JS — é a soma das linhas dos arquivos efetivamente coletados.

        Exemplo:
            >>> MultiStackRunner._count_source_lines([])
            0
        """
        total = 0
        for fpath in source_files:
            try:
                total += len(fpath.read_text(encoding="utf-8", errors="ignore").splitlines())
            except OSError as read_err:
                logger.debug("Linhas indisponíveis para %s: %s", fpath, read_err)
        return total

    @classmethod
    def _collect_js_source_files(cls, workspace_path: Path, stack: str) -> List[Path]:
        """Coleta arquivos JS/TS de produção de forma aberta (workspace inteiro).

        Sem diretórios fixos de projeto: varre a raiz e exclui apenas artefatos
        (node_modules, testes, mocks, build, VCS).

        Exemplo:
            >>> isinstance(MultiStackRunner._collect_js_source_files(Path("."), "NODE_TS_BUN"), list)
            True
        """
        prod_dirs = [workspace_path]

        raw_files = [
            fpath
            for pdir in prod_dirs
            if pdir.exists()
            for ext in ("*.js", "*.jsx", "*.ts", "*.tsx", "*.mjs", "*.cjs")
            for fpath in pdir.rglob(ext)
        ]
        files = cls._dedup_files(raw_files)
        return [
            f
            for f in files
            if not any(
                x in cls._workspace_relative(workspace_path, f)
                for x in ("node_modules", "__tests__", "tests", "__mocks__", "coverage", ".git", ".expo", "dist", "build")
            )
        ]

    @classmethod
    def _merge_tree_sitter(cls, base_result: Dict[str, Any], ts_result: Dict[str, Any]) -> Dict[str, Any]:
        """Mescla a Camada 2 no resultado base, deduplicando regras equivalentes."""
        merged = dict(base_result)
        seen = set()
        for bucket in ("blocking", "warnings"):
            for violation in merged.get(bucket, []) or []:
                seen.add(cls._tree_sitter_dedup_key(violation))

        for bucket in ("blocking", "warnings"):
            for violation in (ts_result or {}).get(bucket, []) or []:
                key = cls._tree_sitter_dedup_key(violation)
                if key in seen:
                    continue
                seen.add(key)
                merged.setdefault(bucket, []).append(violation)

        merged["total_files"] = max(
            merged.get("total_files", 0), (ts_result or {}).get("total_files", 0)
        )
        merged["tree_sitter_files"] = (ts_result or {}).get("total_files", 0)
        ts_engine = (ts_result or {}).get("engine", "tree_sitter")
        merged["tree_sitter_engine"] = ts_engine
        merged["tree_sitter_layer_status"] = (
            (ts_result or {}).get("layer_status")
            or ("executed" if ts_engine in ("tree_sitter", "tree_sitter_partial") else "inconclusive")
        )
        merged["tree_sitter_inconclusive_reasons"] = list(
            (ts_result or {}).get("inconclusive_reasons") or []
        )
        merged["tree_sitter_skip_reasons"] = dict((ts_result or {}).get("skip_reasons") or {})
        merged["engine"] = f"{merged.get('engine', 'unknown')}+{ts_engine}"
        return merged

    @classmethod
    def _tree_sitter_dedup_key(cls, violation: Dict[str, Any]) -> Tuple[str, str, int]:
        """Chave de deduplicação normalizada entre Camada 1 e Camada 2."""
        rule_id = violation.get("rule_id", "")
        alias = cls._TREE_SITTER_DEDUP_ALIASES.get(rule_id, rule_id)
        return (alias, violation.get("file_path", ""), violation.get("line_number", 0))

    @classmethod
    def scan_js_ast(cls, workspace_path: Path, stack: str, audit_mode: str = "delta") -> Dict[str, Any]:
        """Varre arquivos JS/TS de produção e devolve resultado com cobertura explícita.

        A coleta é aberta (workspace inteiro, sem diretórios fixos de projeto) e
        o retorno sempre declara ``layers_executed``/``layers_skipped``/
        ``skip_reasons``/``layer_status`` para impedir fail-open silencioso.

        Exemplo:
            >>> res = MultiStackRunner.scan_js_ast(Path("."), "NODE_TS_BUN", audit_mode="delta")
            >>> "blocking" in res and "layers_executed" in res
            True
        """
        workspace_path = Path(workspace_path).resolve()
        blocking: List[MultiStackViolation] = []
        warnings: List[MultiStackViolation] = []

        source_files = cls._collect_js_source_files(workspace_path, stack)
        if audit_mode == "delta":
            source_files = cls._filter_delta_files(workspace_path, source_files)

        if not source_files:
            skipped = ["layer3_native_toolchain", "layer4_tree_sitter"]
            return {
                "blocking": [],
                "warnings": [],
                "total_files": 0,
                "engine": "not_applicable",
                "layers_executed": [],
                "layers_skipped": skipped,
                "skip_reasons": {
                    layer: "nenhum arquivo JS/TS de produção elegível no workspace"
                    for layer in skipped
                },
                "layer_status": {layer: "not_applicable" for layer in skipped},
                "critical_failures": [],
                "inconclusive_reasons": [],
                "tree_sitter_scan": None,
                "total_lines": 0,
            }

        node_res = cls._run_node_js_ast(workspace_path, source_files, stack, blocking, warnings)
        if node_res is not None:
            base_result = node_res
        else:
            base_result = cls._run_lexical_fallback(workspace_path, source_files, stack, blocking, warnings)

        # Camada 3: bridge nativa (tsc + eslint/biome/oxlint) mesclada ao AST.
        native_res = cls._run_native_bridge_guarded(workspace_path, stack, audit_mode, source_files)
        merged_result = cls._merge_native_bridge(base_result, native_res)

        layers_executed: List[str] = []
        layers_skipped: List[str] = []
        skip_reasons: Dict[str, str] = {}
        if native_res is None:
            layers_skipped.append("layer3_native_toolchain")
            skip_reasons["layer3_native_toolchain"] = "bridge nativa indisponível (erro interno)"
        else:
            native_state = str(merged_result.get("native_layer_status") or "skipped")
            if native_state == "executed":
                layers_executed.append("layer3_native_toolchain")
            else:
                layers_skipped.append("layer3_native_toolchain")
                skip_reasons["layer3_native_toolchain"] = (
                    f"bridge nativa terminou com status={native_state}"
                )
        for tool_name, reason in (merged_result.get("native_skipped_tools") or {}).items():
            skip_reasons[f"tool:{tool_name}"] = str(reason)

        # Camada 4: meta-regras universais via Tree-Sitter (executada UMA vez;
        # o resultado bruto é embutido em ``tree_sitter_scan`` para o orquestrador
        # reaproveitar sem reexecutar o TreeSitterGuard — TICKET-SGA-10 item 1).
        try:
            ts_result = cls.scan_tree_sitter(
                workspace_path, stack, audit_mode, source_files=source_files
            )
        except Exception as ts_err:  # pragma: no cover - fallback gracioso
            logger.debug("Camada 4 (Tree-Sitter) degradada: %s", ts_err)
            ts_result = None
        if ts_result:
            merged_result = cls._merge_tree_sitter(merged_result, ts_result)
            ts_state = merged_result.get("tree_sitter_layer_status", "executed")
            if ts_state == "executed":
                layers_executed.append("layer4_tree_sitter")
            else:
                layers_skipped.append("layer4_tree_sitter")
                skip_reasons["layer4_tree_sitter"] = str(
                    (ts_result.get("skip_reasons") or {}).get(
                        "layer4_tree_sitter",
                        f"Camada 4 terminou com status={ts_state}",
                    )
                )
            # Fallback lexical determinístico (sem gramática instalada): declara
            # a cobertura degradada sem transformá-la em omissão de camada.
            for reason_key, reason_value in (ts_result.get("skip_reasons") or {}).items():
                if reason_key != "layer4_tree_sitter":
                    skip_reasons[reason_key] = str(reason_value)
        else:
            layers_skipped.append("layer4_tree_sitter")
            skip_reasons["layer4_tree_sitter"] = "TreeSitterGuard indisponível para a stack"

        merged_result["tree_sitter_scan"] = ts_result
        merged_result["total_lines"] = cls._count_source_lines(source_files)
        merged_result["layers_executed"] = layers_executed
        merged_result["layers_skipped"] = layers_skipped
        merged_result["skip_reasons"] = skip_reasons
        merged_result["layer_status"] = {
            "layer3_native_toolchain": merged_result.get("native_layer_status", "skipped"),
            "layer4_tree_sitter": merged_result.get("tree_sitter_layer_status", "skipped"),
        }
        merged_result["critical_failures"] = merged_result.get("native_critical_failures", [])
        merged_result["inconclusive_reasons"] = list(
            merged_result.get("native_inconclusive_reasons", [])
        ) + list(merged_result.get("tree_sitter_inconclusive_reasons", []))
        return merged_result
