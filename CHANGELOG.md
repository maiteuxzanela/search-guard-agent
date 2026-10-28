# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-10-27

Patch release consolidando os refinamentos pós-auditoria Diana (Portão 2) e melhorias de portabilidade da matriz CI (TICKET-SGA-10).

### Fixed
- Eliminação de dupla execução do TreeSitterGuard no pipeline de verificação JS/TS.
- Simetria estrita de inconclusive_reasons e veredito entre os ramos JS/TS e Python.
- Validação de suíte Go exigindo ok_packages > 0 para evitar falso positivo.
- Correção da contagem de total_files_scanned durante o fail-fast em Python.
- Resiliência e integridade da execução no ambiente sem Tree-Sitter instalado.
- Correção da detecção da bridge nativa de toolchains no macOS (macos-latest).

## [0.2.0] - 2026-10-22

Major contract and execution-integrity upgrade (`TICKET-SGA-09`) plus the
unified package, sandbox and exclusion capabilities built since `0.1.1`.

### Added
- **Explicit Verdict & Applicability Model (`VerdictState`):**
  `agent_quality_guard` now emits `verdict_state` (`VERIFIED_PASS`,
  `VERIFIED_FAIL`, `NOT_APPLICABLE`, `INCONCLUSIVE`, `EMERGENCY_RELEASE`),
  `evidence_executed` (`bool`), `layers_executed`, `layers_skipped`,
  `skip_reasons`, `layer_status`, and `inconclusive_reasons`. No execution path
  can produce `passed: true` without executed verification evidence (except the
  explicitly labeled 5th-attempt `EMERGENCY_RELEASE`).
- **Capability-based polyglot & monorepo test discovery:** detects project
  stacks and workspace packages directly from manifests (`pyproject.toml`,
  `package.json`, `tsconfig.json`, `bun.lock`/`bun.lockb`, `Cargo.toml`,
  `go.mod`, `pubspec.yaml`), including package-by-package test execution on
  monorepos with root test guards (`do not run tests from root`).
- `.cirurgicignore` support: workspace-root exclusion file (loaded only if
  present) with a gitignore-subset syntax — one pattern per line, `#` for
  full-line comments, blank lines ignored, trailing `\r` tolerated. A bare
  name (e.g. `screenshots`) matches at any level as file or directory; a
  pattern containing `/` is anchored to the workspace root; a trailing `/`
  marks directory-only. Negation (`!`) is not supported (such lines are
  ignored). Patterns are summed with the universal builtins and per-call
  `exclude_patterns`.
- `exclude_patterns` parameter (`Optional[List[str]]`, a single string is also
  accepted) on the search tools `cirurgic_search` and `qmd_search` (same
  signature): per-call, gitignore-style exclusions applied identically on the
  lexical (ripgrep/grep) and AST legs of the hybrid engine. Not a parameter of
  `agent_quality_guard`.
- `SECURITY.md` with a responsible disclosure policy, supported-version matrix
  and sandbox best practices for the community.
- `CONTRIBUTING.md` with local development setup, Conventional Commits
  guidance, the real-deterministic-test (no-mocks) requirement and the
  `agent_quality_guard` quality verification workflow.

### Changed
- **Eliminated project-fingerprint detection and fabricated test counters:**
  removed all hardcoded repository names from `MultiStackRunner.detect_stack()`
  and purged synthetic test-count fallbacks (`336`, `514`, `+= 5`).
- **Elevated toolchain & test execution timeout to 600 seconds:** replaced the
  legacy `0.8s`/`5.0s` budget caps with a `600s` budget across native compilers,
  linters and test suites.
- **Eliminated silent fail-open across Layers 0–4:** missing binaries, timeouts
  and fail-fast short-circuits now report explicit `INCONCLUSIVE` or
  `not_executed` layer status instead of returning empty violation lists
  indistinguishable from clean scans.
- Package identity unified as `search-guard-agent` for both the NPM package and
  the Python distribution. The NPM `bin` map ships `search-guard-agent` and the
  backward-compatible `agent-quality-guard` alias; the historical
  `agent_quality_guard` (underscore) alias was retired. Repository, homepage,
  issue tracker and security-advisory URLs now point to
  `github.com/maiteuxzanela/search-guard-agent`. Historical JEV documentation
  was archived under `docs/archive/`.
- `depth` on `cirurgic_search`/`qmd_search` now defaults to `None`
  (unlimited): the value caps the directory-tree walk depth (ripgrep
  `--max-depth` / walk cutoff) and, when applicable, the AST relational depth
  (`relational_depth`, valid range 0–5). Explicit callers remain 100%
  backward compatible.
- `CHARS_PER_TOKEN = 3.8` is now a named constant defined once in
  `agent_quality_guard/core/schemas.py` (single source of truth) and imported
  by the search and needle engines. Documented as a high-fidelity heuristic
  approximation for fast snippet-budget calculation — not a strict guarantee
  of exact token count.
- README rewritten around the unified **search-guard-agent** identity:
  canonical tools (`cirurgic_search`, `agent_quality_guard`), configurable
  exclusions and the token-budget heuristic are now described explicitly.

### Deprecated
- The standalone NPM packages `cirurgic-search-mcp` and `agent-quality-guard`
  are deprecated in favor of the unified `search-guard-agent` package. Existing
  installs keep working, but no new features will land under the old names.
- The search alias `qmd_search` is deprecated in favor of the canonical
  `cirurgic_search`. It remains a fully compatible alias routing to the same
  implementation; callers should migrate at their convenience.
- The delivery-verification alias `needle_verify_delivery` remains available
  for backward compatibility; new integrations should call
  `agent_quality_guard` directly.

## [0.1.1] - 2026-10-20

Patch release consolidating the CI portability and repository-hygiene fixes
shipped after `0.1.0`, with strict version parity (`0.1.1`) across
`package.json`, `pyproject.toml` and `agent_quality_guard/__init__.py`.

### Fixed
- Cross-platform `ripgrep` provisioning in `.github/workflows/ci.yml`: the
  `test` and `test-without-tree-sitter` jobs now install the `rg` binary on
  both Linux (`sudo apt-get install -y ripgrep`) and macOS
  (`brew install ripgrep`), removing the platform-dependent failure of the
  lexical search lane and of the version-parity step (TICKET-CI-01,
  TICKET-CI-02).
- Python 3.10 support: `tomli>=1.1.0; python_version < "3.11"` is declared as
  a dependency, and both the release-integrity tests and the `version parity`
  heredoc scripts in `ci.yml`/`publish.yml` fall back to
  `try: import tomllib / except ModuleNotFoundError: import tomli as tomllib`
  (TICKET-CI-01).
- `.gitignore` sanitation: `docs/` is no longer globally ignored so
  `docs/archive/` is versioned, and the explicit negation
  `!docs/archive/*_audit_report.md` re-includes the legacy audit report that
  the generic `*_audit_report.md` pattern used to swallow — verified with
  `git check-ignore docs/archive/projeto_jev_audit_report.md` returning no hit
  (TICKET-CI-03).
- TAP reporter pinned as `node --test --test-reporter=tap` plus a parser that
  accepts both classic (`# pass`) and modern (`✔ pass`) Node output, so the
  Node/Bun lane reports deterministically on Node v18–v22+ (TICKET-CI-02).

### Changed
- Version parity bumped to `0.1.1` in `package.json`, `pyproject.toml` and
  `agent_quality_guard/__init__.py`; the canonical manifest assertion in
  `tests/test_sga06_identity_sandbox_hardening.py` now expects `0.1.1`, and
  `test_real_repository_versions_are_in_parity` keeps enforcing equality
  across all three files.

## [0.1.0] - 2026-10-20

First unified release of the deterministic code-quality and surgical-search
MCP server, consolidating the former `cirurgic-search-mcp` and
`agent-quality-guard` codebases into a single project.

### Added
- `cirurgic_search` (canonical): hybrid code search combining exact lexical
  matching (Ripgrep) with semantic/AST extraction (Needle 3 / Tree-Sitter) and
  Reciprocal Rank Fusion (RRF) ranking, running 100% on CPU with no ML/GPU
  dependency. Parameters: `query` (required), `workspace`, `scope`,
  `max_tokens` (default 1200), `depth` (default 2).
- `agent_quality_guard` (canonical): fully deterministic delivery-verification
  pipeline executed fail-fast in order:
  1. real deterministic test suites (Pytest in sequential chunks, Node/Bun,
     Flutter/Dart, Cargo/Go/C# native runners);
  2. NeedleASTGuard Layer 1 — anti-mock in production, SQL injection,
     dangerous `eval`/`exec`, hardcoded secrets, `verify=False`;
  3. native compiler/linter bridge (`tsc`, `eslint`, `biome`, `ruff`, `mypy`,
     `cargo clippy`, `go vet`);
  4. universal Tree-Sitter AST meta-rules (floating promises, silent catch,
     resource leaks).
- Universal 5-attempt circuit breaker with deterministic `EMERGENCY_RELEASE`
  on the fifth attempt to eliminate agent retry deadlocks.
- Backward-compatible aliases `qmd_search` and `needle_verify_delivery`
  routing to the exact same implementations as their canonical counterparts.
- Multi-language AST extraction via Tree-Sitter (Python, TypeScript/TSX,
  JavaScript, Go, Rust, Dart) with graceful lexical fallback.
- MCP server over standard `stdio`, plus entry points for NPM
  (`agent-quality-guard` bin) and Python (`agent-quality-guard-mcp` console
  script).

### Changed
- Packaging synchronized across imports, test suite, CI/publish workflows and
  LICENSE (TICKET-AQG-02).
- Dead code and references to the retired Laya/RLCD probabilistic decision
  engine purged; the repository is 100% deterministic with no local neural
  inference path.

### Removed
- Legacy probabilistic primitives (`choice`, `noul`, `consulta`) and the
  `needle3_search` tool definition (TICKET-HARNESS-47).
- Torch/Transformers dependencies and the obsolete `openjev_score_and_diagnose`
  tool from the ecosystem.

[Unreleased]: https://github.com/maiteuxzanela/search-guard-agent/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/maiteuxzanela/search-guard-agent/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/maiteuxzanela/search-guard-agent/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/maiteuxzanela/search-guard-agent/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/maiteuxzanela/search-guard-agent/releases/tag/v0.1.0
