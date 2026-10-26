"""Contratos de dados Pydantic estritos para as 3 camadas do sistema JEV:
1. Memória & Busca (Needle 3)
2. Decisão & Roteamento (Open Jev: Noul, Choice, Score)
3. Execução & Orquestração
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# TICKET-SGA-05 — Heurística de orçamento de snippets.
# `CHARS_PER_TOKEN = 3.8` é uma APROXIMAÇÃO heurística de alta fidelidade para
# cálculo orçamentário rápido de trechos (mistura de código/inglês), usada para
# estimar `total_tokens`, o teto `max_tokens` e o `token_savings_pct`.
# NÃO é — e não pretende ser — uma garantia criptográfica estrita de contagem
# de tokens: a tokenização real varia por modelo/provedor. Constante única
# (fonte canônica) importada por `memory/qmd_search.py` e `memory/local_needle.py`.
CHARS_PER_TOKEN: float = 3.8


# ==============================================================================
# Exceções Tipadas Estritas (Fail-Fast Policy - Sem Mocks/Simulações)
# ==============================================================================

class OpenJevError(Exception):
    """Exceção base para falhas do subsistema Open Jev."""
    pass


class OpenJevNotReadyError(OpenJevError):
    """Lançada quando a engine/modelo não foi inicializada ou os pesos não estão carregados."""
    pass


class HardwareUnavailableError(OpenJevError):
    """Lançada quando a GPU/CPU requisitada não atende aos requisitos ou está inacessível."""
    pass


class InferenceError(OpenJevError):
    """Lançada quando a inferência do tensor falha durante a execução matemática."""
    pass


class Needle3Error(Exception):
    """Exceção base para falhas no motor de recuperação Needle 3."""
    pass


class Needle3NotAvailableError(Needle3Error):
    """Lançada quando o serviço ou API Cactus Needle 3 não está alcançável."""
    pass


# ==============================================================================
# Schemas Open Jev: Primitiva Noul (Avaliação Booleana / Guardrail)
# ==============================================================================

class NoulRequest(BaseModel):
    """Requisição para avaliação de condição lógica booleana."""
    premise: str = Field(
        ...,
        description="Contexto fático, estado atual ou comando proposto para execução."
    )
    hypothesis: str = Field(
        ...,
        description="Condição de segurança, regra de negócio ou predicado a validar (deve ser verdadeiro/falso)."
    )
    threshold: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description="Limiar de probabilidade mínima para considerar a condição verdadeira (guardrail calibrado 0.78)."
    )


class NoulResponse(BaseModel):
    """Resposta tipada da primitiva Noul."""
    passed: bool = Field(
        ...,
        description="Indica se a probabilidade calculada foi maior ou igual ao threshold estipulado."
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade calibrada P(hipótese é verdadeira dado o contexto)."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Métrica de certeza da decisão (|probabilidade - 0.5| * 2)."
    )
    threshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Limiar de corte utilizado nesta avaliação."
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo de latência da inferência em milissegundos."
    )


# ==============================================================================
# Schemas Open Jev: Primitiva Choice (Roteamento Categórico de Especialistas)
# ==============================================================================

class ChoiceRequest(BaseModel):
    """Requisição para seleção de alternativa em espaço discreto de opções."""
    premise: str = Field(
        ...,
        description="Descrição da tarefa, objetivo ou contexto a ser categorizado."
    )
    options: List[str] = Field(
        ...,
        min_length=2,
        description="Lista de categorias ou subagentes disponíveis (ex.: backend, security, frontend, docs)."
    )
    instructions: Optional[str] = Field(
        default=None,
        description=(
            "Instrução específica do choice. Se None, usa o padrão de roteamento "
            "('Which specialist agent should handle this task?')."
        ),
    )
    criteria: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Mapa opção -> descrição descritiva servida ao modelo como critérios. "
            "Se None, usa o próprio nome da opção como critério (comportamento legado)."
        ),
    )
    temperature: float = Field(
        default=1.0,
        gt=0.0,
        le=5.0,
        description="Temperatura para calibrar a distribuição softmax das probabilidades."
    )

    @model_validator(mode="after")
    def validate_unique_options(self):
        if len(self.options) != len(set(self.options)):
            raise ValueError("As opções de escolha devem ser mutuamente exclusivas e sem duplicatas.")
        return self


class ChoiceResponse(BaseModel):
    """Resposta tipada da primitiva Choice."""
    selected: str = Field(
        ...,
        description="Opção que obteve a maior probabilidade discriminativa."
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Distribuição completa de probabilidades sobre todas as opções fornecidas (soma ~ 1.0)."
    )
    top_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade da opção vencedora."
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo de inferência em milissegundos."
    )


# ==============================================================================
# Schemas Open Jev: Consulta Direta (Indício Textual — sem booleano de aprovação)
# ==============================================================================

class ConsultaRequest(BaseModel):
    """Requisição de consulta direta (ex.: 'esse script funciona?').

    A resposta é um INDÍCIO, não um veredito. Reformular a premissa para
    arrancar uma resposta pré-definida é premise gaming e está proibido
    pela governança (AGENTS.md).
    """
    model_config = ConfigDict(extra="forbid")

    premise: str = Field(
        ...,
        min_length=1,
        description=(
            "Evidência objetiva: trecho do script, schema do banco, amostra de dados "
            "ou estado real observado."
        ),
    )
    question: str = Field(
        ...,
        min_length=1,
        description="Pergunta direta a ser respondida como indício textual.",
    )


class ConsultaResponse(BaseModel):
    """Resposta tipada da consulta direta: indício textual SEM campo `passed`.

    `extra='forbid'` garante contrato rígido — nunca haverá booleano de
    aprovação que convide o agente a praticar premise gaming.
    """
    model_config = ConfigDict(extra="forbid")

    indicio: str = Field(
        ...,
        min_length=1,
        description=(
            "Indício textual derivado da probabilidade (ex.: 'Indício forte a favor...', "
            "'Indício fraco contra...'). Não é veredito nem aprovação."
        ),
    )
    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade calibrada da questão ser verdadeira dado o contexto.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiança da avaliação (|probabilidade - 0.5| * 2).",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo de latência da inferência em milissegundos.",
    )


# ==============================================================================
# Schemas Open Jev: Primitiva Score (Avaliação e Calibração Ordinal de Entrega)
# ==============================================================================

# Catálogo FECHADO de rubricas — única fonte de verdade (anti rubric gaming).
# O MCP resolve por `task_type`; qualquer `rubric` livre do cliente é ignorada.
RUBRIC_CATALOG: Dict[str, str] = {
    "codigo_producao": (
        "Avalie o código para produção de 1.0 a 5.0 considerando ausência de mocks, stubs ou "
        "artefatos de teste no caminho de produção, tratamento de exceções e fechamento de recursos, "
        "sanitização de entrada, ausência de credenciais hardcoded e cobertura de testes reais."
    ),
}

# Constantes do Critério Probabilístico de Aprovação na Fronteira (Borderline Mass Gate)
BORDERLINE_HEALTHY_MIN_MASS: float = 0.85
BORDERLINE_FLAWED_MAX_MASS: float = 0.12



class ScoreRequest(BaseModel):
    """Requisição para avaliação de qualidade e conformidade contra uma rubrica."""
    premise: str = Field(
        ...,
        description="O artefato produzido, diff de código, resposta ou plano a ser avaliado."
    )
    rubric: str = Field(
        default=RUBRIC_CATALOG["codigo_producao"],
        description="Critérios de conformidade e rubrica de qualidade esperada (via catálogo fechado no MCP)."
    )
    min_score: float = Field(
        default=1.0,
        description="Nota mínima da escala (ex.: 1.0)."
    )
    max_score: float = Field(
        default=5.0,
        description="Nota máxima da escala (ex.: 5.0)."
    )
    threshold: Optional[float] = Field(
        default=3.40,
        description="Nota mínima para aprovação (calibrado Laya 3.40; se fornecido, retorna se foi aprovado)."
    )

    @model_validator(mode="after")
    def validate_scale(self):
        if self.min_score >= self.max_score:
            raise ValueError("min_score deve ser estritamente menor que max_score.")
        if self.threshold is not None:
            if not (self.min_score <= self.threshold <= self.max_score):
                raise ValueError("threshold deve estar contido no intervalo [min_score, max_score].")
        return self


class ScoreResponse(BaseModel):
    """Resposta tipada da primitiva Score."""
    score: float = Field(
        ...,
        description="Nota contínua interpolada calculada pelo modelo (ex.: 4.2 em escala de 1 a 5)."
    )
    normalized_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Nota normalizada proporcionalmente no intervalo [0.0, 1.0]."
    )
    passed: Optional[bool] = Field(
        default=None,
        description="Indica se a nota obtida atingiu ou superou o threshold configurado."
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribuição de pesos por nível ou dimensão da rubrica."
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Tempo de latência da inferência em milissegundos."
    )
    score_margin: Optional[float] = Field(
        default=None,
        description="Margem de probabilidade em relação ao threshold: p_acima - p_abaixo em [-1.0, 1.0]."
    )
    borderline: Optional[bool] = Field(
        default=None,
        description="True quando ≥35% da massa probabilística está de cada lado do threshold (nota na fronteira)."
    )
    approved_as_borderline: Optional[bool] = Field(
        default=None,
        description="True quando a entrega é aprovada na fronteira via Borderline Mass Gate."
    )
    approved_with_caveat: Optional[bool] = Field(
        default=None,
        description=(
            "True quando a nota caiu na banda de indiferença "
            "[threshold - CAVEAT_BAND, threshold) e a entrega foi aprovada com ressalva."
        ),
    )
    device: Optional[str] = Field(
        default=None,
        description=(
            "Dispositivo efetivo da inferência (cuda:0/cpu). fp16 GPU × fp32 CPU "
            "podem deslocar score/borderline perto do corte — observabilidade P3.4."
        ),
    )
    dtype: Optional[str] = Field(
        default=None,
        description="dtype efetivo (float16/float32) — Pascal sem bfloat16; migração em OOM muda o numérico."
    )


# ==============================================================================
# Schemas Open Jev: Score + Diagnóstico (Screener Needle 3 + Noul — 10 categorias)
# ==============================================================================
# NOTA (set/2026): o choice de diagnóstico flat foi removido do fluxo de score.
# Quem aponta a categoria agora é o Screener estrutural (needle_defect_screener)
# e quem a confirma/arbitra é o JEV Noul (P >= 0.78). Restam 10 categorias
# técnicas reais — as duas nulas saíram do universo de decisão.
# Expansão (set/2026, decisão da fundadora): +2 categorias de ANÁLISE/qualidade
# (`tipagem_ausente`, `docstring_ausente`) entram no limite calibrado de <= 10
# opções do head do Laya (>= 11 colapsam — ver skill calibracao-score).
# Calibração histórica (ago/2026, ainda relevante para escolhas genéricas de
# openjev_choice): o head do Laya discrimina bem com <= 10 opções; >= 11 opções
# colapsam na opção-savor (precisão 1/4 no golden de defeitos).
DEFECT_OPTIONS: List[str] = [
    "logica_ou_regra",
    "robustez_excecoes_recursos",
    "seguranca",
    "qualidade_de_testes",
    "contrato_ambiente",
    "performance_escopo_estilo",
    "requisito_incompleto_ou_ausente",
    "documentacao_configuracao",
    "tipagem_ausente",
    "docstring_ausente",
]

# Tier de ANÁLISE (categorias de qualidade estrutural, set/2026): suspeitas
# dessas categorias NUNCA reprovam sozinhas uma entrega com score >= threshold
# (decisão "Noul P >= 0.78 reprova sempre" vale para as 8 categorias duras);
# com score < threshold elas reprovam e viram diagnosis para refino. Fonte
# única usada pelo mcp_server para separar confirmed hard vs quality.
QUALITY_DEFECT_CATEGORIES: List[str] = [
    "tipagem_ausente",
    "docstring_ausente",
]

# Instrução do choice — usada apenas por ferramentas de auditoria externas que
# ainda disparam evaluate_choice diretamente (ex.: scripts/audit_mocks.py).
# Texto curto: render "id: desc" + instructions deve caber em head_max_len=192.
DIAG_CHOICE_INSTRUCTIONS = (
    "Audit code artifact for defects. Use descriptions."
)

# Descrições de cada categoria para feedback legível ao agente E como critérios
# de ChoiceRequest.criteria. Textos EN curtos — alongar exige re-medir tokens
# (scripts/measure_diag_choice_tokens.py).
DEFECT_DESCRIPTIONS: Dict[str, str] = {
    "logica_ou_regra": "wrong result, off-by-one rule",
    "robustez_excecoes_recursos": "bare except, null, race, leak",
    "seguranca": "injection, secrets, missing authz",
    "qualidade_de_testes": "missing tests, no error path",
    "contrato_ambiente": "signature or schema mismatch",
    "performance_escopo_estilo": "bottleneck, scope drift, lint",
    "requisito_incompleto_ou_ausente": "missing requested feature",
    "documentacao_configuracao": "docs or config wrong",
    "tipagem_ausente": "params or return without type hints",
    "docstring_ausente": "public function/class without docstring",
}

# Folhas (defeitos finos) candidatas de cada categoria — servem de dica para o
# agente refinar a auto-correção após a categoria apontada pelo Noul.
DEFECT_MEMBERS: Dict[str, List[str]] = {
    "logica_ou_regra": [
        "erro_off_by_one",
        "regra_de_negocio_incompleta",
        "estado_mutavel_inconsistente",
        "inversao_de_predicado",
    ],
    "robustez_excecoes_recursos": [
        "falha_de_tratamento_de_excecoes",
        "falta_tratamento_nulo_ou_vazio",
        "estouro_de_limites_numericos",
        "condicao_de_corrida_deadlock",
        "vazamento_de_recursos",
    ],
    "seguranca": [
        "injecao_e_sanitizacao",
        "credenciais_hardcoded",
        "autorizacao_e_autenticacao_frouxa",
    ],
    "qualidade_de_testes": [
        "ausencia_de_testes_unitarios",
        "teste_tautologico_ou_falso_positivo",
        "falta_cobertura_caminhos_de_erro",
        "testes_nao_reprodutiveis_flaky",
    ],
    "contrato_ambiente": [
        "dependencia_ou_import_inexistente",
        "incompatibilidade_de_assinatura",
        "violacao_de_schema_de_dados",
        "quebra_de_retrocompatibilidade",
        "incompatibilidade_de_ambiente_ou_runtime",
    ],
    "performance_escopo_estilo": [
        "gargalo_de_performance_ou_complexidade",
        "alucinacao_de_efeitos_colaterais_externos",
        "desvio_de_escopo_ou_overengineering",
        "inconsistencia_estilistica_ou_linting_estrito",
    ],
    "requisito_incompleto_ou_ausente": [
        "funcionalidade_do_pedido_ausente",
        "requisito_do_prompt_ignorado",
        "caso_de_borda_do_pedido_faltando",
    ],
    "documentacao_configuracao": [
        "documentacao_ausente_ou_incorreta",
        "configuracao_ou_governanca_incorreta",
    ],
    "tipagem_ausente": [
        "parametro_sem_anotacao_de_tipo",
        "retorno_sem_anotacao_de_tipo",
        "generico_nao_especificado",
    ],
    "docstring_ausente": [
        "docstring_de_funcao_ausente",
        "docstring_de_classe_ausente",
        "contrato_de_uso_nao_documentado",
    ],
}


class DiagnosisResult(BaseModel):
    """Resultado do diagnóstico de defeito via Screener Needle 3 + Noul (10 categorias)."""
    selected: str = Field(
        ...,
        description="Categoria de defeito apontada pelo Screener e arbitrada pelo Noul (10 categorias).",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilidade do Noul para a hipótese desta categoria (0.0–1.0).",
    )
    probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Probabilidade por categoria — vencedora + demais suspeitas do Screener, quando houver.",
    )
    description: str = Field(
        ...,
        description="Descrição legível da categoria identificada para feedback ao agente.",
    )
    candidate_defects: List[str] = Field(
        default_factory=list,
        description=(
            "Folhas (defeitos finos) candidatas da categoria selecionada "
            "(DEFECT_MEMBERS) — dica para o agente refinar a auto-correção."
        ),
    )
    diagnosis_latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Latência da arbitragem (Noul/Screener) em milissegundos.",
    )


class DiffStageResult(BaseModel):
    """Resultado da avaliação de uma etapa/chunk individual de diff de código."""
    stage_id: str = Field(..., description="Identificador único da etapa de diff.")
    file_path: str = Field(..., description="Caminho do arquivo correspondente.")
    part_index: int = Field(default=1, description="Índice da parte/bloco do arquivo.")
    total_parts: int = Field(default=1, description="Total de blocos gerados para o arquivo.")
    score: float = Field(..., description="Nota do Laya para a etapa (1.0 a 5.0).")
    passed: bool = Field(..., description="Se a etapa foi aprovada individualmente.")
    borderline: bool = Field(default=False, description="Se a etapa caiu na fronteira probabilística.")
    cached: bool = Field(default=False, description="Se foi reaproveitada de validação anterior sem nova inferência.")
    diagnosis: Optional["DiagnosisResult"] = Field(default=None, description="Diagnóstico de defeito se a etapa foi reprovada.")


class ScoreDiagnoseResponse(BaseModel):
    """Resposta unificada: Score + Diagnóstico via Screener Needle 3 + Noul.

    Se passed=True e diagnosis=None: aprovação plena (score >= threshold).
    Se passed=True com approved_with_caveat=True: o Needle 3 não apontou
    NENHUMA das 10 categorias e o score caiu na banda de indiferença
    [threshold - CAVEAT_BAND, threshold) (ressalva de fronteira).
    Se passed=False com diagnosis: defeito confirmado pelo Noul (P >= 0.78)
    OU suspeita não-resolvida com score < threshold (investigar mesmo com
    Noul negativo).
    Se passed=False com diagnosis=None: caminho mono (sem símbolos AST) ou
    nota abaixo da banda de indiferença sem suspeita a diagnosticar.
    """
    # --- Campos do Score (sempre presentes) ---
    score: float = Field(..., description="Nota contínua calculada pelo Laya (escala 1.0–5.0).")
    max_score: float = Field(default=5.0, description="Nota máxima da escala.")
    normalized_score: float = Field(..., ge=0.0, le=1.0, description="Nota normalizada [0.0, 1.0].")
    passed: bool = Field(
        ...,
        description=(
            "True se não há defeito duro confirmado pelo Noul E (score >= threshold "
            "OU score na banda de indiferença sem nenhuma suspeita estrutural)."
        ),
    )
    threshold: float = Field(..., description="Nota de corte configurada.")
    breakdown: Dict[str, float] = Field(default_factory=dict, description="Pesos por nível de qualidade.")
    score_latency_ms: float = Field(..., ge=0.0, description="Latência da inferência de score em ms.")
    score_margin: Optional[float] = Field(
        default=None,
        description="Margem de probabilidade em relação ao threshold: p_acima - p_abaixo em [-1.0, 1.0].",
    )
    borderline: Optional[bool] = Field(
        default=None,
        description="True quando ≥35% da massa probabilística está de cada lado do threshold.",
    )
    task_type: Optional[str] = Field(
        default=None,
        description="Chave do catálogo fechado de rubricas usada (atualmente: codigo_producao).",
    )
    rubric_source: Optional[str] = Field(
        default=None,
        description=(
            "Origem da rubrica aplicada, sempre 'catalog:<task_type>' — a rubrica livre "
            "é ignorada pelo servidor (rubric gaming proibido)."
        ),
    )

    # --- Diagnóstico ---
    diagnosis: Optional["DiagnosisResult"] = Field(
        default=None,
        description=(
            "Defeito apontado pelo Screener Needle 3 e arbitrado pelo Noul. "
            "Presente quando a entrega é reprovada (confirmado pelo Noul ou "
            "suspeita não-resolvida com score < threshold)."
        ),
    )
    diagnosis_suppressed: Optional[str] = Field(
        default=None,
        description="DEPRECATED — supressão de diagnóstico não existe mais. Campo mantido por compatibilidade.",
    )
    approved_as_borderline: bool = Field(
        default=False,
        description=(
            "True quando a aprovação com ressalva na fronteira também passou pelo "
            "Borderline Mass Gate (p_saudavel >= 0.85 e p_falhas <= 0.12)."
        ),
    )

    # --- Avaliação Multi-Etapas (Diffs Reais) ---
    stages: List[DiffStageResult] = Field(
        default_factory=list,
        description="Lista detalhada das etapas de diff avaliadas.",
    )
    total_stages: int = Field(default=1, description="Total de etapas geradas a partir da diff do repositório.")
    passed_stages: int = Field(default=1, description="Quantidade de etapas aprovadas.")
    failed_stages: int = Field(default=0, description="Quantidade de etapas reprovadas.")

    # --- Controle Anti-Gaming ---
    attempts: int = Field(default=1, description="Tentativas consecutivas sem aprovação.")
    max_attempts: int = Field(default=5, description="Máximo antes da saída de emergência (MAX_ATTEMPTS).")
    emergency_exit_unlocked: bool = Field(default=False, description="True após MAX_ATTEMPTS tentativas sem refatoração.")
    message: Optional[str] = Field(default=None, description="Mensagem de status ou instrução de Circuit Breaker.")
    approved_with_caveat: bool = Field(default=False, description="True quando o Needle 3 não apontou NENHUMA suspeita estrutural e a nota caiu na banda de indiferença [threshold - CAVEAT_BAND, threshold) (aprovação com ressalva de fronteira).")

    # --- Needle 3 AST Guardrail (Fail-Fast Determinístico) ---
    ast_blocked: bool = Field(default=False, description="True se a entrega foi reprovada deterministamente pelo Needle 3 AST Guardrail.")
    ast_violations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Violações determinísticas de AST detectadas pelo Needle 3.")


class NeedleVerifyDeliveryRequest(BaseModel):
    """Requisição para homologação de entrega via tool needle_verify_delivery."""
    task_type: str = "codigo_producao"
    workspace_path: str = "."
    attempt_number: int = Field(default=1, ge=1, le=5)
    pedido: Optional[str] = None
    resultado: Optional[str] = None
    audit_mode: str = Field(default="delta", description="Modo de varredura: 'delta' (git diff) ou 'full' (workspace completo).")


class PytestChunkResult(BaseModel):
    """Resultado da execução de um chunk isolado da suíte de testes pytest."""
    chunk_id: str
    passed: bool
    total_passed: int
    total_failed: int
    execution_time_seconds: float
    failure_details: List[str] = Field(default_factory=list)


class ASTViolation(BaseModel):
    """Violação ou alerta de análise estática de AST."""
    rule_id: str
    category: str
    file_path: str
    line_number: int
    code_snippet: str
    message: str
    suggestion: Optional[str] = None


class VerdictState(str, Enum):
    """Estados explícitos de veredito do quality gate (TICKET-SGA-09).

    A regra de ouro é: ``passed=True`` só existe em ``VERIFIED_PASS`` (evidência
    real executada e aprovada) ou ``EMERGENCY_RELEASE`` (circuit breaker), nunca
    por ausência de runner/camada. ``NOT_APPLICABLE`` e ``INCONCLUSIVE`` são
    sempre ``passed=False``.
    """

    VERIFIED_PASS = "VERIFIED_PASS"
    VERIFIED_FAIL = "VERIFIED_FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INCONCLUSIVE = "INCONCLUSIVE"
    EMERGENCY_RELEASE = "EMERGENCY_RELEASE"


class NeedleVerifyDeliveryResponse(BaseModel):
    """Resposta unificada do quality gate needle_verify_delivery."""
    passed: bool
    status: str
    attempt_number: int
    pytest_success: bool
    pytest_chunks: List[PytestChunkResult]
    blocking_ast_violations: List[ASTViolation]
    structural_ast_warnings: List[ASTViolation]
    confirmed_defects: List[ASTViolation] = Field(
        default_factory=list,
        description="Defeitos estruturais confirmados pelo Laya RLCD (JEV Noul P >= 0.78)."
    )
    dismissed_warnings: List[ASTViolation] = Field(
        default_factory=list,
        description="Alertas estruturais descartados como falsos-positivos pelo Laya RLCD (JEV Noul P < 0.78)."
    )
    total_files_scanned: Optional[int] = Field(
        default=None,
        description=(
            "Arquivos efetivamente escaneados. ``None`` significa NÃO MEDIDO "
            "(fail-fast/ausência de evidência) — nunca confundir com ``0`` "
            "medido e vazio (TICKET-SGA-11 F5)."
        ),
    )
    total_lines_scanned: Optional[int] = Field(
        default=None,
        description=(
            "Linhas efetivamente escaneadas. ``None`` significa NÃO MEDIDO "
            "(fail-fast) — nunca ``0`` como sinônimo de medição vazia."
        ),
    )
    approved_with_caveat: bool = Field(
        default=False,
        description=(
            "Legado da Saída de Emergência. TICKET-SGA-11 F4: EMERGENCY_RELEASE "
            "sem evidência executada NUNCA aprova (``passed=False`` e este campo "
            "permanece False); o circuito terminal emite Laudo de Inviabilidade."
        ),
    )
    # --- Veredito explícito e cobertura das camadas (TICKET-SGA-09) ---
    verdict_state: str = Field(
        default=VerdictState.INCONCLUSIVE.value,
        description=(
            "Estado explícito: VERIFIED_PASS | VERIFIED_FAIL | NOT_APPLICABLE | "
            "INCONCLUSIVE | EMERGENCY_RELEASE. Default honesto é INCONCLUSIVE."
        ),
    )
    evidence_executed: bool = Field(
        default=False,
        description=(
            "true apenas quando alguma evidência real (teste/linter/AST) foi de fato "
            "executada. Nenhum caminho produz passed=True com este campo false, "
            "exceto EMERGENCY_RELEASE."
        ),
    )
    layers_executed: List[str] = Field(
        default_factory=list,
        description="Camadas efetivamente executadas (ex.: layer0_tests, layer3_native_toolchain).",
    )
    layers_skipped: List[str] = Field(
        default_factory=list,
        description=(
            "Camadas não executadas por omissão real (fail-fast, ausência de binário, "
            "timeout). Camadas que não se aplicam à stack vivem em "
            "``layers_not_applicable`` (TICKET-SGA-11 F3)."
        ),
    )
    layers_not_applicable: List[str] = Field(
        default_factory=list,
        description=(
            "Camadas deliberadamente não aplicáveis à stack detectada (ex.: AST Guard "
            "Python em stack JS). Não são omissões de cobertura (TICKET-SGA-11 F3)."
        ),
    )
    not_applicable_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Motivo declarado por camada não aplicável (TICKET-SGA-11 F3).",
    )
    skip_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Motivo explícito por camada/ferramenta não executada.",
    )
    layer_status: Dict[str, str] = Field(
        default_factory=dict,
        description="Status por camada: executed | skipped | timeout | error | not_applicable | not_executed.",
    )
    inconclusive_reasons: List[str] = Field(
        default_factory=list,
        description="Motivos que impedem um veredito VERIFIED_PASS (timeout, ferramenta ausente, saída sem contagem).",
    )
    pedido_audit: Optional[str] = Field(
        default=None,
        description="Transcrição auditável do pedido/requisito declarado na homologação."
    )
    resultado_audit: Optional[str] = Field(
        default=None,
        description="Resumo factual das alterações entregues para rastreabilidade de compliance."
    )
    message: str


# ==============================================================================
# Schemas Agent Quality Guard: Pipeline 100% Determinístico de Homologação
# Camada 0: testes reais | Camada 1: Guard + toolchain nativa | Camada 2: meta-regras
# (Laya RLCD / JEV Noul NÃO participam do caminho crítico de validação de entrega.)
# ==============================================================================

# Categorias canônicas emitidas pela bridge nativa de compiladores/linters
# (src/orchestrator/multi_stack_runner.py, Camada 1). As duas primeiras são
# BLOQUEANTES; `lint_warning` é alerta estrutural (telemetria, não reprova).
NATIVE_TOOLCHAIN_BLOCKING_CATEGORIES: List[str] = [
    "type_error",
    "lint_error",
]

# Categorias canônicas das meta-regras universais de AST via Tree-Sitter
# (src/memory/tree_sitter_guard.py, Camada 2). Regras de segurança/anti-mock
# em produção são bloqueantes; as demais são alertas estruturais.
META_RULE_BLOCKING_CATEGORIES: List[str] = [
    "anti_mock_producao",
    "security",
]


class ToolchainToolStatus(BaseModel):
    """Telemetria de disponibilidade/skip de uma ferramenta nativa (Camada 3).

    Honestidade de cobertura: ferramenta ausente vira ``skipped`` com motivo;
    timeout/erro operacional vira ``error``/``timeout`` e alimenta
    ``inconclusive_reasons`` — nunca é silenciado como "zero erros".
    """
    name: str = Field(
        ...,
        description="Identificador da ferramenta nativa (tsc, eslint, biome, ruff, mypy, clippy, go vet, dart...).",
    )
    available: bool = Field(
        default=True,
        description="True quando o binário foi localizado no ambiente.",
    )
    skipped: bool = Field(
        default=False,
        description="True quando a execução da ferramenta foi deliberadamente pulada.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Motivo do skip/indisponibilidade (ex.: binário ausente, timeout, sem arquivos alvo).",
    )
    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Tempo de execução da ferramenta em milissegundos.",
    )


class LinterDiagnostic(ASTViolation):
    """Diagnóstico de compilador/linter nativo da Camada 1.

    Herda o contrato canônico de ``ASTViolation`` (rule_id, category,
    file_path, line_number, code_snippet, message, suggestion) e acrescenta
    metadados opcionais da ferramenta de origem.
    """
    tool: Optional[str] = Field(
        default=None,
        description="Ferramenta que emitiu o diagnóstico (tsc, eslint, biome, ruff, mypy, clippy...).",
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severidade nativa reportada pela ferramenta (error | warning | info).",
    )


class MetaRuleViolation(ASTViolation):
    """Violação de meta-regra universal de AST via Tree-Sitter (Camada 2).

    Herda o contrato de ``ASTViolation`` e acrescenta a classificação de
    severidade da meta-regra e a linguagem analisada.
    """
    severity: Optional[str] = Field(
        default=None,
        description="Classificação da meta-regra: 'blocking' (anti-mock/segurança) ou 'warning' (estrutural).",
    )
    language: Optional[str] = Field(
        default=None,
        description="Linguagem do arquivo analisado (python, typescript, javascript, go, rust, dart...).",
    )


class AgentQualityGuardResponse(NeedleVerifyDeliveryResponse):
    """Resposta canônica do quality gate ``agent_quality_guard``.

    Superset 100% retrocompatível de ``NeedleVerifyDeliveryResponse``: todo
    cliente legado que desserializa o payload antigo continua funcionando
    (campos extras são ignorados por padrão pelo Pydantic). O pipeline é
    100% determinístico e não carrega pesos de ML:

    - Camada 0: testes reais (pytest em chunks / runner multi-stack);
    - Camada 1: NeedleASTGuard + bridge nativa de compiladores/linters;
    - Camada 2: meta-regras universais de AST via Tree-Sitter.

    ``confirmed_defects``/``dismissed_warnings`` permanecem no schema por
    retrocompatibilidade, porém a arbitragem Laya/Noul foi removida do
    caminho crítico — esses buckets permanecem vazios.
    """
    tool_name: str = Field(
        default="agent_quality_guard",
        description="Nome canônico da ferramenta que produziu a resposta.",
    )
    # --- Camada 1: bridge nativa de compiladores/linters ---
    layer1_toolchain_diagnostics: List[LinterDiagnostic] = Field(
        default_factory=list,
        description=(
            "Diagnósticos da Camada 1 (type_error/lint_error/lint_warning) com prefixos de "
            "rule_id nativos (TSC_/ESLINT_/RUFF_/MYPY_/CLIPPY_/GOVET_...)."
        ),
    )
    toolchain_status: List[ToolchainToolStatus] = Field(
        default_factory=list,
        description="Telemetria de disponibilidade/skip das ferramentas nativas (fail-open).",
    )
    # --- Camada 2: meta-regras universais via Tree-Sitter ---
    layer2_meta_rule_violations: List[MetaRuleViolation] = Field(
        default_factory=list,
        description=(
            "Violações das meta-regras universais de AST (anti-mock em produção, floating "
            "promises, silent catch, resource leaks, SQL/eval perigosos)."
        ),
    )


class AgentQualityGuardRequest(NeedleVerifyDeliveryRequest):
    """Requisição canônica do ``agent_quality_guard`` (alias de ``NeedleVerifyDeliveryRequest``).

    Mantida como subclasse vazia para espelhar o rebatismo do lado da resposta,
    preservando 100% de retrocompatibilidade de payload com clientes legados.
    """


# ==============================================================================
# Schemas Cactus Needle 3: Busca & Recuperação Cirúrgica Relacional
# ==============================================================================

class Needle3Query(BaseModel):
    """Requisição de busca cirúrgica para o Cactus Needle 3.

    Contrato de profundidade (TICKET-SGA-05): ``max_depth`` limita a árvore de
    diretórios da varredura e ``relational_depth`` limita o grafo de relações de
    AST. Ambos são opcionais; ``max_depth=None`` (padrão) significa profundidade
    ILIMITADA de diretórios — 100% retrocompatível com clientes que não os enviam.
    """
    query: str = Field(
        ...,
        description="Termo ou pergunta técnica de busca."
    )
    scope: Optional[List[str]] = Field(
        default=None,
        description="Filtro opcional de diretórios, pacotes ou arquivos-alvo."
    )

    @field_validator("scope", mode="before")
    @classmethod
    def coerce_scope_to_list(cls, v: Any) -> Optional[List[str]]:
        """Permite que chamadores passem string única (ex: 'src/decision') ou lista de strings."""
        if v is None:
            return None
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean:
                return None
            if "," in v_clean:
                parts = [p.strip() for p in v_clean.split(",") if p.strip()]
                return parts if parts else None
            return [v_clean]
        if isinstance(v, (list, tuple, set)):
            items = []
            for item in v:
                item_str = str(item).strip()
                if not item_str:
                    continue
                if "," in item_str:
                    items.extend([p.strip() for p in item_str.split(",") if p.strip()])
                else:
                    items.append(item_str)
            return items if items else None
        return None
    max_tokens: int = Field(
        default=1000,
        gt=0,
        le=4096,
        description=(
            "Limite máximo estrito de tokens para o recorte retornado (canônico 1000; "
            "faixa AGENTS 500–1500). O cálculo usa CHARS_PER_TOKEN = 3.8, aproximação "
            "heurística de alta fidelidade para orçamento rápido de snippets — não é "
            "garantia criptográfica estrita de contagem de tokens."
        )
    )
    relational_depth: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Profundidade do grafo de relações AST (chamadores, implementações, imports, "
            "referências de tipo). Padrão canônico 2. O parâmetro `depth` da tool MCP "
            "sobrepõe este valor quando enviado; `depth=None` mantém este padrão."
        )
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Profundidade máxima da árvore de diretórios da varredura (níveis abaixo de "
            "cada alvo de escopo). None (padrão) = profundidade ilimitada. Espelha o "
            "parâmetro `depth` da tool MCP quando enviado."
        )
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Globs opcionais de exclusão somados às exclusões universais padrão e ao "
            "arquivo `.cirurgicignore` da raiz do workspace (se existir). Mesma semântica: "
            "nome puro casa em qualquer nível; padrão com '/' é ancorado na raiz; '/' final "
            "marca diretório; negação com '!' não é suportada."
        )
    )

    @field_validator("exclude_patterns", mode="before")
    @classmethod
    def coerce_exclude_patterns_to_list(cls, v: Any) -> Optional[List[str]]:
        """Permite string única, lista ou None para `exclude_patterns` (paridade com `scope`)."""
        if v is None:
            return None
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean:
                return None
            parts = [p.strip() for p in v_clean.split(",") if p.strip()]
            return parts if parts else None
        if isinstance(v, (list, tuple, set)):
            items = [str(item).strip() for item in v if str(item).strip()]
            return items or None
        return None
    preserve_relational_integrity: bool = Field(
        default=True,
        description="Se verdadeiro, garante que dependências relacionais do recorte sejam incluídas sem truncamento arbitrário."
    )


class Needle3Snippet(BaseModel):
    """Fragmento cirúrgico recuperado pelo motor Needle 3."""
    file_path: str = Field(..., description="Caminho relativo ou absoluto do arquivo de origem.")
    start_line: int = Field(..., ge=1, description="Linha inicial do trecho.")
    end_line: int = Field(..., ge=1, description="Linha final do trecho.")
    content: str = Field(..., description="Conteúdo exato do fragmento de código ou documentação.")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Score de relevância semântica calculada pelo Needle 3.")
    relations: List[str] = Field(
        default_factory=list,
        description="Relações detectadas no grafo de AST (ex: 'caller: auth_handler', 'implements: BaseRepository')."
    )


class Needle3Response(BaseModel):
    """Resposta agregada da busca cirúrgica do Needle 3.

    Os totais de tokens são estimativas heurísticas baseadas em
    ``CHARS_PER_TOKEN = 3.8`` (aproximação de alta fidelidade para orçamento
    rápido de snippets, não garantia criptográfica estrita).
    """
    snippets: List[Needle3Snippet] = Field(..., description="Lista de trechos cirúrgicos recuperados.")
    total_tokens: int = Field(..., ge=0, description="Total de tokens consumidos pelos trechos.")
    query_latency_ms: float = Field(..., ge=0.0, description="Latência da busca no Cactus Compute.")
    token_savings_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentual real de tokens economizados vs. indexar todos os blocos candidatos.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Avisos operacionais e alertas de degradação.",
    )
    degraded_lanes: List[str] = Field(
        default_factory=list,
        description="Lanes de busca que falharam ou sofreram degradação/timeout.",
    )


class NeedleSearchHttpRequest(BaseModel):
    """Requisição HTTP para consulta do Needle 3 com seleção de repositório (workspace_path)."""
    query: str = Field(..., description="Termo ou pergunta técnica de busca em linguagem natural.")
    workspace_path: Optional[str] = Field(default=".", description="Caminho do repositório a ser consultado.")
    scope: Optional[List[str]] = Field(default=None, description="Filtro opcional de diretórios/pacotes/arquivos.")
    max_tokens: int = Field(
        default=1000,
        gt=0,
        le=4096,
        description=(
            "Limite máximo de tokens do recorte retornado (heurística CHARS_PER_TOKEN = 3.8, "
            "aproximação de alta fidelidade — não é garantia criptográfica estrita)."
        ),
    )
    relational_depth: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Profundidade do grafo relacional AST (padrão canônico 2).",
    )
    max_depth: Optional[int] = Field(
        default=None,
        ge=0,
        description="Profundidade máxima da árvore de diretórios da varredura (None = ilimitado).",
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None,
        description="Globs de exclusão somados às exclusões universais e ao `.cirurgicignore` da raiz.",
    )

    @field_validator("scope", mode="before")
    @classmethod
    def coerce_scope_to_list(cls, v: Any) -> Optional[List[str]]:
        """Normaliza o campo scope para lista, reaproveitando a regra do schema Needle3Query.

        Exemplo:
            >>> NeedleSearchHttpRequest.coerce_scope_to_list("src/decision")
            ['src/decision']
        """
        return Needle3Query.coerce_scope_to_list(v)

    @field_validator("exclude_patterns", mode="before")
    @classmethod
    def coerce_exclude_patterns_to_list(cls, v: Any) -> Optional[List[str]]:
        """Normaliza `exclude_patterns`, reaproveitando a regra do schema Needle3Query."""
        return Needle3Query.coerce_exclude_patterns_to_list(v)


class CirurgicSearchResponse(BaseModel):
    """Contrato documentado da resposta do MCP `cirurgic_search` (+ alias `qmd_search`).

    O runtime MCP retorna um ``dict`` plain (para preservar paridade byte-a-byte
    entre o nome canônico e o alias legado e não quebrar clientes que fazem
    ``res["status"]``); este schema existe como contrato de referência validável
    e para clientes Python que queiram ``CirurgicSearchResponse.model_validate(res)``.

    Os campos ``status``/``warnings``/``degraded_lanes``/``timeout_s`` são
    aditivos e retrocompatíveis: construções antigas que não os informam
    recebem os defaults (`success`, listas vazias, `None`).
    """
    status: Literal["success", "partial", "timeout", "error"] = Field(
        default="success",
        description=(
            "success = todas as lanes ok; partial = pelo menos uma lane degradada "
            "(snippets ainda úteis); timeout = timeout global da busca híbrida "
            "atingido (snippets vazios); error = falha total (ex: violação de sandbox)."
        ),
    )
    degraded_lanes: List[str] = Field(
        default_factory=list,
        description="Lanes de busca que falharam ou sofreram degradação/timeout (ex: 'ripgrep_lexical', 'needle_ast', 'sandbox').",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Avisos operacionais e alertas de degradação (ex: timeout de lane, token budget insuficiente).",
    )
    latency_ms: float = Field(default=0.0, ge=0.0, description="Latência total da busca híbrida em milissegundos.")
    timeout_s: Optional[float] = Field(
        default=None,
        description="Timeout global da busca em segundos, quando aplicável.",
    )
    total_tokens: int = Field(default=0, ge=0, description="Total de tokens consumidos pelos snippets retornados.")
    token_savings_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentual estimado de tokens economizados vs. indexar todos os candidatos brutos.",
    )
    snippets_count: int = Field(default=0, ge=0, description="Quantidade de snippets no recorte final.")
    files_found: int = Field(default=0, ge=0, description="Quantidade de arquivos candidatos rankeados.")
    snippets: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Snippets cirúrgicos brutos (dict por arquivo com rrf_score, source, relations, exact_matches, content).",
    )
    error: Optional[str] = Field(
        default=None,
        description="Código/identificador do erro quando status='error' (ex: 'sandbox_violation').",
    )
    message: Optional[str] = Field(
        default=None,
        description="Mensagem humana explicando o erro ou a degradação quando aplicável.",
    )


# ==============================================================================
# Schemas Subagentes Especialistas: Invocação com Contexto Isolado
# ==============================================================================

class SpawnSubagentRequest(BaseModel):
    """Requisição para despacho e execução de subagente especialista em contexto limpo."""
    persona: str = Field(
        ...,
        description="Identificador da persona especialista (ex: 'lumi', 'robin', 'kael', 'ariel', 'alex') ou caminho do arquivo."
    )
    task: str = Field(
        ...,
        description="Instrução cirúrgica e detalhada da tarefa técnica a ser executada pelo subagente."
    )
    workspace: Optional[str] = Field(
        default=None,
        description="Diretório absoluto da raiz do repositório alvo. Se omitido, assume o workspace corrente."
    )
    auto_validate: bool = Field(
        default=True,
        description="Se True, submete a entrega do subagente à auditoria fail-fast do Needle AST e ao Score JEV."
    )
    timeout: Optional[float] = Field(
        default=None,
        description="Timeout em segundos para a execução do subagente. Se None ou <= 0, o timeout fica desativado."
    )



class SpawnSubagentResponse(BaseModel):
    """Resposta do subagente especialista após execução em janela de contexto isolada."""
    success: bool = Field(..., description="Indica se o subagente completou a tarefa sem erros não tratados.")
    persona: str = Field(..., description="Nome da persona executada.")
    workspace: str = Field(..., description="Caminho do repositório onde a tarefa foi executada.")
    summary: str = Field(..., description="Relatório resumido e factual da entrega produzida pelo subagente.")
    files_modified: List[str] = Field(default_factory=list, description="Lista de arquivos modificados ou criados no repositório.")
    ast_blocked: bool = Field(default=False, description="Indica se a entrega foi interceptada pelo Needle AST Guardrail.")
    ast_violations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Violações detectadas pelo Needle AST se ast_blocked for True.")
    score: Optional[float] = Field(default=None, description="Nota de conformidade atribuída pelo Open Jev Score (1.00 a 5.00).")
    passed: Optional[bool] = Field(default=None, description="Status de aprovação no quality gate JEV (score >= 3.40).")
    execution_time_ms: float = Field(..., ge=0.0, description="Tempo total de execução do subagente em milissegundos.")
    context_tokens_used: int = Field(default=0, ge=0, description="Estimativa de tokens de contexto consumidos na janela isolada.")

