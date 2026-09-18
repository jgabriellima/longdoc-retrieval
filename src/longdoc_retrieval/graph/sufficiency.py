import asyncio
from typing import Any

from pydantic import BaseModel, Field

from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.evidence import Evidence
from longdoc_retrieval.domain.plan import RetrievalPlan
from longdoc_retrieval.domain.sufficiency import SufficiencyDecision
from longdoc_retrieval.graph.budgets import should_stop
from longdoc_retrieval.graph.llm import StructuredLLM, StructuredOutputError
from longdoc_retrieval.graph.state import RetrievalState
from longdoc_retrieval.graph.types import Node

_ROLE_LABEL_NOTE = (
    "Documentos administrativos e contratuais costumam definir, uma unica "
    "vez em algum ponto do texto, um rotulo generico para cada parte "
    "nomeada (um padrao do tipo '[Nome Proprio], doravante/ora denominad[ao] "
    "[ROTULO GENERICO]') e depois se referem a essa parte so pelo rotulo "
    "generico no restante do documento, sem repetir o nome - os rotulos "
    "exatos variam por tipo de documento (podem ser CONCEDENTE/CONVENENTE, "
    "CONTRATANTE/CONTRATADA, OUTORGANTE/OUTORGADO, ou outros - nao ha uma "
    "lista fixa). Quando a pergunta usar o nome proprio de uma parte mas o "
    "trecho relevante usar apenas o rotulo generico dela (ou vice-versa), "
    "use o CONTEXTO DE APOIO abaixo para localizar essa definicao e "
    "resolver a equivalencia - isso e leitura normal do documento, nao "
    "inferencia arriscada."
)


class _PrimaryExtraction(BaseModel):
    answer_excerpt: str | None = None


class _BatchSignal(BaseModel):
    answer_excerpt: str | None = None
    missing_information: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_queries: list[str] = Field(default_factory=list)


def _render_primary_evidence_prompt(
    question: str, evidence_list: list[Evidence], primary_idx: int
) -> str:
    primary = evidence_list[primary_idx - 1]
    lines = [
        f"Pergunta: {question}",
        (
            "Este documento pode reunir textos de MULTIPLOS instrumentos/convenios "
            "publicados juntos. A EVIDENCIA PRINCIPAL abaixo e a UNICA que voce "
            "deve julgar se responde a pergunta. As demais evidencias sao apenas "
            "CONTEXTO DE APOIO (para resolver nomes/rotulos cruzados) - a presenca "
            "de conteudo de outro instrumento no contexto de apoio NAO deve, por "
            "si so, fazer voce recusar uma resposta que a evidencia principal ja "
            "contem."
        ),
        (
            f"EVIDENCIA PRINCIPAL (offsets {primary.start_offset}-{primary.end_offset}):\n"
            f"{primary.text}"
        ),
    ]
    support_lines = [
        f"[apoio, offsets {evidence.start_offset}-{evidence.end_offset}] {evidence.text}"
        for idx, evidence in enumerate(evidence_list, start=1)
        if idx != primary_idx
    ]
    if support_lines:
        lines.append("CONTEXTO DE APOIO (outras evidencias do lote):\n" + "\n\n".join(support_lines))
    lines.append(_ROLE_LABEL_NOTE)
    lines.append(
        "A propria EVIDENCIA PRINCIPAL pode, ela mesma, misturar num unico "
        "bloco de texto trechos de MAIS DE UM instrumento/convenio (nao so "
        "entre evidencias diferentes, como acima - tambem DENTRO do mesmo "
        "bloco, por exemplo o final de uma clausula de um convenio seguido, "
        "sem separacao clara, do inicio de outro). Nesse caso, use os "
        "identificadores que a propria pergunta ja menciona (numero do "
        "convenio, orgaos, municipio etc.) para localizar, dentro da "
        "evidencia principal, a parte que de fato se refere ao instrumento "
        "perguntado, e extraia o trecho literal dessa parte - a presenca de "
        "outra parte do mesmo bloco pertencente a outro instrumento nao e "
        "motivo para deixar de responder quando a parte correta esta ali."
    )
    lines.append(
        "A EVIDENCIA PRINCIPAL, por si so (usando o contexto de apoio apenas "
        "para resolver nomes/rotulos, nunca para invalidar o que ela ja diz), "
        "responde direta e especificamente a pergunta? Se sim, copie "
        "literalmente (palavra por palavra, sem resumir ou parafrasear) o "
        "trecho DA EVIDENCIA PRINCIPAL que responde, em 'answer_excerpt'. Se "
        "nao, deixe 'answer_excerpt' como null."
    )
    return "\n\n".join(lines)


def _render_batch_signal_prompt(question: str, evidence_list: list[Evidence]) -> str:
    lines = [f"Pergunta: {question}", "Evidencias coletadas ate agora:"]
    if not evidence_list:
        lines.append("(nenhuma evidencia coletada ainda)")
    for idx, evidence in enumerate(evidence_list, start=1):
        lines.append(
            f"EV-{idx:03d} [offsets {evidence.start_offset}-{evidence.end_offset}] "
            f"{evidence.text}"
        )
    lines.append(
        "Olhando para TODAS as evidencias juntas: se alguma delas (mesmo que "
        "isoladamente uma avaliacao anterior nao tenha detectado isso) "
        "contiver a resposta direta e especifica a pergunta, copie o trecho "
        "literal em 'answer_excerpt' (indicando de qual evidencia veio). Isso "
        "vale mesmo que a MESMA evidencia tambem contenha, misturado no mesmo "
        "bloco de texto, conteudo de outro instrumento/convenio - use os "
        "identificadores que a propria pergunta ja menciona para localizar a "
        "parte correta e extraia o trecho literal dela; a mistura NAO e "
        "motivo para deixar 'answer_excerpt' como null quando a parte certa "
        "esta ali. Se nenhuma evidencia contiver a resposta, deixe "
        "'answer_excerpt' como null. De todo modo, liste o que falta "
        "('missing_information'), indicios de que parte do lote pertence a "
        "outro instrumento/convenio ('contradictions'), e quais consultas de "
        "busca ajudariam na proxima iteracao ('recommended_queries')."
    )
    return "\n\n".join(lines)


async def _try_extract(
    llm: StructuredLLM, prompt: str
) -> tuple[_PrimaryExtraction, int, int] | None:
    try:
        result = await llm.invoke(prompt, _PrimaryExtraction)
    except StructuredOutputError:
        return None
    return result.value, result.input_tokens, result.output_tokens


async def _try_batch_signal(
    llm: StructuredLLM, prompt: str
) -> tuple[_BatchSignal, int, int] | None:
    try:
        result = await llm.invoke(prompt, _BatchSignal)
    except StructuredOutputError:
        return None
    return result.value, result.input_tokens, result.output_tokens


def evaluate_sufficiency_node(llm: StructuredLLM, config: RetrievalConfig) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        evidence_list = state["evidence"]
        question = state["question"]
        metrics = state["metrics"]
        n_samples = max(1, config.sufficiency_samples)

        extraction_calls = [
            (idx, _try_extract(llm, _render_primary_evidence_prompt(question, evidence_list, idx)))
            for idx in range(1, len(evidence_list) + 1)
            for _ in range(n_samples)
        ]
        extraction_results = (
            await asyncio.gather(*(call for _, call in extraction_calls)) if extraction_calls else []
        )

        sufficient = False
        extraction_input_tokens = 0
        extraction_output_tokens = 0
        for (idx, _), result in zip(extraction_calls, extraction_results, strict=True):
            if result is None:
                continue
            extraction, input_tokens, output_tokens = result
            extraction_input_tokens += input_tokens
            extraction_output_tokens += output_tokens
            excerpt = extraction.answer_excerpt
            if not sufficient and excerpt and excerpt.strip() in evidence_list[idx - 1].text:
                sufficient = True

        batch_prompt = _render_batch_signal_prompt(question, evidence_list)
        batch_result = await _try_batch_signal(llm, batch_prompt)

        total_calls = len(extraction_calls) + 1
        if batch_result is not None:
            signal, batch_input, batch_output = batch_result
            metrics = metrics.model_copy(
                update={
                    "llm_calls": metrics.llm_calls + total_calls,
                    "llm_input_tokens": metrics.llm_input_tokens
                    + extraction_input_tokens
                    + batch_input,
                    "llm_output_tokens": metrics.llm_output_tokens
                    + extraction_output_tokens
                    + batch_output,
                }
            )
            if not sufficient and signal.answer_excerpt and signal.answer_excerpt.strip():
                candidate = signal.answer_excerpt.strip()
                sufficient = any(candidate in evidence.text for evidence in evidence_list)
        else:
            signal = _BatchSignal(missing_information=["sufficiency signal unavailable"])
            metrics = metrics.model_copy(
                update={
                    "llm_calls": metrics.llm_calls + total_calls,
                    "llm_input_tokens": metrics.llm_input_tokens + extraction_input_tokens,
                    "llm_output_tokens": metrics.llm_output_tokens + extraction_output_tokens,
                }
            )

        decision = SufficiencyDecision(
            sufficient=sufficient,
            confidence=0.9 if sufficient else 0.3,
            missing_information=[] if sufficient else signal.missing_information,
            contradictions=signal.contradictions,
            recommended_queries=[] if sufficient else signal.recommended_queries,
        )

        partial: dict[str, Any] = {
            "sufficient": decision.sufficient,
            "iteration": state["iteration"] + 1,
            "unresolved_questions": decision.missing_information,
            "contradictions": decision.contradictions,
            "last_sufficiency": decision,
            "metrics": metrics,
        }
        merged_view: RetrievalState = {**state, **partial}  # type: ignore[typeddict-item]
        stop, reason = should_stop(merged_view, config)
        partial["stop_reason"] = reason if stop else None
        return partial

    return _node


def refine_strategy_node() -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        last = state["last_sufficiency"]
        previous_plan = state["plan"]

        new_plan = RetrievalPlan(
            objective=previous_plan.objective if previous_plan else state["question"],
            concepts=previous_plan.concepts if previous_plan else [],
            exact_terms=previous_plan.exact_terms if previous_plan else [],
            structural_hints=previous_plan.structural_hints if previous_plan else [],
            evidence_types=previous_plan.evidence_types if previous_plan else [],
            queries=last.recommended_queries if last else [],
        )
        return {"plan": new_plan}

    return _node
