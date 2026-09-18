"""
Synthesizer module for the retrieval graph.
"""
from typing import Any

from pydantic import BaseModel, Field

from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.evidence import Evidence
from longdoc_retrieval.graph.llm import StructuredLLM, StructuredOutputError
from longdoc_retrieval.graph.state import RetrievalState
from longdoc_retrieval.graph.types import Node


class _SynthesizedAnswer(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)


def _render_prompt(
    question: str,
    evidence_list: list[Evidence],
    status: str,
    unresolved_questions: list[str],
    contradictions: list[str],
) -> str:
    lines = [
        f"Pergunta: {question}",
        (
            "Evidencias abaixo, cada uma com seu 'evidence_id' - use APENAS "
            "essas evidencias para responder, nunca conhecimento externo ou "
            "suposicoes sobre o documento."
        ),
    ]
    for evidence in evidence_list:
        lines.append(f"[evidence_id={evidence.evidence_id}] {evidence.text}")
    if status != "sufficient":
        lines.append(
            f"Status da coleta: {status} - a busca pode nao ter encontrado tudo. "
            "Responda com o que as evidencias acima efetivamente sustentam, e "
            "deixe explicito na resposta o que permanece incerto ou nao "
            "encontrado, em vez de completar a lacuna com suposicao."
        )
    if unresolved_questions:
        lines.append("Informacao que a busca NAO encontrou: " + "; ".join(unresolved_questions))
    if contradictions:
        lines.append("Contradicoes identificadas entre evidencias: " + "; ".join(contradictions))
    lines.append(
        "Escreva uma resposta direta e objetiva a pergunta, em portugues, "
        "baseada exclusivamente nas evidencias acima. Em 'citations', liste "
        "os 'evidence_id' das evidencias efetivamente usadas na resposta."
    )
    return "\n\n".join(lines)


def synthesize_answer_node(llm: StructuredLLM, config: RetrievalConfig) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        package = state["evidence_package"]
        assert package is not None, "synthesize_answer must run after build_evidence_package"

        evidence_list = state["evidence"]
        metrics = state["metrics"]

        if not evidence_list:
            return {"evidence_package": package}

        prompt = _render_prompt(
            state["question"],
            evidence_list,
            package.status,
            package.unresolved_questions,
            package.contradictions,
        )
        try:
            result = await llm.invoke(prompt, _SynthesizedAnswer)
        except StructuredOutputError:
            return {"evidence_package": package}

        metrics = metrics.model_copy(
            update={
                "llm_calls": metrics.llm_calls + 1,
                "llm_input_tokens": metrics.llm_input_tokens + result.input_tokens,
                "llm_output_tokens": metrics.llm_output_tokens + result.output_tokens,
            }
        )
        valid_evidence_ids = {evidence.evidence_id for evidence in evidence_list}
        citations = [
            evidence_id for evidence_id in result.value.citations if evidence_id in valid_evidence_ids
        ]
        updated_package = package.model_copy(
            update={"answer": result.value.answer, "answer_citations": citations, "metrics": metrics}
        )
        return {"metrics": metrics, "evidence_package": updated_package}

    return _node
