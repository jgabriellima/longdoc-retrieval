from typing import Any

from pydantic import BaseModel

from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.evaluation import EvaluatedCandidate
from longdoc_retrieval.graph.llm import StructuredLLM, StructuredOutputError
from longdoc_retrieval.graph.state import RetrievalState
from longdoc_retrieval.graph.types import Node


class _EvaluationBatch(BaseModel):
    evaluations: list[EvaluatedCandidate]


def _render_evaluation_prompt(question: str, candidates: list[Any]) -> str:
    lines = [
        f"Pergunta: {question}",
        (
            "Avalie a relevancia de cada candidato (irrelevant, possibly_relevant, "
            "relevant, critical) e decida se vale a pena ler o trecho na integra "
            "(should_read). Responda para TODOS os candidate_id listados."
        ),
    ]
    for candidate in candidates:
        lines.append(
            f"- candidate_id={candidate.candidate_id} metodo={candidate.retrieval_method} "
            f"preview={candidate.preview!r}"
        )
    return "\n".join(lines)


def _fallback_missing(candidate_ids: set[str], covered: set[str]) -> list[EvaluatedCandidate]:
    return [
        EvaluatedCandidate(
            candidate_id=candidate_id,
            relevance="possibly_relevant",
            reason="not evaluated by model (missing from LLM response)",
            should_read=False,
        )
        for candidate_id in candidate_ids
        if candidate_id not in covered
    ]


def evaluate_candidates_node(llm: StructuredLLM, config: RetrievalConfig) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        candidates = state["candidates"]
        if not candidates:
            return {"evaluated_candidates": []}

        input_ids = {c.candidate_id for c in candidates}
        prompt = _render_evaluation_prompt(state["question"], candidates)
        metrics = state["metrics"]

        try:
            result = await llm.invoke(prompt, _EvaluationBatch)
            evaluated = [e for e in result.value.evaluations if e.candidate_id in input_ids]
            covered = {e.candidate_id for e in evaluated}
            evaluated.extend(_fallback_missing(input_ids, covered))
            metrics = metrics.model_copy(
                update={
                    "llm_calls": metrics.llm_calls + 1,
                    "llm_input_tokens": metrics.llm_input_tokens + result.input_tokens,
                    "llm_output_tokens": metrics.llm_output_tokens + result.output_tokens,
                }
            )
        except StructuredOutputError:
            evaluated = [
                EvaluatedCandidate(
                    candidate_id=c.candidate_id,
                    relevance="possibly_relevant",
                    reason="fallback: candidate evaluation unavailable",
                    should_read=True,
                )
                for c in candidates
            ]
            metrics = metrics.model_copy(update={"llm_calls": metrics.llm_calls + 1})

        return {"evaluated_candidates": evaluated, "metrics": metrics}

    return _node
