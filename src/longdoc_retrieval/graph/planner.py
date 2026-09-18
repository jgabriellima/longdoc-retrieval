"""
Planner module for the retrieval graph.
"""
import time
from typing import Any

from longdoc_retrieval.api.schemas import OutlineNode
from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.metrics import RetrievalMetrics
from longdoc_retrieval.domain.plan import RetrievalPlan
from longdoc_retrieval.graph.llm import StructuredLLM, StructuredOutputError
from longdoc_retrieval.graph.state import RetrievalState
from longdoc_retrieval.graph.types import Node
from longdoc_retrieval.tokenize import approx_token_count


def render_outline(node: OutlineNode, indent: int = 0) -> str:
    prefix = "  " * indent
    title = node.title or "(sem titulo)"
    lines = [f"{prefix}- {title} [{node.node_id}] (~{node.token_count} tokens)"]
    for child in node.children:
        lines.append(render_outline(child, indent + 1))
    return "\n".join(lines)


def understand_request_node(service: RetrievalService) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        outline = await service.document_outline(state["document_id"], depth=2)
        metadata = await service.get_metadata(state["document_id"])
        outline_summary = render_outline(outline.root)

        metrics = RetrievalMetrics(
            document_id=state["document_id"],
            document_tokens=outline.root.token_count,
            question_tokens=approx_token_count(state["question"]),
        )

        return {
            "outline_summary": outline_summary,
            "document_metadata": metadata.model_dump(),
            "metrics": metrics,
            "started_at": time.monotonic(),
            "evidence_count_history": [],
            "iteration": 0,
            "plan": None,
            "executed_queries": [],
            "candidates": [],
            "evaluated_candidates": [],
            "evidence": [],
            "pending_evidence": [],
            "unresolved_questions": [],
            "contradictions": [],
            "sufficient": False,
            "stop_reason": None,
            "last_sufficiency": None,
            "evidence_package": None,
        }

    return _node


def _render_plan_prompt(state: RetrievalState) -> str:
    parts = [
        (
            "Voce e um planejador de recuperacao de evidencias em documentos "
            "longos (relatorios, contratos, processos administrativos)."
        ),
        f"Pergunta: {state['question']}",
        "Estrutura do documento (profundidade limitada):",
        state["outline_summary"],
    ]
    if state.get("document_metadata"):
        parts.append(f"Metadados conhecidos: {state['document_metadata']}")

    last = state.get("last_sufficiency")
    if last is not None:
        parts.append(
            "A iteracao anterior considerou a evidencia insuficiente. "
            f"Informacao faltante: {last.missing_information}. "
            f"Consultas recomendadas: {last.recommended_queries}."
        )

    parts.append(
        "Produza um plano de recuperacao: objetivo, conceitos-chave, termos "
        "exatos (numeros de processo/contrato, valores, datas, CNPJ/CPF, "
        "'Art. N'), pistas estruturais (secoes prováveis), tipos de "
        "evidencia esperados, e uma lista de consultas de busca."
    )
    return "\n\n".join(parts)


def plan_retrieval_node(llm: StructuredLLM, config: RetrievalConfig) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        prompt = _render_plan_prompt(state)
        metrics = state["metrics"]
        try:
            result = await llm.invoke(prompt, RetrievalPlan)
            plan = result.value
            metrics = metrics.model_copy(
                update={
                    "llm_calls": metrics.llm_calls + 1,
                    "llm_input_tokens": metrics.llm_input_tokens + result.input_tokens,
                    "llm_output_tokens": metrics.llm_output_tokens + result.output_tokens,
                }
            )
        except StructuredOutputError:
            plan = RetrievalPlan(objective=state["question"], queries=[])
            metrics = metrics.model_copy(update={"llm_calls": metrics.llm_calls + 1})

        return {"plan": plan, "metrics": metrics}

    return _node
