import time
from typing import Any, Literal

from longdoc_retrieval.domain.package import EvidencePackage
from longdoc_retrieval.graph.state import RetrievalState
from longdoc_retrieval.graph.types import Node


def _status(state: RetrievalState) -> Literal["sufficient", "partial", "insufficient"]:
    if state["sufficient"]:
        return "sufficient"
    if state["evidence"]:
        return "partial"
    return "insufficient"


def _summary(state: RetrievalState) -> str:
    return (
        f"{len(state['evidence'])} evidence item(s) found across "
        f"{state['iteration']} iteration(s); stop_reason={state['stop_reason']}."
    )


def build_evidence_package_node() -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        metrics = state["metrics"].model_copy(
            update={
                "latency_ms": (time.monotonic() - state["started_at"]) * 1000,
                "iterations": state["iteration"],
                "stop_reason": state["stop_reason"],
            }
        )
        package = EvidencePackage(
            request_id=state["request_id"],
            document_id=state["document_id"],
            question=state["question"],
            status=_status(state),
            evidence=state["evidence"],
            unresolved_questions=state["unresolved_questions"],
            contradictions=state["contradictions"],
            retrieval_summary=_summary(state),
            metrics=metrics,
        )
        return {"metrics": metrics, "evidence_package": package}

    return _node
