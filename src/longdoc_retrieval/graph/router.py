from typing import Literal

from longdoc_retrieval.graph.state import RetrievalState

RouteTarget = Literal["build_evidence_package", "refine_strategy"]


def route_after_sufficiency(state: RetrievalState) -> RouteTarget:
    return "build_evidence_package" if state.get("stop_reason") else "refine_strategy"
