"""Conditional edge after evaluate_sufficiency. Reads `stop_reason`, which
`evaluate_sufficiency_node` already computed via `should_stop` - the router
never re-derives the stop decision itself.
"""

from typing import Literal

from longdoc_retrieval.graph.state import RetrievalState

RouteTarget = Literal["build_evidence_package", "refine_strategy"]


def route_after_sufficiency(state: RetrievalState) -> RouteTarget:
    return "build_evidence_package" if state.get("stop_reason") else "refine_strategy"
