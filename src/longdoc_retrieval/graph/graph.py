"""
Graph for the retrieval API.
"""
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.graph.evaluator import evaluate_candidates_node
from longdoc_retrieval.graph.fusion import merge_candidates_node
from longdoc_retrieval.graph.llm import LLMClients
from longdoc_retrieval.graph.package import build_evidence_package_node
from longdoc_retrieval.graph.planner import plan_retrieval_node, understand_request_node
from longdoc_retrieval.graph.reader import read_evidence_node, update_evidence_ledger_node
from longdoc_retrieval.graph.router import route_after_sufficiency
from longdoc_retrieval.graph.search import execute_searches_node
from longdoc_retrieval.graph.state import RetrievalInput, RetrievalOutput, RetrievalState
from longdoc_retrieval.graph.sufficiency import evaluate_sufficiency_node, refine_strategy_node
from longdoc_retrieval.graph.synthesizer import synthesize_answer_node


def build_retrieval_graph(
    service: RetrievalService,
    llm_clients: LLMClients,
    config: RetrievalConfig,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    builder = StateGraph(RetrievalState, input_schema=RetrievalInput, output_schema=RetrievalOutput)

    builder.add_node("understand_request", understand_request_node(service))  # type: ignore[call-overload]
    builder.add_node("plan_retrieval", plan_retrieval_node(llm_clients.planner, config))  # type: ignore[call-overload]
    builder.add_node("execute_searches", execute_searches_node(service, config))  # type: ignore[call-overload]
    builder.add_node("merge_candidates", merge_candidates_node(service, config))  # type: ignore[call-overload]
    builder.add_node(  # type: ignore[call-overload]
        "evaluate_candidates", evaluate_candidates_node(llm_clients.evaluator, config)
    )
    builder.add_node("read_evidence", read_evidence_node(service))  # type: ignore[call-overload]
    builder.add_node(  # type: ignore[call-overload]
        "update_evidence_ledger", update_evidence_ledger_node(service, config)
    )
    builder.add_node(  # type: ignore[call-overload]
        "evaluate_sufficiency", evaluate_sufficiency_node(llm_clients.sufficiency, config)
    )
    builder.add_node("refine_strategy", refine_strategy_node())  # type: ignore[call-overload]
    builder.add_node("build_evidence_package", build_evidence_package_node())  # type: ignore[call-overload]
    builder.add_node(  # type: ignore[call-overload]
        "synthesize_answer", synthesize_answer_node(llm_clients.synthesizer, config)
    )

    builder.add_edge(START, "understand_request")
    builder.add_edge("understand_request", "plan_retrieval")
    builder.add_edge("plan_retrieval", "execute_searches")
    builder.add_edge("execute_searches", "merge_candidates")
    builder.add_edge("merge_candidates", "evaluate_candidates")
    builder.add_edge("evaluate_candidates", "read_evidence")
    builder.add_edge("read_evidence", "update_evidence_ledger")
    builder.add_edge("update_evidence_ledger", "evaluate_sufficiency")
    builder.add_conditional_edges(
        "evaluate_sufficiency",
        route_after_sufficiency,
        {
            "build_evidence_package": "build_evidence_package",
            "refine_strategy": "refine_strategy",
        },
    )
    builder.add_edge("refine_strategy", "execute_searches")
    builder.add_edge("build_evidence_package", "synthesize_answer")
    builder.add_edge("synthesize_answer", END)

    if checkpointer is not None:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()
