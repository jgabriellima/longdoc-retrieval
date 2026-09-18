import uuid

from langgraph.checkpoint.memory import MemorySaver

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.package import EvidencePackage
from longdoc_retrieval.graph.graph import build_retrieval_graph
from longdoc_retrieval.graph.llm import LLMClients


async def run_retrieval(
    service: RetrievalService,
    document_id: str,
    question: str,
    config: RetrievalConfig | None = None,
    llm_clients: LLMClients | None = None,
) -> EvidencePackage:
    config = config or RetrievalConfig()
    llm_clients = llm_clients or LLMClients.from_config(config)
    graph = build_retrieval_graph(service, llm_clients, config, checkpointer=MemorySaver())

    request_id = str(uuid.uuid4())
    initial_state = {
        "request_id": request_id,
        "document_id": document_id,
        "question": question,
    }
    final_state = await graph.ainvoke(
        initial_state, config={"configurable": {"thread_id": request_id}}
    )
    package = final_state["evidence_package"]
    if package is None:
        raise RuntimeError("graph completed without producing an EvidencePackage")
    return package
