from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.document import Document
from longdoc_retrieval.graph.graph import build_retrieval_graph
from longdoc_retrieval.graph.llm import LLMClients
from longdoc_retrieval.indexes.db import connect

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLE_CORPUS_DIR = REPO_ROOT / "examples/legalbench_rag/corpus"
LOCAL_DOCS_DIR = REPO_ROOT / "docs/evaluation"

LOCAL_DOCUMENTS = {
    "convenio_169_2022": "TERMO_CONVENIO_169-2022_OCR.md",
}


def _ingest_examples(service: RetrievalService) -> None:
    for path in sorted(EXAMPLE_CORPUS_DIR.rglob("*.txt")):
        document_id = str(path.relative_to(EXAMPLE_CORPUS_DIR))
        service.ingest(Document(document_id=document_id, content=path.read_text(encoding="utf-8")))


def _ingest_local_docs(service: RetrievalService) -> None:
    if not LOCAL_DOCS_DIR.is_dir():
        return
    for document_id, filename in LOCAL_DOCUMENTS.items():
        path = LOCAL_DOCS_DIR / filename
        if path.is_file():
            service.ingest(Document(document_id=document_id, content=path.read_text(encoding="utf-8")))


def make_graph() -> CompiledStateGraph:
    conn = connect()
    service = RetrievalService(conn)
    _ingest_examples(service)
    _ingest_local_docs(service)

    config = RetrievalConfig.from_env()
    llm_clients = LLMClients.from_config(config)

    return build_retrieval_graph(service, llm_clients, config)
