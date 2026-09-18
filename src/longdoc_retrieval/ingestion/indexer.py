import sqlite3
from typing import TYPE_CHECKING

from longdoc_retrieval.domain.document import Document
from longdoc_retrieval.indexes import sparse, structural
from longdoc_retrieval.ingestion.node_builder import build_nodes, build_retrieval_units
from longdoc_retrieval.ingestion.normalizer import normalize
from longdoc_retrieval.ingestion.structure_parser import parse_structure

if TYPE_CHECKING:
    from longdoc_retrieval.retrieval.search import SparseBackend


def ingest_document(
    conn: sqlite3.Connection,
    document: Document,
    sparse_backend: "SparseBackend | None" = None,
) -> None:
    normalized_content = normalize(document.content)
    normalized_document = document.model_copy(update={"content": normalized_content})

    root = parse_structure(normalized_content)
    nodes = build_nodes(document.document_id, normalized_content, root)
    units = build_retrieval_units(document.document_id, normalized_content, nodes)

    structural.insert_document(conn, normalized_document)
    structural.insert_nodes(conn, nodes)
    structural.insert_retrieval_units(conn, units)

    if sparse_backend is not None:
        sparse_backend.index_units(normalized_content, units)
    else:
        sparse.index_units(conn, normalized_content, units)
