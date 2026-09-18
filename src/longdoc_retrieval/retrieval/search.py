"""
Search for the retrieval API.
"""
import sqlite3
from typing import TYPE_CHECKING, Protocol

from longdoc_retrieval.domain.node import DocumentNode
from longdoc_retrieval.domain.retrieval import RetrievalCandidate
from longdoc_retrieval.indexes import sparse
from longdoc_retrieval.indexes.structural import get_document_content, get_node
from longdoc_retrieval.ingestion.node_builder import RetrievalUnit
from longdoc_retrieval.retrieval.candidate_id import make_candidate_id
from longdoc_retrieval.tokenize import approx_token_count

if TYPE_CHECKING:
    import tantivy


class SparseRetriever(Protocol):
    async def search(
        self,
        document_id: str,
        query: str,
        limit: int,
    ) -> list[RetrievalCandidate]: ...


class SparseBackend(Protocol):
    def index_units(self, content: str, units: list[RetrievalUnit]) -> None: ...

    async def search(
        self,
        document_id: str,
        query: str,
        limit: int,
    ) -> list[RetrievalCandidate]: ...


def _candidates_from_hits(
    hits: list[sparse.SparseHit],
    conn: sqlite3.Connection,
    document_id: str,
    query: str,
) -> list[RetrievalCandidate]:
    content = get_document_content(conn, document_id) if hits else ""
    candidates = []
    for ordinal, hit in enumerate(hits):
        node: DocumentNode | None = get_node(conn, document_id, hit.node_id)
        candidates.append(
            RetrievalCandidate(
                candidate_id=make_candidate_id("sparse", document_id, ordinal, query),
                document_id=document_id,
                node_id=hit.node_id,
                start_offset=hit.start_offset,
                end_offset=hit.end_offset,
                start_page=node.start_page if node else None,
                end_page=node.end_page if node else None,
                retrieval_method="sparse",
                retrieval_query=query,
                lexical_score=hit.score,
                token_count=approx_token_count(content[hit.start_offset : hit.end_offset]),
                preview=hit.snippet,
            )
        )
    return candidates


class SqliteSparseRetriever:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def index_units(self, content: str, units: list[RetrievalUnit]) -> None:
        sparse.index_units(self._conn, content, units)

    async def search(
        self, document_id: str, query: str, limit: int
    ) -> list[RetrievalCandidate]:
        hits = sparse.search(self._conn, document_id, query, limit)
        return _candidates_from_hits(hits, self._conn, document_id, query)


class TantivySparseRetriever:
    def __init__(self, conn: sqlite3.Connection) -> None:
        from longdoc_retrieval.indexes import sparse_tantivy

        self._conn = conn
        self._backend = sparse_tantivy
        self._index: tantivy.Index = sparse_tantivy.build_index()

    def index_units(self, content: str, units: list[RetrievalUnit]) -> None:
        self._backend.index_units(self._index, content, units)

    async def search(
        self, document_id: str, query: str, limit: int
    ) -> list[RetrievalCandidate]:
        hits = self._backend.search(self._index, document_id, query, limit)
        return _candidates_from_hits(hits, self._conn, document_id, query)
