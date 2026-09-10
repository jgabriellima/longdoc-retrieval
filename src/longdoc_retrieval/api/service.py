"""RetrievalService: the deterministic retrieval API's 7 methods
(document_outline, search, find_exact, inspect_node, expand_node, read_node,
read_range), as an async facade over the ingestion/index/retrieval layers.

Methods are `async def` to match the SparseRetriever Protocol convention,
but the SQLite calls underneath are synchronous - ops here are ms-scale,
and concurrent fan-out across queries is the agentic loop's concern
(graph/search.py), not this layer's. `document_id` scoping is trusted, not
authorized, by this layer - callers must have already checked access.
"""

import sqlite3

from longdoc_retrieval.api.schemas import DocumentOutline, NodeSummary, OutlineNode
from longdoc_retrieval.domain.document import Document, DocumentMetadata
from longdoc_retrieval.domain.retrieval import RetrievalCandidate
from longdoc_retrieval.indexes.structural import (
    count_units_for_node,
    get_children,
    get_document_metadata,
    get_node,
    get_root,
    get_top_level_ancestor,
)
from longdoc_retrieval.ingestion.indexer import ingest_document
from longdoc_retrieval.retrieval.exact import find_exact as _find_exact
from longdoc_retrieval.retrieval.reader import ReadResult
from longdoc_retrieval.retrieval.reader import read_node as _read_node
from longdoc_retrieval.retrieval.reader import read_range as _read_range
from longdoc_retrieval.retrieval.search import SparseBackend, SqliteSparseRetriever

INSPECT_PREVIEW_CHARS = 300


class RetrievalService:
    def __init__(self, conn: sqlite3.Connection, sparse: SparseBackend | None = None) -> None:
        self._conn = conn
        self._sparse: SparseBackend = sparse if sparse is not None else SqliteSparseRetriever(conn)

    def ingest(self, document: Document) -> None:
        ingest_document(self._conn, document, self._sparse)

    async def get_metadata(self, document_id: str) -> DocumentMetadata:
        return get_document_metadata(self._conn, document_id)

    async def get_top_level_ancestor(self, document_id: str, node_id: str) -> str:
        return get_top_level_ancestor(self._conn, document_id, node_id)

    async def document_outline(self, document_id: str, depth: int | None = None) -> DocumentOutline:
        root = get_root(self._conn, document_id)
        outline_root = self._build_outline(document_id, root, depth)
        return DocumentOutline(document_id=document_id, root=outline_root)

    def _build_outline(self, document_id: str, node, depth: int | None) -> OutlineNode:
        children = []
        if depth is None or node.depth < depth:
            for child in get_children(self._conn, document_id, node.node_id):
                children.append(self._build_outline(document_id, child, depth))
        return OutlineNode(
            node_id=node.node_id,
            title=node.title,
            depth=node.depth,
            token_count=node.token_count,
            children=children,
        )

    async def search(self, document_id: str, query: str, limit: int = 20) -> list[RetrievalCandidate]:
        return await self._sparse.search(document_id, query, limit)

    async def find_exact(
        self, document_id: str, expression: str, limit: int = 20
    ) -> list[RetrievalCandidate]:
        return _find_exact(self._conn, document_id, expression, limit)

    async def inspect_node(self, document_id: str, node_id: str) -> NodeSummary:
        node = get_node(self._conn, document_id, node_id)
        if node is None:
            raise KeyError(f"node not found: {document_id}/{node_id}")
        preview = _read_range(
            self._conn, document_id, node.start_offset, node.end_offset, token_limit=None
        ).text[:INSPECT_PREVIEW_CHARS]
        return NodeSummary(
            node_id=node.node_id,
            title=node.title,
            depth=node.depth,
            start_offset=node.start_offset,
            end_offset=node.end_offset,
            start_page=node.start_page,
            end_page=node.end_page,
            token_count=node.token_count,
            unit_count=count_units_for_node(self._conn, document_id, node_id),
            preview=preview,
        )

    async def expand_node(self, document_id: str, node_id: str) -> list[NodeSummary]:
        children = get_children(self._conn, document_id, node_id)
        return [await self.inspect_node(document_id, child.node_id) for child in children]

    async def read_node(
        self, document_id: str, node_id: str, token_limit: int | None = None
    ) -> ReadResult:
        return _read_node(self._conn, document_id, node_id, token_limit=token_limit)

    async def read_range(self, document_id: str, start_offset: int, end_offset: int) -> ReadResult:
        return _read_range(self._conn, document_id, start_offset, end_offset)
