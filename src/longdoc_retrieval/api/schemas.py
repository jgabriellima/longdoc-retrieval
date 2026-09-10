"""Response DTOs for the deterministic retrieval API. Kept out of domain/
because nothing outside the API layer needs them - they exist purely to
shape what `RetrievalService` returns without ever exposing full document
text by default.
"""

from pydantic import BaseModel

from longdoc_retrieval.retrieval.reader import ReadResult

__all__ = ["DocumentOutline", "NodeSummary", "OutlineNode", "ReadResult"]


class OutlineNode(BaseModel):
    node_id: str
    title: str | None
    depth: int
    token_count: int
    children: list["OutlineNode"] = []


class DocumentOutline(BaseModel):
    document_id: str
    root: OutlineNode


class NodeSummary(BaseModel):
    node_id: str
    title: str | None
    depth: int
    start_offset: int
    end_offset: int
    start_page: int | None
    end_page: int | None
    token_count: int
    unit_count: int
    preview: str
