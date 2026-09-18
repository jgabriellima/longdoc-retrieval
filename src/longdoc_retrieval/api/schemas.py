"""
Schemas for the retrieval API endpoints.
"""
from pydantic import BaseModel

from longdoc_retrieval.retrieval.reader import ReadResult

__all__ = ["DocumentOutline", "DocumentRecord", "NodeSummary", "OutlineNode", "ReadResult"]


class DocumentRecord(BaseModel):
    document_id: str
    source: str | None = None
    token_count: int


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
