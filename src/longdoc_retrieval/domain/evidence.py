"""
Evidence domain model.
"""
from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str
    document_id: str
    node_id: str
    start_offset: int
    end_offset: int
    start_page: int | None = None
    end_page: int | None = None
    text: str
    retrieval_query: str
    retrieval_method: str
    relevance_reason: str
    supports: list[str] = Field(default_factory=list)
    token_count: int
