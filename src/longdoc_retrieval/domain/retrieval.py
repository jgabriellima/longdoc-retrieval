"""
Retrieval domain model.
"""
from typing import Literal

from pydantic import BaseModel

RetrievalMethod = Literal["sparse", "exact", "structural", "expanded"]


class RetrievalCandidate(BaseModel):
    candidate_id: str
    document_id: str
    node_id: str
    start_offset: int
    end_offset: int
    start_page: int | None = None
    end_page: int | None = None
    retrieval_method: RetrievalMethod
    retrieval_query: str
    lexical_score: float | None = None
    token_count: int
    preview: str
