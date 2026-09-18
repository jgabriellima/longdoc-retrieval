"""
Evidence package domain model.
"""
from typing import Literal

from pydantic import BaseModel, Field

from longdoc_retrieval.domain.evidence import Evidence
from longdoc_retrieval.domain.metrics import RetrievalMetrics


class EvidencePackage(BaseModel):
    request_id: str
    document_id: str
    question: str
    status: Literal["sufficient", "partial", "insufficient"]
    evidence: list[Evidence] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    retrieval_summary: str
    answer: str | None = None
    answer_citations: list[str] = Field(default_factory=list)
    metrics: RetrievalMetrics
