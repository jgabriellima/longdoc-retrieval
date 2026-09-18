"""
State for the retrieval graph.
"""
from typing import Any, TypedDict

from pydantic import BaseModel

from longdoc_retrieval.domain.evaluation import EvaluatedCandidate
from longdoc_retrieval.domain.evidence import Evidence
from longdoc_retrieval.domain.metrics import RetrievalMetrics
from longdoc_retrieval.domain.package import EvidencePackage
from longdoc_retrieval.domain.plan import RetrievalPlan
from longdoc_retrieval.domain.retrieval import RetrievalCandidate
from longdoc_retrieval.domain.sufficiency import SufficiencyDecision


class SearchExecution(BaseModel):
    query: str
    retrieval_method: str
    candidate_count: int
    iteration: int


class RetrievalState(TypedDict):
    request_id: str
    document_id: str
    question: str
    plan: RetrievalPlan | None
    iteration: int
    executed_queries: list[SearchExecution]
    candidates: list[RetrievalCandidate]
    evaluated_candidates: list[EvaluatedCandidate]
    evidence: list[Evidence]
    unresolved_questions: list[str]
    sufficient: bool
    stop_reason: str | None
    metrics: RetrievalMetrics
    outline_summary: str
    document_metadata: dict[str, Any]
    last_sufficiency: SufficiencyDecision | None
    contradictions: list[str]
    pending_evidence: list[Evidence]
    evidence_package: EvidencePackage | None
    started_at: float
    evidence_count_history: list[int]


class RetrievalInput(TypedDict):
    request_id: str
    document_id: str
    question: str


class RetrievalOutput(TypedDict):
    evidence_package: EvidencePackage | None
