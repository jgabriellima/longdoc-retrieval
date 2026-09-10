"""RetrievalState: everything the retrieval graph threads between nodes.
Holds structured/raw data only, never concatenated prompts, and never the
full document text, putting the document itself in state would recreate
the same context-growth problem this architecture exists to avoid.
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
    """One executed query, kept for observability (query_count) and for
    detecting "no new information discovered" across iterations.
    """

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

    # Planner input, computed once by `understand_request` and reused
    # across iterations rather than recomputed every loop.
    outline_summary: str
    document_metadata: dict[str, Any]

    # Full sufficiency verdict, not just the flattened `sufficient` bool -
    # `refine_strategy` needs `recommended_queries`, and `contradictions`
    # needs to survive into the final `EvidencePackage`.
    last_sufficiency: SufficiencyDecision | None
    contradictions: list[str]

    # Transient handoff between `read_evidence` and `update_evidence_ledger`
    # (kept as two separate graph nodes) - freshly-read Evidence not yet
    # merged/deduped into the ledger. Always cleared by
    # `update_evidence_ledger` in the same step it's produced.
    pending_evidence: list[Evidence]

    # Set only by the terminal `build_evidence_package` node.
    evidence_package: EvidencePackage | None

    # Wall-clock budget and "no new evidence" stop condition - both are pure
    # functions of state history, computed here rather than as ad hoc checks
    # scattered across nodes.
    started_at: float
    evidence_count_history: list[int]


class RetrievalInput(TypedDict):
    """External input to the graph: everything a caller must supply.

    Every other `RetrievalState` field is initialized by
    `understand_request`, the first node, so a caller (or the LangGraph
    Studio input form) only ever needs to provide these three.
    """

    request_id: str
    document_id: str
    question: str


class RetrievalOutput(TypedDict):
    """External output of the graph: what a caller actually needs back."""

    evidence_package: EvidencePackage | None
