from typing import Literal

from pydantic import BaseModel, Field

from longdoc_retrieval.domain.evidence import Evidence
from longdoc_retrieval.domain.metrics import RetrievalMetrics


class EvidencePackage(BaseModel):
    """The retrieval graph's final output - what the calling application
    receives. Never the document itself, and never anything the graph
    didn't actually read: retrieval failure is a valid result
    (`status="insufficient"`), evidence is never fabricated.

    `answer` extends that same guarantee one step further: it's a synthesis
    over `evidence` only (never a fresh read of the document), and stays
    `None` rather than being fabricated whenever there's no evidence to
    synthesize from, or the synthesis call itself fails - exactly the
    no-fabrication policy `evidence`/`status` already follow.
    """

    request_id: str
    document_id: str

    question: str

    status: Literal["sufficient", "partial", "insufficient"]

    evidence: list[Evidence] = Field(default_factory=list)

    unresolved_questions: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)

    retrieval_summary: str

    # The actual answer to `question`, synthesized from `evidence` by
    # `graph/synthesizer.py`. `answer_citations` are `evidence_id`s the
    # answer draws from, verified against `evidence` in code (never trusted
    # blindly from the model's own claim - same pattern as
    # `evaluate_sufficiency`'s substring check).
    answer: str | None = None
    answer_citations: list[str] = Field(default_factory=list)

    metrics: RetrievalMetrics
