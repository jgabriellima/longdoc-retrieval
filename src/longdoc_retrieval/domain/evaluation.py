from typing import Literal

from pydantic import BaseModel


class EvaluatedCandidate(BaseModel):
    """Verdict on one search/exact-match candidate: worth reading in full, or
    not. Produced in batches (one LLM call per fused candidate set, not one
    call per candidate) over candidate previews - never full node text, so
    judging relevance stays cheap even when a search turns up many hits.
    """

    candidate_id: str

    relevance: Literal["irrelevant", "possibly_relevant", "relevant", "critical"]

    reason: str

    should_read: bool
