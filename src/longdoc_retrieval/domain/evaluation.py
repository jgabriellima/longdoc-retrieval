"""
Evaluation domain model.
"""
from typing import Literal

from pydantic import BaseModel


class EvaluatedCandidate(BaseModel):
    candidate_id: str
    relevance: Literal["irrelevant", "possibly_relevant", "relevant", "critical"]
    reason: str
    should_read: bool
