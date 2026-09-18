"""
Sufficiency domain model.
"""
from pydantic import BaseModel, Field


class SufficiencyDecision(BaseModel):
    sufficient: bool
    confidence: float
    missing_information: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    recommended_queries: list[str] = Field(default_factory=list)
