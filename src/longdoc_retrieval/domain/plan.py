"""
Retrieval plan domain model.
"""
from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    objective: str
    concepts: list[str] = Field(default_factory=list)
    exact_terms: list[str] = Field(default_factory=list)
    structural_hints: list[str] = Field(default_factory=list)
    evidence_types: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
