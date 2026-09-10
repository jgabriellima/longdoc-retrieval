from pydantic import BaseModel, Field


class RetrievalPlan(BaseModel):
    """What the planner decided to search for: lexical queries, exact
    terms/identifiers, and structural hints, derived from the question.
    """

    objective: str

    concepts: list[str] = Field(default_factory=list)
    exact_terms: list[str] = Field(default_factory=list)
    structural_hints: list[str] = Field(default_factory=list)

    evidence_types: list[str] = Field(default_factory=list)

    queries: list[str] = Field(default_factory=list)
