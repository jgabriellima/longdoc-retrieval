from pydantic import BaseModel, Field


class SufficiencyDecision(BaseModel):
    """Whether the evidence gathered so far answers the question. Not
    merely a confidence score - the model must explicitly name what's still
    missing, which is what `refine_strategy` turns into the next
    iteration's queries without re-invoking the planner.
    """

    sufficient: bool

    confidence: float

    missing_information: list[str] = Field(default_factory=list)

    contradictions: list[str] = Field(default_factory=list)

    recommended_queries: list[str] = Field(default_factory=list)
