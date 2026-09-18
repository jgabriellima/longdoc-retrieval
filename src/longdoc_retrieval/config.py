import os

from pydantic import BaseModel

_OPENAI_PLANNER_MODEL = "gpt-5.6-luna"
_OPENAI_EVALUATOR_MODEL = "gpt-5.6-luna"
_OPENAI_SUFFICIENCY_MODEL = "gpt-5.6-luna"
_OPENAI_SYNTHESIZER_MODEL = "gpt-5.6-luna"


class RetrievalConfig(BaseModel):
    max_iterations: int = 6
    max_queries_per_iteration: int = 8
    max_candidates_per_query: int = 10
    max_candidates_for_llm_evaluation: int = 20
    max_evidence_items: int = 10
    max_evidence_tokens: int = 10_000
    max_total_llm_calls: int = 10
    max_time_seconds: float = 240.0

    planner_model: str = "gpt-5.6-luna"
    evaluator_model: str = "gpt-5.6-luna"
    sufficiency_model: str = "gpt-5.6-luna"
    synthesizer_model: str = "gpt-5.6-luna"

    sufficiency_samples: int = 1

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        if os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
            return cls()
        return cls(
            planner_model=_OPENAI_PLANNER_MODEL,
            evaluator_model=_OPENAI_EVALUATOR_MODEL,
            sufficiency_model=_OPENAI_SUFFICIENCY_MODEL,
            synthesizer_model=_OPENAI_SYNTHESIZER_MODEL,
        )
