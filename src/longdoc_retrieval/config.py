"""Loop budgets and per-component model names. Each field maps to a stop
condition in graph/budgets.py.
"""

import os

from pydantic import BaseModel

# Used by from_env() when only OPENAI_API_KEY is set.
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
    # Sufficiency issues ~5-7 LLM calls per iteration (1 evaluator + 1
    # isolated extraction per evidence item + 1 signal). 30 covers a full
    # max_iterations=6 run at typical evidence counts.
    max_total_llm_calls: int = 10
    # Wall-clock ceiling for the full multi-iteration EvidencePackage
    # build. ~20s/call headroom at the previous 12-call budget; not a
    # latency target for single search()/find_exact() calls.
    max_time_seconds: float = 240.0

    # Judgment (planner, sufficiency, synthesizer) vs batch classification
    # (evaluator).
    planner_model: str = "claude-sonnet-5"
    evaluator_model: str = "claude-haiku-4-5-20251001"
    sufficiency_model: str = "claude-sonnet-5"
    synthesizer_model: str = "claude-sonnet-5"

    # Parallel extraction samples per evidence item (graph/sufficiency.py).
    # Isolation is the primary defense against cross-item contamination;
    # raise this only if a deployment still sees per-item sampling variance.
    sufficiency_samples: int = 1

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Anthropic defaults unless only OPENAI_API_KEY is set."""

        if os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("OPENAI_API_KEY"):
            return cls()
        return cls(
            planner_model=_OPENAI_PLANNER_MODEL,
            evaluator_model=_OPENAI_EVALUATOR_MODEL,
            sufficiency_model=_OPENAI_SUFFICIENCY_MODEL,
            synthesizer_model=_OPENAI_SYNTHESIZER_MODEL,
        )
