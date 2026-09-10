"""should_stop(): a single, pure source of truth for every stop condition,
so `router.py` (which decides whether to loop) and `package.py` (which
reports `stop_reason`) can never disagree.
"""

import time

from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.graph.state import RetrievalState


def should_stop(state: RetrievalState, config: RetrievalConfig) -> tuple[bool, str | None]:
    if state["sufficient"]:
        return True, "sufficient"

    if state["iteration"] >= config.max_iterations:
        return True, "max_iterations"

    metrics = state["metrics"]
    if metrics.evidence_tokens >= config.max_evidence_tokens:
        return True, "evidence_token_budget"

    if metrics.llm_calls >= config.max_total_llm_calls:
        return True, "llm_call_budget"

    if time.monotonic() - state["started_at"] >= config.max_time_seconds:
        return True, "time_budget"

    history = state["evidence_count_history"]
    if len(history) >= 2 and history[-1] == history[-2]:
        return True, "no_new_information"

    return False, None
