from pydantic import BaseModel


class RetrievalMetrics(BaseModel):
    """Observability and cost/efficiency attributes tracked per request.
    Populated incrementally as the graph runs rather than computed after
    the fact, so a request that stops early (budget/error) still reports
    accurate partial metrics.
    """

    document_id: str
    document_tokens: int
    question_tokens: int

    iterations: int = 0

    query_count: int = 0
    candidate_count: int = 0
    evidence_count: int = 0

    retrieved_tokens: int = 0
    evidence_tokens: int = 0

    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0

    latency_ms: float = 0.0
    # Sum of individual service.search()/find_exact() call durations across
    # every iteration (graph/search.py) - NOT wall-clock time, since calls
    # within one iteration run concurrently via asyncio.gather. This isolates
    # the sparse-retrieval backend's own cost from LLM latency, which
    # dominates `latency_ms` and would otherwise hide any difference between
    # backends (e.g. SQLite FTS5 vs Tantivy).
    retrieval_latency_ms: float = 0.0

    model: str = ""
    stop_reason: str | None = None
