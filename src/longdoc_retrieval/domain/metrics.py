"""
Metrics domain model.
"""
from pydantic import BaseModel


class RetrievalMetrics(BaseModel):
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
    retrieval_latency_ms: float = 0.0
    model: str = ""
    stop_reason: str | None = None
