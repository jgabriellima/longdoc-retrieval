from pydantic import BaseModel, Field


class Evidence(BaseModel):
    """A candidate that was actually read in full and judged relevant to the
    question, as opposed to `RetrievalCandidate` (a raw search hit that
    hasn't been read yet). `relevance_reason`/`supports` require an LLM
    judgment, so this type is only ever populated by the agentic retrieval
    loop (`graph/`), never by the deterministic search layer on its own.
    """

    evidence_id: str

    document_id: str
    node_id: str

    start_offset: int
    end_offset: int

    start_page: int | None = None
    end_page: int | None = None

    text: str

    retrieval_query: str
    retrieval_method: str

    relevance_reason: str

    supports: list[str] = Field(default_factory=list)

    token_count: int
