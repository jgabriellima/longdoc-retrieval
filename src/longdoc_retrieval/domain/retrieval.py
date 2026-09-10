from typing import Literal

from pydantic import BaseModel

RetrievalMethod = Literal["sparse", "exact", "structural", "expanded"]


class RetrievalCandidate(BaseModel):
    """A raw hit from search()/find_exact() - not yet read in full, not yet
    judged relevant (see `Evidence` for that).

    candidate_id is deterministic:
    f"{retrieval_method}:{document_id}:{match_ordinal}:{query_hash8}" - this
    is also the identity key candidate fusion (deduping/merging overlapping
    hits from different queries) needs, so the format is fixed here.

    lexical_score's sign/scale is engine-specific (this repo negates
    SQLite's bm25() so that higher always means "more relevant", matching
    Tantivy's native convention); the `SparseRetriever` Protocol abstracts
    the search engine but not the numeric meaning of the score, so a future
    engine swap must re-normalize this the same way.

    node_id is the deepest common ancestor when a candidate spans multiple
    merged sibling paragraphs/retrieval units - a known, accepted precision
    loss (the exact span is still recoverable via start_offset/end_offset).
    """

    candidate_id: str

    document_id: str
    node_id: str

    start_offset: int
    end_offset: int

    start_page: int | None = None
    end_page: int | None = None

    retrieval_method: RetrievalMethod

    retrieval_query: str

    lexical_score: float | None = None

    token_count: int

    preview: str
