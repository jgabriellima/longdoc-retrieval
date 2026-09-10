from typing import Any

from pydantic import BaseModel, Field


class DocumentNode(BaseModel):
    """One node in a document's structural tree (a heading/section and the
    text under it, before that text is split into retrieval units).

    node_id is deterministic (f"{document_id}#{ordinal:06d}" from a pre-order
    traversal), not a random UUID, so repeated ingestion of the same document
    is idempotent - re-ingesting doesn't create duplicate nodes.

    token_count on a non-leaf node is the cumulative subtree count, not just
    this node's own heading/text - a top-level section can span tens of
    thousands of tokens across all its children, and that total (not the
    heading's own few words) is what drives the retrieval-unit chunking
    decision.
    """

    node_id: str
    document_id: str

    parent_id: str | None
    children_ids: list[str] = Field(default_factory=list)

    title: str | None

    start_offset: int
    end_offset: int

    start_page: int | None = None
    end_page: int | None = None

    token_count: int

    depth: int

    metadata: dict[str, Any] = Field(default_factory=dict)
