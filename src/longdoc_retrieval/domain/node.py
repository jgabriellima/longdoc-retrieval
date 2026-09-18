from typing import Any

from pydantic import BaseModel, Field


class DocumentNode(BaseModel):
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
