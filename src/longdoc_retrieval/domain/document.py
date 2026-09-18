"""
Document domain model.
"""
from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    source: str | None = None
    pages: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    document_id: str
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
