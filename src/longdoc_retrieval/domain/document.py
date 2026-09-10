from typing import Any

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Free-form metadata carried through from the upstream parser."""

    source: str | None = None
    pages: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """The canonical input to ingestion: an id, raw text, and metadata."""

    document_id: str
    content: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
