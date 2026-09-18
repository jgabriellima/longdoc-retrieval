"""
Ingest module for the retrieval API.
"""
from dataclasses import dataclass
from pathlib import Path

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.app.errors import UserError
from longdoc_retrieval.domain.document import Document, DocumentMetadata

_PDF_SUFFIXES = {".pdf"}


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    replaced: bool
    ingested: bool


def document_id_from_path(path: Path, override: str | None) -> str:
    if override:
        return override
    return path.stem


def read_text_file(path: Path) -> str:
    if not path.is_file():
        raise UserError(f"file not found: {path}")
    if path.suffix.lower() in _PDF_SUFFIXES:
        raise UserError("PDF is not supported. convert to UTF-8 text (.txt or .md)")
    data = path.read_bytes()
    if b"\x00" in data:
        raise UserError(f"binary file rejected: {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UserError(f"file is not UTF-8: {path}") from exc


def ingest_file(service: RetrievalService, path: Path, document_id: str) -> IngestResult:
    content = read_text_file(path)
    replaced = service.has_document(document_id)
    if replaced:
        service.delete_document(document_id)
    service.ingest(
        Document(
            document_id=document_id,
            content=content,
            metadata=DocumentMetadata(source=str(path)),
        )
    )
    return IngestResult(document_id=document_id, replaced=replaced, ingested=True)


def ingest_if_needed(service: RetrievalService, path: Path, document_id: str) -> IngestResult:
    if service.has_document(document_id):
        return IngestResult(document_id=document_id, replaced=False, ingested=False)
    return ingest_file(service, path, document_id)
