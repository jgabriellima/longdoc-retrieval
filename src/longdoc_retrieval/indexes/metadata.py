"""
Metadata for the retrieval indexes.
"""
import sqlite3

from longdoc_retrieval.domain.document import DocumentMetadata
from longdoc_retrieval.indexes.structural import get_document_metadata


def get_metadata(conn: sqlite3.Connection, document_id: str) -> DocumentMetadata:
    return get_document_metadata(conn, document_id)
