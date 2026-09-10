"""Thin accessor over documents.metadata (the DocumentMetadata JSON blob).

Kept separate from structural.py's node/document storage so callers that
only care about metadata (e.g. an authorization check on `source`) don't
need to import the structural-tree query surface.
"""

import sqlite3

from longdoc_retrieval.domain.document import DocumentMetadata
from longdoc_retrieval.indexes.structural import get_document_metadata


def get_metadata(conn: sqlite3.Connection, document_id: str) -> DocumentMetadata:
    return get_document_metadata(conn, document_id)
