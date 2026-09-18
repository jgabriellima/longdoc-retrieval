"""
Database for the retrieval indexes.
"""
import sqlite3

from longdoc_retrieval.indexes.sparse import create_schema as create_sparse_schema
from longdoc_retrieval.indexes.structural import create_schema as create_structural_schema


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    create_structural_schema(conn)
    create_sparse_schema(conn)
    return conn
