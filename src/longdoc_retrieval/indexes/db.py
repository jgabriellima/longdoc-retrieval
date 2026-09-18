import sqlite3

from longdoc_retrieval.indexes import sparse, structural


def connect(path: str = ":memory:") -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    structural.create_schema(conn)
    sparse.create_schema(conn)
    return conn
