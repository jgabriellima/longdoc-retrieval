import json
import sqlite3
from dataclasses import dataclass

from longdoc_retrieval.domain.document import Document, DocumentMetadata
from longdoc_retrieval.domain.node import DocumentNode
from longdoc_retrieval.ingestion.node_builder import RetrievalUnit

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    parent_id TEXT,
    title TEXT,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    start_page INTEGER,
    end_page INTEGER,
    token_count INTEGER NOT NULL,
    depth INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    children_ids TEXT NOT NULL,
    metadata TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_nodes_doc_span
    ON nodes (document_id, start_offset, end_offset);
CREATE INDEX IF NOT EXISTS idx_nodes_doc_parent
    ON nodes (document_id, parent_id);

CREATE TABLE IF NOT EXISTS retrieval_units (
    unit_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    token_count INTEGER NOT NULL,
    ordinal INTEGER NOT NULL,
    truncated_split INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_units_doc_ordinal
    ON retrieval_units (document_id, ordinal);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_document(conn: sqlite3.Connection, document: Document) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO documents (document_id, content, metadata) VALUES (?, ?, ?)",
        (document.document_id, document.content, document.metadata.model_dump_json()),
    )
    conn.commit()


def insert_nodes(conn: sqlite3.Connection, nodes: list[DocumentNode]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO nodes
            (node_id, document_id, parent_id, title, start_offset, end_offset,
             start_page, end_page, token_count, depth, ordinal, children_ids, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                n.node_id,
                n.document_id,
                n.parent_id,
                n.title,
                n.start_offset,
                n.end_offset,
                n.start_page,
                n.end_page,
                n.token_count,
                n.depth,
                idx,
                json.dumps(n.children_ids),
                json.dumps(n.metadata),
            )
            for idx, n in enumerate(nodes)
        ],
    )
    conn.commit()


def insert_retrieval_units(conn: sqlite3.Connection, units: list[RetrievalUnit]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO retrieval_units
            (unit_id, document_id, node_id, start_offset, end_offset,
             token_count, ordinal, truncated_split)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                u.unit_id,
                u.document_id,
                u.node_id,
                u.start_offset,
                u.end_offset,
                u.token_count,
                u.ordinal,
                int(u.truncated_split),
            )
            for u in units
        ],
    )
    conn.commit()


def _row_to_node(row: sqlite3.Row) -> DocumentNode:
    return DocumentNode(
        node_id=row["node_id"],
        document_id=row["document_id"],
        parent_id=row["parent_id"],
        children_ids=json.loads(row["children_ids"]),
        title=row["title"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        start_page=row["start_page"],
        end_page=row["end_page"],
        token_count=row["token_count"],
        depth=row["depth"],
        metadata=json.loads(row["metadata"]),
    )


def get_document_content(conn: sqlite3.Connection, document_id: str) -> str:
    row = conn.execute(
        "SELECT content FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"document not found: {document_id}")
    return row["content"]


def get_document_metadata(conn: sqlite3.Connection, document_id: str) -> DocumentMetadata:
    row = conn.execute(
        "SELECT metadata FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"document not found: {document_id}")
    return DocumentMetadata.model_validate_json(row["metadata"])


def get_node(conn: sqlite3.Connection, document_id: str, node_id: str) -> DocumentNode | None:
    row = conn.execute(
        "SELECT * FROM nodes WHERE document_id = ? AND node_id = ?", (document_id, node_id)
    ).fetchone()
    return _row_to_node(row) if row else None


def get_root(conn: sqlite3.Connection, document_id: str) -> DocumentNode:
    row = conn.execute(
        "SELECT * FROM nodes WHERE document_id = ? AND parent_id IS NULL", (document_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"no root node for document: {document_id}")
    return _row_to_node(row)


def get_children(conn: sqlite3.Connection, document_id: str, node_id: str) -> list[DocumentNode]:
    rows = conn.execute(
        "SELECT * FROM nodes WHERE document_id = ? AND parent_id = ? ORDER BY ordinal",
        (document_id, node_id),
    ).fetchall()
    return [_row_to_node(r) for r in rows]


def get_all_nodes(conn: sqlite3.Connection, document_id: str) -> list[DocumentNode]:
    rows = conn.execute(
        "SELECT * FROM nodes WHERE document_id = ? ORDER BY ordinal", (document_id,)
    ).fetchall()
    return [_row_to_node(r) for r in rows]


def get_top_level_ancestor(conn: sqlite3.Connection, document_id: str, node_id: str) -> str:
    current = get_node(conn, document_id, node_id)
    if current is None:
        return node_id
    while current.parent_id is not None and current.depth > 1:
        parent = get_node(conn, document_id, current.parent_id)
        if parent is None:
            break
        current = parent
    return current.node_id


def resolve_node_at_offset(conn: sqlite3.Connection, document_id: str, offset: int) -> str | None:
    row = conn.execute(
        """
        SELECT node_id FROM nodes
        WHERE document_id = ? AND start_offset <= ? AND end_offset > ?
        ORDER BY depth DESC LIMIT 1
        """,
        (document_id, offset, offset),
    ).fetchone()
    return row["node_id"] if row else None


def get_retrieval_unit(
    conn: sqlite3.Connection, document_id: str, unit_id: str
) -> RetrievalUnit | None:
    row = conn.execute(
        "SELECT * FROM retrieval_units WHERE document_id = ? AND unit_id = ?",
        (document_id, unit_id),
    ).fetchone()
    if row is None:
        return None
    return RetrievalUnit(
        unit_id=row["unit_id"],
        document_id=row["document_id"],
        node_id=row["node_id"],
        start_offset=row["start_offset"],
        end_offset=row["end_offset"],
        token_count=row["token_count"],
        ordinal=row["ordinal"],
        truncated_split=bool(row["truncated_split"]),
    )


def count_units_for_node(conn: sqlite3.Connection, document_id: str, node_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM retrieval_units WHERE document_id = ? AND node_id = ?",
        (document_id, node_id),
    ).fetchone()
    return row["n"] if row else 0


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    source: str | None
    token_count: int


def document_exists(conn: sqlite3.Connection, document_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()
    return row is not None


def list_documents(conn: sqlite3.Connection) -> list[StoredDocument]:
    rows = conn.execute(
        "SELECT document_id, metadata FROM documents ORDER BY document_id"
    ).fetchall()
    documents: list[StoredDocument] = []
    for row in rows:
        metadata = DocumentMetadata.model_validate_json(row["metadata"])
        root = conn.execute(
            "SELECT token_count FROM nodes WHERE document_id = ? AND parent_id IS NULL",
            (row["document_id"],),
        ).fetchone()
        documents.append(
            StoredDocument(
                document_id=row["document_id"],
                source=metadata.source,
                token_count=int(root["token_count"]) if root else 0,
            )
        )
    return documents


def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
    conn.execute("DELETE FROM retrieval_units WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM nodes WHERE document_id = ?", (document_id,))
    conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
    conn.commit()
