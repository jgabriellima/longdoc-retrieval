"""
Sparse index for the retrieval API.
"""
import sqlite3

from longdoc_retrieval.ingestion.node_builder import RetrievalUnit
from longdoc_retrieval.tokenize import tokenize

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS retrieval_units_fts USING fts5(
    content,
    unit_id UNINDEXED,
    document_id UNINDEXED,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(FTS_SCHEMA_SQL)
    conn.commit()


def delete_document(conn: sqlite3.Connection, document_id: str) -> None:
    rows = conn.execute(
        "SELECT rowid FROM retrieval_units_fts WHERE document_id = ?",
        (document_id,),
    ).fetchall()
    for row in rows:
        conn.execute("DELETE FROM retrieval_units_fts WHERE rowid = ?", (row["rowid"],))
    conn.commit()


def index_units(conn: sqlite3.Connection, document_text: str, units: list[RetrievalUnit]) -> None:
    conn.executemany(
        "INSERT INTO retrieval_units_fts (content, unit_id, document_id) VALUES (?, ?, ?)",
        [
            (
                document_text[u.start_offset : u.end_offset],
                u.unit_id,
                u.document_id,
            )
            for u in units
        ],
    )
    conn.commit()


def _build_match_query(query: str) -> str | None:
    terms = [t for t in tokenize(query) if t.isalnum() or "_" in t]
    if not terms:
        return None
    quoted = [f'"{t}"' for t in dict.fromkeys(terms)]
    return " OR ".join(quoted)


class SparseHit:
    __slots__ = ("end_offset", "node_id", "score", "snippet", "start_offset", "unit_id")

    def __init__(
        self,
        unit_id: str,
        node_id: str,
        start_offset: int,
        end_offset: int,
        score: float,
        snippet: str,
    ) -> None:
        self.unit_id = unit_id
        self.node_id = node_id
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.score = score
        self.snippet = snippet


def search(conn: sqlite3.Connection, document_id: str, query: str, limit: int) -> list[SparseHit]:
    match_query = _build_match_query(query)
    if match_query is None:
        return []

    try:
        rows = conn.execute(
            """
            SELECT
                ru.unit_id AS unit_id,
                ru.node_id AS node_id,
                ru.start_offset AS start_offset,
                ru.end_offset AS end_offset,
                bm25(retrieval_units_fts) AS raw_score,
                snippet(retrieval_units_fts, 0, '', '', '...', 100) AS preview
            FROM retrieval_units_fts
            JOIN retrieval_units ru ON ru.unit_id = retrieval_units_fts.unit_id
            WHERE retrieval_units_fts.document_id = ?
              AND retrieval_units_fts MATCH ?
            ORDER BY raw_score
            LIMIT ?
            """,
            (document_id, match_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        SparseHit(
            unit_id=row["unit_id"],
            node_id=row["node_id"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            score=-row["raw_score"],
            snippet=row["preview"],
        )
        for row in rows
    ]
