import re
import sqlite3

from longdoc_retrieval.domain.retrieval import RetrievalCandidate
from longdoc_retrieval.indexes.structural import get_document_content, resolve_node_at_offset
from longdoc_retrieval.retrieval.candidate_id import make_candidate_id
from longdoc_retrieval.tokenize import approx_token_count

PREVIEW_CONTEXT_CHARS = 120


def _build_pattern(expression: str) -> re.Pattern[str] | None:
    expression = expression.strip()
    if not expression:
        return None
    escaped = re.escape(expression)
    flexible = re.sub(r"(\\\s)+", r"\\s+", escaped)
    try:
        return re.compile(flexible, re.IGNORECASE)
    except re.error:
        return re.compile(re.escape(expression), re.IGNORECASE)


def find_exact(
    conn: sqlite3.Connection, document_id: str, expression: str, limit: int = 20
) -> list[RetrievalCandidate]:
    pattern = _build_pattern(expression)
    if pattern is None:
        return []

    content = get_document_content(conn, document_id)
    candidates = []
    for ordinal, match in enumerate(pattern.finditer(content)):
        if ordinal >= limit:
            break
        start, end = match.span()
        node_id = resolve_node_at_offset(conn, document_id, start) or ""
        preview_start = max(0, start - PREVIEW_CONTEXT_CHARS)
        preview_end = min(len(content), end + PREVIEW_CONTEXT_CHARS)
        candidates.append(
            RetrievalCandidate(
                candidate_id=make_candidate_id("exact", document_id, ordinal, expression),
                document_id=document_id,
                node_id=node_id,
                start_offset=start,
                end_offset=end,
                retrieval_method="exact",
                retrieval_query=expression,
                lexical_score=None,
                token_count=approx_token_count(content[start:end]),
                preview=content[preview_start:preview_end],
            )
        )
    return candidates
