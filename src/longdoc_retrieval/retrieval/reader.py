import re
import sqlite3

from pydantic import BaseModel

from longdoc_retrieval.indexes.structural import get_document_content, get_node
from longdoc_retrieval.tokenize import approx_token_count

MAX_RANGE_CHARS = 50_000

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+")


class ReadResult(BaseModel):
    text: str
    start_offset: int
    end_offset: int
    truncated: bool
    continuation_offset: int | None = None


def _cut_at_token_limit(content: str, start: int, end: int, token_limit: int) -> int:
    window = content[start:end]
    pos = None
    for tokens, token_match in enumerate(re.finditer(r"\w+|[^\w\s]", window), start=1):
        if tokens >= token_limit:
            pos = token_match.end()
            break
    if pos is None:
        return end

    last_sentence_end = None
    for m in _SENTENCE_BOUNDARY_RE.finditer(window, 0, pos):
        last_sentence_end = m.end()

    return start + last_sentence_end if last_sentence_end else start + pos


def read_node(
    conn: sqlite3.Connection,
    document_id: str,
    node_id: str,
    token_limit: int | None = None,
) -> ReadResult:
    node = get_node(conn, document_id, node_id)
    if node is None:
        raise KeyError(f"node not found: {document_id}/{node_id}")
    return read_range(conn, document_id, node.start_offset, node.end_offset, token_limit=token_limit)


def read_range(
    conn: sqlite3.Connection,
    document_id: str,
    start_offset: int,
    end_offset: int,
    token_limit: int | None = None,
) -> ReadResult:
    content = get_document_content(conn, document_id)
    start_offset = max(0, start_offset)
    end_offset = min(len(content), end_offset)

    effective_end = end_offset
    truncated = False

    if effective_end - start_offset > MAX_RANGE_CHARS:
        effective_end = start_offset + MAX_RANGE_CHARS
        truncated = True

    if token_limit is not None:
        token_count = approx_token_count(content[start_offset:effective_end])
        if token_count > token_limit:
            effective_end = _cut_at_token_limit(content, start_offset, effective_end, token_limit)
            truncated = True

    return ReadResult(
        text=content[start_offset:effective_end],
        start_offset=start_offset,
        end_offset=effective_end,
        truncated=truncated,
        continuation_offset=effective_end if truncated and effective_end < end_offset else None,
    )
