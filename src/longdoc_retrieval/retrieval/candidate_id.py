"""Deterministic candidate_id construction, shared by search() and
find_exact() so candidate fusion (deduping/merging hits across queries) gets
a stable identity key without re-deriving this format itself.
"""

import hashlib


def query_hash8(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]


def make_candidate_id(retrieval_method: str, document_id: str, match_ordinal: int, query: str) -> str:
    return f"{retrieval_method}:{document_id}:{match_ordinal}:{query_hash8(query)}"
