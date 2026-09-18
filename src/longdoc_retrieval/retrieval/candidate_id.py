import hashlib


def query_hash8(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]


def make_candidate_id(retrieval_method: str, document_id: str, match_ordinal: int, query: str) -> str:
    return f"{retrieval_method}:{document_id}:{match_ordinal}:{query_hash8(query)}"
