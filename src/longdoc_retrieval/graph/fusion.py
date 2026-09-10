"""merge_candidates node: normalize identity, remove duplicate ranges,
combine overlapping ranges, preserve provenance/scores, and prioritize
diversity across sections rather than a flat global top-K - ten passages
from the same paragraph are usually less useful than evidence distributed
across multiple sections of the document.
"""

from collections import defaultdict
from typing import Any

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.retrieval import RetrievalCandidate
from longdoc_retrieval.graph.state import RetrievalState
from longdoc_retrieval.graph.types import Node

# Higher wins when two candidates' ranges overlap and must be merged into
# one - exact matches are the most precise signal, sparse/structural next,
# expanded (already-read) candidates last.
_METHOD_PRIORITY = {"exact": 3, "structural": 2, "sparse": 1, "expanded": 0}


def dedupe_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    seen: set[str] = set()
    result = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        result.append(candidate)
    return result


def merge_overlapping_ranges(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    by_node: dict[str, list[RetrievalCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_node[candidate.node_id].append(candidate)

    merged: list[RetrievalCandidate] = []
    for group in by_node.values():
        group.sort(key=lambda c: c.start_offset)
        current: RetrievalCandidate | None = None
        for candidate in group:
            if current is None:
                current = candidate
                continue
            if candidate.start_offset < current.end_offset:
                current_priority = _METHOD_PRIORITY[current.retrieval_method]
                candidate_priority = _METHOD_PRIORITY[candidate.retrieval_method]
                winner = current if current_priority >= candidate_priority else candidate
                if current_priority == candidate_priority:
                    # Same method (e.g. two sparse hits overlapping) - no
                    # higher-fidelity span to prefer, so keep the union as
                    # before.
                    current = winner.model_copy(
                        update={
                            "start_offset": min(current.start_offset, candidate.start_offset),
                            "end_offset": max(current.end_offset, candidate.end_offset),
                            "token_count": max(current.token_count, candidate.token_count),
                        }
                    )
                else:
                    # Different methods - keep the higher-priority
                    # candidate's own (tighter) span rather than widening it
                    # to the lower-priority one's: a precise exact/structural
                    # match shouldn't balloon back out to a whole sparse
                    # retrieval unit's range just because the two overlap.
                    current = winner
            else:
                merged.append(current)
                current = candidate
        if current is not None:
            merged.append(current)
    return merged


def round_robin_select(
    buckets: list[list[RetrievalCandidate]], limit: int
) -> list[RetrievalCandidate]:
    selected: list[RetrievalCandidate] = []
    while len(selected) < limit and any(buckets):
        for bucket in buckets:
            if len(selected) >= limit:
                break
            if bucket:
                selected.append(bucket.pop(0))
    return selected


def merge_candidates_node(service: RetrievalService, config: RetrievalConfig) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        deduped = dedupe_candidates(state["candidates"])
        merged = merge_overlapping_ranges(deduped)

        buckets_by_ancestor: dict[str, list[RetrievalCandidate]] = defaultdict(list)
        for candidate in merged:
            ancestor = await service.get_top_level_ancestor(
                state["document_id"], candidate.node_id
            )
            buckets_by_ancestor[ancestor].append(candidate)

        for bucket in buckets_by_ancestor.values():
            bucket.sort(key=lambda c: (c.lexical_score is None, -(c.lexical_score or 0.0)))

        selected = round_robin_select(
            list(buckets_by_ancestor.values()), config.max_candidates_for_llm_evaluation
        )

        metrics = state["metrics"].model_copy(
            update={"candidate_count": state["metrics"].candidate_count + len(selected)}
        )
        return {"candidates": selected, "metrics": metrics}

    return _node
