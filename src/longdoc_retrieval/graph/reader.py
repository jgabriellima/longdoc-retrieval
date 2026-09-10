"""read_evidence + update_evidence_ledger nodes. Only should_read=True
candidates get expanded ("search is cheap, reading consumes context") via
RetrievalService.read_range (reusing the deterministic layer's
continuation-metadata behavior). evidence_id is deterministic from
(document_id, node_id, offsets) so re-reading the same range in a later
iteration can never duplicate a ledger entry.
"""

import asyncio
from typing import Any

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.domain.evidence import Evidence
from longdoc_retrieval.graph.state import RetrievalState
from longdoc_retrieval.graph.types import Node
from longdoc_retrieval.tokenize import approx_token_count


def make_evidence_id(document_id: str, node_id: str, start_offset: int, end_offset: int) -> str:
    return f"ev:{document_id}:{node_id}:{start_offset}:{end_offset}"


# Max character gap between two evidence spans to treat them as one
# continuous passage worth merging (the same "combine overlapping ranges"
# idea candidate fusion applies, applied here to the Evidence Ledger).
# Observed real case: a clause split by the 2000-token retrieval-unit cap
# right at a page-break marker had a 2-character gap between the two
# halves ("...para" | 2 chars | "dirimir litígios...").
_ADJACENCY_GAP_CHARS = 12

# Characters that plausibly end a complete sentence/clause/paragraph.
_SENTENCE_END_CHARS = ".!?:;\"')"


def _is_artificial_cut(preceding_text: str) -> bool:
    """True only when `preceding_text` looks like it was sliced mid-clause
    (no terminal punctuation) rather than ending at a natural sentence or
    paragraph boundary.

    This check is the actual gate on merging - NOT mere adjacency.
    Retrieval units tile the whole document contiguously, so any two
    retrieval units that both get marked should_read=True are almost
    always "adjacent" by offset alone, even when they are two complete,
    independently-relevant units with no relationship to each other. An
    earlier version of this merge only checked the character gap, which
    glued unrelated neighboring units into single evidence items of
    15-30K+ characters on a real document (confirmed: only 7 retrieval
    units existed for a ~45K-char file, several thousand chars each - the
    merge was combining whole neighboring units, not stitching a genuine
    mid-sentence split). Gating on "does the text end cleanly" restricts
    merging back to the narrow case it was built for.
    """

    stripped = preceding_text.rstrip()
    return bool(stripped) and stripped[-1] not in _SENTENCE_END_CHARS


def _find_adjacent(existing: list[Evidence], candidate: Evidence) -> Evidence | None:
    for item in existing:
        if item.document_id != candidate.document_id:
            continue
        gap_after = candidate.start_offset - item.end_offset
        gap_before = item.start_offset - candidate.end_offset
        if 0 <= gap_after <= _ADJACENCY_GAP_CHARS and _is_artificial_cut(item.text):
            return item
        if 0 <= gap_before <= _ADJACENCY_GAP_CHARS and _is_artificial_cut(candidate.text):
            return item
    return None


async def _merge_evidence(
    service: RetrievalService, document_id: str, a: Evidence, b: Evidence
) -> Evidence:
    start = min(a.start_offset, b.start_offset)
    end = max(a.end_offset, b.end_offset)
    result = await service.read_range(document_id, start, end)
    supports = list(dict.fromkeys([*a.supports, *b.supports]))
    reasons = [r for r in (a.relevance_reason, b.relevance_reason) if r]
    return Evidence(
        evidence_id=make_evidence_id(document_id, a.node_id, result.start_offset, result.end_offset),
        document_id=document_id,
        node_id=a.node_id,
        start_offset=result.start_offset,
        end_offset=result.end_offset,
        start_page=a.start_page,
        end_page=b.end_page,
        text=result.text,
        retrieval_query=a.retrieval_query,
        retrieval_method=a.retrieval_method,
        relevance_reason=" | ".join(dict.fromkeys(reasons)),
        supports=supports,
        token_count=approx_token_count(result.text),
    )


def read_evidence_node(service: RetrievalService) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        evaluated_by_id = {e.candidate_id: e for e in state["evaluated_candidates"]}
        to_read = [
            c
            for c in state["candidates"]
            if evaluated_by_id.get(c.candidate_id) and evaluated_by_id[c.candidate_id].should_read
        ]

        plan = state.get("plan")
        supports = [plan.objective] if plan is not None else []

        async def read_one(candidate: Any) -> Evidence:
            result = await service.read_range(
                state["document_id"], candidate.start_offset, candidate.end_offset
            )
            evaluation = evaluated_by_id[candidate.candidate_id]
            return Evidence(
                evidence_id=make_evidence_id(
                    state["document_id"], candidate.node_id, result.start_offset, result.end_offset
                ),
                document_id=state["document_id"],
                node_id=candidate.node_id,
                start_offset=result.start_offset,
                end_offset=result.end_offset,
                start_page=candidate.start_page,
                end_page=candidate.end_page,
                text=result.text,
                retrieval_query=candidate.retrieval_query,
                retrieval_method=candidate.retrieval_method,
                relevance_reason=evaluation.reason,
                supports=supports,
                token_count=approx_token_count(result.text),
            )

        if to_read:
            read_results = list(await asyncio.gather(*(read_one(c) for c in to_read)))
        else:
            read_results = []

        metrics = state["metrics"].model_copy(
            update={
                "retrieved_tokens": state["metrics"].retrieved_tokens
                + sum(e.token_count for e in read_results)
            }
        )
        return {"pending_evidence": read_results, "metrics": metrics}

    return _node


def update_evidence_ledger_node(service: RetrievalService, config: RetrievalConfig) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        existing_ids = {e.evidence_id for e in state["evidence"]}
        merged = list(state["evidence"])
        total_tokens = sum(e.token_count for e in merged)
        document_id = state["document_id"]

        for evidence in state["pending_evidence"]:
            if evidence.evidence_id in existing_ids:
                continue

            adjacent = _find_adjacent(merged, evidence)
            if adjacent is not None:
                combined = await _merge_evidence(service, document_id, adjacent, evidence)
                if combined.evidence_id != adjacent.evidence_id and combined.evidence_id in existing_ids:
                    # Merging converged to a span some other ledger entry
                    # already covers exactly (can happen once merge chains
                    # from different starting points meet, e.g. on a small
                    # document) - drop this one rather than inserting a
                    # second entry with a duplicate evidence_id.
                    continue
                idx = merged.index(adjacent)
                total_tokens += combined.token_count - adjacent.token_count
                merged[idx] = combined
                existing_ids.discard(adjacent.evidence_id)
                existing_ids.add(combined.evidence_id)
                continue

            if len(merged) >= config.max_evidence_items:
                break
            if total_tokens + evidence.token_count > config.max_evidence_tokens:
                continue
            merged.append(evidence)
            existing_ids.add(evidence.evidence_id)
            total_tokens += evidence.token_count

        history = [*state["evidence_count_history"], len(merged)]
        metrics = state["metrics"].model_copy(
            update={"evidence_count": len(merged), "evidence_tokens": total_tokens}
        )
        return {
            "evidence": merged,
            "pending_evidence": [],
            "evidence_count_history": history,
            "metrics": metrics,
        }

    return _node
