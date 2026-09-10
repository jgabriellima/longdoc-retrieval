"""execute_searches node: independent sparse/exact searches run concurrently
via asyncio.gather - for a batch of independent queries, there's no reason
to run them one after another when nothing depends on the previous result.
"""

import asyncio
import time
from typing import Any

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.config import RetrievalConfig
from longdoc_retrieval.graph.state import RetrievalState, SearchExecution
from longdoc_retrieval.graph.types import Node


def execute_searches_node(service: RetrievalService, config: RetrievalConfig) -> Node:
    async def _node(state: RetrievalState) -> dict[str, Any]:
        plan = state["plan"]
        document_id = state["document_id"]

        # `concepts` are searched too, not just `queries`: a query phrased
        # to match the question's own wording can miss a clause phrased
        # differently (observed: "valor total" never matched "O valor [...]
        # totaliza R$ 520.861,60" via BM25) - concepts give the sparse index
        # more lexical variety to match against without an extra iteration.
        combined = (
            [("sparse", q) for q in (plan.queries if plan else [])]
            + [("sparse", c) for c in (plan.concepts if plan else [])]
            + [("exact", t) for t in (plan.exact_terms if plan else [])]
        )
        seen: set[tuple[str, str]] = set()
        tasks_spec = []
        for item in combined:
            if item not in seen:
                seen.add(item)
                tasks_spec.append(item)
        tasks_spec = tasks_spec[: config.max_queries_per_iteration]

        async def run(method: str, query: str):
            started = time.monotonic()
            if method == "sparse":
                results = await service.search(
                    document_id, query, limit=config.max_candidates_per_query
                )
            else:
                results = await service.find_exact(
                    document_id, query, limit=config.max_candidates_per_query
                )
            duration_ms = (time.monotonic() - started) * 1000
            return method, query, results, duration_ms

        results = (
            await asyncio.gather(*(run(method, query) for method, query in tasks_spec))
            if tasks_spec
            else []
        )

        new_candidates = list(state["candidates"])
        executed = list(state["executed_queries"])
        retrieval_duration_ms = 0.0
        for method, query, candidates, duration_ms in results:
            executed.append(
                SearchExecution(
                    query=query,
                    retrieval_method=method,
                    candidate_count=len(candidates),
                    iteration=state["iteration"],
                )
            )
            new_candidates.extend(candidates)
            retrieval_duration_ms += duration_ms

        metrics = state["metrics"].model_copy(
            update={
                "query_count": state["metrics"].query_count + len(tasks_spec),
                "retrieval_latency_ms": state["metrics"].retrieval_latency_ms
                + retrieval_duration_ms,
            }
        )

        return {
            "candidates": new_candidates,
            "executed_queries": executed,
            "metrics": metrics,
        }

    return _node
