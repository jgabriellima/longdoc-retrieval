"""
Ask module for the retrieval API.
"""
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from longdoc_retrieval.api.service import RetrievalService
from longdoc_retrieval.app.errors import UserError
from longdoc_retrieval.domain.package import EvidencePackage

AskRunner = Callable[..., Awaitable[EvidencePackage]]


def require_api_key(env: Mapping[str, str] | None = None) -> None:
    import os

    environment = os.environ if env is None else env
    if environment.get("ANTHROPIC_API_KEY", "").strip() or environment.get(
        "OPENAI_API_KEY", ""
    ).strip():
        return
    raise UserError("no API key. set ANTHROPIC_API_KEY or OPENAI_API_KEY")


def load_agent_runner() -> AskRunner:
    try:
        from longdoc_retrieval.graph.run import run_retrieval
    except ImportError as exc:
        raise UserError(
            'ask requires the agent extra. run: pip install -e ".[agent]"'
        ) from exc
    return run_retrieval


async def run_ask(
    service: RetrievalService,
    document_id: str,
    question: str,
    runner: AskRunner | None = None,
    **kwargs: Any,
) -> EvidencePackage:
    execute = runner if runner is not None else load_agent_runner()
    return await execute(service, document_id, question, **kwargs)
