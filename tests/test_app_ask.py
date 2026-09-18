import os

import pytest

from longdoc_retrieval.app.ask import require_api_key
from longdoc_retrieval.app.errors import UserError
from longdoc_retrieval.domain.metrics import RetrievalMetrics
from longdoc_retrieval.domain.package import EvidencePackage


def test_require_api_key_rejects_empty_env():
    with pytest.raises(UserError, match="API key"):
        require_api_key({})


def test_require_api_key_accepts_either_provider():
    require_api_key({"ANTHROPIC_API_KEY": "sk-test"})
    require_api_key({"OPENAI_API_KEY": "sk-test"})


@pytest.mark.asyncio
async def test_run_ask_uses_injected_runner(ingested):
    from longdoc_retrieval.app.ask import run_ask

    async def fake_runner(service, document_id, question, config=None, llm_clients=None):
        return EvidencePackage(
            request_id="req",
            document_id=document_id,
            question=question,
            status="insufficient",
            retrieval_summary="none",
            metrics=RetrievalMetrics(
                document_id=document_id,
                document_tokens=10,
                question_tokens=4,
            ),
        )

    package = await run_ask(ingested, "convenio", "qual o valor?", runner=fake_runner)
    assert package.status == "insufficient"
    assert package.document_id == "convenio"


def test_require_api_key_does_not_read_process_env_when_mapping_passed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-process")
    with pytest.raises(UserError):
        require_api_key({})
    # sanity: process env still set
    assert os.environ["ANTHROPIC_API_KEY"] == "from-process"
