"""
LLM module for the retrieval graph.
"""
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from pydantic import BaseModel

from longdoc_retrieval.config import RetrievalConfig

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    pass


@dataclass
class StructuredLLMResult(Generic[T]):
    value: T
    input_tokens: int
    output_tokens: int


class StructuredLLM(Protocol):
    async def invoke(self, prompt: str, schema: type[T]) -> StructuredLLMResult[T]: ...


async def _invoke_structured(
    chat_model: Any, prompt: str, schema: type[T], max_retries: int
) -> StructuredLLMResult[T]:
    structured = chat_model.with_structured_output(schema, include_raw=True)
    last_error: BaseException | None = None
    for _ in range(max_retries + 1):
        response = cast(dict[str, Any], await structured.ainvoke(prompt))
        parsed = response.get("parsed")
        parsing_error = response.get("parsing_error")
        if parsed is not None and parsing_error is None:
            raw = response.get("raw")
            usage = getattr(raw, "usage_metadata", None) or {}
            return StructuredLLMResult(
                value=parsed,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
        last_error = parsing_error

    raise StructuredOutputError(
        f"failed to obtain valid {schema.__name__} after {max_retries + 1} attempt(s)"
    ) from last_error


class AnthropicStructuredLLM:
    def __init__(self, model: str, max_retries: int = 1) -> None:
        self._model = model
        self._max_retries = max_retries

    async def invoke(self, prompt: str, schema: type[T]) -> StructuredLLMResult[T]:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=self._model)  # type: ignore[call-arg]
        return await _invoke_structured(llm, prompt, schema, self._max_retries)


class OpenAIStructuredLLM:
    def __init__(self, model: str, max_retries: int = 1) -> None:
        self._model = model
        self._max_retries = max_retries

    async def invoke(self, prompt: str, schema: type[T]) -> StructuredLLMResult[T]:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=self._model)  # type: ignore[call-arg]
        return await _invoke_structured(llm, prompt, schema, self._max_retries)


def make_structured_llm(model: str, max_retries: int = 1) -> StructuredLLM:
    if model.startswith("claude-"):
        return AnthropicStructuredLLM(model, max_retries)
    if model.startswith("gpt-"):
        return OpenAIStructuredLLM(model, max_retries)
    raise ValueError(f"cannot infer LLM provider from model name: {model!r}")


@dataclass
class LLMClients:
    planner: StructuredLLM
    evaluator: StructuredLLM
    sufficiency: StructuredLLM
    synthesizer: StructuredLLM

    @classmethod
    def from_config(cls, config: RetrievalConfig) -> "LLMClients":
        return cls(
            planner=make_structured_llm(config.planner_model),
            evaluator=make_structured_llm(config.evaluator_model),
            sufficiency=make_structured_llm(config.sufficiency_model),
            synthesizer=make_structured_llm(config.synthesizer_model),
        )
