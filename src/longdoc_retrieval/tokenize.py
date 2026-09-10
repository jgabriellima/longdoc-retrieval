"""Dependency-free approximate tokenizer.

Deliberately not tiktoken or any model-specific BPE tokenizer: the
retrieval layer must not be coupled to a specific LLM vendor's vocabulary,
and the 150-400 token/unit chunking target has enough tolerance that a
regex approximation is sufficient. Every `token_count` field in this
codebase (DocumentNode, RetrievalCandidate, chunk sizing) uses this same
approximation consistently.
"""

import re

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


def approx_token_count(text: str) -> int:
    return len(_TOKEN_PATTERN.findall(text))
