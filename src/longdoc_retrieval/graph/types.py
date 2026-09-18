from collections.abc import Awaitable, Callable
from typing import Any

from longdoc_retrieval.graph.state import RetrievalState

Node = Callable[[RetrievalState], Awaitable[dict[str, Any]]]
