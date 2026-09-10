"""Shared node function type, so individual node modules don't depend on
each other just to borrow a type alias.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from longdoc_retrieval.graph.state import RetrievalState

Node = Callable[[RetrievalState], Awaitable[dict[str, Any]]]
