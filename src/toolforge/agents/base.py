"""Agent abstract base class — F4.1.

All agents hold a reference to a shared LLMClient and declare a class-level
`name` for logging. Structured-output agents additionally define an
`output_schema` class attribute; free-text agents (UserSimulator) do not.

Design note: Agent is NOT Generic[T]. Parameterising with Generic[T] causes
friction with mypy and LangGraph type inference without meaningful benefit at
this stage. Each agent's concrete method signature is the typed contract.
"""

from __future__ import annotations

from abc import ABC


class Agent(ABC):
    """Abstract base for all toolforge agents.

    Subclasses must set `name` as a class-level string constant.
    `system_prompt` should be set at class-definition time (e.g. loaded from
    a prompts/*.md file).  Both are used by LLMClient for cache-key context
    and structured logging.
    """

    name: str  # class-level constant — must be defined by every subclass
    system_prompt: str = ""  # overridden by concrete agents

    def __init__(self, client: "LLMClient") -> None:  # noqa: F821
        self._client = client
