"""SessionState dataclass and ToolOutput — F3.1.

SessionState is the centerpiece of P3 (Grounding over hallucination).
All mutable state is owned and updated exclusively by the Executor.

Design notes:
- resolved_entities uses tuple[str, Any] keys in memory (matching the spec).
  String serialisation happens only at the SessionState export boundary
  via session_to_dict(), NOT inside ToolOutput.to_dict().
- ToolOutput.timestamp is deterministic ("turn-N") in Phase 3.
  Set to len(state.tool_outputs) BEFORE appending the output.
  This applies to BOTH success and failure outputs — all calls are recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolOutput:
    """Record of a single tool call.

    Exactly one of (response, error) is non-None per call:
    - Success: response is a dict, error is None.
    - Failure: response is None, error is a structured message string.

    timestamp is deterministic in Phase 3: "turn-N" where N is the index
    of this call in the session (0-based). Set before appending to state.
    """

    endpoint_id: str
    arguments: dict[str, Any]
    response: dict[str, Any] | None
    error: str | None
    timestamp: str  # "turn-N" in Phase 3; may become ISO-8601 in Phase 4

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict of this ToolOutput's own fields.

        Does NOT serialise SessionState.resolved_entities — that is handled
        exclusively by session_to_dict().
        """
        return {
            "endpoint_id": self.endpoint_id,
            "arguments": self.arguments,
            "response": self.response,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    def is_success(self) -> bool:
        return self.error is None


@dataclass
class SessionState:
    """Full mutable state for one conversation.

    Mutated ONLY by the Executor. All fields are initialised to empty
    collections by make_session(); never touch them directly outside the
    executor.
    """

    conversation_id: str
    seed: int

    available_values_by_type: dict[str, list[Any]]
    """semantic_type → list of observed values produced by prior tool calls.

    The Executor appends to this pool when a CHAIN_ONLY-typed response field
    is extracted. The grounding check reads from this pool before accepting
    a CHAIN_ONLY argument.
    """

    resolved_entities: dict[tuple[str, Any], dict]
    """(semantic_type, value) → full entity dict.

    Tuple keys in memory (matching the FEATURES.md spec).
    Converted to "{sem_type}:{value}" strings only by session_to_dict().
    Not populated in Phase 3 — field exists for Phase 4/5 agent use.
    """

    created_entities: list[dict]
    """Bookings, orders, and other transactional entities created in this session."""

    tool_outputs: list[ToolOutput]
    """Ordered log of ALL tool calls in the conversation — success and failure alike.

    Both successful calls (response set, error=None) and failed calls
    (response=None, error set) are appended by the Executor. Timestamps are
    deterministic: "turn-N" where N = len(tool_outputs) BEFORE the append,
    so failures advance the turn index exactly like successes.
    """

    private_user_knowledge: dict[str, Any]
    """Fields the planner omitted from the initial query; revealed on request."""


def make_session(conversation_id: str, seed: int) -> SessionState:
    """Factory that initialises all mutable fields to empty collections."""
    return SessionState(
        conversation_id=conversation_id,
        seed=seed,
        available_values_by_type={},
        resolved_entities={},
        created_entities=[],
        tool_outputs=[],
        private_user_knowledge={},
    )


def session_to_dict(state: SessionState) -> dict[str, Any]:
    """Serialise SessionState to a JSON-safe dict.

    This is the only place where resolved_entities tuple keys are converted
    to "{sem_type}:{value}" strings. Called by the I/O layer (F7), not by
    individual ToolOutputs.
    """
    return {
        "conversation_id": state.conversation_id,
        "seed": state.seed,
        "available_values_by_type": state.available_values_by_type,
        "resolved_entities": {
            f"{k[0]}:{k[1]}": v
            for k, v in state.resolved_entities.items()
        },
        "created_entities": state.created_entities,
        "tool_outputs": [o.to_dict() for o in state.tool_outputs],
        "private_user_knowledge": state.private_user_knowledge,
    }
