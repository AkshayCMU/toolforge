"""ConversationState TypedDict and Conversation output model — F4.6.

ConversationState is the shared mutable bag that flows through the LangGraph
graph.  Every node receives the full state and returns a partial dict; LangGraph
merges the partial dict into the state (last-writer-wins per key).

SessionState (inside ConversationState) is a mutable dataclass that the
Executor mutates in-place.  LangGraph 0.2 in-memory mode does NOT deep-copy
the state dict between nodes, so in-place mutations on session_state are
visible to all subsequent nodes without being included in the node's return
dict.

Endpoint-ID normalisation helper
---------------------------------
Graph node IDs use the "ep:" prefix (e.g. "ep:Travel/Hotels/createBooking").
Executor registry keys use bare Endpoint.id (e.g. "Travel/Hotels/createBooking").
_to_registry_id() strips the prefix when present; it is a no-op otherwise.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel
from typing_extensions import TypedDict

from toolforge.agents.assistant import AssistantTurn
from toolforge.agents.judge import JudgeResult
from toolforge.agents.planner import TaskPlan
from toolforge.agents.user_sim import Message
from toolforge.execution.session import SessionState
from toolforge.graph.sampler import ChainConstraints
from toolforge.registry.models import Endpoint


# ---------------------------------------------------------------------------
# ID normalisation
# ---------------------------------------------------------------------------

def _to_registry_id(node_id: str) -> str:
    """Convert a graph node ID to a bare Endpoint.id suitable for executor lookup.

    Strips the "ep:" prefix that build.py adds to endpoint node IDs.
    Conditional — if the prefix is absent (e.g. in hand-built test graphs)
    the string is returned unchanged.

    Examples:
        "ep:Travel/Hotels/createBooking" → "Travel/Hotels/createBooking"
        "Travel/Hotels/createBooking"    → "Travel/Hotels/createBooking"
    """
    return node_id.removeprefix("ep:")


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ConversationState(TypedDict):
    """Full mutable state that flows through the LangGraph conversation loop.

    Fields added beyond the FEATURES.md spec are marked with # added.
    """

    # Identity
    conversation_id: str
    seed: int
    constraints: ChainConstraints          # added — consumed by plan_node

    # Chain (set by plan_node, read-only afterward)
    sampled_chain: list[str]               # bare Endpoint.id values, NO "ep:" prefix
    chain_endpoints: list[Endpoint]        # Endpoint objects for prompt enrichment + distractors
    all_endpoints: list[Endpoint]          # added — full registry pool for distractor selection

    # Plan
    plan: TaskPlan | None                  # None before plan_node runs

    # Live conversation
    messages: list[Message]
    session_state: SessionState            # mutated in-place by executor_node ONLY
    turn_count: int                        # incremented in user_turn_node
    chain_index: int                       # added — on-chain successful call count

    # Inter-node communication
    last_assistant_turn: AssistantTurn | None  # added — set by assistant_turn, read by executor

    # Results
    judge_result: JudgeResult | None
    repair_attempts: int
    status: Literal["running", "done", "failed", "needs_repair"]


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class Conversation(BaseModel):
    """Immutable record of one completed conversation.

    Produced by generate_one(); consumed by F7 (JSONL I/O) and F5 (validation).
    session_summary is the full session_to_dict() output — exact fields stabilise in F7.
    """

    conversation_id: str
    seed: int
    sampled_chain: list[str]            # bare endpoint IDs
    messages: list[Message]
    session_summary: dict[str, Any]     # from execution.session.session_to_dict()
    judge_result: JudgeResult | None
    status: str                         # "done" | "failed"
    repair_attempts: int = 0            # number of repair cycles applied (F5.2 / F7.1)
