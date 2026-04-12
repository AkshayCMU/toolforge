"""Public entrypoint for single-conversation generation — F4.6.

generate_one() is the sole entry point for the conversation loop.
It constructs the initial ConversationState and delegates to a
pre-built ConversationGenerator (which holds the compiled LangGraph).

The caller is responsible for constructing the ConversationGenerator
with the appropriate agents and executor (wired in the CLI at F7).
"""

from __future__ import annotations

from typing import Any

from toolforge.execution.session import make_session, session_to_dict
from toolforge.generator.graph import ConversationGenerator
from toolforge.generator.state import Conversation, ConversationState
from toolforge.graph.sampler import ChainConstraints


def generate_one(
    seed: int,
    constraints: ChainConstraints,
    generator: ConversationGenerator,
    *,
    conversation_id: str | None = None,
) -> Conversation:
    """Run the LangGraph loop for one seed and return an immutable Conversation.

    Args:
        seed: RNG seed.  Same seed + same generator state → same Conversation.
        constraints: Chain-sampling constraints (length, categories, etc.).
        generator: Pre-built ConversationGenerator holding all agents and deps.
        conversation_id: Optional override for the conversation ID.
            Defaults to "conv-{seed}".

    Returns:
        Conversation with all messages, session_summary, and judge_result.

    Raises:
        RuntimeError: If the ChainSampler returns a truncated/failed result.
    """
    cid = conversation_id or f"conv-{seed}"

    initial: ConversationState = {
        "conversation_id": cid,
        "seed": seed,
        "constraints": constraints,
        # Fields set by plan_node:
        "sampled_chain": [],
        "chain_endpoints": [],
        "all_endpoints": generator._all_endpoints,
        "plan": None,
        # Live conversation:
        "messages": [],
        "session_state": make_session(cid, seed),
        "turn_count": 0,
        "chain_index": 0,
        "last_assistant_turn": None,
        # Results:
        "judge_result": None,
        "repair_attempts": 0,
        "status": "running",
    }

    final_state: ConversationState = generator.run(initial)  # type: ignore[assignment]
    return _state_to_conversation(final_state)


def _state_to_conversation(state: ConversationState) -> Conversation:
    """Convert a completed ConversationState to an immutable Conversation record."""
    return Conversation(
        conversation_id=state["conversation_id"],
        seed=state["seed"],
        sampled_chain=state["sampled_chain"],
        messages=state["messages"],
        session_summary=session_to_dict(state["session_state"]),
        judge_result=state["judge_result"],
        status=state["status"],
    )
