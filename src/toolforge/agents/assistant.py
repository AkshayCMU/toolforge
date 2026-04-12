"""Assistant agent — F4.4.

Reads the conversation history and SessionState, then produces the next
assistant action: either a clarifying message or a grounded tool call.

Design decisions (see plan §6):
  - AssistantTurn is a FLAT Pydantic model (not a discriminated union).
    Pydantic v2's anyOf schema for discriminated unions is inconsistent under
    Anthropic tool-use forcing. A flat model with a Literal "type" field
    produces a simpler, more reliable JSON Schema.
  - Post-parse validation enforces field presence for each turn type.
  - Distractor selection: 3 endpoints from the same categories as the chain,
    seeded by session_state.seed, to give the assistant plausible alternatives.

Uses LLMClient.call() with structured output (AssistantTurn schema).
Model: claude-haiku-4-5-20251001, temperature=0.0.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from toolforge.agents.base import Agent
from toolforge.agents.llm_client import LLMClient
from toolforge.agents.user_sim import Message
from toolforge.execution.session import SessionState
from toolforge.registry.models import Endpoint

_PROMPT_PATH = Path(__file__).parent / "prompts" / "assistant.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")

# Maximum number of prior tool outputs to include in the session registry view.
_MAX_REGISTRY_TURNS = 5


class AssistantTurn(BaseModel):
    """Structured output of the Assistant agent.

    Flat model (not a discriminated union) for reliable JSON Schema under
    Anthropic tool-use forcing. Post-parse validator enforces field presence.

      type == "message"   → content must be non-empty; endpoint/arguments ignored
      type == "tool_call" → endpoint must be non-empty; arguments is the call dict
    """

    type: Literal["message", "tool_call"]
    content: str = ""
    endpoint: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_fields(self) -> "AssistantTurn":
        if self.type == "message" and not self.content:
            raise ValueError("AssistantTurn with type='message' must have non-empty content")
        if self.type == "tool_call" and not self.endpoint:
            raise ValueError("AssistantTurn with type='tool_call' must have non-empty endpoint")
        return self


class Assistant(Agent):
    """Produces the next assistant action given conversation context.

    Usage::

        client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.0)
        assistant = Assistant(client)
        turn = assistant.act(history, session_state, chain_endpoints, distractors)
    """

    name = "assistant"

    def act(
        self,
        history: list[Message],
        session_state: SessionState,
        chain_endpoints: list[Endpoint],
        distractors: list[Endpoint],
    ) -> AssistantTurn:
        """Produce the next assistant turn.

        Args:
            history: Conversation so far (user/assistant messages).
            session_state: Current SessionState — source of truth for
                           available_values_by_type and tool_outputs.
            chain_endpoints: Endpoints from the sampled chain.
            distractors: 3 additional endpoints (same-category alternates).
        """
        all_endpoints = chain_endpoints + distractors
        system_prompt = self._build_system_prompt(all_endpoints, session_state)
        user_prompt = self._build_user_prompt(history)
        return self._client.call(
            system_prompt,
            user_prompt,
            AssistantTurn,
            prompt_version="v1",
            agent_name=self.name,
        )

    # ------------------------------------------------------------------
    # Prompt construction (pure Python, P2)
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        endpoints: list[Endpoint],
        session_state: SessionState,
    ) -> str:
        catalog = _format_endpoint_catalog(endpoints)
        registry = _format_session_registry(session_state)
        return _PROMPT_TEMPLATE.format(
            endpoint_catalog=catalog,
            session_registry=registry,
        )

    def _build_user_prompt(self, history: list[Message]) -> str:
        if not history:
            return "The user has not yet said anything. Wait for their message."
        lines: list[str] = ["## Conversation", ""]
        for msg in history:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        lines.append("")
        lines.append("What is your next action?")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Distractor selection (pure Python, P2)
# ---------------------------------------------------------------------------

def select_distractors(
    chain_endpoints: list[Endpoint],
    all_endpoints: list[Endpoint],
    seed: int,
    n: int = 3,
) -> list[Endpoint]:
    """Select n distractor endpoints for the assistant's filtered catalog.

    Strategy:
    1. Collect endpoints from the same categories as the chain, excluding
       chain endpoints themselves.
    2. If fewer than n candidates exist, fill from other categories.
    3. Sample deterministically using seeded RNG.
    """
    chain_ids = {ep.id for ep in chain_endpoints}
    chain_categories = {ep.id.split("/")[0] for ep in chain_endpoints if "/" in ep.id}

    same_cat = [
        ep for ep in all_endpoints
        if ep.id not in chain_ids and ep.id.split("/")[0] in chain_categories
    ]
    other_cat = [
        ep for ep in all_endpoints
        if ep.id not in chain_ids and ep.id.split("/")[0] not in chain_categories
    ]

    rng = random.Random(seed)
    same_cat_sorted = sorted(same_cat, key=lambda e: e.id)
    other_cat_sorted = sorted(other_cat, key=lambda e: e.id)

    # Take same-category distractors first (true priority — P5 domain coherence).
    # Only fall back to cross-category when same-category pool is exhausted.
    same_take = min(n, len(same_cat_sorted))
    selected = rng.sample(same_cat_sorted, same_take) if same_take > 0 else []

    remaining = n - len(selected)
    if remaining > 0 and other_cat_sorted:
        other_take = min(remaining, len(other_cat_sorted))
        selected = selected + rng.sample(other_cat_sorted, other_take)

    return selected


# ---------------------------------------------------------------------------
# Prompt formatting helpers (pure Python, P2)
# ---------------------------------------------------------------------------

def _format_endpoint_catalog(endpoints: list[Endpoint]) -> str:
    if not endpoints:
        return "(no endpoints available)"
    blocks: list[str] = []
    for ep in endpoints:
        lines = [f"ENDPOINT: {ep.id}", f"  Description: {ep.description}"]
        required = [p for p in ep.parameters if p.required]
        optional = [p for p in ep.parameters if not p.required]
        if required:
            req_str = ", ".join(f"{p.name} ({p.type})" for p in required)
            lines.append(f"  Required: {req_str}")
        if optional:
            opt_str = ", ".join(f"{p.name} ({p.type})" for p in optional)
            lines.append(f"  Optional: {opt_str}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_session_registry(state: SessionState) -> str:
    lines: list[str] = []

    if state.available_values_by_type:
        lines.append("Available values from prior tool calls:")
        for sem_type, values in sorted(state.available_values_by_type.items()):
            lines.append(f"  {sem_type}: {values}")
    else:
        lines.append("No values from prior tool calls yet.")

    recent = state.tool_outputs[-_MAX_REGISTRY_TURNS:]
    if recent:
        lines.append("")
        lines.append(f"Last {len(recent)} tool output(s):")
        for out in recent:
            status = "success" if out.is_success() else f"error: {out.error}"
            lines.append(f"  [{out.timestamp}] {out.endpoint_id} → {status}")

    return "\n".join(lines)
