"""Planner agent — F4.2.

Generates a TaskPlan (user persona + initial query) for a sampled chain of endpoints.
~40% of plans deliberately omit ≥1 required parameter from the initial query to drive
the multi-turn disambiguation requirement.

Uses LLMClient.call() with structured output (TaskPlan schema).
Model: claude-haiku-4-5-20251001, temperature=0.7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from toolforge.agents.base import Agent
from toolforge.agents.llm_client import LLMClient

_PROMPT_PATH = Path(__file__).parent / "prompts" / "planner.md"


class TaskPlan(BaseModel):
    """Structured output of the Planner agent."""

    user_persona: str
    """Who the user is: role, context, goals."""

    initial_query: str
    """The user's first message. Natural language; may omit params in private_user_knowledge."""

    clarification_points: list[str]
    """Things the assistant should clarify before calling tools. May be empty."""

    expected_final_outcome: str
    """What success looks like for this conversation."""

    chain_rationale: str
    """One sentence: why this API sequence is natural for this scenario."""

    private_user_knowledge: dict[str, Any]
    """Required parameter values omitted from initial_query.
    Empty when the user stated everything upfront (~60% of plans).
    Non-empty when disambiguation is triggered (~40% of plans).
    """


class Planner(Agent):
    """Scenario planner: given a sampled chain of endpoints, produce a TaskPlan.

    Usage::

        client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.7)
        planner = Planner(client)
        plan = planner.plan(chain=["Travel/Hotels/createBooking", ...], persona_seed=42)
    """

    name = "planner"
    system_prompt: str = _PROMPT_PATH.read_text(encoding="utf-8")

    def plan(
        self,
        chain: list[str],
        persona_seed: int,
        diversity_hints: list[str] | None = None,
    ) -> TaskPlan:
        """Generate a TaskPlan for the given endpoint chain.

        Args:
            chain: Ordered list of endpoint IDs from ChainSampler.
            persona_seed: Integer seed used to personalise the scenario.
                          Injected into the user prompt for variety.
            diversity_hints: Optional list of archetype strings to avoid
                             (injected by F6.1 CorpusDiversityTracker).
                             Stub in this slice — ignored if None.
        """
        user_prompt = self._build_user_prompt(chain, persona_seed, diversity_hints)
        return self._client.call(
            self.system_prompt,
            user_prompt,
            TaskPlan,
            prompt_version="v1",
        )

    # ------------------------------------------------------------------
    # Prompt construction (pure Python, P2)
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        chain: list[str],
        persona_seed: int,
        diversity_hints: list[str] | None,
    ) -> str:
        lines: list[str] = []
        lines.append(f"Persona seed: {persona_seed}")
        lines.append("")
        lines.append("## API call sequence")
        lines.append("")
        for i, ep_id in enumerate(chain, start=1):
            lines.append(f"{i}. {_summarise_endpoint(ep_id)}")
        lines.append("")
        if diversity_hints:
            lines.append("## Archetypes to avoid (already in the dataset)")
            for hint in diversity_hints:
                lines.append(f"  - {hint}")
            lines.append("")
        lines.append(
            "Generate a realistic user scenario requiring this exact sequence of API calls."
        )
        return "\n".join(lines)


def _summarise_endpoint(endpoint_id: str) -> str:
    """Format an endpoint ID for the planner prompt.

    Input: 'Travel/Hotels/createBooking'
    Output: 'Travel / Hotels / createBooking'

    We do not inject the full Endpoint object here because the Planner is
    constructed without a registry reference (keeping it stateless). The
    endpoint ID alone is sufficient for the LLM to infer the operation from
    the naming convention.
    """
    parts = endpoint_id.split("/")
    if len(parts) == 3:
        category, tool, ep_name = parts
        return f"{category} › {tool} › {ep_name}"
    return endpoint_id
