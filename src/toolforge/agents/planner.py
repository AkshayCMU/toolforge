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
from toolforge.registry.models import Endpoint

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
        chain_endpoints: list[Endpoint] | None = None,
    ) -> TaskPlan:
        """Generate a TaskPlan for the given endpoint chain.

        Args:
            chain: Ordered list of endpoint IDs from ChainSampler.
            persona_seed: Integer seed used to personalise the scenario.
                          Injected into the user prompt for variety.
            diversity_hints: Optional list of archetype strings to avoid
                             (injected by F6.1 CorpusDiversityTracker).
                             Stub in this slice — ignored if None.
            chain_endpoints: Optional Endpoint objects for the chain, used to
                             enrich the prompt with descriptions and parameter
                             names. Falls back to ID-only formatting when None.
        """
        user_prompt = self._build_user_prompt(
            chain, persona_seed, diversity_hints, chain_endpoints
        )
        return self._client.call(
            self.system_prompt,
            user_prompt,
            TaskPlan,
            prompt_version="v1",
            agent_name=self.name,
        )

    # ------------------------------------------------------------------
    # Prompt construction (pure Python, P2)
    # ------------------------------------------------------------------

    def _build_user_prompt(
        self,
        chain: list[str],
        persona_seed: int,
        diversity_hints: list[str] | None,
        chain_endpoints: list[Endpoint] | None = None,
    ) -> str:
        ep_map = {ep.id: ep for ep in (chain_endpoints or [])}
        lines: list[str] = []
        lines.append(f"Persona seed: {persona_seed}")
        lines.append("")
        lines.append("## API call sequence")
        lines.append("")
        for i, ep_id in enumerate(chain, start=1):
            lines.append(f"{i}. {_summarise_endpoint(ep_id, ep_map.get(ep_id))}")
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


def _summarise_endpoint(endpoint_id: str, endpoint: Endpoint | None = None) -> str:
    """Format an endpoint for the planner prompt.

    When a full Endpoint object is available, includes the description and
    required parameter names so the LLM has concrete context for planning.
    Falls back to formatting the ID only when no Endpoint is provided.

    Input (ID only): 'Travel/Hotels/createBooking'
    Output (ID only): 'Travel › Hotels › createBooking'

    Input (with Endpoint): same ID + endpoint.description + required params
    Output: multi-line block with label, description, and required params.
    """
    parts = endpoint_id.split("/")
    label = (
        f"{parts[0]} › {parts[1]} › {parts[2]}" if len(parts) == 3 else endpoint_id
    )
    if endpoint is None:
        return label
    lines = [label]
    if endpoint.description:
        lines.append(f"   description: {endpoint.description}")
    required = [p.name for p in endpoint.parameters if p.required]
    if required:
        lines.append(f"   required params: {', '.join(required)}")
    return "\n".join(lines)
