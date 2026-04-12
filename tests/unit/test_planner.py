"""Unit tests for F4.2 — Planner agent.

All tests use dry_run=True + pre-seeded cache. No live LLM calls.
"""

from __future__ import annotations

import pytest

from toolforge.agents.llm_client import LLMClient
from toolforge.agents.planner import Planner, TaskPlan, _summarise_endpoint
from toolforge.registry.models import Endpoint, Parameter, ParamProvenance
from tests.unit.test_llm_client import seed_cache


CHAIN = [
    "Travel/Hotels/createBooking",
    "Travel/Hotels/fetchBookingDetails",
    "Travel/Hotels/updateBooking",
]

_PLAN_DATA = {
    "user_persona": "A frequent traveller who books hotels monthly",
    "initial_query": "I need to book a hotel in Paris for next week",
    "clarification_points": ["What are your check-in and check-out dates?"],
    "expected_final_outcome": "Hotel booked and confirmed with updated guest preferences",
    "chain_rationale": "User creates a booking, checks details, then updates preferences",
    "private_user_knowledge": {},
}

_PLAN_DATA_WITH_PRIVATE = {
    "user_persona": "A business traveller on a corporate account",
    "initial_query": "Book me a hotel",
    "clarification_points": ["Which city?", "What dates?"],
    "expected_final_outcome": "Corporate hotel booking created",
    "chain_rationale": "Standard booking flow with ambiguous initial request",
    "private_user_knowledge": {"city_name": "London", "check_in_date": "2025-06-01"},
}


def _make_planner(tmp_path) -> Planner:
    client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.7,
                       cache_dir=tmp_path, dry_run=True)
    return Planner(client)


def test_plan_returns_task_plan_instance(tmp_path) -> None:
    planner = _make_planner(tmp_path)
    user_prompt = planner._build_user_prompt(CHAIN, persona_seed=42, diversity_hints=None)
    seed_cache(planner._client, planner.system_prompt, user_prompt, TaskPlan, _PLAN_DATA)

    result = planner.plan(CHAIN, persona_seed=42)
    assert isinstance(result, TaskPlan)


def test_plan_fields_populated(tmp_path) -> None:
    planner = _make_planner(tmp_path)
    user_prompt = planner._build_user_prompt(CHAIN, persona_seed=42, diversity_hints=None)
    seed_cache(planner._client, planner.system_prompt, user_prompt, TaskPlan, _PLAN_DATA)

    result = planner.plan(CHAIN, persona_seed=42)
    assert result.user_persona
    assert result.initial_query
    assert result.expected_final_outcome
    assert result.chain_rationale


def test_private_knowledge_nonempty_for_disambiguation_seeds(tmp_path) -> None:
    """Pre-seed 10 plans; at least 3 have non-empty private_user_knowledge (~40%)."""
    client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.7,
                       cache_dir=tmp_path, dry_run=True)
    planner = Planner(client)

    # 7 plans without private knowledge, 3 with — simulates ~30% disambiguation rate
    for seed in range(7):
        up = planner._build_user_prompt(CHAIN, persona_seed=seed, diversity_hints=None)
        seed_cache(client, planner.system_prompt, up, TaskPlan, _PLAN_DATA)
    for seed in range(7, 10):
        up = planner._build_user_prompt(CHAIN, persona_seed=seed, diversity_hints=None)
        seed_cache(client, planner.system_prompt, up, TaskPlan, _PLAN_DATA_WITH_PRIVATE)

    with_private = sum(
        1 for s in range(10)
        if planner.plan(CHAIN, persona_seed=s).private_user_knowledge
    )
    assert with_private >= 3


def test_diversity_hints_none_does_not_crash(tmp_path) -> None:
    planner = _make_planner(tmp_path)
    user_prompt = planner._build_user_prompt(CHAIN, persona_seed=0, diversity_hints=None)
    seed_cache(planner._client, planner.system_prompt, user_prompt, TaskPlan, _PLAN_DATA)

    result = planner.plan(CHAIN, persona_seed=0, diversity_hints=None)
    assert isinstance(result, TaskPlan)


def test_chain_summary_in_prompt(tmp_path) -> None:
    planner = _make_planner(tmp_path)
    user_prompt = planner._build_user_prompt(CHAIN, persona_seed=42, diversity_hints=None)

    # All three endpoint names must appear in the built prompt
    assert "createBooking" in user_prompt
    assert "fetchBookingDetails" in user_prompt
    assert "updateBooking" in user_prompt


def test_diversity_hints_appear_in_prompt(tmp_path) -> None:
    planner = _make_planner(tmp_path)
    hints = ["hotel booking for leisure trip", "flight search scenario"]
    user_prompt = planner._build_user_prompt(CHAIN, persona_seed=1, diversity_hints=hints)

    assert "hotel booking for leisure trip" in user_prompt
    assert "flight search scenario" in user_prompt


# ---------------------------------------------------------------------------
# F4.2 — endpoint metadata enrichment
# ---------------------------------------------------------------------------

def _prov() -> ParamProvenance:
    return ParamProvenance(raw_required_field="required_parameters", raw_type_string="STRING")


def _ep(ep_id: str, description: str = "", params: list[Parameter] | None = None) -> Endpoint:
    return Endpoint(
        id=ep_id,
        name=ep_id.split("/")[-1],
        description=description,
        parameters=params or [],
    )


def _param(name: str, required: bool = True) -> Parameter:
    return Parameter(
        name=name, type="string", description=f"The {name}",
        required=required, provenance=_prov(),
    )


def test_summarise_endpoint_id_only() -> None:
    """Without an Endpoint object, returns the formatted ID only."""
    result = _summarise_endpoint("Travel/Hotels/createBooking")
    assert result == "Travel › Hotels › createBooking"
    assert "description" not in result


def test_summarise_endpoint_with_metadata() -> None:
    """With an Endpoint object, includes description and required params."""
    ep = _ep(
        "Travel/Hotels/createBooking",
        description="Create a hotel booking",
        params=[_param("city_name"), _param("check_in_date"), _param("notes", required=False)],
    )
    result = _summarise_endpoint("Travel/Hotels/createBooking", ep)
    assert "Travel › Hotels › createBooking" in result
    assert "Create a hotel booking" in result
    assert "city_name" in result
    assert "check_in_date" in result
    assert "notes" not in result  # optional params not listed


def test_endpoint_description_in_prompt_when_endpoints_provided(tmp_path) -> None:
    """When chain_endpoints are passed to plan(), prompt includes their descriptions."""
    planner = _make_planner(tmp_path)
    endpoints = [
        _ep(ep_id, description=f"Desc for {ep_id.split('/')[-1]}", params=[_param("param1")])
        for ep_id in CHAIN
    ]
    user_prompt = planner._build_user_prompt(
        CHAIN, persona_seed=42, diversity_hints=None, chain_endpoints=endpoints
    )
    assert "Desc for createBooking" in user_prompt
    assert "Desc for fetchBookingDetails" in user_prompt
    assert "param1" in user_prompt


def test_plan_without_chain_endpoints_still_works(tmp_path) -> None:
    """chain_endpoints=None falls back to ID-only formatting; existing cache tests still pass."""
    planner = _make_planner(tmp_path)
    user_prompt = planner._build_user_prompt(CHAIN, persona_seed=42, diversity_hints=None)
    seed_cache(planner._client, planner.system_prompt, user_prompt, TaskPlan, _PLAN_DATA)

    result = planner.plan(CHAIN, persona_seed=42)
    assert isinstance(result, TaskPlan)
