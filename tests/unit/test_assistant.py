"""Unit tests for F4.4 — Assistant agent.

All tests use dry_run=True + pre-seeded cache. No live LLM calls.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolforge.agents.assistant import (
    Assistant,
    AssistantTurn,
    _format_endpoint_catalog,
    _format_session_registry,
    select_distractors,
)
from toolforge.agents.llm_client import LLMClient
from toolforge.agents.user_sim import Message
from toolforge.execution.session import SessionState, ToolOutput, make_session
from toolforge.registry.models import Endpoint, Parameter, ParamProvenance, ResponseField
from tests.unit.test_llm_client import seed_cache


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _prov() -> ParamProvenance:
    return ParamProvenance(raw_required_field="required_parameters", raw_type_string="STRING")


def _ep(ep_id: str, name: str = "", description: str = "A test endpoint",
        params: tuple[Parameter, ...] = ()) -> Endpoint:
    return Endpoint(id=ep_id, name=name or ep_id.split("/")[-1],
                    description=description, parameters=params)


def _param(name: str, required: bool = True, type_: str = "string") -> Parameter:
    return Parameter(name=name, type=type_, description=f"Param {name}",
                     required=required, provenance=_prov())


CHAIN_EPS = [
    _ep("Travel/Hotels/createBooking", params=(_param("city_name"), _param("check_in_date"))),
    _ep("Travel/Hotels/fetchBookingDetails", params=(_param("booking_id"),)),
]

HISTORY: list[Message] = [
    {"role": "user", "content": "I need to book a hotel in Paris"},
    {"role": "assistant", "content": "What dates would you like?"},
    {"role": "user", "content": "June 1st through the 5th"},
]


def _make_assistant(tmp_path) -> Assistant:
    client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.0,
                       cache_dir=tmp_path, dry_run=True)
    return Assistant(client)


def _seed_turn(asst: Assistant, history, state: SessionState, data: dict, tmp_path) -> None:
    """Pre-seed a cache entry for assistant.act()."""
    system = asst._build_system_prompt(CHAIN_EPS, state)
    user_prompt = asst._build_user_prompt(history)
    seed_cache(asst._client, system, user_prompt, AssistantTurn, data)


# ---------------------------------------------------------------------------
# AssistantTurn model validation
# ---------------------------------------------------------------------------

def test_message_turn_valid() -> None:
    turn = AssistantTurn(type="message", content="What city?")
    assert turn.type == "message"
    assert turn.content == "What city?"


def test_tool_call_turn_valid() -> None:
    turn = AssistantTurn(type="tool_call", endpoint="Travel/Hotels/createBooking",
                         arguments={"city_name": "Paris"})
    assert turn.type == "tool_call"
    assert turn.endpoint == "Travel/Hotels/createBooking"


def test_tool_call_without_endpoint_raises() -> None:
    with pytest.raises(ValidationError, match="endpoint"):
        AssistantTurn(type="tool_call", endpoint="", arguments={})


def test_message_without_content_raises() -> None:
    with pytest.raises(ValidationError, match="content"):
        AssistantTurn(type="message", content="")


# ---------------------------------------------------------------------------
# act() via dry_run cache
# ---------------------------------------------------------------------------

def test_act_returns_assistant_turn(tmp_path) -> None:
    asst = _make_assistant(tmp_path)
    state = make_session("conv-1", seed=42)
    _seed_turn(asst, HISTORY, state,
               {"type": "message", "content": "Let me create that booking for you."}, tmp_path)

    result = asst.act(HISTORY, state, CHAIN_EPS, distractors=[])
    assert isinstance(result, AssistantTurn)


def test_message_turn_has_content(tmp_path) -> None:
    asst = _make_assistant(tmp_path)
    state = make_session("conv-1", seed=42)
    _seed_turn(asst, HISTORY, state,
               {"type": "message", "content": "Which city should I book?"}, tmp_path)

    result = asst.act(HISTORY, state, CHAIN_EPS, distractors=[])
    assert result.type == "message"
    assert result.content


def test_tool_call_turn_has_endpoint_and_args(tmp_path) -> None:
    asst = _make_assistant(tmp_path)
    state = make_session("conv-1", seed=42)
    _seed_turn(asst, HISTORY, state, {
        "type": "tool_call",
        "endpoint": "Travel/Hotels/createBooking",
        "arguments": {"city_name": "Paris", "check_in_date": "2025-06-01"},
    }, tmp_path)

    result = asst.act(HISTORY, state, CHAIN_EPS, distractors=[])
    assert result.type == "tool_call"
    assert result.endpoint
    assert isinstance(result.arguments, dict)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_session_registry_in_prompt(tmp_path) -> None:
    asst = _make_assistant(tmp_path)
    state = make_session("conv-1", seed=42)
    state.available_values_by_type["booking_id"] = ["uuid-001"]

    system = asst._build_system_prompt(CHAIN_EPS, state)
    assert "booking_id" in system
    assert "uuid-001" in system


def test_grounding_rule_in_prompt(tmp_path) -> None:
    asst = _make_assistant(tmp_path)
    state = make_session("conv-1", seed=42)

    system = asst._build_system_prompt(CHAIN_EPS, state)
    assert "exact value" in system.lower() or "do not invent" in system.lower()


def test_endpoint_catalog_in_prompt(tmp_path) -> None:
    asst = _make_assistant(tmp_path)
    state = make_session("conv-1", seed=42)

    system = asst._build_system_prompt(CHAIN_EPS, state)
    assert "createBooking" in system
    assert "fetchBookingDetails" in system
    assert "city_name" in system


def test_tool_outputs_in_registry_view() -> None:
    state = make_session("conv-1", seed=42)
    state.tool_outputs.append(ToolOutput(
        endpoint_id="Travel/Hotels/createBooking",
        arguments={"city_name": "Paris"},
        response={"booking_id": "uuid-001"},
        error=None,
        timestamp="turn-0",
    ))
    registry = _format_session_registry(state)
    assert "turn-0" in registry
    assert "createBooking" in registry
    assert "success" in registry


# ---------------------------------------------------------------------------
# select_distractors
# ---------------------------------------------------------------------------

def test_select_distractors_returns_correct_count() -> None:
    chain = [_ep("Travel/Hotels/createBooking")]
    all_eps = [
        _ep("Travel/Hotels/fetchBookingDetails"),
        _ep("Travel/Hotels/updateBooking"),
        _ep("Travel/Flights/searchFlights"),
        _ep("Financial/Payments/createPayment"),
        _ep("Financial/Payments/getPayment"),
    ]
    result = select_distractors(chain, all_eps, seed=42, n=3)
    assert len(result) == 3


def test_select_distractors_excludes_chain_endpoints() -> None:
    chain = [_ep("Travel/Hotels/createBooking")]
    all_eps = [
        _ep("Travel/Hotels/createBooking"),  # must be excluded
        _ep("Travel/Hotels/fetchBookingDetails"),
        _ep("Travel/Flights/searchFlights"),
    ]
    result = select_distractors(chain, all_eps, seed=42, n=2)
    chain_ids = {ep.id for ep in chain}
    for ep in result:
        assert ep.id not in chain_ids


def test_select_distractors_is_deterministic() -> None:
    chain = [_ep("Travel/Hotels/createBooking")]
    all_eps = [_ep(f"Travel/Hotels/ep{i}") for i in range(10)]
    r1 = select_distractors(chain, all_eps, seed=7, n=3)
    r2 = select_distractors(chain, all_eps, seed=7, n=3)
    assert [e.id for e in r1] == [e.id for e in r2]
