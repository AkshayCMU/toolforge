"""Unit tests for F4.3 — UserSimulator agent.

All tests use dry_run=True + pre-seeded cache. No live LLM calls.
"""

from __future__ import annotations

import pytest

from toolforge.agents.llm_client import LLMClient
from toolforge.agents.planner import TaskPlan
from toolforge.agents.user_sim import Message, UserSimulator
from tests.unit.test_llm_client import seed_cache


_PLAN = TaskPlan(
    user_persona="A frequent traveller who books hotels monthly",
    initial_query="I need to book a hotel in Paris",
    clarification_points=["What are your check-in and check-out dates?"],
    expected_final_outcome="Hotel booked in Paris with confirmed dates",
    chain_rationale="User creates a booking then reviews the confirmation",
    private_user_knowledge={"check_in_date": "2025-06-01", "check_out_date": "2025-06-05"},
)

_PLAN_NO_PRIVATE = TaskPlan(
    user_persona="A business traveller",
    initial_query="Book me a hotel in London for June 1–3",
    clarification_points=[],
    expected_final_outcome="Hotel booked in London",
    chain_rationale="Complete booking with all details",
    private_user_knowledge={},
)


def _make_sim(tmp_path) -> UserSimulator:
    client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.7,
                       cache_dir=tmp_path, dry_run=True)
    return UserSimulator(client)


def test_respond_returns_string(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    system = sim._build_system_prompt(_PLAN)
    user_prompt = sim._build_user_prompt([])
    seed_cache(sim._client, system, user_prompt, None, "I need to book a hotel in Paris")

    result = sim.respond(_PLAN, history=[])
    assert isinstance(result, str)
    assert result  # non-empty


def test_text_call_uses_text_sentinel(tmp_path) -> None:
    """call_text uses __text__ sentinel — verify the cached key does NOT collide
    with a hypothetical structured call using the same prompts."""
    sim = _make_sim(tmp_path)
    system = sim._build_system_prompt(_PLAN)
    user_prompt = sim._build_user_prompt([])

    text_key = sim._client._cache_key(system, user_prompt, "__text__", None, "v1")
    # Fabricate a structured key for a dummy schema using the same prompts
    from pydantic import BaseModel
    class _Dummy(BaseModel):
        x: str
    struct_key = sim._client._cache_key(system, user_prompt, "_Dummy",
                                         _Dummy.model_json_schema(), "v1")

    assert text_key != struct_key
    assert "__text__" not in struct_key  # sentinel is not in structured keys either


def test_history_formatted_in_user_prompt(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    history: list[Message] = [
        {"role": "user", "content": "I need a hotel"},
        {"role": "assistant", "content": "Which city?"},
    ]
    user_prompt = sim._build_user_prompt(history)

    assert "I need a hotel" in user_prompt
    assert "Which city?" in user_prompt
    assert "You:" in user_prompt or "user" in user_prompt.lower()
    assert "Assistant:" in user_prompt


def test_private_knowledge_in_system_prompt(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    system = sim._build_system_prompt(_PLAN)

    # Both private knowledge keys must appear in the system prompt
    assert "check_in_date" in system
    assert "check_out_date" in system
    assert "2025-06-01" in system


def test_no_private_knowledge_shows_none_message(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    system = sim._build_system_prompt(_PLAN_NO_PRIVATE)

    assert "none" in system.lower() or "shared everything" in system.lower()


def test_empty_history_produces_start_prompt(tmp_path) -> None:
    sim = _make_sim(tmp_path)
    user_prompt = sim._build_user_prompt([])
    assert "initial" in user_prompt.lower() or "start" in user_prompt.lower()
