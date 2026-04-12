"""Integration test for F5.2 — repair loop (REQUIRED by assignment).

Scenario: hallucinated hotel_id in a tool_call → grounding validator fires →
RepairAgent (live LLM) suggests a regenerate_turn fix → judge re-passes.

Run with:
    pytest tests/integration/test_repair_loop.py -m integration -v

Requires ANTHROPIC_API_KEY in the environment (or .env file).
Calls are cached to .cache/llm/ so subsequent runs are free.

Design notes
------------
The conversation is hand-crafted (deterministic, no LangGraph) to isolate
the repair+judge path.  The session has:
  - A successful searchHotels call that produced hotel_id "hotel-real-42".
  - A failed createBooking call that used a hallucinated "hotel-FAKE-99".
validate_grounding fires on the failed call.
RepairAgent must detect the wrong ID and suggest regenerate_turn with the
correct ID from the prior result.
"""

from __future__ import annotations

import os

import pytest

from toolforge.agents.judge import Judge
from toolforge.agents.llm_client import LLMClient
from toolforge.agents.repair import RepairAgent
from toolforge.evaluation.repair import run_repair
from toolforge.evaluation.validators import validate_conversation
from toolforge.execution.session import SessionState, ToolOutput, make_session
from toolforge.generator.state import Conversation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_failing_conv() -> tuple[Conversation, SessionState]:
    """Build a hand-crafted conversation with a grounding failure."""
    session = make_session("integ-conv-repair-001", seed=99)

    # Turn 0: successful search that produces hotel_id "hotel-real-42"
    session.tool_outputs.append(ToolOutput(
        endpoint_id="Travel/Hotels/searchHotels",
        arguments={"city_name": "Barcelona"},
        response={"hotels": [{"hotel_id": "hotel-real-42", "name": "Hotel Arts Barcelona"}]},
        error=None,
        timestamp="turn-0",
    ))
    session.available_values_by_type["hotel_id"] = ["hotel-real-42"]

    # Turn 1: failed createBooking with hallucinated ID
    session.tool_outputs.append(ToolOutput(
        endpoint_id="Travel/Hotels/createBooking",
        arguments={"hotel_id": "hotel-FAKE-99", "check_in": "2025-08-01"},
        response=None,
        error=(
            "Invalid hotel_id: 'hotel-FAKE-99' not in session. "
            "Valid values: ['hotel-real-42']"
        ),
        timestamp="turn-1",
    ))

    messages = [
        {"role": "user", "content": "Book a hotel in Barcelona for August 1st."},
        {
            "role": "assistant",
            "content": '[tool_call: Travel/Hotels/searchHotels, args={"city_name": "Barcelona"}]',
        },
        {
            "role": "user",
            "content": (
                '[tool_result: {"hotels": [{"hotel_id": "hotel-real-42", '
                '"name": "Hotel Arts Barcelona"}]}]'
            ),
        },
        {
            "role": "assistant",
            "content": (
                '[tool_call: Travel/Hotels/createBooking, '
                'args={"hotel_id": "hotel-FAKE-99", "check_in": "2025-08-01"}]'
            ),
        },
        {
            "role": "user",
            "content": (
                '[tool_result: {"error": "Invalid hotel_id: hotel-FAKE-99 not in session"}]'
            ),
        },
        # No final assistant summary either → also fails validate_completeness
    ]

    conv = Conversation(
        conversation_id="integ-conv-repair-001",
        seed=99,
        sampled_chain=[
            "Travel/Hotels/searchHotels",
            "Travel/Hotels/createBooking",
        ],
        messages=messages,
        session_summary={},
        judge_result=None,
        status="failed",
    )
    return conv, session


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="Requires ANTHROPIC_API_KEY for live LLM integration test",
)
def test_hallucinated_id_repaired_and_judge_passes() -> None:
    """Full repair loop: grounding failure → repair → judge passes.

    This is the REQUIRED integration test from FEATURES.md F5.2.
    Uses live LLM calls (cached after first run).
    """
    conv, session = _build_failing_conv()

    # Pre-repair: assert grounding validator fires.
    results_before = validate_conversation(conv, session)
    grounding = next(r for r in results_before if r.stage == "grounding")
    assert not grounding.passed, (
        f"Expected grounding failure before repair, got: {grounding.errors}"
    )

    # Build agents with live LLM (cached).
    client_sonnet = LLMClient(model="claude-sonnet-4-6", temperature=0.0)
    repair_agent = RepairAgent(client_sonnet)
    judge = Judge(client_sonnet)

    # Run the repair loop.
    repaired_conv, judge_result, attempts_used = run_repair(
        conv=conv,
        state=session,
        repair_agent=repair_agent,
        judge=judge,
    )

    # Post-repair: grounding must pass.
    results_after = validate_conversation(repaired_conv, session)
    grounding_after = next(r for r in results_after if r.stage == "grounding")
    assert grounding_after.passed, (
        f"Grounding still failing after repair:\n{grounding_after.errors}"
    )

    # Judge must pass.
    assert judge_result is not None, (
        "Expected a JudgeResult after repair — hard validators passed but judge was not called"
    )
    assert judge_result.overall_pass, (
        f"Judge did not pass after repair.\n"
        f"  naturalness={judge_result.naturalness.score}\n"
        f"  tool_correctness={judge_result.tool_correctness.score}\n"
        f"  chain_coherence={judge_result.chain_coherence.score}\n"
        f"  task_completion={judge_result.task_completion.score}\n"
        f"  mean={judge_result.mean_score():.2f}\n"
        f"  failure_modes={judge_result.failure_modes}"
    )

    assert attempts_used >= 1, "Expected at least one repair attempt"
