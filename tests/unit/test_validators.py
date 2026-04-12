"""Unit tests for F5.1 — stage-level structural validation pipeline.

All tests are pure — no live LLM calls, no file I/O, no registry access.
Hand-crafted Conversation and SessionState fixtures cover every validator
branch.
"""

from __future__ import annotations

import pytest

from toolforge.agents.judge import DimensionScore, JudgeResult
from toolforge.evaluation.validators import (
    ValidationResult,
    validate_completeness,
    validate_constraints,
    validate_conversation,
    validate_grounding,
    validate_structure,
    validate_tool_calls,
)
from toolforge.execution.session import SessionState, ToolOutput, make_session
from toolforge.generator.state import Conversation


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tool_output(
    endpoint_id: str = "Travel/Hotels/searchHotels",
    *,
    success: bool = True,
    error: str | None = None,
    turn: int = 0,
) -> ToolOutput:
    if success:
        return ToolOutput(
            endpoint_id=endpoint_id,
            arguments={"city_name": "Paris"},
            response={"hotels": [{"hotel_id": "hotel-001"}]},
            error=None,
            timestamp=f"turn-{turn}",
        )
    return ToolOutput(
        endpoint_id=endpoint_id,
        arguments={},
        response=None,
        error=error or "Some error",
        timestamp=f"turn-{turn}",
    )


def _good_state(tool_outputs: list[ToolOutput] | None = None) -> SessionState:
    state = make_session("conv-test", seed=42)
    if tool_outputs is not None:
        state.tool_outputs = tool_outputs
    return state


def _good_judge_result() -> JudgeResult:
    ds = DimensionScore(score=4, rationale="good")
    return JudgeResult(
        naturalness=ds,
        tool_correctness=ds,
        chain_coherence=ds,
        task_completion=ds,
    )


def _good_conv(messages: list[dict] | None = None) -> Conversation:
    if messages is None:
        messages = [
            {"role": "user", "content": "Book me a hotel in Paris."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city_name": "Paris"}]'},
            {"role": "user", "content": '[tool_result: {"hotels": [{"hotel_id": "hotel-001"}]}]'},
            {"role": "assistant", "content": "I found a hotel for you in Paris. Booking confirmed."},
        ]
    return Conversation(
        conversation_id="conv-test",
        seed=42,
        sampled_chain=["Travel/Hotels/searchHotels"],
        messages=messages,
        session_summary={},
        judge_result=None,
        status="done",
    )


# ---------------------------------------------------------------------------
# ValidationResult model
# ---------------------------------------------------------------------------

class TestValidationResult:
    def test_hard_stages_are_hard(self) -> None:
        for stage in ("structure", "tool_calls", "grounding", "completeness"):
            r = ValidationResult(stage=stage, passed=True, errors=[], warnings=[])
            assert r.is_hard is True, f"{stage} should be hard"

    def test_constraints_is_soft(self) -> None:
        r = ValidationResult(stage="constraints", passed=True, errors=[], warnings=[])
        assert r.is_hard is False

    def test_unknown_stage_is_soft(self) -> None:
        r = ValidationResult(stage="custom_stage", passed=True, errors=[], warnings=[])
        assert r.is_hard is False


# ---------------------------------------------------------------------------
# validate_structure
# ---------------------------------------------------------------------------

class TestValidateStructure:
    def test_passes_well_formed_conversation(self) -> None:
        conv = _good_conv()
        state = _good_state()
        result = validate_structure(conv, state)
        assert result.passed
        assert result.errors == []
        assert result.stage == "structure"
        assert result.is_hard is True

    def test_fails_empty_messages(self) -> None:
        conv = _good_conv(messages=[])
        state = _good_state()
        result = validate_structure(conv, state)
        assert not result.passed
        assert any("empty" in e.lower() for e in result.errors)

    def test_fails_first_message_not_user(self) -> None:
        messages = [
            {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Hi."},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_structure(conv, state)
        assert not result.passed
        assert any("First message" in e for e in result.errors)

    def test_fails_consecutive_same_role(self) -> None:
        messages = [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
            {"role": "assistant", "content": "Also hi."},  # duplicate
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_structure(conv, state)
        assert not result.passed
        assert any("consecutive" in e for e in result.errors)

    def test_fails_invalid_role(self) -> None:
        # Conversation's Pydantic model normally rejects invalid roles, so we
        # use model_construct() to bypass validation and test the validator's
        # defensive check directly.
        bad_messages = [
            {"role": "user", "content": "Hello."},
            {"role": "system", "content": "Bad role."},
        ]
        conv = Conversation.model_construct(
            conversation_id="conv-test",
            seed=42,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=bad_messages,
            session_summary={},
            judge_result=None,
            status="done",
        )
        state = _good_state()
        result = validate_structure(conv, state)
        assert not result.passed
        assert any("invalid role" in e.lower() for e in result.errors)

    def test_fails_empty_content(self) -> None:
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Reply"},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_structure(conv, state)
        assert not result.passed
        assert any("content" in e.lower() for e in result.errors)

    def test_fails_empty_sampled_chain(self) -> None:
        conv = Conversation(
            conversation_id="conv-test",
            seed=42,
            sampled_chain=[],
            messages=[
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello."},
            ],
            session_summary={},
            judge_result=None,
            status="done",
        )
        state = _good_state()
        result = validate_structure(conv, state)
        assert not result.passed
        assert any("sampled_chain" in e for e in result.errors)

    def test_fails_invalid_status(self) -> None:
        conv = Conversation(
            conversation_id="conv-test",
            seed=42,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=[
                {"role": "user", "content": "Hi."},
                {"role": "assistant", "content": "Hello."},
            ],
            session_summary={},
            judge_result=None,
            status="running",  # invalid for a finished conversation
        )
        state = _good_state()
        result = validate_structure(conv, state)
        assert not result.passed
        assert any("status" in e for e in result.errors)


# ---------------------------------------------------------------------------
# validate_tool_calls
# ---------------------------------------------------------------------------

class TestValidateToolCalls:
    def test_passes_valid_tool_call_pair(self) -> None:
        conv = _good_conv()
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert result.passed
        assert result.errors == []
        assert result.is_hard is True

    def test_passes_pure_dialog_no_tool_calls(self) -> None:
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I'm not sure, I can't check live data."},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert result.passed

    def test_fails_malformed_tool_call_no_args(self) -> None:
        messages = [
            {"role": "user", "content": "Search hotels."},
            {"role": "assistant", "content": "[tool_call: Travel/Hotels/searchHotels]"},  # missing args=
            {"role": "user", "content": '[tool_result: {"hotels": []}]'},
            {"role": "assistant", "content": "No hotels found."},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert not result.passed
        assert any("malformed" in e.lower() for e in result.errors)

    def test_fails_args_not_valid_json(self) -> None:
        messages = [
            {"role": "user", "content": "Search hotels."},
            {"role": "assistant", "content": "[tool_call: Travel/Hotels/searchHotels, args={broken json}]"},
            {"role": "user", "content": '[tool_result: {"hotels": []}]'},
            {"role": "assistant", "content": "Done."},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert not result.passed
        assert any("not valid JSON" in e or "json" in e.lower() for e in result.errors)

    def test_fails_tool_call_not_followed_by_result(self) -> None:
        messages = [
            {"role": "user", "content": "Search hotels."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            # Missing tool_result — next message is another assistant turn (consecutive would
            # be caught by structure, but let's make it a user msg without tool_result prefix)
            {"role": "user", "content": "Actually never mind."},
            {"role": "assistant", "content": "OK."},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert not result.passed
        assert any("not followed by a tool_result" in e for e in result.errors)

    def test_fails_tool_call_at_end_of_conversation(self) -> None:
        messages = [
            {"role": "user", "content": "Search hotels."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            # No following message at all.
        ]
        # Note: this also fails validate_structure (no final assistant summary),
        # but validate_tool_calls should also flag it independently.
        conv = Conversation(
            conversation_id="conv-test",
            seed=42,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=messages,
            session_summary={},
            judge_result=None,
            status="done",
        )
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert not result.passed
        assert any("end of conversation" in e for e in result.errors)

    def test_fails_endpoint_id_has_no_slash(self) -> None:
        messages = [
            {"role": "user", "content": "Do something."},
            {"role": "assistant", "content": '[tool_call: hallucinated_endpoint, args={"q": "x"}]'},
            {"role": "user", "content": '[tool_result: {"ok": true}]'},
            {"role": "assistant", "content": "Done."},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert not result.passed
        assert any("no '/'" in e or "slash" in e.lower() or "endpoint_id" in e for e in result.errors)

    def test_multiple_tool_calls_all_checked(self) -> None:
        """Two valid tool_call/result pairs → passes."""
        messages = [
            {"role": "user", "content": "Find and book a hotel."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            {"role": "user", "content": '[tool_result: {"hotels": [{"hotel_id": "h-1"}]}]'},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/createBooking, args={"hotel_id": "h-1"}]'},
            {"role": "user", "content": '[tool_result: {"booking_id": "bk-1"}]'},
            {"role": "assistant", "content": "Booking confirmed."},
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_tool_calls(conv, state)
        assert result.passed


# ---------------------------------------------------------------------------
# validate_grounding
# ---------------------------------------------------------------------------

class TestValidateGrounding:
    def test_passes_no_errors_in_tool_outputs(self) -> None:
        state = _good_state([
            _tool_output("Travel/Hotels/searchHotels", success=True, turn=0),
            _tool_output("Travel/Hotels/createBooking", success=True, turn=1),
        ])
        conv = _good_conv()
        result = validate_grounding(conv, state)
        assert result.passed
        assert result.errors == []
        assert result.is_hard is True

    def test_fails_grounding_error_in_tool_outputs(self) -> None:
        state = _good_state([
            _tool_output("Travel/Hotels/searchHotels", success=True, turn=0),
            _tool_output(
                "Travel/Hotels/createBooking",
                success=False,
                error="Invalid hotel_id: 'fake-123' not in session. Valid values: ['hotel-001']",
                turn=1,
            ),
        ])
        conv = _good_conv()
        result = validate_grounding(conv, state)
        assert not result.passed
        assert len(result.errors) == 1
        assert "not in session" in result.errors[0]
        assert "turn-1" in result.errors[0]

    def test_passes_structural_error_is_not_grounding(self) -> None:
        """Executor 'Missing required parameter' is not a grounding failure."""
        state = _good_state([
            _tool_output(
                "Travel/Hotels/searchHotels",
                success=False,
                error="Missing required parameter: 'city_name'",
                turn=0,
            ),
        ])
        conv = _good_conv()
        result = validate_grounding(conv, state)
        assert result.passed

    def test_passes_unknown_endpoint_is_not_grounding(self) -> None:
        state = _good_state([
            _tool_output(
                "Travel/Hotels/searchHotels",
                success=False,
                error="Unknown endpoint: 'Travel/Hotels/hallucinated'. Known (sample): [...]",
                turn=0,
            ),
        ])
        conv = _good_conv()
        result = validate_grounding(conv, state)
        assert result.passed

    def test_multiple_grounding_errors_all_reported(self) -> None:
        state = _good_state([
            _tool_output(
                "A/B/c",
                success=False,
                error="Invalid hotel_id: 'x' not in session. Valid values: []",
                turn=0,
            ),
            _tool_output(
                "A/B/d",
                success=False,
                error="Invalid booking_id: 'y' not in session. Valid values: []",
                turn=1,
            ),
        ])
        conv = _good_conv()
        result = validate_grounding(conv, state)
        assert not result.passed
        assert len(result.errors) == 2

    def test_empty_tool_outputs_passes(self) -> None:
        state = _good_state([])
        conv = _good_conv()
        result = validate_grounding(conv, state)
        assert result.passed


# ---------------------------------------------------------------------------
# validate_completeness
# ---------------------------------------------------------------------------

class TestValidateCompleteness:
    def test_passes_ends_with_assistant_message(self) -> None:
        conv = _good_conv()
        state = _good_state()
        result = validate_completeness(conv, state)
        assert result.passed
        assert result.errors == []
        assert result.is_hard is True

    def test_fails_ends_with_tool_result_user_message(self) -> None:
        messages = [
            {"role": "user", "content": "Search hotels."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            {"role": "user", "content": '[tool_result: {"hotels": []}]'},
            # Conversation ends here — no final assistant summary.
        ]
        conv = Conversation(
            conversation_id="conv-test",
            seed=42,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=messages,
            session_summary={},
            judge_result=None,
            status="done",
        )
        state = _good_state()
        result = validate_completeness(conv, state)
        assert not result.passed
        assert any("role='user'" in e or "role=user" in e or "user" in e.lower() for e in result.errors)

    def test_fails_ends_with_tool_call_assistant(self) -> None:
        messages = [
            {"role": "user", "content": "Search hotels."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            # Missing tool_result; for completeness check purposes we just check the last msg.
        ]
        conv = Conversation(
            conversation_id="conv-test",
            seed=42,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=messages,
            session_summary={},
            judge_result=None,
            status="done",
        )
        state = _good_state()
        result = validate_completeness(conv, state)
        assert not result.passed
        assert any("tool_call" in e for e in result.errors)

    def test_fails_trivial_final_message(self) -> None:
        messages = [
            {"role": "user", "content": "Done?"},
            {"role": "assistant", "content": "Yes"},  # 3 chars — too short
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_completeness(conv, state)
        assert not result.passed
        assert any("trivially short" in e or "short" in e.lower() for e in result.errors)

    def test_passes_exactly_eleven_char_message(self) -> None:
        """Boundary: 11 chars should pass (> 10 threshold)."""
        messages = [
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": "Done indeed"},  # 11 chars
        ]
        conv = _good_conv(messages=messages)
        state = _good_state()
        result = validate_completeness(conv, state)
        assert result.passed

    def test_fails_empty_messages(self) -> None:
        conv = _good_conv(messages=[])
        state = _good_state()
        result = validate_completeness(conv, state)
        assert not result.passed


# ---------------------------------------------------------------------------
# validate_constraints
# ---------------------------------------------------------------------------

class TestValidateConstraints:
    def test_always_passes(self) -> None:
        """validate_constraints is soft — always passed=True."""
        conv = _good_conv()
        state = _good_state()
        result = validate_constraints(conv, state)
        assert result.passed is True
        assert result.errors == []
        assert result.is_hard is False

    def test_warns_fewer_than_3_successful_calls(self) -> None:
        state = _good_state([
            _tool_output(success=True, turn=0),
            _tool_output(success=True, turn=1),
        ])
        conv = _good_conv()
        result = validate_constraints(conv, state)
        assert result.passed is True
        assert any("3" in w for w in result.warnings)

    def test_warns_fewer_than_2_distinct_endpoints(self) -> None:
        state = _good_state([
            _tool_output("Travel/Hotels/searchHotels", success=True, turn=0),
            _tool_output("Travel/Hotels/searchHotels", success=True, turn=1),
            _tool_output("Travel/Hotels/searchHotels", success=True, turn=2),
        ])
        conv = _good_conv()
        result = validate_constraints(conv, state)
        assert result.passed is True
        assert any("distinct" in w for w in result.warnings)

    def test_warns_majority_calls_failed(self) -> None:
        state = _good_state([
            _tool_output(success=False, error="Error A", turn=0),
            _tool_output(success=False, error="Error B", turn=1),
            _tool_output(success=True, turn=2),
        ])
        conv = _good_conv()
        result = validate_constraints(conv, state)
        assert result.passed is True
        # Should warn about majority failures
        assert any("failed" in w.lower() for w in result.warnings)

    def test_no_warnings_for_healthy_conversation(self) -> None:
        state = _good_state([
            _tool_output("Travel/Hotels/searchHotels", success=True, turn=0),
            _tool_output("Travel/Hotels/createBooking", success=True, turn=1),
            _tool_output("Travel/Hotels/getBookingDetails", success=True, turn=2),
        ])
        conv = Conversation(
            conversation_id="conv-test",
            seed=42,
            sampled_chain=["Travel/Hotels/searchHotels", "Travel/Hotels/createBooking"],
            messages=_good_conv().messages,
            session_summary={},
            judge_result=None,
            status="done",
        )
        result = validate_constraints(conv, state)
        assert result.passed is True
        # No warnings expected for a healthy 3-call, 3-endpoint conversation
        # that completed its 2-step chain.
        assert len(result.warnings) == 0


# ---------------------------------------------------------------------------
# validate_conversation (integration of all five)
# ---------------------------------------------------------------------------

class TestValidateConversation:
    def test_returns_exactly_five_results(self) -> None:
        conv = _good_conv()
        state = _good_state([_tool_output(success=True, turn=0)])
        results = validate_conversation(conv, state)
        assert len(results) == 5

    def test_stage_order(self) -> None:
        conv = _good_conv()
        state = _good_state()
        results = validate_conversation(conv, state)
        assert [r.stage for r in results] == [
            "structure",
            "tool_calls",
            "grounding",
            "completeness",
            "constraints",
        ]

    def test_all_pass_on_good_conversation(self) -> None:
        conv = _good_conv()
        state = _good_state([_tool_output(success=True, turn=0)])
        results = validate_conversation(conv, state)
        # All should pass for a well-formed conversation.
        for r in results:
            if r.stage != "constraints":
                assert r.passed, f"Expected {r.stage} to pass, got errors: {r.errors}"

    def test_all_validators_run_even_on_failure(self) -> None:
        """Even if structure fails, all five validators run (no short-circuit)."""
        conv = _good_conv(messages=[])  # empty → structure fails
        state = _good_state()
        results = validate_conversation(conv, state)
        assert len(results) == 5
        assert not results[0].passed  # structure fails
        # Other validators also ran (completeness will fail too on empty messages)

    def test_is_hard_correct_per_stage(self) -> None:
        conv = _good_conv()
        state = _good_state()
        results = validate_conversation(conv, state)
        hardness = {r.stage: r.is_hard for r in results}
        assert hardness["structure"] is True
        assert hardness["tool_calls"] is True
        assert hardness["grounding"] is True
        assert hardness["completeness"] is True
        assert hardness["constraints"] is False

    def test_deliberate_grounding_failure_fires_only_grounding(self) -> None:
        """A conversation with only a grounding error fires only validate_grounding."""
        state = _good_state([
            _tool_output(
                success=False,
                error="Invalid hotel_id: 'fake' not in session. Valid values: []",
                turn=0,
            ),
        ])
        conv = _good_conv()
        results = validate_conversation(conv, state)
        results_by_stage = {r.stage: r for r in results}
        assert not results_by_stage["grounding"].passed
        # Other hard validators should still pass for this well-formed conversation.
        assert results_by_stage["structure"].passed
        assert results_by_stage["tool_calls"].passed
        assert results_by_stage["completeness"].passed
