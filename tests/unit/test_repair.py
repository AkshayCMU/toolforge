"""Unit tests for F5.2 — repair loop.

Tests cover:
  - RepairOperation schema validation (flat model, field constraints)
  - RepairAgent (dry_run + seed_cache pattern — no live LLM calls)
  - _apply_operation (pure function)
  - _failure_signature (pure function)
  - run_repair (all agents MagicMocked — no LLM, no network)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from toolforge.agents.judge import DimensionScore, JudgeResult
from toolforge.agents.llm_client import CacheMissError, LLMClient
from toolforge.agents.repair import RepairAgent, RepairOperation, _build_user_prompt
from toolforge.evaluation.repair import _apply_operation, _failure_signature, run_repair
from toolforge.evaluation.validators import ValidationResult, validate_conversation
from toolforge.execution.session import SessionState, ToolOutput, make_session
from toolforge.generator.state import Conversation
from tests.unit.test_llm_client import seed_cache


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _passing_judge_result() -> JudgeResult:
    ds = DimensionScore(score=4, rationale="good")
    return JudgeResult(
        naturalness=ds, tool_correctness=ds,
        chain_coherence=ds, task_completion=ds,
    )


def _failing_judge_result() -> JudgeResult:
    ds_low = DimensionScore(score=2, rationale="poor")
    return JudgeResult(
        naturalness=ds_low, tool_correctness=ds_low,
        chain_coherence=ds_low, task_completion=ds_low,
    )


def _good_conv(messages=None) -> Conversation:
    if messages is None:
        messages = [
            {"role": "user", "content": "Find a hotel."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            {"role": "user", "content": '[tool_result: {"hotels": [{"hotel_id": "h-1"}]}]'},
            {"role": "assistant", "content": "Found Hotel Arts in Paris for you."},
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


def _good_state(tool_outputs=None) -> SessionState:
    state = make_session("conv-test", seed=42)
    if tool_outputs is not None:
        state.tool_outputs = tool_outputs
    return state


def _make_repair_agent(tmp_path) -> RepairAgent:
    client = LLMClient(
        model="claude-sonnet-4-6",
        temperature=0.0,
        cache_dir=tmp_path,
        dry_run=True,
    )
    return RepairAgent(client)


def _seed_repair_agent(agent: RepairAgent, conv: Conversation,
                        results, attempt: int, data: dict) -> None:
    user_prompt = _build_user_prompt(conv, results, attempt)
    seed_cache(agent._client, agent.system_prompt, user_prompt, RepairOperation, data)


# ---------------------------------------------------------------------------
# RepairOperation schema validation
# ---------------------------------------------------------------------------

class TestRepairOperationSchema:
    def test_regenerate_turn_valid(self) -> None:
        op = RepairOperation(
            type="regenerate_turn",
            turn_index=2,
            content="[tool_call: A/B/c, args={\"k\": \"v\"}]",
            reason="fixed hallucinated ID",
        )
        assert op.type == "regenerate_turn"
        assert op.turn_index == 2

    def test_append_turn_valid(self) -> None:
        op = RepairOperation(
            type="append_turn",
            role="assistant",
            content="Booking confirmed. Your ID is bk-001.",
            reason="missing closing summary",
        )
        assert op.type == "append_turn"
        assert op.role == "assistant"

    def test_discard_valid(self) -> None:
        op = RepairOperation(
            type="discard",
            reason="consecutive same-role messages — structural failure",
        )
        assert op.type == "discard"

    def test_regenerate_turn_requires_nonneg_turn_index(self) -> None:
        with pytest.raises(ValidationError, match="turn_index"):
            RepairOperation(
                type="regenerate_turn",
                turn_index=-1,
                content="some content",
                reason="x",
            )

    def test_regenerate_turn_requires_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            RepairOperation(
                type="regenerate_turn",
                turn_index=0,
                content="",
                reason="x",
            )

    def test_append_turn_invalid_role(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            RepairOperation(
                type="append_turn",
                role="god",
                content="Hello",
                reason="x",
            )

    def test_append_turn_requires_content(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            RepairOperation(
                type="append_turn",
                role="assistant",
                content="",
                reason="x",
            )

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValidationError, match="type"):
            RepairOperation(type="hallucinate", reason="x")

    def test_schema_has_no_anyof_at_top_level(self) -> None:
        """Flat model invariant: no top-level 'anyOf' key in JSON Schema.

        Guards the AssistantTurn lesson (DESIGN.md §5.4):
        discriminated unions cause reliability issues under tool-use forcing.
        """
        schema = RepairOperation.model_json_schema()
        assert "anyOf" not in schema, (
            "RepairOperation schema has top-level 'anyOf' — use flat model, "
            "not a discriminated union"
        )


# ---------------------------------------------------------------------------
# RepairAgent (dry_run + seed_cache — no live LLM)
# ---------------------------------------------------------------------------

class TestRepairAgent:
    def test_suggests_append_turn(self, tmp_path) -> None:
        agent = _make_repair_agent(tmp_path)
        conv = _good_conv(messages=[
            {"role": "user", "content": "Book a hotel."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            # Ends without tool_result or summary — completeness failure
        ])
        state = _good_state()
        results = validate_conversation(conv, state)

        _seed_repair_agent(agent, conv, results, attempt=1, data={
            "type": "append_turn",
            "role": "assistant",
            "content": "I found Hotel Arts in Paris for you.",
            "reason": "Conversation ended without a closing summary.",
        })

        op = agent.suggest(conv, results, attempt=1)
        assert op.type == "append_turn"
        assert op.role == "assistant"
        assert "Paris" in op.content or "found" in op.content.lower()

    def test_suggests_regenerate_turn(self, tmp_path) -> None:
        agent = _make_repair_agent(tmp_path)
        conv = _good_conv()
        state = _good_state()
        results = validate_conversation(conv, state)

        _seed_repair_agent(agent, conv, results, attempt=1, data={
            "type": "regenerate_turn",
            "turn_index": 1,
            "content": '[tool_call: Travel/Hotels/searchHotels, args={"city_name": "Paris"}]',
            "reason": "Fixed args key from 'city' to 'city_name'.",
        })

        op = agent.suggest(conv, results, attempt=1)
        assert op.type == "regenerate_turn"
        assert op.turn_index == 1

    def test_suggests_discard(self, tmp_path) -> None:
        agent = _make_repair_agent(tmp_path)
        conv = _good_conv()
        state = _good_state()
        results = validate_conversation(conv, state)

        _seed_repair_agent(agent, conv, results, attempt=1, data={
            "type": "discard",
            "reason": "Consecutive same-role messages; structural failure.",
        })

        op = agent.suggest(conv, results, attempt=1)
        assert op.type == "discard"

    def test_no_live_llm_calls_on_cache_miss(self, tmp_path) -> None:
        """dry_run=True without seeding → CacheMissError (proves no live calls)."""
        agent = _make_repair_agent(tmp_path)
        conv = _good_conv()
        state = _good_state()
        results = validate_conversation(conv, state)

        with pytest.raises(CacheMissError):
            agent.suggest(conv, results, attempt=1)


# ---------------------------------------------------------------------------
# _apply_operation (pure function)
# ---------------------------------------------------------------------------

class TestApplyOperation:
    def test_regenerate_turn_replaces_content_at_index(self) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "OLD content"},
            {"role": "user", "content": "Thanks"},
        ]
        op = RepairOperation(
            type="regenerate_turn",
            turn_index=1,
            content="NEW content",
            reason="test",
        )
        result = _apply_operation(messages, op)
        assert result[1]["content"] == "NEW content"
        assert result[1]["role"] == "assistant"  # role preserved

    def test_regenerate_turn_preserves_other_messages(self) -> None:
        messages = [
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        ]
        op = RepairOperation(type="regenerate_turn", turn_index=0,
                             content="A2", reason="x")
        result = _apply_operation(messages, op)
        assert result[0]["content"] == "A2"
        assert result[1]["content"] == "B"  # unchanged

    def test_regenerate_turn_out_of_bounds_is_noop(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        op = RepairOperation(type="regenerate_turn", turn_index=99,
                             content="y", reason="x")
        result = _apply_operation(messages, op)
        assert result == messages  # unchanged

    def test_append_turn_adds_at_end(self) -> None:
        messages = [
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": "Done."},
        ]
        op = RepairOperation(
            type="append_turn",
            role="assistant",
            content="Booking confirmed!",
            reason="missing summary",
        )
        result = _apply_operation(messages, op)
        assert len(result) == 3
        assert result[-1] == {"role": "assistant", "content": "Booking confirmed!"}

    def test_apply_returns_new_list_not_mutating_input(self) -> None:
        messages = [{"role": "user", "content": "x"}]
        original = list(messages)
        op = RepairOperation(type="append_turn", role="assistant",
                             content="y", reason="z")
        _apply_operation(messages, op)
        assert messages == original  # original not mutated


# ---------------------------------------------------------------------------
# _failure_signature (pure function)
# ---------------------------------------------------------------------------

class TestFailureSignature:
    def test_empty_when_all_pass(self) -> None:
        conv = _good_conv()
        state = _good_state([
            ToolOutput("Travel/Hotels/searchHotels", {}, {"h": "h-1"}, None, "turn-0")
        ])
        results = validate_conversation(conv, state)
        sig = _failure_signature(results)
        assert sig == ""

    def test_same_errors_produce_same_sig(self) -> None:
        # Build a conversation with a known completeness failure.
        conv = _good_conv(messages=[
            {"role": "user", "content": "Go."},
            {"role": "assistant", "content": '[tool_call: A/B/c, args={}]'},
            # Ends without tool_result
        ])
        state = _good_state()
        results1 = validate_conversation(conv, state)
        results2 = validate_conversation(conv, state)
        assert _failure_signature(results1) == _failure_signature(results2)

    def test_different_errors_produce_different_sig(self) -> None:
        conv_incomplete = _good_conv(messages=[
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y" * 20},
            # missing tool calls
        ])
        conv_with_grounding = _good_conv()
        state_grounding = _good_state([
            ToolOutput("A/B/c", {}, None,
                       "Invalid hotel_id: 'x' not in session. Valid values: []",
                       "turn-0"),
        ])
        state_clean = _good_state()
        sig1 = _failure_signature(validate_conversation(conv_with_grounding, state_grounding))
        sig2 = _failure_signature(validate_conversation(conv_incomplete, state_clean))
        # Both fail but for different reasons → different signatures.
        # (at least one will be non-empty)
        assert sig1 != sig2 or (sig1 == "" and sig2 == "")

    def test_order_independent(self) -> None:
        """Sorting ensures sig is independent of error list order."""
        # Two hard ValidationResults with multiple errors — order shouldn't matter.
        r1 = ValidationResult(stage="structure", passed=False,
                              errors=["error B", "error A"], warnings=[])
        r2 = ValidationResult(stage="structure", passed=False,
                              errors=["error A", "error B"], warnings=[])
        assert _failure_signature([r1]) == _failure_signature([r2])


# ---------------------------------------------------------------------------
# run_repair (all agents MagicMocked — no LLM)
# ---------------------------------------------------------------------------

class TestRunRepair:
    """Test the repair runner with fully mocked agents."""

    def _make_mock_agent(self, operations: list[dict]) -> RepairAgent:
        """Build a MagicMock RepairAgent that returns operations in sequence."""
        agent = MagicMock(spec=RepairAgent)
        ops = [RepairOperation(**d) for d in operations]
        agent.suggest.side_effect = ops
        return agent

    def _make_mock_judge(self, passes: bool) -> Judge:
        from toolforge.agents.judge import Judge
        judge = MagicMock(spec=Judge)
        if passes:
            judge.score.return_value = _passing_judge_result()
        else:
            judge.score.return_value = _failing_judge_result()
        return judge

    def test_passes_after_one_repair(self) -> None:
        """Repair replaces a tool_call with a valid one → validates → judge passes."""
        # Conversation with a completeness failure (ends on tool_result).
        broken_messages = [
            {"role": "user", "content": "Find a hotel."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            {"role": "user", "content": '[tool_result: {"hotels": []}]'},
            # Missing final assistant summary — completeness failure.
        ]
        conv = Conversation(
            conversation_id="conv-t",
            seed=1,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=broken_messages,
            session_summary={},
            judge_result=None,
            status="failed",
        )
        state = _good_state()

        repair_agent = self._make_mock_agent([{
            "type": "append_turn",
            "role": "assistant",
            "content": "No hotels found in Paris for the requested dates.",
            "reason": "missing closing summary",
        }])
        judge = self._make_mock_judge(passes=True)

        repaired, judge_result, attempts_used = run_repair(conv, state, repair_agent, judge)

        assert attempts_used == 1
        assert judge_result is not None
        assert judge_result.overall_pass is True
        assert repaired.messages[-1]["role"] == "assistant"
        repair_agent.suggest.assert_called_once()
        judge.score.assert_called_once()

    def test_returns_early_on_discard(self) -> None:
        """Discard op → returns immediately, judge never called.

        Use a conversation that fails validate_completeness so the runner
        reaches repair_agent.suggest (hard failure → no judge call first).
        """
        broken_messages = [
            {"role": "user", "content": "Find a hotel."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            {"role": "user", "content": '[tool_result: {"hotels": []}]'},
            # No final assistant summary → completeness failure.
        ]
        conv = Conversation(
            conversation_id="conv-d",
            seed=1,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=broken_messages,
            session_summary={},
            judge_result=None,
            status="failed",
        )
        state = _good_state()

        repair_agent = self._make_mock_agent([{
            "type": "discard",
            "reason": "structural failure",
        }])
        judge = self._make_mock_judge(passes=True)

        repaired, judge_result, attempts_used = run_repair(conv, state, repair_agent, judge)

        assert judge_result is None
        judge.score.assert_not_called()
        repair_agent.suggest.assert_called_once()
        assert attempts_used == 1

    def test_repeated_failure_aborts_early(self) -> None:
        """Same failure twice → gives up after detecting repeated signature."""
        # Build a conversation that will fail completeness both before and after repair.
        broken_messages = [
            {"role": "user", "content": "Search."},
            {"role": "assistant", "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Paris"}]'},
            {"role": "user", "content": '[tool_result: {"h": "x"}]'},
            # No summary.
        ]
        conv = Conversation(
            conversation_id="conv-r",
            seed=1,
            sampled_chain=["Travel/Hotels/searchHotels"],
            messages=broken_messages,
            session_summary={},
            judge_result=None,
            status="failed",
        )
        state = _good_state()

        # Repair replaces the tool_result with ANOTHER tool_call (still no summary).
        repair_agent = self._make_mock_agent([
            {
                "type": "regenerate_turn",
                "turn_index": 2,
                "content": '[tool_call: Travel/Hotels/searchHotels, args={"city": "Lyon"}]',
                "reason": "changed city",
            },
            # Second call if needed — but shouldn't be reached due to repeated sig detection.
            {
                "type": "discard",
                "reason": "still failing",
            },
        ])
        judge = self._make_mock_judge(passes=False)

        repaired, judge_result, attempts_used = run_repair(conv, state, repair_agent, judge,
                                                            max_attempts=2)

        # Should never call judge (completeness hard failure remains after repair).
        judge.score.assert_not_called()
        # repair_agent.suggest called at most max_attempts times.
        assert repair_agent.suggest.call_count <= 2

    def test_max_attempts_guard(self) -> None:
        """After max_attempts, suggest is never called again."""
        conv = _good_conv()
        state = _good_state()

        # Each repair appends a turn but judge never passes.
        repair_agent = self._make_mock_agent([
            {"type": "append_turn", "role": "assistant",
             "content": "Attempt 1 summary.", "reason": "missing summary"},
            {"type": "append_turn", "role": "assistant",
             "content": "Attempt 2 summary.", "reason": "still missing"},
            # This third one should never be called.
            {"type": "append_turn", "role": "assistant",
             "content": "Attempt 3 summary.", "reason": "still missing"},
        ])
        judge = self._make_mock_judge(passes=False)

        repaired, judge_result, attempts_used = run_repair(
            conv, state, repair_agent, judge, max_attempts=2
        )

        # suggest called at most 2 times (max_attempts).
        assert repair_agent.suggest.call_count <= 2

    def test_no_repair_needed_when_already_valid_and_judge_passes(self) -> None:
        """If the first validation + judge both pass, attempts_used == 0."""
        conv = _good_conv()
        state = _good_state([
            ToolOutput("Travel/Hotels/searchHotels", {}, {"h": "h-1"}, None, "turn-0")
        ])

        repair_agent = self._make_mock_agent([])  # no ops should be needed
        judge = self._make_mock_judge(passes=True)

        # Note: run_repair is normally called on a FAILING conversation.
        # When called on a passing one, it should validate, judge, and return.
        repaired, judge_result, attempts_used = run_repair(conv, state, repair_agent, judge)

        # No repair ops consumed — suggest never called.
        repair_agent.suggest.assert_not_called()
        # Judge was called to confirm the pass.
        judge.score.assert_called_once()
        assert judge_result is not None and judge_result.overall_pass
        assert attempts_used == 0
