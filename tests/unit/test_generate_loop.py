"""Unit tests for generate_batch helpers and _run_batch — F7.1.

All tests are offline-safe: no LLM calls, no artifact files.
A mock ConversationGenerator returns hand-crafted ConversationState dicts so
_run_batch() can be exercised end-to-end without touching the network or disk.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from toolforge.agents.judge import DimensionScore, JudgeResult
from toolforge.generator.loop import (
    _conv_to_record,
    _length_bucket,
    _run_batch,
    _session_from_summary,
)
from toolforge.generator.state import Conversation
from toolforge.graph.sampler import ChainConstraints
from toolforge.memory.corpus_stats import NoOpDiversityTracker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_judge_result(score: int = 4) -> JudgeResult:
    dim = DimensionScore(score=score, rationale="ok")
    return JudgeResult(
        naturalness=dim,
        tool_correctness=dim,
        chain_coherence=dim,
        task_completion=dim,
    )


def _make_session_summary(
    tool_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "conversation_id": "conv-000042",
        "seed": 42,
        "available_values_by_type": {},
        "resolved_entities": {},
        "created_entities": [],
        "tool_outputs": tool_outputs or [],
        "private_user_knowledge": {},
    }


_NO_JUDGE = object()  # sentinel for "explicitly pass None judge_result"


def _make_conversation(
    chain: list[str] | None = None,
    messages: list[dict] | None = None,
    tool_outputs: list[dict] | None = None,
    status: str = "done",
    judge_result: JudgeResult | None | object = None,
    repair_attempts: int = 0,
) -> Conversation:
    if chain is None:
        chain = ["Sports/Football/getLeagues", "Sports/Football/getStandings"]
    if messages is None:
        messages = [
            {"role": "user", "content": "Show me the standings."},
            {"role": "assistant", "content": "[tool_call: Sports/Football/getLeagues, args={}]"},
            {"role": "user", "content": '[tool_result: {"leagues": ["Premier League"]}]'},
            {"role": "assistant", "content": "Here are the standings."},
        ]
    resolved_jr: JudgeResult | None
    if judge_result is _NO_JUDGE:
        resolved_jr = None
    elif judge_result is None:
        resolved_jr = _make_judge_result()
    else:
        resolved_jr = judge_result  # type: ignore[assignment]
    return Conversation(
        conversation_id="conv-000042",
        seed=42,
        sampled_chain=chain,
        messages=messages,
        session_summary=_make_session_summary(tool_outputs),
        judge_result=resolved_jr,
        status=status,
        repair_attempts=repair_attempts,
    )


# ---------------------------------------------------------------------------
# _length_bucket
# ---------------------------------------------------------------------------

def test_length_bucket_short() -> None:
    assert _length_bucket(1) == "short"
    assert _length_bucket(2) == "short"


def test_length_bucket_medium() -> None:
    assert _length_bucket(3) == "medium"
    assert _length_bucket(4) == "medium"


def test_length_bucket_long() -> None:
    assert _length_bucket(5) == "long"
    assert _length_bucket(10) == "long"


# ---------------------------------------------------------------------------
# _session_from_summary
# ---------------------------------------------------------------------------

def test_session_from_summary_empty_tool_outputs() -> None:
    summary = _make_session_summary(tool_outputs=[])
    state = _session_from_summary(summary)
    assert state.tool_outputs == []


def test_session_from_summary_reconstructs_tool_output() -> None:
    tool_outputs = [
        {
            "endpoint_id": "Sports/Football/getLeagues",
            "arguments": {},
            "response": {"leagues": ["PL"]},
            "error": None,
            "timestamp": "turn-0",
        }
    ]
    state = _session_from_summary(_make_session_summary(tool_outputs))
    assert len(state.tool_outputs) == 1
    o = state.tool_outputs[0]
    assert o.endpoint_id == "Sports/Football/getLeagues"
    assert o.response == {"leagues": ["PL"]}
    assert o.error is None


def test_session_from_summary_failed_call() -> None:
    tool_outputs = [
        {
            "endpoint_id": "Sports/Football/getLeagues",
            "arguments": {},
            "response": None,
            "error": "Invalid hotel_id: 'x' not in session. Valid values: []",
            "timestamp": "turn-0",
        }
    ]
    state = _session_from_summary(_make_session_summary(tool_outputs))
    assert state.tool_outputs[0].error is not None
    assert not state.tool_outputs[0].is_success()


# ---------------------------------------------------------------------------
# _conv_to_record
# ---------------------------------------------------------------------------

def test_conv_to_record_required_top_level_keys() -> None:
    conv = _make_conversation()
    record = _conv_to_record(conv, [], was_steered=True)
    for key in ("conversation_id", "messages", "tool_calls", "tool_outputs",
                "judge_scores", "validation_results", "metadata"):
        assert key in record, f"Missing top-level key: {key!r}"


def test_conv_to_record_required_metadata_keys() -> None:
    conv = _make_conversation()
    record = _conv_to_record(conv, [], was_steered=True)
    meta = record["metadata"]
    for key in ("seed", "sampled_chain", "pattern", "length_bucket",
                "repair_attempts", "was_steered", "tools_used", "num_turns"):
        assert key in meta, f"Missing metadata key: {key!r}"


def test_conv_to_record_was_steered_flag() -> None:
    conv = _make_conversation()
    assert _conv_to_record(conv, [], was_steered=True)["metadata"]["was_steered"] is True
    assert _conv_to_record(conv, [], was_steered=False)["metadata"]["was_steered"] is False


def test_conv_to_record_tools_used_extraction() -> None:
    conv = _make_conversation(
        chain=["Sports/Football/getLeagues", "Sports/Football/getStandings"]
    )
    record = _conv_to_record(conv, [], was_steered=True)
    # Both endpoints belong to tool "Football"
    assert record["metadata"]["tools_used"] == ["Football"]


def test_conv_to_record_tools_used_multi_tool() -> None:
    conv = _make_conversation(
        chain=["Sports/Football/getLeagues", "Sports/Tennis/getTournaments"]
    )
    record = _conv_to_record(conv, [], was_steered=True)
    assert sorted(record["metadata"]["tools_used"]) == ["Football", "Tennis"]


def test_conv_to_record_length_bucket_medium() -> None:
    conv = _make_conversation(chain=["A/B/c", "D/E/f", "G/H/i"])
    record = _conv_to_record(conv, [], was_steered=True)
    assert record["metadata"]["length_bucket"] == "medium"


def test_conv_to_record_num_turns() -> None:
    conv = _make_conversation()
    record = _conv_to_record(conv, [], was_steered=True)
    assert record["metadata"]["num_turns"] == len(conv.messages)


def test_conv_to_record_repair_attempts() -> None:
    conv = _make_conversation(repair_attempts=2)
    record = _conv_to_record(conv, [], was_steered=True)
    assert record["metadata"]["repair_attempts"] == 2


def test_conv_to_record_judge_scores_present() -> None:
    conv = _make_conversation(judge_result=_make_judge_result(score=4))
    record = _conv_to_record(conv, [], was_steered=True)
    scores = record["judge_scores"]
    assert scores["naturalness"] == 4
    assert "mean" in scores
    assert "overall_pass" in scores


def test_conv_to_record_judge_scores_empty_when_no_judge() -> None:
    conv = _make_conversation(judge_result=_NO_JUDGE)
    record = _conv_to_record(conv, [], was_steered=True)
    assert record["judge_scores"] == {}


def test_conv_to_record_tool_calls_from_session_summary() -> None:
    tool_outputs = [
        {
            "endpoint_id": "Sports/Football/getLeagues",
            "arguments": {"season": "2024"},
            "response": {"leagues": []},
            "error": None,
            "timestamp": "turn-0",
        }
    ]
    conv = _make_conversation(tool_outputs=tool_outputs)
    record = _conv_to_record(conv, [], was_steered=True)
    assert len(record["tool_calls"]) == 1
    tc = record["tool_calls"][0]
    assert tc["endpoint_id"] == "Sports/Football/getLeagues"
    assert tc["arguments"] == {"season": "2024"}


# ---------------------------------------------------------------------------
# _run_batch with mock generator
# ---------------------------------------------------------------------------

def _make_mock_generator(convs: list[Conversation]) -> MagicMock:
    """Return a mock ConversationGenerator whose run() cycles through convs."""
    mock_gen = MagicMock()
    mock_gen._all_endpoints = []

    call_count = [0]

    def fake_run(state: dict) -> dict:
        idx = call_count[0] % len(convs)
        call_count[0] += 1
        conv = convs[idx]
        # Return a ConversationState-like dict that _state_to_conversation reads.
        from toolforge.execution.session import make_session, session_to_dict
        session = make_session(conv.conversation_id, conv.seed)
        return {
            "conversation_id": conv.conversation_id,
            "seed": conv.seed,
            "sampled_chain": conv.sampled_chain,
            "messages": list(conv.messages),
            "session_state": session,
            "judge_result": conv.judge_result,
            "repair_attempts": conv.repair_attempts,
            "status": conv.status,
        }

    mock_gen.run.side_effect = fake_run
    return mock_gen


def test_run_batch_returns_n_records() -> None:
    conv = _make_conversation()
    generator = _make_mock_generator([conv])
    tracker = NoOpDiversityTracker()
    constraints = ChainConstraints(length=2)

    records = _run_batch(
        n=3,
        seed=42,
        generator=generator,
        tracker=tracker,
        constraints=constraints,
        was_steered=False,
    )
    assert len(records) == 3


def test_run_batch_seeds_are_sequential() -> None:
    """Each generated conversation gets a distinct seed (seed + i)."""
    seeds_seen: list[int] = []
    mock_gen = MagicMock()
    mock_gen._all_endpoints = []

    def fake_run(state: dict) -> dict:
        seeds_seen.append(state["seed"])
        from toolforge.execution.session import make_session
        return {
            "conversation_id": state["conversation_id"],
            "seed": state["seed"],
            "sampled_chain": ["A/B/c"],
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "done"}],
            "session_state": make_session(state["conversation_id"], state["seed"]),
            "judge_result": _make_judge_result(),
            "repair_attempts": 0,
            "status": "done",
        }

    mock_gen.run.side_effect = fake_run
    tracker = NoOpDiversityTracker()
    _run_batch(
        n=3, seed=10, generator=mock_gen, tracker=tracker,
        constraints=ChainConstraints(length=1), was_steered=False
    )
    assert seeds_seen == [10, 11, 12]


def test_run_batch_steering_off_flag_propagates() -> None:
    conv = _make_conversation()
    generator = _make_mock_generator([conv])
    tracker = NoOpDiversityTracker()
    records = _run_batch(
        n=1, seed=42, generator=generator, tracker=tracker,
        constraints=ChainConstraints(length=2), was_steered=False,
    )
    assert records[0]["metadata"]["was_steered"] is False


def test_run_batch_steering_on_flag_propagates() -> None:
    conv = _make_conversation()
    generator = _make_mock_generator([conv])
    tracker = NoOpDiversityTracker()
    records = _run_batch(
        n=1, seed=42, generator=generator, tracker=tracker,
        constraints=ChainConstraints(length=2), was_steered=True,
    )
    assert records[0]["metadata"]["was_steered"] is True


def test_run_batch_skips_on_generator_error() -> None:
    """If generate_one() raises, the record is skipped (not the whole batch)."""
    mock_gen = MagicMock()
    mock_gen._all_endpoints = []

    call_count = [0]

    def flaky_run(state: dict) -> dict:
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("simulated failure")
        from toolforge.execution.session import make_session
        return {
            "conversation_id": state["conversation_id"],
            "seed": state["seed"],
            "sampled_chain": ["A/B/c"],
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "assistant", "content": "done"}],
            "session_state": make_session(state["conversation_id"], state["seed"]),
            "judge_result": _make_judge_result(),
            "repair_attempts": 0,
            "status": "done",
        }

    mock_gen.run.side_effect = flaky_run
    tracker = NoOpDiversityTracker()
    records = _run_batch(
        n=3, seed=0, generator=mock_gen, tracker=tracker,
        constraints=ChainConstraints(length=1), was_steered=False,
    )
    # 3 requested, 1 failed → 2 records produced
    assert len(records) == 2


def test_run_batch_validation_results_in_record() -> None:
    conv = _make_conversation()
    generator = _make_mock_generator([conv])
    tracker = NoOpDiversityTracker()
    records = _run_batch(
        n=1, seed=42, generator=generator, tracker=tracker,
        constraints=ChainConstraints(length=2), was_steered=False,
    )
    val = records[0]["validation_results"]
    # validate_conversation always returns exactly 5 results.
    assert len(val) == 5
    stages = [v["stage"] for v in val]
    assert stages == ["structure", "tool_calls", "grounding", "completeness", "constraints"]
