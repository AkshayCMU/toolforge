"""Unit tests for F4.5 — Judge agent.

All tests use dry_run=True + pre-seeded cache. No live LLM calls.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolforge.agents.judge import (
    DimensionScore,
    Judge,
    JudgeResult,
    _build_user_prompt,
)
from toolforge.agents.llm_client import LLMClient
from toolforge.agents.user_sim import Message
from tests.unit.test_llm_client import seed_cache


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

_HIGH_QUALITY_CONV: list[Message] = [
    {"role": "user", "content": "Book a hotel in Paris for June 1–5."},
    {"role": "assistant", "content": "Creating your booking now."},
    {"role": "assistant", "content": "[tool_call: Travel/Hotels/createBooking, city_name=Paris]"},
    {"role": "user", "content": "[tool_result: booking_id=bk-001, status=confirmed]"},
    {"role": "assistant", "content": "Done! Your booking ID is bk-001."},
    {"role": "user", "content": "Thanks!"},
]

_BAD_CONV: list[Message] = [
    {"role": "user", "content": "cancel my subscription"},
    {"role": "assistant", "content": "[tool_call: Account/Settings/deleteAccount]"},
    {"role": "user", "content": "[tool_result: status=account_deleted]"},
    {"role": "assistant", "content": "Done, your account has been deleted."},
]

_ENDPOINTS = [
    "Travel/Hotels/createBooking",
    "Travel/Hotels/fetchBookingDetails",
    "Travel/Hotels/updateBooking",
]

_PASS_RESULT = {
    "naturalness": {"score": 5, "rationale": "Natural dialogue"},
    "tool_correctness": {"score": 4, "rationale": "Correct endpoint"},
    "chain_coherence": {"score": 5, "rationale": "IDs flow correctly"},
    "task_completion": {"score": 5, "rationale": "Task fully resolved"},
    "failure_modes": [],
    "overall_pass": False,  # will be overwritten by validator
}

_FAIL_RESULT = {
    "naturalness": {"score": 2, "rationale": "Vague user request"},
    "tool_correctness": {"score": 1, "rationale": "Wrong endpoint called"},
    "chain_coherence": {"score": 3, "rationale": "No chaining required"},
    "task_completion": {"score": 2, "rationale": "Task not resolved"},
    "failure_modes": ["Wrong endpoint: deleteAccount instead of cancelSubscription"],
    "overall_pass": True,  # will be overwritten to False by validator
}


def _make_judge(tmp_path) -> Judge:
    client = LLMClient(
        model="claude-sonnet-4-6",
        temperature=0.0,
        cache_dir=tmp_path,
        dry_run=True,
    )
    return Judge(client)


def _seed_judge(judge: Judge, conv: list[Message], data: dict, tmp_path) -> None:
    user_prompt = _build_user_prompt(conv, _ENDPOINTS)
    seed_cache(judge._client, judge.system_prompt, user_prompt, JudgeResult, data)


# ---------------------------------------------------------------------------
# JudgeResult model
# ---------------------------------------------------------------------------

def test_judge_result_pass_computed_correctly() -> None:
    result = JudgeResult(
        naturalness=DimensionScore(score=4, rationale="ok"),
        tool_correctness=DimensionScore(score=4, rationale="ok"),
        chain_coherence=DimensionScore(score=4, rationale="ok"),
        task_completion=DimensionScore(score=3, rationale="ok"),
        failure_modes=[],
        overall_pass=False,
    )
    # mean = 3.75 >= 3.5, min = 3 >= 2.5 → should pass
    assert result.overall_pass is True


def test_judge_result_fail_when_mean_below_threshold() -> None:
    result = JudgeResult(
        naturalness=DimensionScore(score=3, rationale="ok"),
        tool_correctness=DimensionScore(score=3, rationale="ok"),
        chain_coherence=DimensionScore(score=3, rationale="ok"),
        task_completion=DimensionScore(score=3, rationale="ok"),
        failure_modes=[],
        overall_pass=True,  # should be overwritten to False
    )
    # mean = 3.0 < 3.5 → fail
    assert result.overall_pass is False


def test_judge_result_fail_when_min_below_threshold() -> None:
    result = JudgeResult(
        naturalness=DimensionScore(score=5, rationale="ok"),
        tool_correctness=DimensionScore(score=5, rationale="ok"),
        chain_coherence=DimensionScore(score=5, rationale="ok"),
        task_completion=DimensionScore(score=2, rationale="catastrophic"),
        failure_modes=["task not done"],
        overall_pass=True,  # should be overwritten to False
    )
    # min = 2 < 2.5 → fail regardless of mean
    assert result.overall_pass is False


def test_dimension_score_rejects_out_of_range() -> None:
    with pytest.raises(ValidationError):
        DimensionScore(score=6, rationale="too high")
    with pytest.raises(ValidationError):
        DimensionScore(score=0, rationale="too low")


def test_mean_score_calculation() -> None:
    result = JudgeResult(
        naturalness=DimensionScore(score=4, rationale="ok"),
        tool_correctness=DimensionScore(score=3, rationale="ok"),
        chain_coherence=DimensionScore(score=5, rationale="ok"),
        task_completion=DimensionScore(score=4, rationale="ok"),
        failure_modes=[],
        overall_pass=False,
    )
    assert result.mean_score() == 4.0


# ---------------------------------------------------------------------------
# overall_pass is always computed (not trusted from LLM)
# ---------------------------------------------------------------------------

def test_overall_pass_overrides_llm_value() -> None:
    """LLM says overall_pass=True but scores are too low — validator must fix it."""
    result = JudgeResult(
        naturalness=DimensionScore(score=2, rationale="bad"),
        tool_correctness=DimensionScore(score=1, rationale="wrong endpoint"),
        chain_coherence=DimensionScore(score=3, rationale="ok"),
        task_completion=DimensionScore(score=2, rationale="incomplete"),
        failure_modes=["wrong endpoint"],
        overall_pass=True,  # LLM hallucinated True — must be overwritten
    )
    assert result.overall_pass is False


# ---------------------------------------------------------------------------
# score() — dry_run, pre-seeded cache
# ---------------------------------------------------------------------------

def test_score_returns_judge_result(tmp_path) -> None:
    judge = _make_judge(tmp_path)
    _seed_judge(judge, _HIGH_QUALITY_CONV, _PASS_RESULT, tmp_path)

    result = judge.score(_HIGH_QUALITY_CONV, _ENDPOINTS)
    assert isinstance(result, JudgeResult)


def test_score_high_quality_conv_passes(tmp_path) -> None:
    judge = _make_judge(tmp_path)
    _seed_judge(judge, _HIGH_QUALITY_CONV, _PASS_RESULT, tmp_path)

    result = judge.score(_HIGH_QUALITY_CONV, _ENDPOINTS)
    assert result.overall_pass is True


def test_score_bad_conv_fails(tmp_path) -> None:
    judge = _make_judge(tmp_path)
    _seed_judge(judge, _BAD_CONV, _FAIL_RESULT, tmp_path)

    result = judge.score(_BAD_CONV, _ENDPOINTS)
    assert result.overall_pass is False


def test_score_reproducible_same_inputs(tmp_path) -> None:
    """Same conversation twice → same scores (temperature=0 reproducibility via cache)."""
    judge = _make_judge(tmp_path)
    _seed_judge(judge, _HIGH_QUALITY_CONV, _PASS_RESULT, tmp_path)

    r1 = judge.score(_HIGH_QUALITY_CONV, _ENDPOINTS)
    r2 = judge.score(_HIGH_QUALITY_CONV, _ENDPOINTS)

    assert r1.naturalness.score == r2.naturalness.score
    assert r1.tool_correctness.score == r2.tool_correctness.score
    assert r1.chain_coherence.score == r2.chain_coherence.score
    assert r1.task_completion.score == r2.task_completion.score
    assert r1.overall_pass == r2.overall_pass


# ---------------------------------------------------------------------------
# _build_user_prompt
# ---------------------------------------------------------------------------

def test_user_prompt_contains_conversation_turns() -> None:
    prompt = _build_user_prompt(_HIGH_QUALITY_CONV, _ENDPOINTS)
    assert "Book a hotel in Paris" in prompt
    assert "Done! Your booking ID is bk-001" in prompt


def test_user_prompt_contains_endpoints() -> None:
    prompt = _build_user_prompt(_HIGH_QUALITY_CONV, _ENDPOINTS)
    assert "Travel/Hotels/createBooking" in prompt


def test_user_prompt_empty_conversation() -> None:
    prompt = _build_user_prompt([], _ENDPOINTS)
    assert "conversation" in prompt
