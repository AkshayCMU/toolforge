"""Unit tests for evaluation metrics — F7.2.

All tests are offline-safe:
  - No LLM calls.
  - No JSONL file I/O (records built in-memory).
  - Embedding dispersion tests mock sentence_transformers.

Covers:
  - compute_quality_metrics: all fields, edge cases (empty, all-scored, zero tools)
  - compute_tool_coverage_entropy: uniform/skewed/empty
  - compute_distinct_bigrams: single/multi/no pairs
  - compute_embedding_dispersion: mocked embeddings
  - compute_all_metrics: integration smoke test
  - report.compare_reports: delta signs, table presence
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from toolforge.evaluation.metrics import (
    _EMBED_UNAVAILABLE,
    _has_disambiguation,
    compute_all_metrics,
    compute_distinct_bigrams,
    compute_embedding_dispersion,
    compute_quality_metrics,
    compute_tool_coverage_entropy,
)
from toolforge.evaluation.report import (
    build_report,
    build_markdown_report,
    compare_reports,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _make_record(
    *,
    tool_calls: list[dict] | None = None,
    tool_outputs: list[dict] | None = None,
    messages: list[dict] | None = None,
    judge_scores: dict | None = None,
    length_bucket: str = "medium",
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Build a minimal JSONL record dict."""
    if tool_calls is None:
        tool_calls = []
    if tool_outputs is None:
        tool_outputs = []
    if messages is None:
        messages = [
            {"role": "user", "content": "Show me something."},
            {"role": "assistant", "content": "Here you go."},
        ]
    if judge_scores is None:
        judge_scores = {
            "naturalness": 4,
            "tool_correctness": 4,
            "chain_coherence": 4,
            "task_completion": 4,
            "mean": 4.0,
            "overall_pass": True,
        }
    base_meta = {"length_bucket": length_bucket}
    if metadata:
        base_meta.update(metadata)
    return {
        "conversation_id": "conv-000001",
        "messages": messages,
        "tool_calls": tool_calls,
        "tool_outputs": tool_outputs,
        "judge_scores": judge_scores,
        "validation_results": [],
        "metadata": base_meta,
    }


def _make_tool_call(ep_id: str, args: dict | None = None) -> dict:
    return {"endpoint_id": ep_id, "arguments": args or {}}


def _make_tool_output(ep_id: str, *, error: str | None = None) -> dict:
    return {
        "endpoint_id": ep_id,
        "arguments": {},
        "response": None if error else {"ok": True},
        "error": error,
        "timestamp": "turn-0",
    }


# ---------------------------------------------------------------------------
# _has_disambiguation
# ---------------------------------------------------------------------------

def test_has_disambiguation_with_clarification() -> None:
    msgs = [
        {"role": "user", "content": "I need help."},
        {"role": "assistant", "content": "Could you clarify your destination?"},
        {"role": "user", "content": "London"},
        {"role": "assistant", "content": "[tool_call: A/B/c, args={}]"},
    ]
    assert _has_disambiguation(msgs) is True


def test_has_disambiguation_without_clarification() -> None:
    msgs = [
        {"role": "user", "content": "I need help."},
        {"role": "assistant", "content": "[tool_call: A/B/c, args={}]"},
    ]
    assert _has_disambiguation(msgs) is False


def test_has_disambiguation_no_tool_calls() -> None:
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello back"},
    ]
    # First assistant message is not a tool call → counts as disambiguation
    assert _has_disambiguation(msgs) is True


def test_has_disambiguation_empty_messages() -> None:
    assert _has_disambiguation([]) is False


# ---------------------------------------------------------------------------
# compute_quality_metrics — empty input
# ---------------------------------------------------------------------------

def test_quality_metrics_empty_returns_none_scores() -> None:
    result = compute_quality_metrics([])
    assert result["n"] == 0
    assert result["mean_judge_score"] is None
    assert result["pass_rate"] is None


# ---------------------------------------------------------------------------
# compute_quality_metrics — basic counts
# ---------------------------------------------------------------------------

def test_quality_metrics_n() -> None:
    records = [_make_record() for _ in range(5)]
    result = compute_quality_metrics(records)
    assert result["n"] == 5


def test_quality_metrics_mean_judge_score() -> None:
    r = _make_record(judge_scores={
        "naturalness": 4, "tool_correctness": 3, "chain_coherence": 4,
        "task_completion": 5, "mean": 4.0, "overall_pass": True,
    })
    result = compute_quality_metrics([r])
    assert result["mean_judge_score"] == pytest.approx(4.0, abs=0.01)


def test_quality_metrics_per_dimension_means() -> None:
    r = _make_record(judge_scores={
        "naturalness": 4, "tool_correctness": 3, "chain_coherence": 4,
        "task_completion": 5, "mean": 4.0, "overall_pass": True,
    })
    result = compute_quality_metrics([r])
    assert result["per_dimension_means"]["naturalness"] == pytest.approx(4.0)
    assert result["per_dimension_means"]["tool_correctness"] == pytest.approx(3.0)


def test_quality_metrics_pass_rate_all_pass() -> None:
    records = [_make_record(judge_scores={
        "naturalness": 4, "tool_correctness": 4, "chain_coherence": 4,
        "task_completion": 4, "mean": 4.0, "overall_pass": True,
    }) for _ in range(3)]
    result = compute_quality_metrics(records)
    assert result["pass_rate"] == pytest.approx(1.0)


def test_quality_metrics_pass_rate_mixed() -> None:
    pass_r = _make_record(judge_scores={
        "naturalness": 4, "tool_correctness": 4, "chain_coherence": 4,
        "task_completion": 4, "mean": 4.0, "overall_pass": True,
    })
    fail_r = _make_record(judge_scores={
        "naturalness": 2, "tool_correctness": 2, "chain_coherence": 2,
        "task_completion": 2, "mean": 2.0, "overall_pass": False,
    })
    result = compute_quality_metrics([pass_r, fail_r])
    assert result["pass_rate"] == pytest.approx(0.5)


def test_quality_metrics_pct_multi_step_with_successful_outputs() -> None:
    many_outputs = [_make_tool_output(f"A/B/{i}") for i in range(3)]
    r = _make_record(tool_outputs=many_outputs)
    result = compute_quality_metrics([r])
    assert result["pct_multi_step"] == pytest.approx(1.0)


def test_quality_metrics_pct_multi_step_with_errors_excluded() -> None:
    outputs = [
        _make_tool_output("A/B/1"),
        _make_tool_output("A/B/2", error="boom"),
        _make_tool_output("A/B/3", error="boom"),
    ]
    r = _make_record(tool_outputs=outputs)
    result = compute_quality_metrics([r])
    # Only 1 success → < 3 → not multi-step
    assert result["pct_multi_step"] == pytest.approx(0.0)


def test_quality_metrics_pct_multi_tool() -> None:
    tcs = [
        _make_tool_call("Sports/Football/getLeagues"),
        _make_tool_call("Sports/Tennis/getTournaments"),
    ]
    r = _make_record(tool_calls=tcs)
    result = compute_quality_metrics([r])
    assert result["pct_multi_tool"] == pytest.approx(1.0)


def test_quality_metrics_pct_multi_tool_single_tool() -> None:
    tcs = [
        _make_tool_call("Sports/Football/getLeagues"),
        _make_tool_call("Sports/Football/getStandings"),
    ]
    r = _make_record(tool_calls=tcs)
    result = compute_quality_metrics([r])
    assert result["pct_multi_tool"] == pytest.approx(0.0)


def test_quality_metrics_disambiguation() -> None:
    messages = [
        {"role": "user", "content": "I need help."},
        {"role": "assistant", "content": "Could you clarify?"},
        {"role": "user", "content": "London."},
        {"role": "assistant", "content": "[tool_call: A/B/c, args={}]"},
    ]
    r = _make_record(messages=messages)
    result = compute_quality_metrics([r])
    assert result["pct_disambiguation"] == pytest.approx(1.0)


def test_quality_metrics_length_distribution() -> None:
    records = [
        _make_record(length_bucket="short"),
        _make_record(length_bucket="medium"),
        _make_record(length_bucket="medium"),
        _make_record(length_bucket="long"),
    ]
    result = compute_quality_metrics(records)
    ld = result["length_distribution"]
    assert ld["short"] == 1
    assert ld["medium"] == 2
    assert ld["long"] == 1


def test_quality_metrics_no_judge_scores_handled() -> None:
    r = _make_record(judge_scores={})
    result = compute_quality_metrics([r])
    # n_scored should be 0, mean_judge_score None
    assert result["n_scored"] == 0
    assert result["mean_judge_score"] is None


# ---------------------------------------------------------------------------
# compute_tool_coverage_entropy
# ---------------------------------------------------------------------------

def test_entropy_empty_records() -> None:
    assert compute_tool_coverage_entropy([]) == 0.0


def test_entropy_single_endpoint() -> None:
    records = [_make_record(tool_calls=[_make_tool_call("A/B/c")]) for _ in range(5)]
    entropy = compute_tool_coverage_entropy(records)
    # p=1 → entropy = -1*log(1) = 0
    assert entropy == pytest.approx(0.0)


def test_entropy_uniform_two_endpoints() -> None:
    records = [
        _make_record(tool_calls=[_make_tool_call("A/B/c")]),
        _make_record(tool_calls=[_make_tool_call("X/Y/z")]),
    ]
    entropy = compute_tool_coverage_entropy(records)
    # p=0.5 each → entropy = -2*(0.5*log(0.5)) = log(2) ≈ 0.693
    assert entropy == pytest.approx(math.log(2), abs=0.001)


def test_entropy_increases_with_variety() -> None:
    # 1 endpoint repeated
    rec1 = [_make_record(tool_calls=[_make_tool_call("A/B/c")]) for _ in range(4)]
    # 4 distinct endpoints equally
    rec4 = [
        _make_record(tool_calls=[_make_tool_call(f"A/B/{i}")])
        for i in range(4)
    ]
    e1 = compute_tool_coverage_entropy(rec1)
    e4 = compute_tool_coverage_entropy(rec4)
    assert e4 > e1


# ---------------------------------------------------------------------------
# compute_distinct_bigrams
# ---------------------------------------------------------------------------

def test_bigrams_empty() -> None:
    assert compute_distinct_bigrams([]) == 0.0


def test_bigrams_no_pairs_single_call_each() -> None:
    records = [_make_record(tool_calls=[_make_tool_call("A/B/c")])]
    assert compute_distinct_bigrams(records) == 0.0


def test_bigrams_all_distinct() -> None:
    # Each conversation has a different pair
    records = [
        _make_record(tool_calls=[_make_tool_call("A/B/x"), _make_tool_call("C/D/y")]),
        _make_record(tool_calls=[_make_tool_call("E/F/p"), _make_tool_call("G/H/q")]),
    ]
    result = compute_distinct_bigrams(records)
    # 2 unique pairs / 2 total = 1.0
    assert result == pytest.approx(1.0)


def test_bigrams_repeated_pair_reduces_score() -> None:
    records = [
        _make_record(tool_calls=[_make_tool_call("A/B/x"), _make_tool_call("C/D/y")]),
        _make_record(tool_calls=[_make_tool_call("A/B/x"), _make_tool_call("C/D/y")]),
    ]
    result = compute_distinct_bigrams(records)
    # 1 unique pair / 2 total = 0.5
    assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_embedding_dispersion (mocked)
# ---------------------------------------------------------------------------

def test_embedding_dispersion_empty() -> None:
    assert compute_embedding_dispersion([]) == 0.0


def test_embedding_dispersion_single_record() -> None:
    r = _make_record(messages=[{"role": "user", "content": "hello"}])
    assert compute_embedding_dispersion([r]) == 0.0


def test_embedding_dispersion_import_error_returns_sentinel() -> None:
    r1 = _make_record(messages=[{"role": "user", "content": "a"}])
    r2 = _make_record(messages=[{"role": "user", "content": "b"}])
    with patch("toolforge.evaluation.metrics._SentenceTransformer", None):
        result = compute_embedding_dispersion([r1, r2])
    assert result == _EMBED_UNAVAILABLE


def test_embedding_dispersion_orthogonal_vectors() -> None:
    """Mocked embeddings that are orthogonal → dispersion ≈ 1.0."""
    import numpy as np

    r1 = _make_record(messages=[{"role": "user", "content": "alpha"}])
    r2 = _make_record(messages=[{"role": "user", "content": "beta"}])

    mock_model = MagicMock()
    # Two orthogonal unit vectors
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])

    with patch("toolforge.evaluation.metrics._SentenceTransformer", return_value=mock_model):
        result = compute_embedding_dispersion([r1, r2])

    # cosine distance = 1 - cos(90°) = 1.0
    assert isinstance(result, float)
    assert result == pytest.approx(1.0, abs=0.001)


def test_embedding_dispersion_identical_vectors() -> None:
    """Mocked embeddings that are identical → dispersion = 0.0."""
    import numpy as np

    r1 = _make_record(messages=[{"role": "user", "content": "same"}])
    r2 = _make_record(messages=[{"role": "user", "content": "same"}])

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]])

    with patch("toolforge.evaluation.metrics._SentenceTransformer", return_value=mock_model):
        result = compute_embedding_dispersion([r1, r2])

    assert isinstance(result, float)
    assert result == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------------------
# compute_all_metrics integration
# ---------------------------------------------------------------------------

def test_compute_all_metrics_quality_key_present() -> None:
    records = [_make_record()]
    result = compute_all_metrics(records, include_diversity=False)
    assert "quality" in result
    assert "diversity" not in result


def test_compute_all_metrics_diversity_key_present() -> None:
    records = [_make_record()]
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
    with patch("toolforge.evaluation.metrics._SentenceTransformer", return_value=mock_model):
        result = compute_all_metrics(records, include_diversity=True)
    assert "diversity" in result


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------

def test_build_report_contains_required_keys() -> None:
    metrics = compute_quality_metrics([_make_record()])
    report = build_report({"quality": metrics}, run_label="test_run")
    assert report["run_label"] == "test_run"
    assert "metrics" in report


def test_build_markdown_report_contains_section_headers() -> None:
    metrics = {"quality": compute_quality_metrics([_make_record()])}
    report = build_report(metrics, run_label="my_run")
    md = build_markdown_report(report)
    assert "# Evaluation Report" in md
    assert "## Quality Metrics" in md


def test_compare_reports_produces_table() -> None:
    metrics_a = {"quality": compute_quality_metrics([_make_record()])}
    metrics_b = {"quality": compute_quality_metrics([_make_record()])}
    ra = build_report(metrics_a, run_label="Run A")
    rb = build_report(metrics_b, run_label="Run B")
    comparison = compare_reports(ra, rb)
    assert "Run A" in comparison
    assert "Run B" in comparison
    assert "Delta" in comparison
    assert "Tradeoff Summary" in comparison


def test_compare_reports_delta_positive_when_b_higher() -> None:
    ma = {"quality": {"n": 2, "mean_judge_score": 3.5, "pass_rate": 0.5,
                      "pct_multi_step": 0.5, "pct_multi_tool": 0.5,
                      "pct_disambiguation": 0.3, "n_scored": 2,
                      "per_dimension_means": {}, "length_distribution": {}}}
    mb = {"quality": {"n": 2, "mean_judge_score": 4.0, "pass_rate": 0.8,
                      "pct_multi_step": 0.6, "pct_multi_tool": 0.6,
                      "pct_disambiguation": 0.4, "n_scored": 2,
                      "per_dimension_means": {}, "length_distribution": {}}}
    ra = build_report(ma, run_label="A")
    rb = build_report(mb, run_label="B")
    comparison = compare_reports(ra, rb)
    # Delta for mean_judge_score should be +0.500
    assert "+0.500" in comparison


def test_compare_reports_no_crash_on_missing_diversity() -> None:
    metrics = {"quality": compute_quality_metrics([_make_record()])}
    ra = build_report(metrics, run_label="A")
    rb = build_report(metrics, run_label="B")
    # Should not raise even without diversity key
    result = compare_reports(ra, rb)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# toolforge evaluate CLI — empty JSONL safety
# ---------------------------------------------------------------------------

def test_evaluate_cli_empty_jsonl_does_not_crash(tmp_path: pytest.fixture) -> None:  # type: ignore[valid-type]
    """toolforge evaluate on an empty JSONL file must exit 0 without crashing.

    Regression test for the :.1% crash on None percentage metrics.
    """
    from typer.testing import CliRunner
    from toolforge.cli import app

    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    out_path = tmp_path / "reports" / "eval.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["evaluate", "--in", str(empty_jsonl), "--out", str(out_path), "--no-diversity"],
    )

    assert result.exit_code == 0, (
        f"CLI crashed on empty JSONL.\nOutput: {result.output}\n"
        f"Exception: {result.exception}"
    )
    assert "n=0" in result.output


def test_evaluate_cli_empty_jsonl_shows_na_for_percentages(tmp_path: pytest.fixture) -> None:  # type: ignore[valid-type]
    """Percentage fields display 'n/a' (not a format crash) on empty input."""
    from typer.testing import CliRunner
    from toolforge.cli import app

    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("", encoding="utf-8")
    out_path = tmp_path / "reports" / "eval.json"

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["evaluate", "--in", str(empty_jsonl), "--out", str(out_path), "--no-diversity"],
    )

    assert result.exit_code == 0
    # All three percentage fields must show n/a, not a Python traceback
    assert "multi-step=n/a" in result.output
    assert "multi-tool=n/a" in result.output
    assert "disambiguation=n/a" in result.output
