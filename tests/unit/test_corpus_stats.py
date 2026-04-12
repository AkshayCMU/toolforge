"""Unit tests for CorpusDiversityTracker and NoOpDiversityTracker — F6.1.

Tests cover:
- sampling_weight inverse-frequency formula
- should_accept: endpoint cap, category cap (onset at ≥10 convs), tool-pair cap
- update: all counter increments
- steering_weights: full dict built from metadata keys
- NoOpDiversityTracker: always accepts, always weight 1.0, update is a no-op
- build_endpoint_metadata: correct extraction from a toy graph
"""

from __future__ import annotations

import networkx as nx
import pytest

from toolforge.memory.corpus_stats import (
    CorpusDiversityTracker,
    NoOpDiversityTracker,
    build_endpoint_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_meta() -> dict[str, dict[str, str]]:
    """Three endpoints across two tools and two categories."""
    return {
        "ep:a": {"category": "Finance", "tool": "tool_alpha"},
        "ep:b": {"category": "Finance", "tool": "tool_alpha"},
        "ep:c": {"category": "Weather", "tool": "tool_beta"},
    }


@pytest.fixture()
def tracker(simple_meta: dict[str, dict[str, str]]) -> CorpusDiversityTracker:
    return CorpusDiversityTracker(simple_meta)


# ---------------------------------------------------------------------------
# sampling_weight
# ---------------------------------------------------------------------------

def test_sampling_weight_unseen_is_one(tracker: CorpusDiversityTracker) -> None:
    assert tracker.sampling_weight("ep:a") == pytest.approx(1.0)


def test_sampling_weight_decreases_with_usage(tracker: CorpusDiversityTracker) -> None:
    tracker.tool_usage["ep:a"] = 3
    expected = 1.0 / (1.0 + 3)
    assert tracker.sampling_weight("ep:a") == pytest.approx(expected)


def test_sampling_weight_unknown_endpoint_is_one(tracker: CorpusDiversityTracker) -> None:
    # An endpoint not in meta still gets the unseen weight.
    assert tracker.sampling_weight("ep:unknown") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# steering_weights
# ---------------------------------------------------------------------------

def test_steering_weights_keys_match_meta(tracker: CorpusDiversityTracker) -> None:
    weights = tracker.steering_weights()
    assert set(weights.keys()) == {"ep:a", "ep:b", "ep:c"}


def test_steering_weights_all_one_when_empty(tracker: CorpusDiversityTracker) -> None:
    weights = tracker.steering_weights()
    assert all(w == pytest.approx(1.0) for w in weights.values())


def test_steering_weights_inverse_after_update(tracker: CorpusDiversityTracker) -> None:
    tracker.update(["ep:a", "ep:c"], "linear", "short")
    weights = tracker.steering_weights()
    # ep:a and ep:c used once → weight = 0.5
    assert weights["ep:a"] == pytest.approx(0.5)
    assert weights["ep:c"] == pytest.approx(0.5)
    # ep:b still unseen → weight = 1.0
    assert weights["ep:b"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# should_accept — endpoint cap
# ---------------------------------------------------------------------------

def test_should_accept_fresh_chain(tracker: CorpusDiversityTracker) -> None:
    ok, reason = tracker.should_accept(["ep:a", "ep:c"])
    assert ok is True
    assert reason == "ok"


def test_should_accept_rejects_on_endpoint_cap(tracker: CorpusDiversityTracker) -> None:
    tracker.tool_usage["ep:a"] = CorpusDiversityTracker.MAX_ENDPOINT_COUNT
    ok, reason = tracker.should_accept(["ep:a"])
    assert ok is False
    assert "endpoint_cap" in reason


def test_should_accept_at_endpoint_cap_minus_one(tracker: CorpusDiversityTracker) -> None:
    # One below cap → still accepted.
    tracker.tool_usage["ep:a"] = CorpusDiversityTracker.MAX_ENDPOINT_COUNT - 1
    ok, _ = tracker.should_accept(["ep:a"])
    assert ok is True


def test_should_accept_empty_chain_rejected(tracker: CorpusDiversityTracker) -> None:
    ok, reason = tracker.should_accept([])
    assert ok is False
    assert reason == "empty_chain"


# ---------------------------------------------------------------------------
# should_accept — category cap
# ---------------------------------------------------------------------------

def _fill_convs(tracker: CorpusDiversityTracker, n: int) -> None:
    """Register n conversations to trigger the category cap enforcement (needs ≥10)."""
    for _ in range(n):
        tracker.length_bucket_usage["short"] += 1


def test_category_cap_not_enforced_below_ten_convs(
    tracker: CorpusDiversityTracker,
) -> None:
    # 9 conversations — cap not enforced yet.
    _fill_convs(tracker, 9)
    # Mark Finance as having appeared in all 9.
    tracker.category_usage["Finance"] = 9
    # Finance fraction would be (9+1)/(9+1) = 1.0 — but cap inactive.
    ok, _ = tracker.should_accept(["ep:a"])  # ep:a is Finance
    assert ok is True


def test_category_cap_enforced_at_ten_convs(
    tracker: CorpusDiversityTracker,
) -> None:
    _fill_convs(tracker, 10)
    # Finance at 2 out of 10 = 0.20 → 0.20 > MAX_CATEGORY_FRACTION (0.15)
    tracker.category_usage["Finance"] = 2
    ok, reason = tracker.should_accept(["ep:a"])  # ep:a is Finance
    assert ok is False
    assert "category_cap" in reason


def test_category_cap_allows_under_threshold(
    tracker: CorpusDiversityTracker,
) -> None:
    _fill_convs(tracker, 10)
    # Finance at 0 out of 10 → prospective = 1/11 ≈ 0.091 < 0.15 → OK.
    tracker.category_usage["Finance"] = 0
    ok, _ = tracker.should_accept(["ep:a"])
    assert ok is True


# ---------------------------------------------------------------------------
# should_accept — tool-pair cap
# ---------------------------------------------------------------------------

def test_tool_pair_cap_fires(tracker: CorpusDiversityTracker) -> None:
    pair = ("tool_alpha", "tool_beta")
    tracker.tool_pair_usage[pair] = CorpusDiversityTracker.MAX_TOOL_PAIR_COUNT
    # Chain includes both ep:a (tool_alpha) and ep:c (tool_beta).
    ok, reason = tracker.should_accept(["ep:a", "ep:c"])
    assert ok is False
    assert "tool_pair_cap" in reason


def test_tool_pair_cap_below_limit(tracker: CorpusDiversityTracker) -> None:
    pair = ("tool_alpha", "tool_beta")
    tracker.tool_pair_usage[pair] = CorpusDiversityTracker.MAX_TOOL_PAIR_COUNT - 1
    ok, _ = tracker.should_accept(["ep:a", "ep:c"])
    assert ok is True


def test_tool_pair_same_tool_no_pair_formed(tracker: CorpusDiversityTracker) -> None:
    # Chain uses only tool_alpha endpoints — no cross-tool pair.
    # Even with pair count at cap, same-tool chains are unaffected.
    pair = ("tool_alpha", "tool_beta")
    tracker.tool_pair_usage[pair] = CorpusDiversityTracker.MAX_TOOL_PAIR_COUNT
    ok, _ = tracker.should_accept(["ep:a", "ep:b"])  # both tool_alpha
    assert ok is True


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

def test_update_increments_tool_usage(tracker: CorpusDiversityTracker) -> None:
    tracker.update(["ep:a", "ep:c"], "linear", "short")
    assert tracker.tool_usage["ep:a"] == 1
    assert tracker.tool_usage["ep:c"] == 1
    assert tracker.tool_usage["ep:b"] == 0


def test_update_increments_category_once_per_unique_category(
    tracker: CorpusDiversityTracker,
) -> None:
    # ep:a and ep:b are both Finance — category_usage["Finance"] should be 1, not 2.
    tracker.update(["ep:a", "ep:b"], "linear", "medium")
    assert tracker.category_usage["Finance"] == 1


def test_update_increments_tool_pair(tracker: CorpusDiversityTracker) -> None:
    tracker.update(["ep:a", "ep:c"], "linear", "long")
    assert tracker.tool_pair_usage[("tool_alpha", "tool_beta")] == 1


def test_update_no_pair_for_single_tool(tracker: CorpusDiversityTracker) -> None:
    tracker.update(["ep:a", "ep:b"], "linear", "short")
    assert len(tracker.tool_pair_usage) == 0


def test_update_increments_pattern_and_bucket(tracker: CorpusDiversityTracker) -> None:
    tracker.update(["ep:a"], "linear", "short")
    assert tracker.chain_pattern_usage["linear"] == 1
    assert tracker.length_bucket_usage["short"] == 1


# ---------------------------------------------------------------------------
# NoOpDiversityTracker
# ---------------------------------------------------------------------------

def test_noop_always_accepts() -> None:
    noop = NoOpDiversityTracker()
    ok, reason = noop.should_accept(["ep:anything"])
    assert ok is True
    assert reason == "ok"


def test_noop_weight_is_one() -> None:
    noop = NoOpDiversityTracker()
    assert noop.sampling_weight("ep:x") == pytest.approx(1.0)


def test_noop_steering_weights_empty() -> None:
    noop = NoOpDiversityTracker()
    assert noop.steering_weights() == {}


def test_noop_update_is_noop() -> None:
    noop = NoOpDiversityTracker()
    noop.update(["ep:a"], "linear", "short")  # must not raise


# ---------------------------------------------------------------------------
# build_endpoint_metadata
# ---------------------------------------------------------------------------

def _make_toy_graph() -> nx.MultiDiGraph:
    G: nx.MultiDiGraph = nx.MultiDiGraph()
    G.add_node(
        "ep:x",
        node_type="endpoint",
        tool="tool_x",
        category="CatA",
        terminal=False,
    )
    G.add_node(
        "ep:y",
        node_type="endpoint",
        tool="tool_y",
        category="CatB",
        terminal=False,
    )
    # Terminal endpoint — should be excluded.
    G.add_node(
        "ep:z",
        node_type="endpoint",
        tool="tool_z",
        category="CatC",
        terminal=True,
    )
    # Category node — should be excluded.
    G.add_node("st:Foo", node_type="semantic_type", label="Foo")
    return G


def test_build_endpoint_metadata_excludes_terminal() -> None:
    G = _make_toy_graph()
    meta = build_endpoint_metadata(G)
    assert "ep:z" not in meta


def test_build_endpoint_metadata_excludes_non_endpoint_nodes() -> None:
    G = _make_toy_graph()
    meta = build_endpoint_metadata(G)
    assert "st:Foo" not in meta


def test_build_endpoint_metadata_includes_non_terminal_endpoints() -> None:
    G = _make_toy_graph()
    meta = build_endpoint_metadata(G)
    assert set(meta.keys()) == {"ep:x", "ep:y"}


def test_build_endpoint_metadata_correct_fields() -> None:
    G = _make_toy_graph()
    meta = build_endpoint_metadata(G)
    assert meta["ep:x"] == {"category": "CatA", "tool": "tool_x"}
    assert meta["ep:y"] == {"category": "CatB", "tool": "tool_y"}
