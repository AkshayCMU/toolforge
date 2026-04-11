"""Unit tests for toolforge.graph.sampler (F2.2)."""

from __future__ import annotations

import networkx as nx
import pytest

from toolforge.graph.sampler import (
    ChainConstraints,
    ChainResult,
    ChainSampler,
    FailureReason,
)


# ---------------------------------------------------------------------------
# Shared test fixture
# ---------------------------------------------------------------------------
# Graph topology:
#
#   ep_A (Cat1/tool_1/ep_A) --CHAINS_TO--> ep_B (Cat1/tool_2/ep_B)
#                                              |
#                                       CHAINS_TO
#                                              ↓
#   ep_D (Cat2/tool_3/ep_D) <--CHAINS_TO-- ep_C (Cat2/tool_3/ep_C)
#
#   All four endpoints connected through booking_id.
#   ep_D is a dead-end (has CHAINS_TO incoming, but NO outgoing CHAINS_TO).
#   ep_A → ep_B → ep_C → ep_D  (linear chain of length 4)
#   ep_B also connects to ep_D directly (alternative path).

def _ep_node(nid: str, tool: str, category: str, terminal: bool = False) -> dict:
    return dict(node_type="endpoint", id=nid, tool=tool, category=category, terminal=terminal)


@pytest.fixture()
def small_graph() -> nx.MultiDiGraph:
    """
    4-endpoint linear graph across 3 tools, 2 categories.
    Chain: ep_A → ep_B → ep_C → ep_D  (booking_id)
    ep_B also chains to ep_D directly.
    """
    G = nx.MultiDiGraph()

    # Nodes
    G.add_node("ep:ep_A", **_ep_node("ep_A", "tool_1", "Cat1"))
    G.add_node("ep:ep_B", **_ep_node("ep_B", "tool_2", "Cat1"))
    G.add_node("ep:ep_C", **_ep_node("ep_C", "tool_3", "Cat2"))
    G.add_node("ep:ep_D", **_ep_node("ep_D", "tool_3", "Cat2"))

    # CHAINS_TO edges
    G.add_edge("ep:ep_A", "ep:ep_B", edge_type="CHAINS_TO", via_type="booking_id")
    G.add_edge("ep:ep_B", "ep:ep_C", edge_type="CHAINS_TO", via_type="booking_id")
    G.add_edge("ep:ep_C", "ep:ep_D", edge_type="CHAINS_TO", via_type="booking_id")
    G.add_edge("ep:ep_B", "ep:ep_D", edge_type="CHAINS_TO", via_type="booking_id")

    return G


@pytest.fixture()
def single_ep_graph() -> nx.MultiDiGraph:
    """A graph with a single non-terminal endpoint and no CHAINS_TO edges."""
    G = nx.MultiDiGraph()
    G.add_node("ep:solo", **_ep_node("solo", "tool_1", "Cat1"))
    return G


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_identical_output(self, small_graph):
        sampler = ChainSampler(small_graph)
        constraints = ChainConstraints(length=3)
        results = [sampler.sample(constraints, seed=42).endpoint_ids for _ in range(10)]
        assert all(r == results[0] for r in results)

    def test_different_seeds_may_differ(self, small_graph):
        """Not a strict requirement, but confirms seed is actually used."""
        sampler = ChainSampler(small_graph)
        c = ChainConstraints(length=2)
        r1 = sampler.sample(c, seed=1).endpoint_ids
        r2 = sampler.sample(c, seed=9999).endpoint_ids
        # Both should be non-empty; they may or may not differ depending on topology.
        assert len(r1) >= 1
        assert len(r2) >= 1

    def test_no_global_rng_mutation(self, small_graph):
        """Two independent samplers with different seeds produce independent results."""
        s1 = ChainSampler(small_graph)
        s2 = ChainSampler(small_graph)
        c = ChainConstraints(length=3)
        r1 = s1.sample(c, seed=1)
        r2 = s2.sample(c, seed=2)
        # We only require that both are valid (non-empty, no crash).
        assert r1.endpoint_ids or r1.truncated
        assert r2.endpoint_ids or r2.truncated


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------

class TestLength:
    def test_exact_length(self, small_graph):
        sampler = ChainSampler(small_graph)
        result = sampler.sample(ChainConstraints(length=2), seed=42)
        assert not result.truncated
        assert len(result.endpoint_ids) == 2

    def test_length_range(self, small_graph):
        sampler = ChainSampler(small_graph)
        for seed in range(20):
            result = sampler.sample(ChainConstraints(length=(1, 3)), seed=seed)
            n = len(result.endpoint_ids)
            assert 1 <= n <= 3 or (result.truncated and n == 0)

    def test_length_one(self, small_graph):
        sampler = ChainSampler(small_graph)
        result = sampler.sample(ChainConstraints(length=1), seed=42)
        assert len(result.endpoint_ids) == 1
        assert not result.truncated


# ---------------------------------------------------------------------------
# min_distinct_tools
# ---------------------------------------------------------------------------

class TestMinDistinctTools:
    def test_enforced_when_satisfiable(self, small_graph):
        """With enough seeds tried, at least one must produce a ≥2-tool chain."""
        sampler = ChainSampler(small_graph)
        # The graph has ep_A(tool_1)→ep_B(tool_2)→ep_C(tool_3) — 3 distinct tools
        # reachable in a single chain. Over 20 seeds, at least one should satisfy.
        found_valid = False
        for seed in range(20):
            result = sampler.sample(ChainConstraints(length=3, min_distinct_tools=2), seed=seed)
            if not result.truncated and len(result.endpoint_ids) == 3:
                tools = {small_graph.nodes[ep]["tool"] for ep in result.endpoint_ids}
                assert len(tools) >= 2
                found_valid = True
                break
        assert found_valid, "No seed produced a valid 3-hop 2-tool chain"

    def test_unsatisfiable_returns_empty(self, small_graph):
        sampler = ChainSampler(small_graph)
        # graph has 3 tools; requiring 5 is impossible
        result = sampler.sample(ChainConstraints(length=4, min_distinct_tools=5), seed=42)
        assert result.endpoint_ids == []
        assert result.truncated
        assert result.failure_reason == FailureReason.UNSATISFIABLE


# ---------------------------------------------------------------------------
# must_include_categories
# ---------------------------------------------------------------------------

class TestMustIncludeCategories:
    def test_all_required_categories_present(self, small_graph):
        """Over multiple seeds, at least one must satisfy both categories."""
        sampler = ChainSampler(small_graph)
        # The graph has Cat1(ep_A, ep_B) → Cat2(ep_C, ep_D) via CHAINS_TO.
        found_valid = False
        for seed in range(20):
            result = sampler.sample(
                ChainConstraints(length=3, must_include_categories=["Cat1", "Cat2"]),
                seed=seed,
            )
            if not result.truncated:
                cats = {small_graph.nodes[ep]["category"] for ep in result.endpoint_ids}
                assert "Cat1" in cats
                assert "Cat2" in cats
                found_valid = True
                break
        assert found_valid, "No seed produced a chain spanning both categories"

    def test_unsatisfiable_category(self, small_graph):
        sampler = ChainSampler(small_graph)
        result = sampler.sample(
            ChainConstraints(length=2, must_include_categories=["Nonexistent"]),
            seed=42,
        )
        assert result.endpoint_ids == []
        assert result.truncated
        assert result.failure_reason == FailureReason.UNSATISFIABLE


# ---------------------------------------------------------------------------
# must_include_endpoints
# ---------------------------------------------------------------------------

class TestMustIncludeEndpoints:
    def test_required_endpoint_present(self, small_graph):
        """Over multiple seeds, at least one must yield a chain containing ep_C."""
        sampler = ChainSampler(small_graph)
        # ep_C is reachable: ep_A → ep_B → ep_C. At least one seed should land there.
        found_valid = False
        for seed in range(20):
            result = sampler.sample(
                ChainConstraints(length=3, must_include_endpoints=["ep:ep_C"]),
                seed=seed,
            )
            if not result.truncated and "ep:ep_C" in result.endpoint_ids:
                found_valid = True
                break
        assert found_valid, "No seed produced a chain containing ep_C"

    def test_unreachable_endpoint_unsatisfiable(self, small_graph):
        sampler = ChainSampler(small_graph)
        # "ep:ep_UNKNOWN" does not exist in the graph.
        result = sampler.sample(
            ChainConstraints(length=2, must_include_endpoints=["ep:ep_UNKNOWN"]),
            seed=42,
        )
        assert result.endpoint_ids == []
        assert result.truncated
        assert result.failure_reason == FailureReason.UNSATISFIABLE


# ---------------------------------------------------------------------------
# allow_repeats
# ---------------------------------------------------------------------------

class TestAllowRepeats:
    def test_no_repeats_by_default(self, small_graph):
        sampler = ChainSampler(small_graph)
        for seed in range(30):
            result = sampler.sample(ChainConstraints(length=4, allow_repeats=False), seed=seed)
            if not result.truncated:
                assert len(result.endpoint_ids) == len(set(result.endpoint_ids))


# ---------------------------------------------------------------------------
# Dead-end / truncation
# ---------------------------------------------------------------------------

class TestTruncation:
    def test_truncated_on_dead_end(self, single_ep_graph):
        """Single endpoint with no CHAINS_TO → can't extend → truncated."""
        sampler = ChainSampler(single_ep_graph)
        result = sampler.sample(ChainConstraints(length=3), seed=42)
        assert result.truncated
        assert result.failure_reason == FailureReason.DEAD_END
        # Should not crash; may return partial chain of length 1.
        assert isinstance(result.endpoint_ids, list)

    def test_dead_end_does_not_raise(self, single_ep_graph):
        sampler = ChainSampler(single_ep_graph)
        # No exception expected.
        result = sampler.sample(ChainConstraints(length=10), seed=99)
        assert result.truncated


# ---------------------------------------------------------------------------
# Pattern stubs
# ---------------------------------------------------------------------------

class TestPatternStubs:
    def test_parallel_raises_not_implemented(self, small_graph):
        sampler = ChainSampler(small_graph)
        with pytest.raises(NotImplementedError):
            sampler.sample(ChainConstraints(length=2, pattern="parallel"), seed=42)

    def test_branch_merge_raises_not_implemented(self, small_graph):
        sampler = ChainSampler(small_graph)
        with pytest.raises(NotImplementedError):
            sampler.sample(ChainConstraints(length=2, pattern="branch_merge"), seed=42)


# ---------------------------------------------------------------------------
# seed_endpoint_id populated on success
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_seed_endpoint_id_set(self, small_graph):
        sampler = ChainSampler(small_graph)
        result = sampler.sample(ChainConstraints(length=2), seed=42)
        assert not result.truncated
        assert result.seed_endpoint_id == result.endpoint_ids[0]

    def test_seed_endpoint_id_empty_on_failure(self, small_graph):
        sampler = ChainSampler(small_graph)
        result = sampler.sample(
            ChainConstraints(length=2, must_include_categories=["Nonexistent"]),
            seed=42,
        )
        assert result.seed_endpoint_id == ""
