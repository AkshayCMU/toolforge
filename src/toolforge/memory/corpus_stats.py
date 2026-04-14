"""CorpusDiversityTracker: frequency counters + diversity steering — F6.1.

Design (P5 — diversity by design):
  Two mechanisms operate on every sample call:

  1. Soft steering: ``steering_weights()`` returns a ``{endpoint_id: float}`` dict
     that is passed to ``ChainSampler.sample(steering=...)``.  Weights follow an
     inverse-frequency formula: ``1/(1+usage_count)``, so the sampler already
     biases toward under-used endpoints before the chain is drawn.

  2. Hard rejection: ``should_accept(chain)`` checks three hard caps and returns
     ``(False, reason)`` if any is violated.  The caller (``_plan_node``) retries
     with seed ``original_seed + attempt`` for up to MAX_ACCEPT_RETRIES attempts.
     All retries exhausted → RuntimeError (no silent fallback, P1).

Stable public API for the metadata mapping:
  ``build_endpoint_metadata(graph)`` reads from ``graph.nodes(data=True)`` using
  only the documented node attributes (``node_type``, ``tool``, ``category``,
  ``terminal``) and returns a plain ``dict[str, dict[str, str]]``.  Neither the
  tracker nor any caller ever touches ``ChainSampler._ep_attrs``.

Determinism guarantee (P2): no randomness, no LLM calls.  Same sequence of
``update()`` calls → same counter state → same weights and acceptance decisions.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    import networkx as nx

log = structlog.get_logger(__name__)

# Maximum retries when should_accept rejects a sampled chain.
# Used by plan_node; defined here so both sides share one constant.
MAX_ACCEPT_RETRIES: int = 10


# ---------------------------------------------------------------------------
# Public helper: build the stable metadata mapping from the graph
# ---------------------------------------------------------------------------

def build_endpoint_metadata(graph: "nx.MultiDiGraph") -> dict[str, dict[str, str]]:
    """Extract a stable ``{node_id: {"category": ..., "tool": ...}}`` mapping.

    Includes only non-terminal endpoint nodes — exactly the set the sampler
    considers.  This is the canonical way to build the mapping that
    ``CorpusDiversityTracker`` expects; it must not be built from
    ``ChainSampler`` internals.

    Args:
        graph: The compiled NetworkX tool graph (from ``build_graph``).

    Returns:
        Mapping of graph node ID → ``{"category": str, "tool": str}``.
    """
    return {
        nid: {
            "category": data["category"],
            "tool": data["tool"],
        }
        for nid, data in graph.nodes(data=True)
        if data.get("node_type") == "endpoint" and not data.get("terminal", False)
    }


# ---------------------------------------------------------------------------
# Stub tracker for --no-cross-conversation-steering mode
# ---------------------------------------------------------------------------

class NoOpDiversityTracker:
    """Stub tracker: always accepts, always weights 1.0, never stores state.

    Used when ``--no-cross-conversation-steering`` is passed on the CLI.
    Every method is a guaranteed no-op so the generator behaves as if no
    corpus memory exists (Run A baseline in the diversity experiment).
    """

    def should_accept(self, chain: list[str]) -> tuple[bool, str]:  # noqa: ARG002
        return (True, "ok")

    def sampling_weight(self, endpoint_id: str) -> float:  # noqa: ARG002
        return 1.0

    def steering_weights(self) -> dict[str, float]:
        return {}

    def update(
        self,
        chain: list[str],  # noqa: ARG002
        pattern: str,  # noqa: ARG002
        length_bucket: str,  # noqa: ARG002
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Real tracker
# ---------------------------------------------------------------------------

class CorpusDiversityTracker:
    """Tracks per-endpoint, per-category, and tool-pair usage across the corpus.

    Provides two outputs consumed by ``ConversationGenerator._plan_node``:
    - ``steering_weights()`` → inverse-frequency dict for the sampler.
    - ``should_accept(chain)`` → hard rejection with reason string.

    Must be updated via ``update()`` after each *accepted* conversation.

    Hard caps re-calibrated for N≈100 conversations with 68 valid chain seeds
    across 8 categories (~8.5 seeds/category).  Original caps were tuned for
    N=120 / 500 endpoints / 40 categories and caused >50% skip rate on the
    smaller 8-category corpus.

    At 100 conversations with 68 seeds across 8 categories:
    - ``MAX_ENDPOINT_COUNT=15`` → any endpoint in at most 15 % of convs.
    - ``MAX_CATEGORY_FRACTION=0.30`` → any category covers at most ~30 convs
       (3.5 convs/seed on average — acceptable given only 8 categories).
    - ``MAX_TOOL_PAIR_COUNT=8`` → allows more pair reuse given the small pool.
    """

    MAX_ENDPOINT_COUNT: int = 15
    MAX_CATEGORY_FRACTION: float = 0.30
    MAX_TOOL_PAIR_COUNT: int = 8

    def __init__(self, endpoint_meta: dict[str, dict[str, str]]) -> None:
        """
        Args:
            endpoint_meta: Mapping of graph node ID → ``{"category": str,
                "tool": str}``.  Build with ``build_endpoint_metadata(graph)``.
        """
        self._meta = endpoint_meta

        # Per-endpoint usage count (key = graph node ID, e.g. ``"ep:xxx"``).
        self.tool_usage: Counter[str] = Counter()
        # Per-category usage count (key = category name string).
        self.category_usage: Counter[str] = Counter()
        # Tool-pair co-occurrence count (key = sorted (tool_a, tool_b) tuple).
        self.tool_pair_usage: Counter[tuple[str, str]] = Counter()
        # Chain traversal pattern distribution ("linear", …).
        self.chain_pattern_usage: Counter[str] = Counter()
        # Coarse chain length distribution ("short", "medium", "long").
        self.length_bucket_usage: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Soft steering
    # ------------------------------------------------------------------

    def sampling_weight(self, endpoint_id: str) -> float:
        """Inverse-frequency weight for a single endpoint.

        Returns 1.0 for unseen endpoints, decreasing toward zero as usage
        grows.  Formula: ``1 / (1 + usage_count)``.
        """
        return 1.0 / (1.0 + self.tool_usage.get(endpoint_id, 0))

    def steering_weights(self) -> dict[str, float]:
        """Build the full ``steering`` dict for all tracked endpoints.

        The returned dict is passed directly to
        ``ChainSampler.sample(steering=...)``.  Only endpoints present in this
        dict have their weight overridden; the sampler defaults to 1.0 for any
        key not present, so an empty dict is safe.
        """
        return {ep_id: self.sampling_weight(ep_id) for ep_id in self._meta}

    # ------------------------------------------------------------------
    # Hard rejection
    # ------------------------------------------------------------------

    def should_accept(self, chain: list[str]) -> tuple[bool, str]:
        """Return ``(True, "ok")`` or ``(False, reason)`` based on hard caps.

        Args:
            chain: Ordered list of graph node IDs as returned by
                   ``ChainResult.endpoint_ids``.

        Returns:
            Tuple of (accepted, reason_string).  ``reason_string`` is ``"ok"``
            on acceptance or a short diagnostic key on rejection.
        """
        if not chain:
            return (False, "empty_chain")

        total_convs: int = sum(self.length_bucket_usage.values())

        # Cap 1: per-endpoint count.
        for ep_id in chain:
            count = self.tool_usage.get(ep_id, 0)
            if count >= self.MAX_ENDPOINT_COUNT:
                log.debug(
                    "diversity.reject",
                    reason="endpoint_cap",
                    endpoint=ep_id,
                    count=count,
                )
                return (False, f"endpoint_cap:{ep_id}")

        # Cap 2: per-category fraction.
        # Enforced only once we have ≥10 conversations so early convs aren't
        # all rejected before any category pattern emerges.
        if total_convs >= 10:
            cats_in_chain: set[str] = {
                self._meta[ep]["category"]
                for ep in chain
                if ep in self._meta
            }
            for cat in cats_in_chain:
                cat_count = self.category_usage.get(cat, 0)
                # Prospective fraction: if we accept this chain.
                prospective = (cat_count + 1) / (total_convs + 1)
                if prospective > self.MAX_CATEGORY_FRACTION:
                    log.debug(
                        "diversity.reject",
                        reason="category_cap",
                        category=cat,
                        prospective_fraction=round(prospective, 3),
                    )
                    return (False, f"category_cap:{cat}")

        # Cap 3: tool-pair co-occurrence count.
        tools_in_chain: list[str] = sorted({
            self._meta[ep]["tool"]
            for ep in chain
            if ep in self._meta
        })
        for i, t1 in enumerate(tools_in_chain):
            for t2 in tools_in_chain[i + 1 :]:
                pair: tuple[str, str] = (t1, t2)
                pair_count = self.tool_pair_usage.get(pair, 0)
                if pair_count >= self.MAX_TOOL_PAIR_COUNT:
                    log.debug(
                        "diversity.reject",
                        reason="tool_pair_cap",
                        pair=pair,
                        count=pair_count,
                    )
                    return (False, f"tool_pair_cap:{t1}+{t2}")

        return (True, "ok")

    # ------------------------------------------------------------------
    # Mutation (call after each accepted conversation)
    # ------------------------------------------------------------------

    def update(self, chain: list[str], pattern: str, length_bucket: str) -> None:
        """Increment all counters to record a newly accepted conversation.

        Args:
            chain: Graph node IDs from the accepted conversation's chain.
            pattern: Chain traversal pattern (``"linear"``, ``"parallel"``, …).
            length_bucket: Coarse length label (``"short"``, ``"medium"``,
                ``"long"``).
        """
        # Per-endpoint.
        for ep_id in chain:
            self.tool_usage[ep_id] += 1

        # Per-category (once per unique category per conversation).
        cats_seen: set[str] = {
            self._meta[ep]["category"]
            for ep in chain
            if ep in self._meta
        }
        for cat in cats_seen:
            self.category_usage[cat] += 1

        # Tool-pair co-occurrences (sorted pair = canonical key).
        tools_seen: list[str] = sorted({
            self._meta[ep]["tool"]
            for ep in chain
            if ep in self._meta
        })
        for i, t1 in enumerate(tools_seen):
            for t2 in tools_seen[i + 1 :]:
                self.tool_pair_usage[(t1, t2)] += 1

        # Pattern + length distribution.
        self.chain_pattern_usage[pattern] += 1
        self.length_bucket_usage[length_bucket] += 1

        log.debug(
            "diversity.update",
            chain_len=len(chain),
            pattern=pattern,
            length_bucket=length_bucket,
            total_convs=sum(self.length_bucket_usage.values()),
        )
