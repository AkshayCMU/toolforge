"""Deterministic constrained chain sampler — F2.2.

ChainSampler.sample() produces a list of endpoint IDs forming a semantically
continuous chain (connected via CHAINS_TO edges). All randomness flows through
a seeded random.Random instance; the same (constraints, seed, steering) triple
always produces the same result.

Design principles:
  P2 — Deterministic: seeded RNG, sorted candidate lists as tie-break.
  P5 — Diversity-ready: steering weights plug in here (F6.1 will populate them).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import networkx as nx
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class FailureReason(str, Enum):
    NONE = "none"
    DEAD_END = "dead_end"           # chain ran out of CHAINS_TO successors before target
    UNSATISFIABLE = "unsatisfiable" # hard constraint cannot be met (bad category, etc.)


@dataclass
class ChainConstraints:
    """Constraints for a single chain-sampling call."""

    length: int | tuple[int, int]
    """Exact chain length, or (min, max) range. Sampled with seeded RNG if a range."""

    must_include_categories: list[str] = field(default_factory=list)
    """ALL listed categories must appear in the final chain (≥1 endpoint each)."""

    must_include_endpoints: list[str] = field(default_factory=list)
    """ALL listed endpoint IDs must appear in the final chain."""

    min_distinct_tools: int = 1
    """Minimum number of distinct tools that must appear in the final chain."""

    pattern: Literal["linear", "parallel", "branch_merge"] = "linear"
    """Traversal pattern. Only 'linear' is implemented; others raise NotImplementedError."""

    allow_repeats: bool = False
    """Whether the same endpoint may appear more than once in the chain."""


@dataclass
class ChainResult:
    """Output of ChainSampler.sample()."""

    endpoint_ids: list[str]
    """Ordered endpoint IDs. Empty when truncated and a hard constraint was violated."""

    truncated: bool = False
    """True when the chain is shorter than the target length, OR when constraints
    could not be satisfied (in which case endpoint_ids is empty)."""

    failure_reason: FailureReason = FailureReason.NONE
    """Distinguishes dead-end (graph ran dry) from unsatisfiable (constraint logic)."""

    seed_endpoint_id: str = ""
    """First endpoint chosen, for debugging. Empty when endpoint_ids is empty."""


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class ChainSampler:
    """Deterministic constrained chain sampler over a CHAINS_TO graph.

    Usage::

        sampler = ChainSampler(graph)
        result = sampler.sample(constraints, seed=42)
        if not result.truncated:
            use(result.endpoint_ids)
    """

    def __init__(self, graph: nx.MultiDiGraph) -> None:
        self._graph = graph

        # Index non-terminal endpoint node IDs, sorted for determinism.
        self._endpoints: list[str] = sorted(
            nid
            for nid, data in graph.nodes(data=True)
            if data.get("node_type") == "endpoint" and not data.get("terminal", False)
        )

        # Subset of endpoints that have ≥1 outgoing CHAINS_TO edge.
        # These are the only valid chain seeds — an endpoint with no CHAINS_TO
        # out-edges can never grow a chain past length 1.
        _chains_to_sources: set[str] = {
            u
            for u, v, d in graph.edges(data=True)
            if d.get("edge_type") == "CHAINS_TO"
        }
        self._chain_seeds: list[str] = sorted(
            ep for ep in self._endpoints if ep in _chains_to_sources
        )

        # Map category → sorted list of (non-terminal) endpoint node IDs.
        self._ep_by_category: dict[str, list[str]] = {}
        for nid in self._endpoints:
            cat = graph.nodes[nid].get("category", "")
            self._ep_by_category.setdefault(cat, []).append(nid)
        # Lists are already sorted because self._endpoints is sorted.

        # Cached attribute dicts for O(1) lookup during sampling.
        self._ep_attrs: dict[str, dict] = {
            nid: dict(graph.nodes[nid]) for nid in self._endpoints
        }

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def sample(
        self,
        constraints: ChainConstraints,
        seed: int,
        steering: dict[str, float] | None = None,
    ) -> ChainResult:
        """Sample a chain satisfying *constraints* using *seed* for reproducibility.

        Args:
            constraints: Length, category, endpoint, and tool requirements.
            seed: RNG seed. Same seed → same output every time.
            steering: Optional map of endpoint node ID → inverse-frequency weight.
                      When None, all endpoints are weighted equally.

        Returns:
            ChainResult with endpoint_ids (may be empty on failure).
        """
        if constraints.pattern != "linear":
            raise NotImplementedError(
                f"pattern={constraints.pattern!r} is not yet implemented. "
                "Only 'linear' is supported in v1."
            )

        rng = random.Random(seed)
        return self._sample_linear(constraints, rng, steering)

    # ------------------------------------------------------------------
    # Linear sampling
    # ------------------------------------------------------------------

    def _sample_linear(
        self,
        constraints: ChainConstraints,
        rng: random.Random,
        steering: dict[str, float] | None,
    ) -> ChainResult:
        # Step 1: resolve target length.
        if isinstance(constraints.length, tuple):
            lo, hi = constraints.length
            target_length = rng.randint(lo, hi)
        else:
            target_length = constraints.length

        # Step 2: build seed candidate set.
        # Prefer endpoints that have outgoing CHAINS_TO edges so the chain
        # can actually grow.  Fall back to all non-terminal endpoints only
        # when no such seeds exist (degenerate graph with no chains at all).
        viable_seeds = self._chain_seeds if self._chain_seeds else self._endpoints

        if constraints.must_include_categories:
            seed_candidates: list[str] = []
            for cat in sorted(constraints.must_include_categories):
                seed_candidates.extend(self._ep_by_category.get(cat, []))
            # Deduplicate while preserving sorted order.
            seen: set[str] = set()
            unique_seeds: list[str] = []
            for ep in seed_candidates:
                if ep not in seen:
                    unique_seeds.append(ep)
                    seen.add(ep)
            # Restrict to viable seeds (those with outgoing CHAINS_TO edges).
            viable_set = set(viable_seeds)
            seed_candidates = [ep for ep in unique_seeds if ep in viable_set] or unique_seeds
        else:
            seed_candidates = list(viable_seeds)

        if not seed_candidates:
            return ChainResult(
                endpoint_ids=[],
                truncated=True,
                failure_reason=FailureReason.UNSATISFIABLE,
            )

        # Step 3: pick seed endpoint (sorted order = deterministic tie-break).
        seed_ep = self._weighted_choice(seed_candidates, steering, rng)
        if seed_ep is None:
            return ChainResult(
                endpoint_ids=[],
                truncated=True,
                failure_reason=FailureReason.UNSATISFIABLE,
            )

        # Step 4 + 5: extend chain via CHAINS_TO, with one backtrack on dead-end.
        chain = self._grow_chain(
            seed_ep, target_length, constraints, steering, rng
        )

        # Step 6: hard constraint validation.
        failure = self._validate_constraints(chain, constraints)
        if failure is not None:
            return ChainResult(
                endpoint_ids=[],
                truncated=True,
                failure_reason=failure,
            )

        # Step 7: return result.
        truncated = len(chain) < target_length
        return ChainResult(
            endpoint_ids=chain,
            truncated=truncated,
            failure_reason=FailureReason.DEAD_END if truncated else FailureReason.NONE,
            seed_endpoint_id=chain[0] if chain else "",
        )

    def _grow_chain(
        self,
        seed_ep: str,
        target_length: int,
        constraints: ChainConstraints,
        steering: dict[str, float] | None,
        rng: random.Random,
    ) -> list[str]:
        """Grow a chain from seed_ep up to target_length via CHAINS_TO.

        Performs one backtrack on dead-end. When backtracking yields no alternative,
        returns the chain as it stood before the dead-end (not the shorter backtracked
        version) so callers see the longest partial chain available.
        """
        chain: list[str] = [seed_ep]
        visited: set[str] = {seed_ep} if not constraints.allow_repeats else set()

        while len(chain) < target_length:
            current = chain[-1]
            candidates = self._next_candidates(current, visited, constraints, steering)

            if candidates:
                chosen = self._weighted_choice(
                    candidates, steering, rng,
                    must_include=constraints.must_include_endpoints,
                )
                chain.append(chosen)
                if not constraints.allow_repeats:
                    visited.add(chosen)
            else:
                # Dead-end. Try one backtrack.
                if len(chain) <= 1:
                    break  # nothing to backtrack to

                best_so_far = list(chain)   # preserve current (longer) chain
                dead_ep = chain.pop()
                if not constraints.allow_repeats:
                    visited.discard(dead_ep)

                # Find alternatives at the backtracked position, excluding dead_ep.
                alt_candidates = self._next_candidates(chain[-1], visited, constraints, steering)
                alt_candidates = [c for c in alt_candidates if c != dead_ep]

                if alt_candidates:
                    # Found an alternative — pick it and continue.
                    chosen = self._weighted_choice(
                        alt_candidates, steering, rng,
                        must_include=constraints.must_include_endpoints,
                    )
                    chain.append(chosen)
                    if not constraints.allow_repeats:
                        visited.add(chosen)
                else:
                    # Backtrack found nothing — return the best partial chain we had.
                    return best_so_far

        return chain

    def _next_candidates(
        self,
        current: str,
        visited: set[str],
        constraints: ChainConstraints,
        steering: dict[str, float] | None,
    ) -> list[str]:
        """Return sorted list of valid CHAINS_TO successors from current."""
        successors: set[str] = set()
        for _, target, data in self._graph.out_edges(current, data=True):
            if data.get("edge_type") == "CHAINS_TO":
                successors.add(target)
        if not constraints.allow_repeats:
            successors -= visited
        # Filter to non-terminal endpoints only (terminal nodes have no out-edges anyway,
        # but be explicit for clarity).
        successors = {s for s in successors if s in self._ep_attrs}
        return sorted(successors)  # sorted = deterministic tie-break

    def _weighted_choice(
        self,
        candidates: list[str],
        steering: dict[str, float] | None,
        rng: random.Random,
        must_include: list[str] | None = None,
    ) -> str | None:
        """Pick one candidate using weights, with sorted input as tie-break."""
        if not candidates:
            return None
        weights = []
        for ep in candidates:
            w = steering.get(ep, 1.0) if steering else 1.0
            if must_include and ep in must_include:
                w *= 2.0
            weights.append(w)
        return rng.choices(candidates, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Hard constraint validation
    # ------------------------------------------------------------------

    def _validate_constraints(
        self, chain: list[str], constraints: ChainConstraints
    ) -> FailureReason | None:
        """Return a FailureReason if the chain violates a hard constraint, else None."""

        # must_include_endpoints: all required endpoints must be present.
        if constraints.must_include_endpoints:
            chain_set = set(chain)
            missing = [
                # The sampler stores node IDs as "ep:{id}"; callers may pass either form.
                ep for ep in constraints.must_include_endpoints
                if ep not in chain_set
            ]
            if missing:
                log.debug("sampler.constraint_fail", reason="missing_endpoints", missing=missing)
                return FailureReason.UNSATISFIABLE

        # must_include_categories: every listed category must appear.
        if constraints.must_include_categories:
            chain_categories = {
                self._ep_attrs[ep]["category"]
                for ep in chain
                if ep in self._ep_attrs
            }
            missing_cats = [
                cat for cat in constraints.must_include_categories
                if cat not in chain_categories
            ]
            if missing_cats:
                log.debug("sampler.constraint_fail", reason="missing_categories",
                          missing=missing_cats)
                return FailureReason.UNSATISFIABLE

        # min_distinct_tools.
        if constraints.min_distinct_tools > 1:
            distinct_tools = {
                self._ep_attrs[ep]["tool"]
                for ep in chain
                if ep in self._ep_attrs
            }
            if len(distinct_tools) < constraints.min_distinct_tools:
                log.debug(
                    "sampler.constraint_fail",
                    reason="min_distinct_tools",
                    have=len(distinct_tools),
                    need=constraints.min_distinct_tools,
                )
                return FailureReason.UNSATISFIABLE

        return None
