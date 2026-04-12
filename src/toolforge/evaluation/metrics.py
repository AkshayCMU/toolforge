"""Evaluation metrics for toolforge generated datasets — F7.2.

All functions are pure (no I/O, no LLM calls).  They operate on the
JSON-safe record dicts produced by _conv_to_record() / the JSONL output
of ``toolforge generate``.

Quality metrics (always computed):
  - Mean + per-dimension judge scores
  - Pass rate at threshold (mean ≥ 3.5 AND min ≥ 2.5)
  - % multi-step (≥3 successful tool calls)
  - % multi-tool (≥2 distinct tools used)
  - % disambiguation (assistant asked ≥1 clarifying question before first tool call)
  - Length distribution (short/medium/long counts + percentages)

Diversity metrics (computed when --diversity):
  - Tool coverage entropy (Shannon entropy of endpoint_id usage distribution)
  - Distinct tool bigrams distinct-2 (unique adjacent endpoint pairs / total pairs)
  - Task embedding dispersion (mean pairwise cosine distance of first user messages)

Determinism guarantee: given the same list of records, all metrics return
the same value on every call (embedding model is local, fixed weights).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Optional dependency — imported at module level for patchability in tests.
# If sentence_transformers is absent, embedding_dispersion returns the sentinel.
try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:
    _SentenceTransformer = None  # type: ignore[assignment,misc]

# Sentinel string for "embedding model unavailable"
_EMBED_UNAVAILABLE = "__unavailable__"

# Score dimensions present in judge_scores dicts
_JUDGE_DIMS = ("naturalness", "tool_correctness", "chain_coherence", "task_completion")

# Pattern to detect a tool-call message
_TOOL_CALL_PREFIX = "[tool_call:"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_disambiguation(messages: list[dict[str, Any]]) -> bool:
    """Return True if the assistant asked a clarifying question before the first tool call.

    Heuristic: any assistant message before the first [tool_call: ...] message is
    classified as a clarification/disambiguation turn.
    """
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "assistant":
            if isinstance(content, str) and content.startswith(_TOOL_CALL_PREFIX):
                # Reached first tool call without a prior non-tool assistant message.
                return False
            # Non-tool-call assistant message before first tool call → disambiguation.
            return True
    return False


def _successful_tool_call_count(record: dict[str, Any]) -> int:
    """Count successful (non-error) tool calls in a record.

    Uses the tool_outputs list which contains raw session output dicts.
    A call is successful when "error" is None or absent.
    """
    return sum(
        1
        for o in record.get("tool_outputs", [])
        if o.get("error") is None
    )


def _distinct_tools_used(record: dict[str, Any]) -> int:
    """Count distinct tool names (middle path component) from tool_calls list."""
    tools: set[str] = set()
    for tc in record.get("tool_calls", []):
        ep_id = tc.get("endpoint_id", "")
        parts = ep_id.split("/")
        if len(parts) >= 2:
            tools.add(parts[1])
    return len(tools)


def _first_user_message(record: dict[str, Any]) -> str:
    """Return the content of the first user message in a record."""
    for msg in record.get("messages", []):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else ""
    return ""


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------

def compute_quality_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute quality metrics from a list of JSONL record dicts.

    Returns a dict with keys:
        n, mean_judge_score, per_dimension_means, pass_rate,
        pct_multi_step, pct_multi_tool, pct_disambiguation,
        length_distribution
    """
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "mean_judge_score": None,
            "per_dimension_means": {},
            "pass_rate": None,
            "pct_multi_step": None,
            "pct_multi_tool": None,
            "pct_disambiguation": None,
            "length_distribution": {"short": 0, "medium": 0, "long": 0},
        }

    # --- Judge scores ---------------------------------------------------------
    scored = [r for r in records if r.get("judge_scores")]
    dim_scores: dict[str, list[float]] = {d: [] for d in _JUDGE_DIMS}
    pass_count = 0

    for r in scored:
        js = r["judge_scores"]
        for d in _JUDGE_DIMS:
            v = js.get(d)
            if isinstance(v, (int, float)):
                dim_scores[d].append(float(v))
        if js.get("overall_pass") is True:
            pass_count += 1

    per_dim_means: dict[str, float] = {}
    all_means: list[float] = []
    for d in _JUDGE_DIMS:
        vals = dim_scores[d]
        if vals:
            m = sum(vals) / len(vals)
            per_dim_means[d] = round(m, 3)
            all_means.append(m)

    mean_judge = round(sum(all_means) / len(all_means), 3) if all_means else None
    pass_rate = round(pass_count / len(scored), 3) if scored else None

    # --- Multi-step / multi-tool / disambiguation -----------------------------
    multi_step = sum(1 for r in records if _successful_tool_call_count(r) >= 3)
    multi_tool = sum(1 for r in records if _distinct_tools_used(r) >= 2)
    disambiguation = sum(1 for r in records if _has_disambiguation(r.get("messages", [])))

    pct_ms = round(multi_step / n, 3)
    pct_mt = round(multi_tool / n, 3)
    pct_da = round(disambiguation / n, 3)

    # --- Length distribution --------------------------------------------------
    length_counter: Counter[str] = Counter()
    for r in records:
        lb = r.get("metadata", {}).get("length_bucket", "unknown")
        length_counter[lb] += 1

    length_dist = {
        "short": length_counter.get("short", 0),
        "medium": length_counter.get("medium", 0),
        "long": length_counter.get("long", 0),
    }

    return {
        "n": n,
        "n_scored": len(scored),
        "mean_judge_score": mean_judge,
        "per_dimension_means": per_dim_means,
        "pass_rate": pass_rate,
        "pct_multi_step": pct_ms,
        "pct_multi_tool": pct_mt,
        "pct_disambiguation": pct_da,
        "length_distribution": length_dist,
    }


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def compute_tool_coverage_entropy(records: list[dict[str, Any]]) -> float:
    """Shannon entropy of endpoint_id usage distribution.

    Range: 0 (one endpoint always used) to log(n_endpoints) (perfectly uniform).
    Returns 0.0 for empty or single-endpoint input.
    """
    counts: Counter[str] = Counter()
    for r in records:
        for tc in r.get("tool_calls", []):
            ep_id = tc.get("endpoint_id", "")
            if ep_id:
                counts[ep_id] += 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)

    return round(entropy, 4)


def compute_distinct_bigrams(records: list[dict[str, Any]]) -> float:
    """Distinct-2: unique adjacent endpoint pairs / total adjacent pairs.

    For each conversation, adjacent (tool_call[i], tool_call[i+1]) pairs are
    extracted in order.  Distinct-2 measures chain-pattern diversity beyond
    individual tool frequency.

    Returns 0.0 when there are no pairs.
    """
    all_pairs: list[tuple[str, str]] = []
    for r in records:
        tc_ids = [tc.get("endpoint_id", "") for tc in r.get("tool_calls", [])]
        for i in range(len(tc_ids) - 1):
            if tc_ids[i] and tc_ids[i + 1]:
                all_pairs.append((tc_ids[i], tc_ids[i + 1]))

    if not all_pairs:
        return 0.0

    distinct = len(set(all_pairs))
    return round(distinct / len(all_pairs), 4)


def compute_embedding_dispersion(
    records: list[dict[str, Any]],
    *,
    model_name: str = "all-MiniLM-L6-v2",
) -> float | str:
    """Mean pairwise cosine distance of first user message embeddings.

    Uses sentence-transformers with the local all-MiniLM-L6-v2 model.
    Returns the mean pairwise cosine *distance* (1 - cosine_similarity),
    which is 0 for identical texts and approaches 1 for orthogonal ones.

    Returns _EMBED_UNAVAILABLE string on import failure.
    Falls back gracefully if fewer than 2 records are present (returns 0.0).
    """
    texts = [_first_user_message(r) for r in records]
    texts = [t for t in texts if t]

    if len(texts) < 2:
        return 0.0

    if _SentenceTransformer is None:
        log.warning("embedding_dispersion.unavailable", error="sentence_transformers not installed")
        return _EMBED_UNAVAILABLE

    try:
        import numpy as np
    except ImportError as exc:
        log.warning("embedding_dispersion.unavailable", error=str(exc))
        return _EMBED_UNAVAILABLE

    model = _SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    # Cosine similarity matrix (embeddings are already L2-normalized)
    sim_matrix = np.dot(embeddings, embeddings.T)
    n = len(embeddings)

    # Mean pairwise distance = mean of (1 - sim) for upper triangle (i < j)
    total_dist = 0.0
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_dist += 1.0 - float(sim_matrix[i, j])
            pair_count += 1

    if pair_count == 0:
        return 0.0

    return round(total_dist / pair_count, 4)


def compute_diversity_metrics(
    records: list[dict[str, Any]],
    *,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Compute all three diversity metrics and return as a dict.

    Keys: tool_coverage_entropy, distinct_bigrams, embedding_dispersion.
    If embedding model is unavailable, embedding_dispersion is the string
    sentinel _EMBED_UNAVAILABLE and a warning is logged.
    """
    entropy = compute_tool_coverage_entropy(records)
    bigrams = compute_distinct_bigrams(records)
    dispersion = compute_embedding_dispersion(records, model_name=model_name)

    return {
        "tool_coverage_entropy": entropy,
        "distinct_bigrams": bigrams,
        "embedding_dispersion": dispersion,
    }


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(
    records: list[dict[str, Any]],
    *,
    include_diversity: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Compute quality metrics (always) and optionally diversity metrics.

    Returns a flat dict suitable for JSON serialisation + Markdown rendering.
    """
    result: dict[str, Any] = {"quality": compute_quality_metrics(records)}
    if include_diversity:
        result["diversity"] = compute_diversity_metrics(
            records, model_name=embedding_model
        )
    return result
