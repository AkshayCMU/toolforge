"""Phase 1 build pipeline — loader through schema inference.

Single entry point: build_registry(). All LLM calls are cached; a warm run
(all cache hits) completes in well under 30 seconds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import structlog

from toolforge.registry.loader import walk_toolbench
from toolforge.registry.models import Tool
from toolforge.registry.normalizer import NormalizationReport, normalize_corpus
from toolforge.registry.schema_infer import infer_corpus
from toolforge.registry.semantic_typing import type_corpus
from toolforge.registry.semantic_vocab import CHAIN_ONLY_VOCAB, USER_PROVIDED_VOCAB
from toolforge.registry.subset import select_subset

log = structlog.get_logger(__name__)


@dataclass
class BuildResult:
    """All outputs from a successful build_registry() call."""

    tools: list[Tool]
    normalization_report: NormalizationReport
    subset_report: dict[str, dict[str, int]]  # category -> {tools, endpoints}
    semantic_types: dict[str, dict]            # type -> {count, tier}
    chain_only_types: list[str]                # producer-only semantic types
    schema_stats: dict[str, int]               # static/schema/llm/empty counts
    accepted_new_types: dict[str, int]
    llm_calls: dict[str, int]                  # typing_calls, schema_calls, total_cache_hits


def build_registry(
    data_dir: Path,
    examples_dir: Path,
    cache_dir: Path,
    client: anthropic.Anthropic,
    seed: int = 42,
    target_endpoints: int = 500,
    categories: list[str] | None = None,
) -> BuildResult:
    """Run the full Phase 1 pipeline. Returns a BuildResult with all derived data."""

    # -------------------------------------------------------------------------
    # Stage 1: Load + normalize
    # -------------------------------------------------------------------------
    log.info("build.stage", stage="normalize")
    tools_all, norm_report = normalize_corpus(walk_toolbench(data_dir))
    log.info(
        "build.normalize_done",
        total_seen=norm_report.total_seen,
        total_kept=norm_report.total_kept,
    )

    # -------------------------------------------------------------------------
    # Stage 1b: Optional category filter
    # -------------------------------------------------------------------------
    if categories:
        allowed = frozenset(categories)
        before = len(tools_all)
        tools_all = [t for t in tools_all if t.category in allowed]
        log.info(
            "build.category_filter",
            allowed=sorted(allowed),
            before=before,
            after=len(tools_all),
        )

    # -------------------------------------------------------------------------
    # Stage 2: Stratified subset selection
    # -------------------------------------------------------------------------
    log.info("build.stage", stage="subset")
    subset = select_subset(tools_all, target_endpoints=target_endpoints, seed=seed)
    total_eps = sum(len(t.endpoints) for t in subset)
    log.info("build.subset_done", tools=len(subset), endpoints=total_eps)

    subset_report: dict[str, dict[str, int]] = {}
    for tool in subset:
        cat = tool.category
        if cat not in subset_report:
            subset_report[cat] = {"tools": 0, "endpoints": 0}
        subset_report[cat]["tools"] += 1
        subset_report[cat]["endpoints"] += len(tool.endpoints)

    # -------------------------------------------------------------------------
    # Stage 3: Schema inference (static + schema + LLM, cached)
    # Must run BEFORE semantic typing so the LLM sees populated response
    # schemas when assigning response-field semantic types.  Running typing
    # first caused the LLM to see mostly empty response_schema tuples and
    # assign null to virtually all response fields, yielding <5 CHAINS_TO
    # edges regardless of corpus size.
    # -------------------------------------------------------------------------
    log.info("build.stage", stage="schema_inference")
    schema_tools, schema_stats = infer_corpus(subset, examples_dir, client, cache_dir)

    # -------------------------------------------------------------------------
    # Stage 4: Semantic typing (LLM, cached)
    # Runs on schema-enriched tools so response fields are populated.
    # -------------------------------------------------------------------------
    log.info("build.stage", stage="semantic_typing")
    typed_tools, accepted_new = type_corpus(schema_tools, client, cache_dir)

    # Count typing cache hits/misses for the build report
    from toolforge.registry.semantic_typing import MODEL as TYPING_MODEL
    from toolforge.registry.semantic_typing import PROMPT_VERSION as TYPING_PV
    from toolforge.registry.semantic_typing import _cache_key as typing_cache_key
    from toolforge.registry.semantic_typing import _load_cache as typing_load_cache

    typing_hits = 0
    typing_calls = 0
    for tool in schema_tools:
        for ep in tool.endpoints:
            key = typing_cache_key(ep, TYPING_MODEL, TYPING_PV)
            cache_path = cache_dir / "llm" / f"{key}.json"
            if typing_load_cache(cache_path) is not None:
                typing_hits += 1
            else:
                typing_calls += 1

    final_tools = typed_tools

    # -------------------------------------------------------------------------
    # Stage 5: Derive vocabulary artifacts
    # -------------------------------------------------------------------------
    log.info("build.stage", stage="derive_vocab")

    # Collect semantic type usage counts (params = consumers, response fields = producers)
    param_type_counts: dict[str, int] = {}
    field_type_counts: dict[str, int] = {}

    for tool in final_tools:
        for ep in tool.endpoints:
            for p in ep.parameters:
                if p.semantic_type:
                    param_type_counts[p.semantic_type] = (
                        param_type_counts.get(p.semantic_type, 0) + 1
                    )
            for f in ep.response_schema:
                if f.semantic_type:
                    field_type_counts[f.semantic_type] = (
                        field_type_counts.get(f.semantic_type, 0) + 1
                    )

    all_types = set(param_type_counts) | set(field_type_counts)
    accepted_vocab = frozenset(accepted_new) | CHAIN_ONLY_VOCAB | USER_PROVIDED_VOCAB

    semantic_types: dict[str, dict] = {}
    for st in sorted(all_types):
        if st in CHAIN_ONLY_VOCAB:
            tier = "CHAIN_ONLY"
        elif st in USER_PROVIDED_VOCAB:
            tier = "USER_PROVIDED"
        elif st in accepted_new:
            tier = "CHAIN_ONLY"  # new types accepted into the corpus are CHAIN_ONLY
        else:
            tier = "unknown"
        semantic_types[st] = {
            "tier": tier,
            "param_count": param_type_counts.get(st, 0),
            "field_count": field_type_counts.get(st, 0),
            "total_count": param_type_counts.get(st, 0) + field_type_counts.get(st, 0),
            "is_new_type": st in accepted_new,
        }

    # Chain-only types = all types in the CHAIN_ONLY semantic tier.
    # Emit the full tier (seed vocab + accepted new types) regardless of whether
    # each type happened to appear in this corpus.  Keying on tier semantics rather
    # than corpus-presence avoids the "produced but never consumed" filter that
    # over-rejects when the corpus is large enough for every type to appear on both
    # parameters and response fields simultaneously.
    chain_only_tier = CHAIN_ONLY_VOCAB | frozenset(accepted_new)
    chain_only_types = sorted(chain_only_tier)

    llm_calls = {
        "typing_calls": typing_calls,
        "typing_cache_hits": typing_hits,
        "schema_calls": schema_stats.get("llm_calls", 0),
        "schema_cache_hits": schema_stats.get("cache_hits", 0),
    }

    return BuildResult(
        tools=final_tools,
        normalization_report=norm_report,
        subset_report=subset_report,
        semantic_types=semantic_types,
        chain_only_types=chain_only_types,
        schema_stats=schema_stats,
        accepted_new_types=accepted_new,
        llm_calls=llm_calls,
    )


def save_artifacts(result: BuildResult, artifacts_dir: Path) -> None:
    """Write all Phase 1 artifacts to disk."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. registry.json — enriched Tool list
    registry_path = artifacts_dir / "registry.json"
    registry_path.write_text(
        json.dumps([t.model_dump() for t in result.tools], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("build.artifact_saved", path=str(registry_path))

    # 2. normalization_report.json
    norm_path = artifacts_dir / "normalization_report.json"
    norm_report = result.normalization_report
    norm_path.write_text(
        json.dumps(
            {
                "total_seen": norm_report.total_seen,
                "total_kept": norm_report.total_kept,
                "drop_reasons": norm_report.drop_reasons,
                "per_category_counts": norm_report.per_category_counts,
                "rule_counts": norm_report.rule_counts,
                "distinct_raw_type_strings": sorted(norm_report.distinct_raw_type_strings),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("build.artifact_saved", path=str(norm_path))

    # 3. subset_report.json
    subset_path = artifacts_dir / "subset_report.json"
    total_tools = sum(v["tools"] for v in result.subset_report.values())
    total_eps = sum(v["endpoints"] for v in result.subset_report.values())
    subset_path.write_text(
        json.dumps(
            {
                "total_tools": total_tools,
                "total_endpoints": total_eps,
                "per_category": result.subset_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("build.artifact_saved", path=str(subset_path))

    # 4. semantic_types.json
    sem_path = artifacts_dir / "semantic_types.json"
    sem_path.write_text(
        json.dumps(
            {
                "accepted_new_types": result.accepted_new_types,
                "all_types": result.semantic_types,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    log.info("build.artifact_saved", path=str(sem_path))

    # 5. chain_only_types.json
    chain_path = artifacts_dir / "chain_only_types.json"
    chain_path.write_text(
        json.dumps(result.chain_only_types, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("build.artifact_saved", path=str(chain_path))

    # 6. build_report.md
    _write_build_report(result, artifacts_dir / "build_report.md")


def _write_build_report(result: BuildResult, path: Path) -> None:
    norm = result.normalization_report
    schema = result.schema_stats
    llm = result.llm_calls

    total_tools = sum(v["tools"] for v in result.subset_report.values())
    total_eps = sum(v["endpoints"] for v in result.subset_report.values())

    lines: list[str] = [
        "# toolforge build report",
        "",
        "## Phase 1 pipeline summary",
        "",
        f"| Stage | Input | Output |",
        f"|-------|-------|--------|",
        f"| Load + Normalize | {norm.total_seen} raw tool files | {norm.total_kept} tools kept |",
        f"| Subset selection | {norm.total_kept} tools | {total_tools} tools, {total_eps} endpoints |",
        f"| Semantic typing | {total_eps} endpoints | {len(result.semantic_types)} distinct types |",
        f"| Schema inference | {total_eps} endpoints | {total_eps} with response_schema |",
        "",
        "## Subset per-category breakdown",
        "",
        "| Category | Tools | Endpoints |",
        "|----------|-------|-----------|",
    ]
    for cat, counts in sorted(result.subset_report.items()):
        lines.append(f"| {cat} | {counts['tools']} | {counts['endpoints']} |")

    lines += [
        "",
        "## LLM call counts",
        "",
        f"| Pass | Calls | Cache hits |",
        f"|------|-------|------------|",
        f"| Semantic typing | {llm['typing_calls']} | {llm['typing_cache_hits']} |",
        f"| Schema inference | {llm['schema_calls']} | {llm['schema_cache_hits']} |",
        "",
        "## Schema inference coverage",
        "",
        f"| Path | Count |",
        f"|------|-------|",
        f"| static (response_examples) | {schema.get('static', 0)} |",
        f"| schema (normalizer) | {schema.get('schema', 0)} |",
        f"| llm (fallback) | {schema.get('llm', 0)} |",
        f"| empty | {schema.get('empty', 0)} |",
        "",
        "## Normalization rule counts",
        "",
        "| Rule tag | Count |",
        "|----------|-------|",
    ]
    for rule, count in sorted(norm.rule_counts.items(), key=lambda x: -x[1]):
        lines.append(f"| `{rule}` | {count} |")

    lines += [
        "",
        "## Drop reasons",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ]
    for reason, count in sorted(norm.drop_reasons.items(), key=lambda x: -x[1]):
        lines.append(f"| {reason} | {count} |")

    lines += [
        "",
        "## Accepted new semantic types (>= 3 occurrences)",
        "",
        "| Type | Count |",
        "|------|-------|",
    ]
    for t, c in sorted(result.accepted_new_types.items(), key=lambda x: -x[1]):
        lines.append(f"| `{t}` | {c} |")

    lines += [
        "",
        "## Producer-only chain types",
        "",
        "Types that appear as response fields but never as parameters.",
        "The executor uses this list to enforce hard grounding.",
        "",
        ", ".join(f"`{t}`" for t in result.chain_only_types) or "(none)",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("build.artifact_saved", path=str(path))
