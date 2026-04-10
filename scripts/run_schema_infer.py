"""Run the F1.6 response schema inference pass on the 500-endpoint subset.

Static + schema paths run first (no LLM). LLM fallback runs only for endpoints
with no example and no existing schema. Results are cached in .cache/llm_schema/.

Usage: python scripts/run_schema_infer.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from toolforge.config import get_settings
from toolforge.registry.loader import walk_toolbench
from toolforge.registry.normalizer import normalize_corpus
from toolforge.registry.schema_infer import infer_corpus
from toolforge.registry.semantic_typing import type_corpus
from toolforge.registry.subset import select_subset

FIXTURE_ROOT = Path("tests/fixtures/toolbench_mini/data/toolenv/tools")
EX_DIR = Path("tests/fixtures/toolbench_mini/response_examples")
ARTIFACT_DIR = Path("artifacts")
SCHEMA_ARTIFACT = ARTIFACT_DIR / "schema_corpus.json"
SCHEMA_STATS = ARTIFACT_DIR / "schema_stats.json"


def main() -> None:
    settings = get_settings()
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # Build 500-endpoint subset (seed=42)
    tools_all, _ = normalize_corpus(walk_toolbench(FIXTURE_ROOT))
    subset = select_subset(tools_all, target_endpoints=500, seed=42)
    total_endpoints = sum(len(t.endpoints) for t in subset)
    print(f"\nSubset: {len(subset)} tools, {total_endpoints} endpoints")

    # Re-apply F1.5 typing pass (100% cache hits -- 0 LLM calls)
    typed_tools, _ = type_corpus(subset, client, settings.cache_dir)
    print("F1.5 typing pass: all cache hits")

    # Pre-scan F1.6 LLM cache
    from toolforge.registry.schema_infer import MODEL, PROMPT_VERSION, _cache_key
    cold_eps = []
    warm_count = 0
    for tool in typed_tools:
        for ep in tool.endpoints:
            if ep.response_schema:
                continue  # will go schema path, no LLM needed
            key = _cache_key(ep, MODEL, PROMPT_VERSION)
            cache_path = settings.cache_dir / "llm_schema" / f"{key}.json"
            if cache_path.exists():
                warm_count += 1
            else:
                cold_eps.append(ep.id)

    cold_count = len(cold_eps)
    print(f"F1.6 cache pre-scan: {warm_count} warm, {cold_count} cold (LLM calls needed)")

    # Run schema inference (static + schema + llm)
    print("\nRunning infer_corpus ...")
    inferred, stats = infer_corpus(typed_tools, EX_DIR, client, settings.cache_dir)

    # Field-path pattern analysis for suspicious high-frequency fields
    path_counter: Counter[str] = Counter()
    for tool in inferred:
        for ep in tool.endpoints:
            if ep.mock_policy == "llm":
                for f in ep.response_schema:
                    # Normalize to leaf token for frequency analysis
                    leaf = f.path.rstrip("[]").rsplit(".", 1)[-1].rsplit("[]", 1)[-1]
                    path_counter[leaf] += 1

    suspicious_threshold = 50
    suspicious = {k: v for k, v in path_counter.most_common(20) if v >= suspicious_threshold}

    # Save artifacts
    ARTIFACT_DIR.mkdir(exist_ok=True)
    SCHEMA_ARTIFACT.write_text(
        json.dumps([t.model_dump() for t in inferred], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    schema_stats_data = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "total_endpoints": total_endpoints,
        "static": stats["static"],
        "schema": stats["schema"],
        "llm": stats["llm"],
        "empty": stats["empty"],
        "llm_calls": stats["llm_calls"],
        "cache_hits": stats["cache_hits"],
        "top_llm_field_paths": path_counter.most_common(30),
        "suspicious_high_freq_paths": suspicious,
    }
    SCHEMA_STATS.write_text(
        json.dumps(schema_stats_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Report
    print("\n" + "=" * 60)
    print("F1.6 SCHEMA INFERENCE REPORT")
    print("=" * 60)
    print(f"  Total endpoints processed : {total_endpoints}")
    print(f"  Static (example match)    : {stats['static']}")
    print(f"  Schema (normalizer schema): {stats['schema']}")
    print(f"  LLM fallback              : {stats['llm']}")
    print(f"  Empty (LLM returned none) : {stats['empty']}")
    print(f"  LLM calls made            : {stats['llm_calls']}")
    print(f"  LLM cache hits            : {stats['cache_hits']}")
    print()

    if suspicious:
        print("  Suspicious high-frequency LLM field paths (>= 50 occurrences):")
        for leaf, count in sorted(suspicious.items(), key=lambda x: -x[1]):
            print(f"    {leaf:<30s} count={count}")
    else:
        print("  No suspiciously high-frequency LLM field paths detected.")

    print()
    print(f"Artifacts saved:")
    print(f"  {SCHEMA_ARTIFACT}")
    print(f"  {SCHEMA_STATS}")
    print("=" * 60)


if __name__ == "__main__":
    main()
