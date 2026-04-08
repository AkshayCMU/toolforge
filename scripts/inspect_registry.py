"""Quick diagnostic: load + normalize + subset the fixture categories, print summary.

Usage:  python scripts/inspect_registry.py
"""

from collections import defaultdict
from pathlib import Path

from toolforge.registry.loader import walk_toolbench
from toolforge.registry.normalizer import normalize_corpus
from toolforge.registry.subset import select_subset

FIXTURE_ROOT = Path("tests/fixtures/toolbench_mini/data/toolenv/tools")


def _category_counts(tools):
    counts = defaultdict(lambda: {"tools": 0, "endpoints": 0})
    for t in tools:
        counts[t.category]["tools"] += 1
        counts[t.category]["endpoints"] += len(t.endpoints)
    return counts


def main() -> None:
    print(f"Loading from: {FIXTURE_ROOT.resolve()}\n")

    tools, report = normalize_corpus(walk_toolbench(FIXTURE_ROOT))

    total_endpoints = sum(len(t.endpoints) for t in tools)
    total_params = sum(
        len(ep.parameters) for t in tools for ep in t.endpoints
    )
    total_response_fields = sum(
        len(ep.response_schema) for t in tools for ep in t.endpoints
    )

    print("=== LOADER + NORMALIZER SUMMARY ===\n")
    print(f"Tools seen (raw files):     {report.total_seen}")
    print(f"Tools kept (normalized):    {report.total_kept}")
    print(f"Endpoints (total):          {total_endpoints}")
    print(f"Parameters (total):         {total_params}")
    print(f"Response fields (total):    {total_response_fields}")

    print(f"\n--- Per-category tool counts (before subset) ---")
    before = _category_counts(tools)
    for cat in sorted(before):
        c = before[cat]
        print(f"  {cat:20s}  tools={c['tools']:4d}  endpoints={c['endpoints']:5d}")

    print(f"\n--- Drop reasons ---")
    if report.drop_reasons:
        for reason, count in sorted(report.drop_reasons.items()):
            print(f"  {reason:40s}  {count}")
    else:
        print("  (none)")

    print(f"\n--- Normalization rules fired ---")
    if report.rule_counts:
        for rule, count in sorted(report.rule_counts.items()):
            print(f"  {rule:30s}  {count}")
    else:
        print("  (none)")

    print(f"\n--- Distinct raw type strings ---")
    for t in sorted(report.distinct_raw_type_strings):
        print(f"  {t}")

    # --- Subset ---
    print(f"\n=== SUBSET FILTER (target_endpoints=500, seed=42) ===\n")
    subset = select_subset(tools, target_endpoints=500, seed=42)

    after = _category_counts(subset)
    total_sub_eps = sum(c["endpoints"] for c in after.values())

    print(f"{'Category':20s}  {'Before tools':>12}  {'Before eps':>10}  {'After tools':>11}  {'After eps':>9}")
    print("-" * 72)
    for cat in sorted(before):
        b = before[cat]
        a = after.get(cat, {"tools": 0, "endpoints": 0})
        print(
            f"  {cat:18s}  {b['tools']:12d}  {b['endpoints']:10d}  "
            f"{a['tools']:11d}  {a['endpoints']:9d}"
        )
    print("-" * 72)
    b_total_tools = sum(c["tools"] for c in before.values())
    b_total_eps = sum(c["endpoints"] for c in before.values())
    a_total_tools = sum(c["tools"] for c in after.values())
    print(
        f"  {'TOTAL':18s}  {b_total_tools:12d}  {b_total_eps:10d}  "
        f"{a_total_tools:11d}  {total_sub_eps:9d}"
    )


if __name__ == "__main__":
    main()
