"""Quick diagnostic: load + normalize the 3 fixture categories, print summary.

Usage:  python scripts/inspect_registry.py
"""

from pathlib import Path

from toolforge.registry.loader import walk_toolbench
from toolforge.registry.normalizer import normalize_corpus

FIXTURE_ROOT = Path("tests/fixtures/toolbench_mini/data/toolenv/tools")


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

    print(f"\n--- Per-category tool counts ---")
    for cat in sorted(report.per_category_counts):
        print(f"  {cat:20s}  {report.per_category_counts[cat]}")

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


if __name__ == "__main__":
    main()
