"""Report generation for toolforge evaluation — F7.2.

Produces two artifacts side-by-side:
  - JSON report (machine-readable, --out PATH)
  - Markdown report (human-readable, same directory, same stem + .md)

Both are derived deterministically from the metrics dict produced by
compute_all_metrics() in evaluation.metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Prior expected values (from FEATURES.md F6.1 / F7.3 spec).
# Used in compare output to show actual vs expected delta direction.
# These are qualitative directional priors, not precise numerical predictions.
_COMPARE_PRIORS: dict[str, str] = {
    "mean_judge_score":     "Run B ≥ Run A (steering improves quality by reducing repetition)",
    "pass_rate":            "Run B ≥ Run A",
    "tool_coverage_entropy":"Run B > Run A (steering flattens endpoint frequency distribution)",
    "distinct_bigrams":     "Run B > Run A (steering promotes varied chain patterns)",
    "embedding_dispersion": "Run B > Run A (steering forces semantic task variety)",
    "pct_multi_step":       "Run B ≈ Run A (both should satisfy ≥50% target)",
    "pct_multi_tool":       "Run B ≈ Run A (structural, not affected by steering)",
}


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def build_report(
    metrics: dict[str, Any],
    run_label: str = "run",
    source_path: str | None = None,
) -> dict[str, Any]:
    """Wrap raw metrics in a report envelope with provenance metadata."""
    return {
        "run_label": run_label,
        "source_path": source_path,
        "metrics": metrics,
    }


def save_json_report(report: dict[str, Any], out_path: Path) -> None:
    """Write the JSON report to *out_path*; parent dirs created automatically."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("report.saved.json", path=str(out_path))


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _md_quality(q: dict[str, Any]) -> str:
    lines = ["## Quality Metrics\n"]

    n = q.get("n", 0)
    n_scored = q.get("n_scored", n)
    lines.append(f"- **Conversations:** {n} total, {n_scored} scored by judge\n")

    mean = q.get("mean_judge_score")
    lines.append(f"- **Mean judge score:** {mean if mean is not None else 'n/a'}\n")

    per_dim = q.get("per_dimension_means", {})
    if per_dim:
        lines.append("\n### Per-Dimension Means\n\n")
        lines.append("| Dimension | Mean |\n")
        lines.append("|---|---|\n")
        for dim, val in per_dim.items():
            lines.append(f"| {dim} | {val} |\n")

    pass_rate = q.get("pass_rate")
    lines.append(f"\n- **Pass rate** (mean ≥ 3.5 AND min ≥ 2.5): "
                 f"{f'{pass_rate:.1%}' if pass_rate is not None else 'n/a'}\n")

    pct_ms = q.get("pct_multi_step")
    pct_mt = q.get("pct_multi_tool")
    pct_da = q.get("pct_disambiguation")
    lines.append(f"- **Multi-step** (≥3 tool calls): "
                 f"{f'{pct_ms:.1%}' if pct_ms is not None else 'n/a'}\n")
    lines.append(f"- **Multi-tool** (≥2 distinct tools): "
                 f"{f'{pct_mt:.1%}' if pct_mt is not None else 'n/a'}\n")
    lines.append(f"- **Disambiguation** (clarifying question before first tool call): "
                 f"{f'{pct_da:.1%}' if pct_da is not None else 'n/a'}\n")

    ld = q.get("length_distribution", {})
    if ld and n > 0:
        lines.append("\n### Length Distribution\n\n")
        lines.append("| Bucket | Count | % |\n")
        lines.append("|---|---|---|\n")
        for bucket in ("short", "medium", "long"):
            cnt = ld.get(bucket, 0)
            pct = cnt / n
            lines.append(f"| {bucket} | {cnt} | {pct:.1%} |\n")

    return "".join(lines)


def _md_diversity(d: dict[str, Any]) -> str:
    lines = ["## Diversity Metrics\n\n"]
    lines.append("| Metric | Value |\n")
    lines.append("|---|---|\n")

    entropy = d.get("tool_coverage_entropy", "n/a")
    bigrams = d.get("distinct_bigrams", "n/a")
    dispersion = d.get("embedding_dispersion", "n/a")

    lines.append(f"| Tool coverage entropy | {entropy} |\n")
    lines.append(f"| Distinct tool bigrams (distinct-2) | {bigrams} |\n")
    lines.append(f"| Task embedding dispersion | {dispersion} |\n")

    return "".join(lines)


def build_markdown_report(report: dict[str, Any]) -> str:
    """Render a human-readable Markdown string from a report dict."""
    label = report.get("run_label", "run")
    source = report.get("source_path", "")
    metrics = report.get("metrics", {})

    parts = [f"# Evaluation Report: {label}\n\n"]
    if source:
        parts.append(f"**Source:** `{source}`\n\n")

    q = metrics.get("quality")
    if q:
        parts.append(_md_quality(q))
        parts.append("\n")

    d = metrics.get("diversity")
    if d:
        parts.append(_md_diversity(d))
        parts.append("\n")

    return "".join(parts)


def save_markdown_report(report: dict[str, Any], out_path: Path) -> None:
    """Write Markdown to *out_path* (same stem as JSON, .md suffix)."""
    md_path = out_path.with_suffix(".md")
    md_path.write_text(build_markdown_report(report), encoding="utf-8")
    log.info("report.saved.markdown", path=str(md_path))


def save_reports(
    report: dict[str, Any],
    out_path: Path,
) -> tuple[Path, Path]:
    """Write both JSON and Markdown reports.  Returns (json_path, md_path)."""
    save_json_report(report, out_path)
    md_path = out_path.with_suffix(".md")
    save_markdown_report(report, out_path)
    return out_path, md_path


# ---------------------------------------------------------------------------
# Compare two reports
# ---------------------------------------------------------------------------

def _safe_delta(a: Any, b: Any) -> str:
    """Return a formatted delta string if both values are numeric, else 'n/a'."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        delta = float(b) - float(a)
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.3f}"
    return "n/a"


def compare_reports(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
) -> str:
    """Produce a side-by-side comparison table + tradeoff summary as Markdown."""
    label_a = report_a.get("run_label", "Run A")
    label_b = report_b.get("run_label", "Run B")

    metrics_a = report_a.get("metrics", {})
    metrics_b = report_b.get("metrics", {})

    q_a = metrics_a.get("quality", {})
    q_b = metrics_b.get("quality", {})
    d_a = metrics_a.get("diversity", {})
    d_b = metrics_b.get("diversity", {})

    lines = [
        f"# Run Comparison: {label_a} vs {label_b}\n\n",
        "> **Prior:** Based on FEATURES.md F6.1 spec (caps designed for N=120; "
        "steering expected to improve diversity metrics while preserving quality).\n\n",
    ]

    # --- Quality table --------------------------------------------------------
    lines.append("## Quality Metrics\n\n")
    lines.append(f"| Metric | {label_a} | {label_b} | Delta (B−A) | Prior direction |\n")
    lines.append("|---|---|---|---|---|\n")

    quality_rows: list[tuple[str, Any, Any]] = [
        ("n_conversations",  q_a.get("n"),              q_b.get("n")),
        ("mean_judge_score", q_a.get("mean_judge_score"), q_b.get("mean_judge_score")),
        ("pass_rate",        q_a.get("pass_rate"),        q_b.get("pass_rate")),
        ("pct_multi_step",   q_a.get("pct_multi_step"),   q_b.get("pct_multi_step")),
        ("pct_multi_tool",   q_a.get("pct_multi_tool"),   q_b.get("pct_multi_tool")),
        ("pct_disambiguation", q_a.get("pct_disambiguation"), q_b.get("pct_disambiguation")),
    ]
    for name, va, vb in quality_rows:
        delta = _safe_delta(va, vb)
        prior = _COMPARE_PRIORS.get(name, "—")
        va_s = f"{va}" if va is not None else "n/a"
        vb_s = f"{vb}" if vb is not None else "n/a"
        lines.append(f"| {name} | {va_s} | {vb_s} | {delta} | {prior} |\n")

    # Per-dimension means
    lines.append("\n### Per-Dimension Judge Scores\n\n")
    lines.append(f"| Dimension | {label_a} | {label_b} | Delta |\n")
    lines.append("|---|---|---|---|\n")
    all_dims = set(q_a.get("per_dimension_means", {}).keys()) | set(q_b.get("per_dimension_means", {}).keys())
    for dim in sorted(all_dims):
        va = q_a.get("per_dimension_means", {}).get(dim)
        vb = q_b.get("per_dimension_means", {}).get(dim)
        delta = _safe_delta(va, vb)
        lines.append(f"| {dim} | {va if va is not None else 'n/a'} | {vb if vb is not None else 'n/a'} | {delta} |\n")

    # Length distribution
    lines.append("\n### Length Distribution\n\n")
    lines.append(f"| Bucket | {label_a} | {label_b} |\n")
    lines.append("|---|---|---|\n")
    n_a = max(q_a.get("n", 1), 1)
    n_b = max(q_b.get("n", 1), 1)
    for bucket in ("short", "medium", "long"):
        ca = q_a.get("length_distribution", {}).get(bucket, 0)
        cb = q_b.get("length_distribution", {}).get(bucket, 0)
        lines.append(f"| {bucket} | {ca} ({ca/n_a:.1%}) | {cb} ({cb/n_b:.1%}) |\n")

    # --- Diversity table ------------------------------------------------------
    if d_a or d_b:
        lines.append("\n## Diversity Metrics\n\n")
        lines.append(f"| Metric | {label_a} | {label_b} | Delta (B−A) | Prior direction |\n")
        lines.append("|---|---|---|---|---|\n")

        div_rows: list[tuple[str, str, Any, Any]] = [
            ("tool_coverage_entropy", "Tool coverage entropy",
             d_a.get("tool_coverage_entropy"), d_b.get("tool_coverage_entropy")),
            ("distinct_bigrams", "Distinct tool bigrams (distinct-2)",
             d_a.get("distinct_bigrams"), d_b.get("distinct_bigrams")),
            ("embedding_dispersion", "Task embedding dispersion",
             d_a.get("embedding_dispersion"), d_b.get("embedding_dispersion")),
        ]
        for key, label, va, vb in div_rows:
            delta = _safe_delta(va, vb)
            prior = _COMPARE_PRIORS.get(key, "—")
            va_s = f"{va}" if va is not None else "n/a"
            vb_s = f"{vb}" if vb is not None else "n/a"
            lines.append(f"| {label} | {va_s} | {vb_s} | {delta} | {prior} |\n")

    # --- Written tradeoff summary --------------------------------------------
    lines.append("\n## Tradeoff Summary\n\n")
    lines.append(_tradeoff_summary(label_a, label_b, q_a, q_b, d_a, d_b))

    return "".join(lines)


def _tradeoff_summary(
    label_a: str,
    label_b: str,
    q_a: dict[str, Any],
    q_b: dict[str, Any],
    d_a: dict[str, Any],
    d_b: dict[str, Any],
) -> str:
    """Generate a concise written tradeoff narrative from the two metric dicts."""
    parts: list[str] = []

    # Quality verdict
    ms_a = q_a.get("mean_judge_score")
    ms_b = q_b.get("mean_judge_score")
    if isinstance(ms_a, float) and isinstance(ms_b, float):
        delta = ms_b - ms_a
        if abs(delta) < 0.05:
            parts.append(
                f"**Quality:** {label_a} and {label_b} achieve near-identical mean judge scores "
                f"({ms_a:.3f} vs {ms_b:.3f}, Δ={delta:+.3f}). "
                "Steering has negligible quality impact at this scale."
            )
        elif delta > 0:
            parts.append(
                f"**Quality:** {label_b} scores higher than {label_a} "
                f"({ms_b:.3f} vs {ms_a:.3f}, Δ={delta:+.3f}). "
                "Consistent with the prior that reducing repetition improves judge scores."
            )
        else:
            parts.append(
                f"**Quality:** {label_b} scores lower than {label_a} "
                f"({ms_b:.3f} vs {ms_a:.3f}, Δ={delta:+.3f}). "
                "Contrary to prior — steering may introduce harder chains that increase failure rate."
            )

    # Diversity verdict
    ent_a = d_a.get("tool_coverage_entropy")
    ent_b = d_b.get("tool_coverage_entropy")
    bg_a  = d_a.get("distinct_bigrams")
    bg_b  = d_b.get("distinct_bigrams")
    disp_a = d_a.get("embedding_dispersion")
    disp_b = d_b.get("embedding_dispersion")

    div_improvements: list[str] = []
    div_regressions: list[str] = []

    for name, va, vb in [
        ("tool coverage entropy", ent_a, ent_b),
        ("distinct bigrams", bg_a, bg_b),
        ("embedding dispersion", disp_a, disp_b),
    ]:
        if isinstance(va, float) and isinstance(vb, float):
            if vb > va + 0.001:
                div_improvements.append(name)
            elif vb < va - 0.001:
                div_regressions.append(name)

    if div_improvements:
        parts.append(
            f"\n\n**Diversity:** {label_b} improves over {label_a} on: "
            + ", ".join(div_improvements) + ". "
            "Cross-conversation steering achieves its intended effect on these axes."
        )
    if div_regressions:
        parts.append(
            f"\n\n**Diversity regression:** {label_b} is *worse* than {label_a} on: "
            + ", ".join(div_regressions) + ". "
            "Possible cause: hard rejection caps force the sampler into a smaller "
            "effective chain space, reducing the variety it can explore."
        )
    if not div_improvements and not div_regressions and (d_a or d_b):
        parts.append(
            f"\n\n**Diversity:** {label_b} and {label_a} are statistically indistinguishable "
            "on all three diversity metrics. Consider running at N≥120 to observe the "
            "effect of the hard rejection caps."
        )

    if not parts:
        parts.append("Insufficient data to produce a tradeoff summary.")

    return " ".join(parts) if len(parts) == 1 else "\n".join(parts)
