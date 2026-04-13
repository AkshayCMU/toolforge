"""CLI entry point for toolforge."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import anthropic
import typer

from toolforge.config import configure_logging, get_settings

app = typer.Typer(
    name="toolforge",
    help="Offline synthetic tool-use conversation generator.",
    no_args_is_help=True,
)


@app.command()
def build(
    data_dir: Optional[Path] = typer.Option(
        None,
        "--data-dir",
        help="Root of ToolBench tool JSON files. Defaults to settings.toolbench_data_dir.",
    ),
    examples_dir: Optional[Path] = typer.Option(
        None,
        "--examples-dir",
        help="Root of response_examples files. Required if not set in environment.",
    ),
    seed: int = typer.Option(42, "--seed", help="RNG seed for subset selection."),
    target_endpoints: int = typer.Option(
        500, "--target-endpoints", help="Target endpoint count for the subset."
    ),
) -> None:
    """Run the Phase 1 build pipeline and write all registry artifacts."""
    from toolforge.registry.pipeline import build_registry, save_artifacts

    configure_logging()
    settings = get_settings()

    resolved_data_dir = data_dir or settings.toolbench_data_dir
    if examples_dir is None:
        typer.echo(
            "Error: --examples-dir is required (no default configured).", err=True
        )
        raise typer.Exit(1)

    typer.echo(f"Building from: {resolved_data_dir}")
    typer.echo(f"Examples dir:  {examples_dir}")
    typer.echo(f"Artifacts dir: {settings.artifacts_dir}")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    result = build_registry(
        data_dir=resolved_data_dir,
        examples_dir=examples_dir,
        cache_dir=settings.cache_dir,
        client=client,
        seed=seed,
        target_endpoints=target_endpoints,
    )

    save_artifacts(result, settings.artifacts_dir)

    typer.echo("\nBuilding tool graph...")
    from toolforge.graph.build import build_graph, save_graph
    graph = build_graph(result.tools, result.chain_only_types)
    save_graph(graph, settings.artifacts_dir)

    typer.echo("\nBuild complete. Artifacts written to:")
    for name in (
        "registry.json",
        "normalization_report.json",
        "subset_report.json",
        "semantic_types.json",
        "chain_only_types.json",
        "build_report.md",
        "graph.pkl",
        "graph_report.json",
    ):
        typer.echo(f"  {settings.artifacts_dir / name}")


@app.command()
def generate(
    n: int = typer.Option(100, "--n", help="Number of conversations to generate."),
    seed: int = typer.Option(42, "--seed", help="RNG seed for reproducibility."),
    out: Path = typer.Option(
        Path("runs/dataset.jsonl"),
        "--out",
        help="Output JSONL path.",
    ),
    no_cross_conversation_steering: bool = typer.Option(
        False,
        "--no-cross-conversation-steering",
        help="Disable cross-conversation diversity steering (Run A baseline).",
        is_flag=True,
    ),
) -> None:
    """Generate a dataset of synthetic tool-use conversations."""
    import json as _json

    from toolforge.generator.loop import generate_batch

    configure_logging()
    settings = get_settings()

    # Cost guardrail: require explicit confirmation for large runs.
    if n > 10:
        typer.echo(
            f"Warning: generating {n} conversations will make ~{n * 17} LLM calls "
            "(estimate; varies with cache hit rate)."
        )
        confirmed = typer.confirm(f"Proceed with n={n}?", default=False)
        if not confirmed:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    out.parent.mkdir(parents=True, exist_ok=True)

    was_steered = not no_cross_conversation_steering
    typer.echo(
        f"Generating {n} conversations (seed={seed}, steering={'on' if was_steered else 'off'}) -> {out}"
    )

    records = generate_batch(
        n=n,
        seed=seed,
        artifacts_dir=settings.artifacts_dir,
        cache_dir=settings.cache_dir,
        was_steered=was_steered,
    )

    with out.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(_json.dumps(record, ensure_ascii=False) + "\n")

    typer.echo(f"Done. Wrote {len(records)} conversations to {out}.")


@app.command()
def evaluate(
    in_path: Path = typer.Option(
        ...,
        "--in",
        help="Input JSONL dataset to evaluate.",
    ),
    diversity: bool = typer.Option(
        True,
        "--diversity/--no-diversity",
        help="Compute diversity metrics in addition to judge scores.",
    ),
    out: Path = typer.Option(
        Path("reports/eval.json"),
        "--out",
        help="Output evaluation report path.",
    ),
) -> None:
    """Validate and score a generated conversation dataset."""
    import json as _json

    from toolforge.evaluation.metrics import compute_all_metrics
    from toolforge.evaluation.report import build_report, save_reports

    configure_logging()

    if not in_path.exists():
        typer.echo(f"Error: input file not found: {in_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading records from {in_path} ...")
    records: list[dict] = []
    with in_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(_json.loads(line))

    typer.echo(f"Loaded {len(records)} records. Computing metrics ...")

    metrics = compute_all_metrics(records, include_diversity=diversity)
    run_label = in_path.stem
    report = build_report(metrics, run_label=run_label, source_path=str(in_path))

    json_path, md_path = save_reports(report, out)

    def _fmt_pct(v: object) -> str:
        return f"{v:.1%}" if isinstance(v, float) else "n/a"

    q = metrics["quality"]
    typer.echo(f"\nResults ({run_label}):")
    typer.echo(f"  n={q['n']}  mean_judge={q.get('mean_judge_score')}  pass_rate={q.get('pass_rate')}")
    typer.echo(f"  multi-step={_fmt_pct(q.get('pct_multi_step'))}  multi-tool={_fmt_pct(q.get('pct_multi_tool'))}  disambiguation={_fmt_pct(q.get('pct_disambiguation'))}")
    if diversity and "diversity" in metrics:
        d = metrics["diversity"]
        typer.echo(f"  entropy={d.get('tool_coverage_entropy')}  distinct-2={d.get('distinct_bigrams')}  dispersion={d.get('embedding_dispersion')}")
    typer.echo(f"\nReports written to:\n  {json_path}\n  {md_path}")


@app.command()
def compare(
    a: Path = typer.Option(..., "--a", help="Path to Run A evaluation report (JSON)."),
    b: Path = typer.Option(..., "--b", help="Path to Run B evaluation report (JSON)."),
    out: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Optional path to save the comparison Markdown. Defaults to reports/comparison.md.",
    ),
) -> None:
    """Compare two evaluation reports (Run A vs Run B experiment)."""
    import json as _json

    from toolforge.evaluation.report import compare_reports

    configure_logging()

    for p in (a, b):
        if not p.exists():
            typer.echo(f"Error: report file not found: {p}", err=True)
            raise typer.Exit(1)

    report_a = _json.loads(a.read_text(encoding="utf-8"))
    report_b = _json.loads(b.read_text(encoding="utf-8"))

    comparison_md = compare_reports(report_a, report_b)

    typer.echo(comparison_md)

    save_path = out or Path("reports/comparison.md")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(comparison_md, encoding="utf-8")
    typer.echo(f"\nComparison saved to: {save_path}")
