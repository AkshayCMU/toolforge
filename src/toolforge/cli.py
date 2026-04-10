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

    typer.echo("\nBuild complete. Artifacts written to:")
    for name in (
        "registry.json",
        "normalization_report.json",
        "subset_report.json",
        "semantic_types.json",
        "chain_only_types.json",
        "build_report.md",
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
    typer.echo("not implemented")
    raise typer.Exit(0)


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
    typer.echo("not implemented")
    raise typer.Exit(0)


@app.command()
def compare(
    a: Path = typer.Option(..., "--a", help="Path to Run A JSONL (steering off)."),
    b: Path = typer.Option(..., "--b", help="Path to Run B JSONL (steering on)."),
) -> None:
    """Compare two runs for diversity and quality (Run A vs Run B experiment)."""
    typer.echo("not implemented")
    raise typer.Exit(0)
