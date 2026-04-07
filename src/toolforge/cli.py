"""CLI entry point for toolforge."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="toolforge",
    help="Offline synthetic tool-use conversation generator.",
    no_args_is_help=True,
)


@app.command()
def build() -> None:
    """Ingest ToolBench data and build all derived artifacts (registry, graph, indexes)."""
    typer.echo("not implemented")
    raise typer.Exit(0)


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
