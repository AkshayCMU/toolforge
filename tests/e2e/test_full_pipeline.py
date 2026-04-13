"""End-to-end pipeline test — F8.1.

Gated behind ``pytest -m e2e``. Do NOT run casually — makes ~1,700 LLM
calls cold (near-zero on a warm cache).

Pipeline under test (all steps invoked as real CLI subprocesses):
  1. graph.pkl rebuild  — pure Python inline (registry.json already exists;
                          avoids re-running toolforge build cold)
  2. toolforge generate --n 100 --seed 42
  3. toolforge evaluate --in ... --diversity --out ...

Pass criteria (FEATURES.md F8.1):
  - ≥100 records written to JSONL
  - mean judge score ≥ 3.5
  - ≥50% conversations have ≥3 tool calls AND ≥2 distinct tools (combined)
  - ≥20% conversations have a disambiguation turn before the first tool call

Log files written to logs/e2e/ (one per step) for post-run analysis.
Each log contains: command, exit code, elapsed time, LLM call summary,
and the full stderr (structlog JSON events).
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    # Allow direct imports from src/ when test runs without editable install.
    sys.path.insert(0, str(_SRC_DIR))

_LOGS_DIR = _PROJECT_ROOT / "logs" / "e2e"
_RUNS_DIR = _PROJECT_ROOT / "runs"
_REPORTS_DIR = _PROJECT_ROOT / "reports"
_ARTIFACTS_DIR = _PROJECT_ROOT / "artifacts"
_CLI = Path(sys.executable).parent / ("toolforge.exe" if sys.platform == "win32" else "toolforge")

# ---------------------------------------------------------------------------
# LLM-call log parser
# ---------------------------------------------------------------------------

def _parse_llm_stats(stderr: str) -> dict:
    """Parse structlog JSON lines from subprocess stderr.

    Returns a dict with:
      live_calls        total cache-miss (actual API) calls
      cache_hits        total cache-hit calls
      by_model          {model_name: {"live": int, "cached": int}}
      by_agent          {agent_name: {"live": int, "cached": int}}
      prompt_tokens     total input tokens (live calls only)
      completion_tokens total output tokens (live calls only)
    """
    live_calls = 0
    cache_hits = 0
    prompt_tokens = 0
    completion_tokens = 0
    by_model: dict[str, dict] = {}
    by_agent: dict[str, dict] = {}

    for raw_line in stderr.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = entry.get("event", "")
        model = entry.get("model", "unknown")
        agent = entry.get("agent_name") or "unknown"

        # Count live calls (cache miss — actual API hit)
        if event == "llm_client.live_call":
            live_calls += 1
            by_model.setdefault(model, {"live": 0, "cached": 0})["live"] += 1
            by_agent.setdefault(agent, {"live": 0, "cached": 0})["live"] += 1

        # Count cache hits and token usage from usage events
        elif event == "llm_client.usage":
            if entry.get("cached") is True:
                cache_hits += 1
                by_model.setdefault(model, {"live": 0, "cached": 0})["cached"] += 1
                by_agent.setdefault(agent, {"live": 0, "cached": 0})["cached"] += 1
            else:
                prompt_tokens += entry.get("prompt_tokens", 0) or 0
                completion_tokens += entry.get("completion_tokens", 0) or 0

    return {
        "live_calls": live_calls,
        "cache_hits": cache_hits,
        "by_model": by_model,
        "by_agent": by_agent,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _run_step(step_name: str, args: list[str], stdin_input: str | None = None) -> dict:
    """Run a CLI subprocess, capture output, save log file, return stats."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _LOGS_DIR / f"{step_name}_{timestamp}.log"

    cmd = [str(_CLI)] + args
    print(f"\n[e2e] Running: toolforge {' '.join(args)}")

    t0 = time.time()
    import subprocess
    result = subprocess.run(
        cmd,
        input=stdin_input,
        capture_output=True,
        text=True,
        cwd=str(_PROJECT_ROOT),
        env={**os.environ},  # pass full env including ANTHROPIC_API_KEY
    )
    elapsed = time.time() - t0

    llm_stats = _parse_llm_stats(result.stderr)

    # Write full log file
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"STEP:       {step_name}\n")
        fh.write(f"COMMAND:    toolforge {' '.join(args)}\n")
        fh.write(f"TIMESTAMP:  {timestamp}\n")
        fh.write(f"ELAPSED:    {elapsed:.1f}s\n")
        fh.write(f"EXIT CODE:  {result.returncode}\n")
        fh.write(f"\n=== LLM CALL SUMMARY ===\n")
        fh.write(f"Live calls (cache miss):  {llm_stats['live_calls']}\n")
        fh.write(f"Cache hits:               {llm_stats['cache_hits']}\n")
        fh.write(f"Prompt tokens:            {llm_stats['prompt_tokens']}\n")
        fh.write(f"Completion tokens:        {llm_stats['completion_tokens']}\n")
        fh.write(f"\nBy model:\n")
        for model, counts in llm_stats["by_model"].items():
            fh.write(f"  {model}: live={counts['live']} cached={counts['cached']}\n")
        fh.write(f"\nBy agent:\n")
        for agent, counts in llm_stats["by_agent"].items():
            fh.write(f"  {agent}: live={counts['live']} cached={counts['cached']}\n")
        fh.write(f"\n=== STDOUT ===\n")
        fh.write(result.stdout or "(empty)\n")
        fh.write(f"\n=== STDERR (structlog JSON) ===\n")
        fh.write(result.stderr or "(empty)\n")

    print(f"[e2e]   exit={result.returncode}  elapsed={elapsed:.1f}s  "
          f"live_calls={llm_stats['live_calls']}  cache_hits={llm_stats['cache_hits']}")
    print(f"[e2e]   log → {log_path}")

    return {
        "step": step_name,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "llm_stats": llm_stats,
        "log_path": log_path,
    }


# ---------------------------------------------------------------------------
# Metric helpers (operate on loaded JSONL records)
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pct_multi_step_and_tool(records: list[dict]) -> float:
    def _qualifies(r: dict) -> bool:
        successful = sum(1 for o in r.get("tool_outputs", []) if o.get("error") is None)
        distinct_tools = len({
            tc.get("endpoint_id", "").split("/")[1]
            for tc in r.get("tool_calls", [])
            if "/" in tc.get("endpoint_id", "")
        })
        return successful >= 3 and distinct_tools >= 2
    return sum(1 for r in records if _qualifies(r)) / len(records)


def _pct_disambiguation(records: list[dict]) -> float:
    def _has_disambig(messages: list[dict]) -> bool:
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("[tool_call:"):
                return False   # hit first tool call with no prior assistant message
            return True        # non-tool assistant message before first tool call
        return False
    return sum(1 for r in records if _has_disambig(r.get("messages", []))) / len(records)


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------

@pytest.mark.e2e
def test_full_pipeline() -> None:
    """End-to-end: graph rebuild → generate 100 → evaluate → assert thresholds."""

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_log = _LOGS_DIR / f"e2e_summary_{run_timestamp}.log"
    steps: list[dict] = []

    # ------------------------------------------------------------------
    # Step 0: Rebuild graph.pkl if missing (pure Python, 0 LLM calls)
    # ------------------------------------------------------------------
    graph_path = _ARTIFACTS_DIR / "graph.pkl"
    if not graph_path.exists():
        print(f"\n[e2e] graph.pkl missing — rebuilding from registry.json (0 LLM calls)")
        t0 = time.time()
        import json as _json
        from toolforge.registry.models import Tool
        from toolforge.graph.build import build_graph, save_graph

        registry_json = _json.loads(
            (_ARTIFACTS_DIR / "registry.json").read_text("utf-8")
        )
        tools = [Tool.model_validate(t) for t in registry_json]
        chain_only_path = _ARTIFACTS_DIR / "chain_only_types.json"
        chain_only = (
            _json.loads(chain_only_path.read_text("utf-8"))
            if chain_only_path.exists() else []
        )
        graph = build_graph(tools, chain_only)
        save_graph(graph, _ARTIFACTS_DIR)
        elapsed = time.time() - t0
        print(f"[e2e]   graph.pkl built in {elapsed:.1f}s")
        steps.append({"step": "graph_build", "returncode": 0,
                      "elapsed": elapsed, "llm_stats": {"live_calls": 0, "cache_hits": 0}})
    else:
        print(f"\n[e2e] graph.pkl already exists — skipping rebuild")
        steps.append({"step": "graph_build", "returncode": 0,
                      "elapsed": 0.0, "llm_stats": {"live_calls": 0, "cache_hits": 0}})

    # ------------------------------------------------------------------
    # Step 1: toolforge generate --n 100 --seed 42
    # ------------------------------------------------------------------
    out_jsonl = _RUNS_DIR / f"e2e_dataset_{run_timestamp}.jsonl"
    gen_result = _run_step(
        "generate",
        ["generate", "--n", "100", "--seed", "42", "--out", str(out_jsonl)],
        stdin_input="y\n",   # confirms the n>10 cost guardrail
    )
    steps.append(gen_result)

    assert gen_result["returncode"] == 0, (
        f"toolforge generate failed (exit {gen_result['returncode']}).\n"
        f"Last 30 lines of stderr:\n"
        + "\n".join(gen_result["stderr"].splitlines()[-30:])
    )
    assert out_jsonl.exists(), f"generate exited 0 but {out_jsonl} was not written"

    # ------------------------------------------------------------------
    # Step 2: toolforge evaluate --in ... --diversity --out ...
    # ------------------------------------------------------------------
    out_report = _REPORTS_DIR / f"e2e_report_{run_timestamp}.json"
    eval_result = _run_step(
        "evaluate",
        ["evaluate", "--in", str(out_jsonl), "--diversity", "--out", str(out_report)],
    )
    steps.append(eval_result)

    assert eval_result["returncode"] == 0, (
        f"toolforge evaluate failed (exit {eval_result['returncode']}).\n"
        f"Last 20 lines of stderr:\n"
        + "\n".join(eval_result["stderr"].splitlines()[-20:])
    )

    # ------------------------------------------------------------------
    # Step 3: Assert quality thresholds
    # ------------------------------------------------------------------
    records = _load_jsonl(out_jsonl)
    assert len(records) >= 100, (
        f"Expected ≥100 records, got {len(records)}. "
        "Check for skipped conversations in the batch loop."
    )

    report = json.loads(out_report.read_text("utf-8"))
    q = report["metrics"]["quality"]

    mean_score = q.get("mean_judge_score")
    assert mean_score is not None, "No judge scores in report — judge agent may have failed"
    assert mean_score >= 3.5, (
        f"Mean judge score {mean_score:.3f} < 3.5 threshold. "
        f"Per-dimension: {q.get('per_dimension_means')}"
    )

    pct_combined = _pct_multi_step_and_tool(records)
    assert pct_combined >= 0.50, (
        f"Only {pct_combined:.1%} of conversations have ≥3 tool calls AND "
        f"≥2 distinct tools (threshold 50%)."
    )

    pct_da = _pct_disambiguation(records)
    assert pct_da >= 0.20, (
        f"Only {pct_da:.1%} of conversations have a disambiguation turn "
        "(threshold 20%)."
    )

    # ------------------------------------------------------------------
    # Step 4: Write summary log and print final report
    # ------------------------------------------------------------------
    total_live = sum(s.get("llm_stats", {}).get("live_calls", 0) for s in steps)
    total_cached = sum(s.get("llm_stats", {}).get("cache_hits", 0) for s in steps)
    total_elapsed = sum(s.get("elapsed", 0.0) for s in steps)

    summary_lines = [
        "=== E2E TEST SUMMARY ===",
        f"Timestamp:        {run_timestamp}",
        f"Total elapsed:    {total_elapsed:.1f}s",
        f"Total LLM calls:  {total_live} live  |  {total_cached} cache hits",
        "",
        "Per-step breakdown:",
    ]
    for s in steps:
        stats = s.get("llm_stats", {})
        summary_lines.append(
            f"  {s['step']:20s}  exit={s['returncode']}  "
            f"elapsed={s.get('elapsed', 0):.1f}s  "
            f"live={stats.get('live_calls', 0)}  cached={stats.get('cache_hits', 0)}"
        )
    summary_lines += [
        "",
        "=== ASSERTION RESULTS ===",
        f"Records:              {len(records)}  (≥100 required)",
        f"Mean judge score:     {mean_score:.3f}  (≥3.5 required)",
        f"Multi-step + tool:    {pct_combined:.1%}  (≥50.0% required)",
        f"Disambiguation:       {pct_da:.1%}  (≥20.0% required)",
        "",
        "=== ARTIFACTS ===",
        f"JSONL:   {out_jsonl}",
        f"Report:  {out_report}",
        f"Logs:    {_LOGS_DIR}/",
    ]

    summary_text = "\n".join(summary_lines)
    summary_log.write_text(summary_text, encoding="utf-8")

    print(f"\n{'='*60}")
    print(summary_text)
    print(f"{'='*60}\n")
