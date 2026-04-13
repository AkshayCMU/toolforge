# toolforge

Offline synthetic multi-turn tool-use conversation generator, grounded in a 500-endpoint
subset of [ToolBench](https://github.com/OpenBMB/ToolBench).

Produces JSONL datasets of multi-turn, multi-tool conversations where:
- Arguments in turn N reference real values from turn N−1 outputs (executor-enforced grounding)
- Chains are sampled from a tool knowledge graph, not hardcoded
- Cross-conversation diversity steering prevents repeated tool patterns
- An LLM-as-judge scores every conversation on 4 dimensions

See [DESIGN.md](DESIGN.md) for architecture, design decisions, and honest failure analysis.

---

## Requirements

- Python 3.11 or 3.12
- `ANTHROPIC_API_KEY` in `.env`
- ToolBench raw data at `../toolbench_raw/data/data/toolenv/tools/` (for `toolforge build` only)

---

## Quickstart

```bash
# 1. Clone and enter the project directory
git clone <repo>
cd toolforge

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install (editable + dev deps)
pip install -e ".[dev]"

# 4. Set your API key
cp .env.example .env               # then edit: ANTHROPIC_API_KEY=<your-key>  # nocheck

# 5. Build the registry and graph artifacts  (~500 LLM calls cold; ~0 warm)
toolforge build

# 6. Generate a small sample to verify the pipeline
toolforge generate --n 5 --seed 42 --out runs/sample.jsonl

# 7. Evaluate quality and diversity
toolforge evaluate --in runs/sample.jsonl --diversity --out reports/sample.json

# 8. Diversity experiment (Run A baseline vs Run B steered)
toolforge generate --n 120 --seed 42 --no-cross-conversation-steering --out runs/run_a.jsonl
toolforge generate --n 120 --seed 42 --out runs/run_b.jsonl
toolforge evaluate --in runs/run_a.jsonl --diversity --out reports/run_a.json
toolforge evaluate --in runs/run_b.jsonl --diversity --out reports/run_b.json
toolforge compare --a reports/run_a.json --b reports/run_b.json
```

> **Cost guardrail:** `toolforge generate` requires explicit confirmation for `--n > 10`.
> The 120 × 2 diversity experiment makes ~4,000 LLM calls. Confirm before running.

---

## CLI Reference

### `toolforge build`

Builds the tool registry and knowledge graph from raw ToolBench data. Outputs artifacts
to `artifacts/` (registry.json, graph.pkl, semantic types, normalization report, etc.).

```
toolforge build [--data-dir PATH] [--examples-dir PATH] [--seed INT] [--target-endpoints INT]
```

Second and subsequent runs hit the LLM cache (0 calls, ~30 s).

---

### `toolforge generate`

Generates a JSONL dataset of multi-turn tool-use conversations.

```
toolforge generate --n 100 --seed 42 --out runs/dataset.jsonl
toolforge generate --n 100 --seed 42 --no-cross-conversation-steering --out runs/run_a.jsonl
```

Flags:
- `--n INT` — number of conversations (default: 100)
- `--seed INT` — RNG seed; same seed → same output (default: 42)
- `--out PATH` — output JSONL path (default: `runs/dataset.jsonl`)
- `--no-cross-conversation-steering` — disable diversity steering (Run A baseline)

**Example output (1 record, abbreviated):**
```json
{
  "conversation_id": "conv-000042",
  "messages": [
    {"role": "user", "content": "I need to check the Premier League standings."},
    {"role": "assistant", "content": "I can help with that. Which season?"},
    {"role": "user", "content": "The current 2024 season."},
    {"role": "assistant", "content": "[tool_call: Sports/Football/getLeagues, args={\"season\": \"2024\"}]"},
    {"role": "user", "content": "[tool_result: {\"leagues\": [{\"id\": \"PL\", \"name\": \"Premier League\"}]}]"},
    {"role": "assistant", "content": "[tool_call: Sports/Football/getStandings, args={\"leagueId\": \"PL\", \"season\": \"2024\"}]"},
    {"role": "user", "content": "[tool_result: {\"standings\": [...]}]"},
    {"role": "assistant", "content": "Here are the current Premier League standings for 2024: ..."}
  ],
  "tool_calls": [
    {"endpoint_id": "Sports/Football/getLeagues", "arguments": {"season": "2024"}},
    {"endpoint_id": "Sports/Football/getStandings", "arguments": {"leagueId": "PL", "season": "2024"}}
  ],
  "tool_outputs": [...],
  "judge_scores": {
    "naturalness": 4, "tool_correctness": 5, "chain_coherence": 4, "task_completion": 5,
    "mean": 4.5, "overall_pass": true
  },
  "validation_results": [
    {"stage": "structure", "passed": true, "is_hard": true, "errors": [], "warnings": []},
    {"stage": "tool_calls", "passed": true, "is_hard": true, "errors": [], "warnings": []},
    {"stage": "grounding", "passed": true, "is_hard": true, "errors": [], "warnings": []},
    {"stage": "completeness", "passed": true, "is_hard": true, "errors": [], "warnings": []},
    {"stage": "constraints", "passed": true, "is_hard": false, "errors": [], "warnings": []}
  ],
  "metadata": {
    "seed": 42, "sampled_chain": ["Sports/Football/getLeagues", "Sports/Football/getStandings"],
    "pattern": "linear", "length_bucket": "medium",
    "repair_attempts": 0, "was_steered": true, "tools_used": ["Football"], "num_turns": 8
  }
}
```

---

### `toolforge evaluate`

Computes quality and diversity metrics on a JSONL run file. Writes a JSON report and
a human-readable Markdown summary side-by-side.

```
toolforge evaluate --in runs/dataset.jsonl --out reports/eval.json [--diversity/--no-diversity]
```

Flags:
- `--in PATH` — input JSONL (required)
- `--out PATH` — JSON report output (default: `reports/eval.json`); Markdown written alongside at same stem + `.md`
- `--diversity / --no-diversity` — include diversity metrics (default: on)

**Example terminal output:**
```
Loading records from runs/sample.jsonl ...
Loaded 5 records. Computing metrics ...

Results (sample):
  n=5  mean_judge=4.2  pass_rate=0.8
  multi-step=60.0%  multi-tool=80.0%  disambiguation=40.0%
  entropy=2.831  distinct-2=1.0  dispersion=0.412

Reports written to:
  reports/sample.json
  reports/sample.md
```

---

### `toolforge compare`

Compares two evaluation report JSON files (produced by `toolforge evaluate`). Prints a
side-by-side table with numeric deltas and a written tradeoff summary.

```
toolforge compare --a reports/run_a.json --b reports/run_b.json [--out reports/comparison.md]
```

Flags:
- `--a PATH` — Run A evaluation report JSON (steering off)
- `--b PATH` — Run B evaluation report JSON (steering on)
- `--out PATH` — save Markdown output (default: `reports/comparison.md`)

> Note: `--a` and `--b` take **evaluation report JSON paths** (from `toolforge evaluate --out`),
> not raw JSONL run files.

**Example output (excerpt):**
```markdown
# Run Comparison: run_a vs run_b

## Quality Metrics

| Metric           | run_a | run_b | Delta (B−A) | Prior direction               |
|---|---|---|---|---|
| mean_judge_score | 3.921 | 4.012 | +0.091      | Run B ≥ Run A                 |
| pass_rate        | 0.750 | 0.783 | +0.033      | Run B ≥ Run A                 |
| pct_multi_step   | 0.567 | 0.583 | +0.017      | Run B ≈ Run A                 |

## Diversity Metrics

| Metric                    | run_a | run_b | Delta  | Prior direction    |
|---|---|---|---|---|
| Tool coverage entropy     | 3.211 | 3.847 | +0.636 | Run B > Run A      |
| Distinct tool bigrams     | 0.612 | 0.781 | +0.169 | Run B > Run A      |
| Task embedding dispersion | 0.341 | 0.378 | +0.037 | Run B > Run A      |
```

---

## Project Layout

```
src/toolforge/
├── registry/     # F1: load, normalise, filter, semantically type ToolBench tools
├── graph/        # F2: tool knowledge graph + constrained chain sampler
├── execution/    # F3: SessionState, Executor (grounding), MockResponder
├── agents/       # F4-F5: Planner, UserSimulator, Assistant, Judge, RepairAgent
│   └── prompts/  # Prompt markdown files (versioned; cache-keyed)
├── generator/    # F4-F7: LangGraph orchestration + batch generation loop
├── memory/       # F6: CorpusDiversityTracker (cross-conversation steering)
├── evaluation/   # F5-F7: validators, repair, metrics, report generation
└── io/           # stub
```

---

## Tests

```bash
# Unit + integration tests (offline-safe, no LLM calls)
python -m pytest tests/unit tests/integration -q

# Run a single test file
python -m pytest tests/unit/test_metrics.py -v
```

> Note: An e2e test suite (`pytest -m e2e`) is not yet implemented (F8.1 pending).

---

## Reproducing the Diversity Experiment

Full grader reproduction from a clean state (requires warm LLM cache from prior `toolforge build`):

```bash
pip install -e ".[dev]"
cp .env.example .env                            # fill in ANTHROPIC_API_KEY

# Build artifacts (~500 LLM calls cold; 0 on warm cache)
toolforge build

# Run A — steering off (baseline)
toolforge generate --n 120 --seed 42 --no-cross-conversation-steering --out runs/run_a.jsonl

# Run B — steering on
toolforge generate --n 120 --seed 42 --out runs/run_b.jsonl

# Evaluate both
toolforge evaluate --in runs/run_a.jsonl --diversity --out reports/run_a.json
toolforge evaluate --in runs/run_b.jsonl --diversity --out reports/run_b.json

# Compare
toolforge compare --a reports/run_a.json --b reports/run_b.json --out reports/comparison.md
cat reports/comparison.md
```

Estimated LLM calls: ~500 (build, cold) + ~1,700 (run A) + ~1,700 (run B) = ~3,900 total.
Warm-cache build + warm-cache generate: near-zero calls.
