# toolforge

Offline synthetic multi-turn tool-use conversation generator, grounded in ToolBench schemas.

## Quickstart

```bash
# 1. Create and activate a Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install
pip install -e ".[dev]"

# 3. Copy and fill in your API key
cp .env.example .env               # edit ANTHROPIC_API_KEY

# 4. Install pre-commit hook (blocks accidental secret commits)
bash scripts/install-hooks.sh

# 5. Build the registry and graph artifacts
toolforge build

# 6. Generate conversations (fixed seed, 100 samples)
toolforge generate --n 100 --seed 42 --out runs/dataset.jsonl

# 7. Evaluate and score
toolforge evaluate --in runs/dataset.jsonl --out reports/eval.json

# 8. Diversity experiment (Run A vs Run B)
toolforge generate --n 100 --seed 42 --no-cross-conversation-steering --out runs/run_a.jsonl
toolforge generate --n 100 --seed 42 --out runs/run_b.jsonl
toolforge compare --a runs/run_a.jsonl --b runs/run_b.jsonl
```

## Requirements

- Python 3.11.x
- `ANTHROPIC_API_KEY` in `.env`

## Tests

```bash
pytest                        # unit + integration tests
pytest -m e2e                 # end-to-end (makes LLM calls, slow)
```

## Project layout

```
src/toolforge/
├── registry/     # F1: load, normalize, filter, semantically type ToolBench tools
├── graph/        # F2: tool graph + constrained chain sampler
├── execution/    # F3: offline execution model + session state
├── agents/       # F4: planner, user-sim, assistant agents
├── generator/    # F4: end-to-end conversation pipeline
├── memory/       # F6: cross-conversation diversity steering
├── evaluation/   # F5: validators, LLM judge, repair loop
└── io/           # F4: JSONL writer
```

## Reproducing the grader's run

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in ANTHROPIC_API_KEY
toolforge build
toolforge generate --n 100 --seed 42 --out runs/dataset.jsonl
toolforge evaluate --in runs/dataset.jsonl --out reports/eval.json
pytest -m e2e
```
