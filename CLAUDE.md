# CLAUDE.md

> Persistent context for Claude Code working in this repo. Read this first, every session.

## What this project is

`toolforge` is an offline synthetic data generation system that produces multi-turn,
multi-tool conversations grounded in a subset of [ToolBench](https://github.com/OpenBMB/ToolBench).
Output is a JSONL dataset suitable for training/evaluating tool-use agents.

This is a production-grade offline data generation system. Build it like a senior AI
engineer who has one focused sprint to ship something real, clean, and defensible.
Code quality, architectural decisions, and empirical results all matter equally:

| Criterion | Weight |
|---|---|
| System design & architecture (DESIGN.md) | 25% |
| Functional correctness | 20% |
| Software engineering practices & code quality | 15% |
| Quality evaluation pipeline (LLM-as-judge) | 10% |
| Context management & diversity–quality analysis | 10% |
| Knowledge graph construction & sampling | 10% |
| Multi-agent system design | 10% |

**Implication:** a working-but-incomplete submission with a strong DESIGN.md is worth more
than a complete-but-shallow one. Always prefer documented tradeoffs over silent shortcuts.

The full design spec is in `PROJECT_PLAN.md`. Feature-by-feature task formulations are in
`FEATURES.md`. When in doubt, read both.

---

## Repo layout

```
SAP/                              ← parent dir, NOT the project root
├── toolbench_raw/                ← READ-ONLY external data, ~16k API files. NEVER scan.
│   └── data/data/toolenv/tools/{Category}/{tool}.json
├── ToolBench/                    ← upstream repo clone, reference only. NEVER scan.
├── reproduction_data/            ← upstream artifacts, reference only. NEVER scan.
├── Literature Survey/            ← background reading, ignore unless asked
└── toolforge/                    ← THIS is the project root. Run `claude` from here.
    ├── CLAUDE.md                 ← you are here
    ├── PROJECT_PLAN.md           ← authoritative design doc
    ├── FEATURES.md               ← feature breakdown with Task Formulation blocks
    ├── .claudeignore
    ├── pyproject.toml
    ├── README.md
    ├── DESIGN.md                 ← the graded document (lab notebook, not spec rewrite)
    ├── src/toolforge/
    │   ├── cli.py
    │   ├── config.py
    │   ├── registry/
    │   ├── graph/
    │   ├── execution/
    │   ├── agents/
    │   │   └── prompts/
    │   ├── generator/
    │   ├── memory/
    │   ├── evaluation/
    │   └── io/
    ├── tests/
    │   ├── unit/
    │   ├── integration/          ← REQUIRED: repair loop test
    │   └── e2e/                  ← REQUIRED: build + generate ≥100 + assert score
    ├── artifacts/                ← built by `toolforge build` (gitignored)
    ├── runs/                     ← generated datasets (gitignored)
    └── reports/                  ← evaluation reports (gitignored)
```

**The toolbench data is at `../toolbench_raw/data/data/toolenv/tools/`** — note the double
`data/data`. Always reference it as a path; never let Claude Code scan the directory.

---

## The five guiding principles

Every design choice cites at least one of these. If you are writing code that doesn't
ladder back to one of them, stop and reconsider.

### P1 — Task-first thinking: normalize chaos into contracts before writing code
ToolBench JSON is deliberately inconsistent. Before any generation logic, define
`Parameter`, `Endpoint`, `Tool` Pydantic models with **provenance fields** so every
normalized value traces back to its raw source. **Every feature in `FEATURES.md` opens
with a Task Formulation block (Inputs / Outputs / Invariants / Failure modes / Principle
citations) that must be filled in before any code is written. That block is where P1
actually happens.**

### P2 — Symbolic control vs free-form generation: know when to stop using the LLM
**Rule of thumb: if the answer has to be right, it's code; if it has to be plausible, it's an LLM.**

Deterministic Python: registry normalization, graph construction, chain sampling (seeded
RNG), mock ID generation, parameter validation, session registry, diversity tracking,
usage-count steering.

LLMs (structured output where possible): scenario planning, user dialogue, assistant
message composition, judge scoring, repair operation selection.

### P3 — Grounding over hallucination: stateful systems by construction
"Arguments in step N reference real values from step N−1 outputs" is a **runtime
invariant**, not a prompt hope. Three layers, all required:
1. `SessionState` owned by the executor (`available_values_by_type`, `resolved_entities`,
   `created_entities`, `tool_outputs`)
2. Soft grounding: session registry serialized into the Assistant prompt before each turn
3. Hard grounding: executor rejects any semantic-typed argument not in
   `available_values_by_type` and returns a structured error listing valid values

If layer 3 is missing, grounding degrades to "whatever the LLM felt like." Non-negotiable.

### P4 — Modular evaluation: stage-level validation
Deterministic `ValidationResult` checkpoints run **before** the LLM judge:
`validate_structure`, `validate_tool_calls`, `validate_grounding`, `validate_completeness`,
`validate_constraints`. The judge only scores structurally-valid conversations. Each
validator failing routes to a different repair path — modular evaluation enables modular
repair.

### P5 — Diversity by design, not post-hoc
Diversity is a sampler input, not an output metric. `CorpusDiversityTracker` does both
soft biasing (inverse-frequency weights at seed selection) and hard rejection (caps on
per-endpoint, per-category-fraction, per-tool-pair counts). The sampler resamples until
`should_accept()` passes. The diversity experiment measures whether these design-time
choices moved the needle.

---

## Model choices

| Role | Model string | Rationale |
|---|---|---|
| Planner, UserSimulator, Assistant | `claude-haiku-4-5-20251001` | Fast, cheap, reliable JSON mode. |
| Judge, Repair | `claude-sonnet-4-6` | Stronger than generator; prevents self-preference bias inflating scores. **Different family is mandatory.** |
| Embeddings (diversity metrics) | `all-MiniLM-L6-v2` (local, via `sentence-transformers`) | Free, deterministic, no API call needed for evaluation. Fall back to `text-embedding-3-small` only if local model unavailable. |

Using the same model family for generator and judge would inflate scores by ~0.3–0.5
(self-preference bias). This separation is non-negotiable. Document it in DESIGN.md §1.3.

---

## Conventions

### Python
- **Python 3.11.x only.** Pin `requires-python = ">=3.11,<3.12"` in `pyproject.toml`.
  Reason: `mem0ai` pulls in `chromadb`/`qdrant-client` with wheel gaps on 3.12+; staying
  on 3.11 avoids source builds on the grader's machine.
- **Pydantic v2** for all data models. No raw dicts crossing module boundaries.
- **Typing discipline (P2 applied to types):** `mypy` must pass on
  `src/toolforge/{registry,graph,execution,evaluation}/` — deterministic code must be
  type-correct. Agent and generator modules may use `# type: ignore` where LangGraph or
  SDK types are awkward.
- **`ruff`** for lint and format. No bare `except:`, no unused imports.
- **`pytest`** for tests. Fixtures in `tests/conftest.py`.
- **`typer`** for CLI.
- **`structlog`** for internal logging; **`typer.echo()`** for user-facing CLI output.
  Never `print`.

### Determinism
- Every stochastic operation takes an explicit `seed: int` argument.
- Use `random.Random(seed)` instances, not `random.seed()` global mutation.
- Faker is seeded per call.
- Diversity experiment must reproduce identically across runs given the same seed
  (modulo the documented mem0 ANN caveat — document this in DESIGN.md §7.2).

### LLM calls
- **Always cache.** Content-addressed cache keyed by `(model, prompt_hash,
  structured_output_schema_hash)`. Cache lives in `.cache/llm/` (gitignored). Without
  this, iteration is prohibitively expensive.
- **Always use structured output** (Pydantic models via SDK) where the consumer needs
  structure. Never parse JSON out of free text — that is a P2 violation.
- **Temperature:** 0 for judge and repair. 0.7 for user simulator and planner.
- **Generator and judge must be different model families** — see Model choices above.
- **`tenacity`** for retries with exponential backoff. Max 3 attempts.

### Cost discipline — call-count thresholds, not dollar guesses

Claude Code can count calls reliably. Rules are in calls, not dollars.

- **Never** read raw ToolBench files into Claude's context — use the loader and pass paths
  around.
- **Before any operation expected to make >20 LLM calls, state the estimated call count
  and expected cache hit rate, then wait for confirmation.**
- Operations that always require confirmation before running:
  - **Semantic typing pass (`toolforge build`)** — ~500 calls cold, ~0 warm.
  - **Full generate run** — ~2,000 calls for 120 conversations.
  - **Diversity experiment (Run A + Run B)** — ~4,000 calls total.
  - **E2E test** — ~1,500 calls.
- If a session enters an unexpected LLM retry loop, stop after 10 consecutive failures
  and report rather than burning through retries.
- A second `toolforge build` on the same source must make zero LLM calls.

### Testing
- Unit tests for every module in `src/toolforge/`. Target >70% coverage on deterministic
  code; LLM-touching code is harder and that's acceptable.
- **Integration test (REQUIRED):** inject a deliberately broken conversation (hallucinated
  ID), assert the validation layer catches it and the repair loop fixes it.
- **E2E test (REQUIRED):** `toolforge build` + `toolforge generate --n 100 --seed 42` +
  assert mean judge score ≥ 3.5. Gate behind `pytest -m e2e`; do not run casually.

### Git
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.
- Small commits. Each completed feature in `FEATURES.md` = one or two commits.
- Never commit: `runs/`, `artifacts/`, `reports/`, `.cache/`, `.env`, `*.jsonl`,
  `__pycache__`, `.venv/`.

### Dependencies — canonical pinned list (do not add without asking)

**Runtime:**
```
pydantic>=2.5,<3
pydantic-settings>=2.0
python-dotenv>=1.0
typer>=0.12
networkx>=3.2
langgraph>=0.2
anthropic>=0.40
openai>=1.40           # documented fallback only, not used in parallel with anthropic
mem0ai>=0.1
sentence-transformers>=3.0   # for local embeddings in diversity metrics
faker>=25
structlog>=24
tenacity>=8.5
jsonschema>=4.22
numpy>=1.26,<2.1
```

**Dev:**
```
pytest>=8
pytest-cov>=5
ruff>=0.6
mypy>=1.11
```

Adding anything outside this list requires raising it with the user first.

---

## Files and directories Claude Code must NEVER scan

Also in `.claudeignore` but stated here because scanning is the most common way to burn
credits on this project:

- `../toolbench_raw/**` — 16k API files, several GB. Read programmatically via the
  loader only; never opened in Claude's context.
- `../ToolBench/**` — upstream reference repo.
- `../reproduction_data/**` — upstream artifacts.
- `../Literature Survey/**` — background reading.
- `runs/`, `artifacts/`, `reports/`, `.cache/` — generated, potentially huge.
- `*.zip`, `*.pdf` — binary, no value to scan.

If you need to understand a single ToolBench tool file as a reference, ask the user for
one specific path and read only that file. Do not glob.

---

## How to work in this repo — session startup ritual

Do this at the start of every session, in order:

1. Run `git status` and `git log --oneline -10` to see where we left off.
2. Open `FEATURES.md`. Find the **first unchecked feature** (the `☐` box) — that is the
   current feature unless the user says otherwise.
3. Read that feature's **Task Formulation block**:
   - If it says `[confirmed]` → you may proceed to code after confirming with the user.
   - If it says `[draft]` → **stop**. Discuss and refine the block with the user until
     both agree it's complete and accurate. Change `[draft]` to `[confirmed]` before
     writing any code.
   - If the block is empty → same as `[draft]`: fill it in with the user first.
4. Confirm the feature with the user. State what you are about to build in one sentence.
5. Build the feature. When done, check its box in `FEATURES.md`, commit with
   `feat: <feature name>`, then stop and wait for the user to confirm before moving to
   the next feature.

**Do not skip features.** The order in `FEATURES.md` encodes dependencies — the graph
needs the registry, the executor needs the graph, etc.

**Do not** run any `find`, `grep`, `ls`, or `cat` against `../toolbench_raw/` or any
other excluded directory.

**Before any operation expected to make >20 LLM calls, state the estimated count +
expected cache hit rate, and wait for confirmation.**

---

## DESIGN.md is a lab notebook, not a spec rewrite

`PROJECT_PLAN.md` is the specification. `DESIGN.md` should answer questions
`PROJECT_PLAN` *can't*:

- What numbers came out of the diversity experiment? (Run A vs Run B, all metrics, three
  decimal places)
- Which prompt iterations failed? What did the failure look like? What was the fix?
- Which design decisions changed under time pressure, and why?
- What surprised me? Where did my prior in PROJECT_PLAN §10.3 turn out wrong?
- What is the actual normalization-decisions table from the registry build, with real
  ToolBench examples grepped from the source?
- Where does this break, honestly, with concrete failure cases observed during the run?

If a DESIGN.md section reads like a paraphrase of PROJECT_PLAN, delete it and replace it
with evidence. The grader has both files and will notice duplication. Evidence is rewarded.

**Maintain a running prompt-iteration log in DESIGN.md as you build.** The assignment
explicitly requires at least one documented prompt failure. Do not reconstruct it at the
end — record failures in real time.

---

## The hard requirements checklist

Explicit in the assignment PDF. Losing any costs significant points:

- [ ] CLI with `build`, `generate`, `evaluate` commands
- [ ] **The graph sampler must drive generation** (not a hardcoded list). Unit test proves it.
- [ ] **At least one agent uses structured output** (we are doing four: planner, assistant,
      judge, repair)
- [ ] **≥50% of conversations have ≥3 tool calls AND ≥2 distinct tools**
- [ ] **Multi-turn disambiguation**: assistant asks clarifying questions before tool calls
      when intent is ambiguous or required fields are missing
- [ ] **Within-conversation grounding**: arguments in step N reference real values from
      step N−1 outputs (enforced by the executor, not hoped for via prompts)
- [ ] **Cross-conversation steering** with `--no-cross-conversation-steering` CLI flag
- [ ] **Diversity experiment**: Run A (steering off) vs Run B (steering on), same seed,
      ≥2 diversity metrics + judge quality reported for both
- [ ] **LLM-as-judge** with ≥3 dimensions, integrated into per-conversation metadata
- [ ] **Retry/repair loop** that fixes failures rather than discarding
- [ ] **Integration test** for the repair loop (deliberately broken conv → fixed)
- [ ] **E2E test** generating ≥100 samples and asserting mean judge score ≥ 3.5
- [ ] **DESIGN.md** with: architecture, agent protocol, context-management design +
      tradeoffs, prompt iteration log with ≥1 documented failure, diversity & quality
      analysis with numbers, honest "where this breaks" sections
- [ ] **README.md** with one-command reproduction for the grader

---

## What Claude Code must NOT do

- Don't run `/init` — this file replaces it.
- Don't read files from outside `toolforge/` unless the user explicitly provides a path.
- Don't add dependencies outside the canonical pinned list above.
- Don't use an LLM where deterministic code works (P2 violation).
- Don't bypass the graph sampler in generation code (hard-requirement violation).
- Don't write conversations to JSONL before they have passed `ValidationResult` checks (P4 violation).
- Don't paraphrase `PROJECT_PLAN` into `DESIGN.md`. DESIGN.md is the lab notebook.
- **Don't start a feature by writing code.** Open the Task Formulation block in
  `FEATURES.md` first. If it is `[draft]` or empty, fill it in with the user. This is P1
  in daily practice — skipping it is the most common way this kind of project goes wrong.
- Don't move to the next feature until the current one is committed and the user confirms.
- Don't invent design decisions when the plan is silent. Ask.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
