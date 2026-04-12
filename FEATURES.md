# FEATURES.md

> Feature-by-feature working plan for `toolforge`. Every feature opens with a **Task
> Formulation** block that must be confirmed *before* any code is written. P1 in daily
> practice.
>
> **How to use this file:**
> 1. Pick the next unchecked `☐` feature in order.
> 2. Read its Task Formulation block.
>    - `[confirmed]` → proceed after telling the user what you are about to build.
>    - `[draft]` → stop, resolve open questions with the user, change to `[confirmed]`.
> 3. Build. When done, check the box, commit `feat: <name>`, wait for user confirmation.
> 4. Do NOT skip ahead. Order encodes dependencies.
>
> **Cost discipline:** never run `toolforge generate` with `--n > 10` without explicit
> user confirmation. Sample runs only during development.

---

# Phase 0 — Setup

## F0.1 — Project skeleton ☑

### Task Formulation [confirmed]
**Inputs:** empty `toolforge/` directory, `CLAUDE.md`, `PROJECT_PLAN.md`,
`.claudeignore` already present.
**Outputs:** `pyproject.toml` with canonical pinned deps (see CLAUDE.md); full
`src/toolforge/` package skeleton with empty `__init__.py` for every module in
PROJECT_PLAN §11; `tests/{unit,integration,e2e}/`; `.gitignore`; stub `README.md`;
stub `DESIGN.md` with section headers only; first commit.
**Invariants:**
- `pip install -e .` succeeds in a fresh venv on Python 3.11.
- `python -c "import toolforge"` succeeds.
- No file contains real implementation — stubs and docstrings only.
**Failure modes:**
- Dep resolution fails → drop/adjust pin, document in DESIGN.md.
- `mem0ai` transitive dep breaks → install `--no-deps`, pin manually, document.
**Principles:** P1 (module boundaries defined before logic), P2 (pinned = deterministic).
**Out of scope:** any functionality; CLI beyond `--help`.
**Done when:**
- [ ] `pip install -e .` works in a fresh venv.
- [ ] All directories from PROJECT_PLAN §11 exist with `__init__.py`.
- [ ] `.gitignore` covers everything in `.claudeignore` + standard Python clutter.
- [ ] `README.md` Quickstart section present (commands stub out, not yet working).
- [ ] `DESIGN.md` has empty section headers.
- [ ] Commit: `chore: initial project skeleton`.

---

## F0.2 — CLI scaffold ☑

### Task Formulation [confirmed]
**Inputs:** `src/toolforge/cli.py` stub from F0.1.
**Outputs:** Typer CLI with four commands: `build`, `generate`, `evaluate`, `compare`.
Each prints "not implemented" and exits 0. All eventual flags declared now so the
signature is locked early:
- `generate`: `--n INT`, `--seed INT`, `--out PATH`, `--no-cross-conversation-steering`
- `evaluate`: `--in PATH`, `--diversity BOOL`, `--out PATH`
- `compare`: `--a PATH`, `--b PATH`
**Invariants:**
- `toolforge --help` lists all four commands after `pip install -e .`.
- `toolforge generate --help` shows all flags.
**Failure modes:** entry point misconfigured → test by reinstalling.
**Principles:** P1 (CLI signature is the user-facing contract, frozen early).
**Out of scope:** any implementation.
**Done when:**
- [ ] `toolforge --help` lists all four commands.
- [ ] `toolforge generate --help` shows all flags.
- [ ] `toolforge build` prints "not implemented" and exits 0.
- [ ] Commit: `feat: CLI scaffold`.

---

## F0.3 — Config + structlog ☑

### Task Formulation [confirmed]
**Inputs:** `.env` (gitignored, user-provided); `src/toolforge/config.py`.
**Outputs:** Pydantic `Settings` loading from `.env`. Fields: `anthropic_api_key`,
`toolbench_data_dir` (default `../toolbench_raw/data/data/toolenv/tools`),
`cache_dir` (default `.cache/`), `artifacts_dir`, `runs_dir`, `reports_dir`,
`log_level`. `get_settings()` accessor (lru_cached). Structlog: JSON in non-TTY,
human-readable in TTY.
**Invariants:**
- Missing `anthropic_api_key` → clear `ValidationError`.
- Default paths resolve relative to CWD.
- `structlog.get_logger(__name__)` works from any module.
**Principles:** P1 (typed config contract), P2 (no LLM in config).
**Out of scope:** secrets management beyond `.env`.
**Done when:**
- [ ] `from toolforge.config import get_settings; get_settings()` works with valid `.env`.
- [ ] Missing key raises `ValidationError` with clear message.
- [ ] Unit test: `tests/unit/test_config.py` covers missing-key and default-paths.
- [ ] Commit: `feat: settings and structlog`.

---

# Phase 1 — Tool Registry (≈2 hours)
# CAPABILITY 1: Data Exploration — understand ToolBench's structure,
# inconsistencies, and build a clean normalized registry.

## F1.1 — Pydantic data models ☑

### Task Formulation [confirmed]
**Inputs:** PROJECT_PLAN §3.1 schema.
**Outputs:** `src/toolforge/registry/models.py` with `Parameter`, `ParamProvenance`,
`ResponseField`, `Endpoint`, `Tool`. All frozen (`model_config = ConfigDict(frozen=True)`).
**Invariants:**
- Every `Parameter` requires a `ParamProvenance` — non-optional.
- Models round-trip through JSON losslessly.
- `Endpoint.id` format: `"{category}/{tool_name}/{endpoint_name}"`.
**Failure modes:** Pydantic v2 frozen + mutable default → use `Field(default_factory=...)`.
**Principles:** P1 (this IS P1 — contracts before any loader code), P2 (deterministic).
**Out of scope:** loader, normalizer, semantic typing.
**Done when:**
- [ ] All five models in `models.py`.
- [ ] Round-trip test: build → dump → parse → assert equality.
- [ ] Provenance test: `Parameter` rejects construction without `ParamProvenance`.
- [ ] Commit: `feat(registry): pydantic data models with provenance`.

---

## F1.2 — Loader: walk and parse raw ToolBench JSON ☑

### Task Formulation [confirmed]
**Inputs:** path to `../toolbench_raw/data/data/toolenv/tools/`; `src/toolforge/registry/loader.py`.
**Outputs:** `walk_toolbench(root: Path) -> Iterator[tuple[str, str, dict]]` yielding
`(category, tool_name, raw_json_dict)`. Generator — never loads all 16k files at once.
Skips `__MACOSX/` junk directories and hidden dirs. Malformed JSON → log warning, skip.
**Invariants:**
- Pure generator, no normalization, no LLM.
- Malformed JSON never crashes the walk.
- Wrong-shape JSON (not a tool def) is still yielded — normalizer decides.
**Failure modes:**
- Path doesn't exist → `FileNotFoundError` with the configured path in message.
- `__MACOSX/` dirs → explicitly skipped.
**Principles:** P2 (pure I/O, no LLM), P4 (raw data passed through; normalization is separate stage).
**Out of scope:** normalization, subset selection.
**Done when:**
- [ ] Walker works on `tests/fixtures/toolbench_mini/` (3 files: 1 valid, 1 bad JSON, 1 wrong shape).
- [ ] Unit test: asserts 2 yielded (valid + wrong-shape), 1 skipped with warning.
- [ ] `__MACOSX/` dirs skipped in test fixture.
- [ ] Commit: `feat(registry): raw toolbench walker`.

---

## F1.3 — Normalizer: raw dict → Tool/Endpoint/Parameter ☑

### Task Formulation [confirmed]
**Inputs:** raw dicts from `walk_toolbench`; `src/toolforge/registry/normalizer.py`.
**Outputs:**
- `normalize_tool(raw, category) -> Tool | None`
- `normalize_corpus(raw_iter) -> tuple[list[Tool], NormalizationReport]`

`NormalizationReport` contains: total_seen, total_kept, drop_reasons (dict),
per_category_counts, distinct_raw_type_strings, distinct_required_field_names.
Saved to `artifacts/normalization_report.json` on every build.

**Normalization rules (all must be recorded in provenance):**
1. Type strings: uppercase → lowercase (`STRING`→`string`, `NUMBER`→`number`,
   `INTEGER`→`integer`). Unknown → `string` + flag `"unknown-type-fallback"`.
2. Required params: check keys `required_parameters`, `params`, `parameters` in that
   order. Record which key was found in `provenance.raw_required_field`.
3. Missing description → synthesize from `"{tool_name} {endpoint_name}"`.
   Set `provenance.synthesized_description = True`.
4. Null defaults → drop silently, record `"null-default-dropped"` in rules.
5. Enum as comma-string → split on comma, strip whitespace → list.
6. Tools with zero endpoints after normalization → drop, record reason.

**Invariants:**
- No LLM calls anywhere in this feature.
- Every `normalization_rules_applied` is non-empty when normalization happened.
- The `NormalizationReport` is the evidence for "5 real inconsistencies" in DESIGN.md.
**Principles:** P1 (chaos→contracts with audit trail), P2 (deterministic), P4 (report is a stage artifact).
**Out of scope:** semantic typing (F1.5), response schemas (F1.6), subset filter (F1.4).
**Done when:**
- [ ] `normalize_tool` produces valid `Tool` from test fixture.
- [ ] Unit tests for all 6 normalization rules above.
- [ ] `NormalizationReport` saved to `artifacts/normalization_report.json`.
- [ ] Commit: `feat(registry): normalizer with provenance and report`.

---

## F1.4 — Subset filter: ~500 endpoints, stratified ☑

### Task Formulation [confirmed]
**Inputs:** full normalized tool list from F1.3; `src/toolforge/registry/subset.py`.
**Outputs:** `select_subset(tools, target_endpoints=500, seed=42) -> list[Tool]`.

**Filter rules (applied in order):**
1. Drop tools with empty description AND all endpoints have empty descriptions.
2. Drop endpoints with zero params (required + optional).
3. Prefer tools with ≥1 required param on at least one endpoint.
4. Stratify: ≥3 tools per category where possible, across all 49 categories.
5. Within category: prefer tools that have a corresponding `response_examples/` file.

**Invariants:**
- Deterministic given same `(tools, target_endpoints, seed)`.
- Endpoint count within ±10% of target.
- All categories represented if usable tools exist.
- Subset report saved to `artifacts/subset_report.json`.
**Failure modes:**
- Category has zero usable tools → log, skip, record in report.
- Total usable tools < target → return all, log warning.
**Principles:** P2 (seeded RNG, deterministic), P5 (stratification = first diversity intervention).
**Out of scope:** semantic typing, response schemas.
**Done when:**
- [ ] `select_subset(all_tools)` returns ~500 endpoints across available categories.
- [ ] `artifacts/subset_report.json` saved with per-category breakdown.
- [ ] Unit test: fixed seed → deterministic output on synthetic mini-corpus.
- [ ] Commit: `feat(registry): stratified subset selection`.

---

## F1.5 — Semantic typing pass (LLM, cached) ☑

### Task Formulation [confirmed]
**Inputs:** subset tools from F1.4; seed vocab in `src/toolforge/registry/semantic_vocab.py`;
`src/toolforge/registry/semantic_typing.py`.

**Seed vocabulary (hand-curated, agree with user before running):**
`hotel_id`, `booking_id`, `user_id`, `order_id`, `product_id`, `listing_id`,
`city_name`, `country_code`, `lat_lng`, `datetime`, `date`, `currency_code`,
`email`, `phone_number`, `url`, `search_query`, `price`, `quantity`

**Outputs:** same tools with `Parameter.semantic_type` and `ResponseField.semantic_type`
populated. `artifacts/semantic_types.json` with vocab + counts.
`artifacts/chain_only_types.json` — types that appear ONLY as response fields
(never as user-provided params). This list feeds the executor's grounding check.

**LLM call design:** one call per endpoint, structured output (Pydantic), cached on
`(endpoint.id, model, prompt_version)`. New types accepted only if they appear ≥3
times across corpus (post-hoc consolidation). LLM can pick `null` for untyped params.

**Invariants:**
- Second build → 0 LLM calls (100% cache hit).
- Always confirm call count with user before running (expected ~500 cold).
- Prompt version is part of cache key — changing the prompt invalidates cache.
**Failure modes:**
- LLM proposes unknown type → accept if ≥3 occurrences, else drop.
- Cache corruption → invalidate entry, re-fetch.
**Principles:** P2 (only LLM touching registry; structured output; cached), P3 (semantic types enable runtime grounding), P4 (vocab artifact is inspectable without rerunning).
**Out of scope:** graph construction (F2.1).

**IMPORTANT — manual spot-check required:** after first run, manually review
first 50 annotations. This is the load-bearing quality check of the whole project.

**Done when:**
- [ ] Confirm seed vocab with user before first run.
- [ ] First run: ~500 calls (confirm count with user first).
- [ ] Second run: 0 LLM calls.
- [ ] `artifacts/semantic_types.json` shows vocab with counts.
- [ ] `artifacts/chain_only_types.json` derived and saved.
- [ ] Manual spot-check of 50 annotations completed, notes added to DESIGN.md.
- [ ] Commit: `feat(registry): semantic typing pass with cache`.

---

## F1.6 — Response schema inference (cached) ☑

### Task Formulation [confirmed]
**Inputs:** subset tools (with semantic types); `response_examples/` dir under raw data;
`src/toolforge/registry/schema_infer.py`.

**Two-tier strategy:**
1. **Static** (no LLM): response example exists → walk JSON tree, extract paths + types,
   mark `mock_policy="static"`.
2. **LLM fallback**: no example → one structured LLM call asking for plausible response
   schema. Mark `mock_policy="llm"`. Cached on `(endpoint_id, model, prompt_version)`.

**Invariants:**
- Static schemas are exact reproductions of example structure.
- Second build → 0 LLM calls.
- Every `ResponseField` with `*_id` / `*Id` naming gets cross-checked against semantic vocab.
- `mock_policy` set on every endpoint (used by executor in F3.2).
**Failure modes:**
- Example exists but malformed → fall to LLM tier, log.
- LLM returns unparseable schema → retry ≤2 times, then drop endpoint.
**Principles:** P2 (prefer deterministic; LLM is explicit fallback), P3 (response schemas with semantic types populate `available_values_by_type`).
**Out of scope:** the mock responder itself (F3.2).
**Done when:**
- [ ] Every endpoint in subset has non-empty `response_schema`.
- [ ] Static vs LLM split logged in `artifacts/build_report.md`.
- [ ] Cache hit rate 100% on second run.
- [ ] Commit: `feat(registry): response schema inference`.

---

## F1.7 — `toolforge build` CLI command ☑

### Task Formulation [confirmed]
**Inputs:** F1.1–F1.6; CLI scaffold from F0.2.
**Outputs:** working `toolforge build` that runs the full pipeline and dumps:
`artifacts/registry.json`, `artifacts/normalization_report.json`,
`artifacts/subset_report.json`, `artifacts/semantic_types.json`,
`artifacts/chain_only_types.json`, `artifacts/build_report.md`.
**Invariants:**
- Cold build < 5 minutes.
- Warm build (cache hit) < 30 seconds, 0 LLM calls.
- All artifacts valid JSON / Markdown.
- Artifacts dir created if missing.
**Principles:** P4 (artifacts are stage outputs inspectable without rerunning).
**Done when:**
- [ ] `toolforge build` produces all six artifacts on a fresh repo.
- [ ] Second `toolforge build` is fast and silent on LLM calls.
- [ ] `build_report.md` includes normalization-decisions table with real ToolBench examples.
- [ ] Commit: `feat(cli): toolforge build end-to-end`.

---

# Phase 2 — Tool Graph + Sampler (≈2 hours)
# CAPABILITY 2: Knowledge Graph — capture semantic relationships between
# tools/endpoints and sample realistic multi-step chains.

## F2.1 — Build the networkx graph from the registry ☑

### Task Formulation [confirmed]
**Inputs:** `artifacts/registry.json`, `artifacts/chain_only_types.json`;
`src/toolforge/graph/build.py`.

**Node types:**
- `Category` (49 max)
- `Tool` (~150)
- `Endpoint` (~500)
- `SemanticType` (~20–60 from vocab)

**Edge types:**
- `Endpoint —BELONGS_TO→ Tool —IN_CATEGORY→ Category`
- `Endpoint —CONSUMES→ SemanticType` (via required/optional params)
- `Endpoint —PRODUCES→ SemanticType` (via response fields)
- `Endpoint —CHAINS_TO→ Endpoint` (materialized: A PRODUCES T AND B CONSUMES T)
- `Endpoint —CO_CATEGORY→ Endpoint` (weight=1 if same category)

**Outputs:** `networkx.MultiDiGraph`, saved to `artifacts/graph.pkl`.
`load_graph() -> nx.MultiDiGraph` loader function.

**Invariants:**
- Every endpoint in registry is a node.
- `CHAINS_TO` edges are the backbone of realistic chain sampling.
- Graph is reproducible: same registry → byte-identical pickle.
- Endpoints producing nothing AND consuming nothing → node flagged `terminal=True`.
**Principles:** P2 (deterministic graph construction), P3 (CHAINS_TO edges are structural precondition for grounded chains).
**Out of scope:** sampler (F2.2).
**Done when:**
- [ ] `build_graph(registry)` produces correct `MultiDiGraph`.
- [ ] Unit test: small synthetic registry → assert correct edge types.
- [ ] `artifacts/graph.pkl` saved.
- [ ] Commit: `feat(graph): build networkx graph from registry`.

---

## F2.2 — Chain sampler (deterministic + constraint interface) ☑

### Task Formulation [confirmed]
**Inputs:** graph from F2.1; `src/toolforge/graph/sampler.py`.

**`ChainConstraints` dataclass:**
```python
@dataclass
class ChainConstraints:
    length: int | tuple[int, int]       # exact or (min, max)
    must_include_categories: list[str] = field(default_factory=list)
    must_include_endpoints: list[str] = field(default_factory=list)
    min_distinct_tools: int = 1
    pattern: Literal["linear", "parallel", "branch_merge"] = "linear"
    allow_repeats: bool = False
```

**`ChainSampler.sample(constraints, seed, steering=None) -> list[str]`**
returns endpoint IDs. Linear pattern is v1 priority; parallel and branch_merge
are stubs raising `NotImplementedError` (cut per PROJECT_PLAN §12 if behind).

**BFS algorithm (linear):**
1. Pick seed endpoint: if steering → weight by `1/(1+usage_count)`, else uniform.
2. BFS along `CHAINS_TO` edges. At each step: prefer endpoints consuming types
   produced by any prior endpoint. Break ties: (a) must-include, (b) steering, (c) co-category.
3. Dead-end before target length → backtrack once → relax to same-category fallback
   → return with `truncated=True` flag.

**Invariants:**
- Same `(constraints, seed, steering)` → same output, every time.
- `random.Random(seed)` instance — no global RNG.
- Never falls back to random when constraints unsatisfiable — returns truncated flag.
- **Unit test must prove the generator cannot run without the sampler** (mock it → crash).
  This satisfies the assignment's hard requirement.
**Principles:** P2 (deterministic; reproducibility for diversity experiment lives here), P5 (inverse-frequency weights plug in here via steering).
**Out of scope:** corpus steering integration (F6.1 plugs in later); parallel/branch_merge patterns (stubs only for v1).
**Done when:**
- [ ] `ChainSampler.sample(...)` produces deterministic linear chains on real graph.
- [ ] Unit test: same seed → 10 identical calls → 10 identical chains.
- [ ] Unit test: constraint satisfaction (length, must-include, min-distinct-tools).
- [ ] Unit test: mocking sampler causes generator (F4.6) to crash — proves dependency.
- [ ] Commit: `feat(graph): deterministic chain sampler`.

---

# Phase 3 — Offline Execution (≈1.5 hours)

## F3.1 — SessionState dataclass + executor scaffold ☑

### Task Formulation [confirmed]
**Inputs:** PROJECT_PLAN §6.3; `src/toolforge/execution/session.py`.
**Outputs:** Full `SessionState` dataclass:
```python
@dataclass
class SessionState:
    conversation_id: str
    seed: int
    available_values_by_type: dict[str, list[Any]]   # semantic_type → observed values
    resolved_entities: dict[tuple[str, Any], dict]   # (sem_type, value) → full entity
    created_entities: list[dict]                      # bookings, orders, etc.
    tool_outputs: list[ToolOutput]                    # full ordered log
    private_user_knowledge: dict[str, Any]            # fields planner omitted from query
```
`ToolOutput` dataclass: `endpoint_id`, `arguments`, `response`, `error`, `timestamp`.
`Executor` class skeleton with `execute(endpoint_id, arguments, state) -> ToolOutput`
body raising `NotImplementedError`.
**Invariants:**
- `SessionState` mutated ONLY by the executor.
- `ToolOutput` is JSON-serializable.
**Principles:** P3 (this IS the centerpiece of P3 — entire grounding story rests here).
**Done when:**
- [x] `SessionState` and `ToolOutput` exist with all fields.
- [x] Unit test: state updates correctly when manually appended.
- [x] Commit: `feat(execution): session state scaffold`.

---

## F3.2 — Mock responder: static + schema-derived + LLM tiers ☑

### Task Formulation [confirmed]
**Inputs:** `SessionState`, `Endpoint` with `response_schema` and `mock_policy`;
`src/toolforge/execution/mock_responder.py`, `src/toolforge/execution/faker_rules.py`.

**Three tiers:**
1. **Static** (`mock_policy="static"`): load response example, substitute entity names.
2. **Schema-derived** (`mock_policy="schema"`): walk `response_schema`, generate values
   with Faker seeded per session. Semantic-typed fields draw from / register into
   session-scoped pools so the same call with same args in same session always
   returns the same IDs.
3. **LLM** (`mock_policy="llm"`): cached on `(endpoint_id, hash(sorted_args))`.

**Invariants:**
- Tiers 1 and 2 are fully deterministic given conversation seed.
- Semantic-typed response fields register into `state.available_values_by_type`.
- LLM tier is the last resort — build report logs the tier distribution.
**Principles:** P2 (prefer deterministic; LLM is explicit fallback), P3 (registering produced values enables grounding).
**Done when:**
- [x] All three tiers implemented; build report shows the split.
- [x] Unit test: same input twice → same output.
- [x] Unit test: produced semantic-typed values land in `state.available_values_by_type`.
- [x] Commit: `feat(execution): three-tier mock responder`.

---

## F3.3 — Executor: grounding check + full contract ☑

### Task Formulation [confirmed]
**Inputs:** F3.1 scaffold, F3.2 responder, `chain_only_types.json`;
`src/toolforge/execution/executor.py`.

**`execute()` contract (pure Python, NO LLM in this hot path):**
1. Structural validation: required params present, types match schema.
2. Semantic grounding check: every argument whose semantic_type is in
   `CHAIN_ONLY_TYPES` must exist in `state.available_values_by_type[type]`.
   If not → return `ToolOutput(error="Invalid {type}: {value!r} not in session.
   Valid values: {state.available_values_by_type.get(type, [])}")`.
3. Generate mock response via responder.
4. Extract semantic-typed values from response → register into state.
5. Append to `state.tool_outputs`.

**`CHAIN_ONLY_TYPES`** = types that can ONLY come from prior tool outputs, never
from a user (`hotel_id`, `booking_id`, `order_id`, `listing_id`, `user_id`, etc.).
Types like `city_name`, `date`, `search_query` are never rejected.

**Invariants:**
- Hallucinated chain-only ID → structured error with valid values listed.
- First tool call in fresh session with valid user-provided params ALWAYS succeeds.
- Error message is structured enough for the Assistant to self-correct.
**Failure modes:**
- `CHAIN_ONLY_TYPES` too strict → first call in every chain fails.
  **Unit test for this exact case is mandatory.**
- `CHAIN_ONLY_TYPES` too loose → hallucination slips through (integration test catches).
**Principles:** P3 (layer 3 of the grounding invariant — without this, P3 is just a wish), P2 (deterministic Python, no LLM in hot path), P4 (structural vs grounding errors are distinguishable → enables targeted repair).
**Done when:**
- [x] Hallucinated `hotel_id` in step 2 → structured error listing valid IDs.
- [x] First call with valid user-provided params → success.
- [x] 3-step sequence updates state correctly end-to-end.
- [x] Unit tests: structural failure path, grounding failure path, success path.
- [x] Commit: `feat(execution): executor with grounding invariant`.

---

# Phase 4 — Multi-Agent Generator (≈3 hours)

## F4.1 — LLM client wrapper with cache ☑

### Task Formulation [confirmed]
**Inputs:** `anthropic` SDK; `src/toolforge/agents/base.py`, `src/toolforge/agents/llm_client.py`.
**Outputs:** `LLMClient` handling: structured output via Pydantic, content-addressed
disk cache (`(model, prompt_hash, schema_hash) → response`), tenacity retries,
temperature config, token usage logging per agent. `dry_run=True` mode raises on cache
miss (used by tests to assert no live LLM calls). `Agent` ABC: name, system prompt,
output schema.
**Invariants:**
- Every LLM call goes through `LLMClient`. No subclass calls `anthropic.Anthropic()` directly.
- Cache key includes schema hash — changing output schema invalidates entries.
**Principles:** P2 (structured output is the contract between LLM and consumer), P4 (per-agent token logging).
**Done when:**
- [x] Cached call works; second call with same input is free.
- [x] Unit test in `dry_run=True`: pre-seed cache → agent runs → assert no live call.
- [x] Commit: `feat(agents): LLM client with structured output and cache`.

---

## F4.2 — Planner agent (structured output) ☑

### Task Formulation [confirmed]
**Inputs:** sampled chain, diversity hints (stub for now), persona_seed;
`src/toolforge/agents/planner.py`, `prompts/planner.md`.
**Outputs:** `TaskPlan` Pydantic model:
```python
class TaskPlan(BaseModel):
    user_persona: str
    initial_query: str           # must omit ≥1 required param ~40% of time
    clarification_points: list[str]
    expected_final_outcome: str
    chain_rationale: str
    private_user_knowledge: dict[str, Any]  # omitted fields the user will reveal on request
```
Model: `claude-haiku-4-5-20251001`. Temperature: 0.7.
**Invariants:**
- ~40% of plans omit ≥1 required param from `initial_query`, stashing it in
  `private_user_knowledge`. This drives the disambiguation requirement.
- Prompt template in `prompts/planner.md`, versioned in git.
**Principles:** P2 (planning is semantic→LLM; output is structured→reliable), P5 (diversity hints inject corpus state).
**Done when:**
- [x] `Planner.plan(...)` returns valid `TaskPlan` for a real sampled chain.
- [x] Unit test (cached LLM): ~40% of seeds have non-empty `private_user_knowledge`.
- [x] Prompt in `prompts/planner.md`.
- [x] Commit: `feat(agents): planner with structured output`.

---

## F4.3 — User simulator agent ☑

### Task Formulation [confirmed]
**Inputs:** `TaskPlan` (with `private_user_knowledge`), conv history;
`src/toolforge/agents/user_simulator.py`, `prompts/user_simulator.md`.
**Outputs:** next user turn as free text. Model: `claude-haiku-4-5-20251001`. Temp: 0.7.
**Invariants:**
- Never confesses to being an LLM (enforced in system prompt).
- Reveals `private_user_knowledge` fields only when assistant asks for them directly.
- Returns short closing message when task is resolved.
**Failure modes:**
- Volunteers all private knowledge upfront → strengthen prompt. Log as prompt iteration in DESIGN.md.
**Principles:** P2 (free text is correct here — structuring defeats the purpose).
**Done when:**
- [x] `UserSimulator.respond(...)` produces plausible turns on a real `TaskPlan`.
- [x] Manual test: clarifying question → simulator reveals the right field.
- [x] Commit: `feat(agents): user simulator with private knowledge`.

---

## F4.4 — Assistant agent (structured tool-call output) ☑

### Task Formulation [confirmed]
**Inputs:** conv history, `SessionState`, filtered endpoint list (sampled chain + 3
distractors); `src/toolforge/agents/assistant.py`, `prompts/assistant.md`.
**Outputs:** discriminated union via structured output:
```python
class MessageTurn(BaseModel):
    type: Literal["message"]
    content: str

class ToolCallTurn(BaseModel):
    type: Literal["tool_call"]
    endpoint: str
    arguments: dict[str, Any]

AssistantTurn = MessageTurn | ToolCallTurn
```
Model: `claude-haiku-4-5-20251001`. Temperature: 0.

**Prompt must include:**
1. Filtered endpoint catalog (NOT all 500 — just chain + 3 distractors).
2. Session registry: `"Available values from prior tool calls: {registry_view}"`.
3. Grounding rule: `"When an argument was produced by a prior tool call, copy the
   exact value from the session registry. Do not invent IDs."`

**Invariants:**
- Always produces valid `AssistantTurn` (structured output enforces).
- Session registry freshly serialized before every turn (P3 soft grounding).
**Principles:** P2 (structured output non-negotiable), P3 (soft grounding in prompt + hard grounding in executor — both required).
**Done when:**
- [x] `Assistant.act(...)` produces valid structured output.
- [x] Unit test (cached, fixed seed): assistant copies an ID from registry rather than inventing.
- [x] Commit: `feat(agents): assistant with grounded tool calls`.

---

## F4.5 — Judge agent (structured 4-dimensional scoring) ☑

### Task Formulation [confirmed]
**Inputs:** finished conversation, available endpoint catalog;
`src/toolforge/agents/judge.py`, `prompts/judge.md`.
**Outputs:** `JudgeResult`:
```python
class DimensionScore(BaseModel):
    score: int          # 1–5
    rationale: str

class JudgeResult(BaseModel):
    naturalness: DimensionScore
    tool_correctness: DimensionScore
    chain_coherence: DimensionScore
    task_completion: DimensionScore
    failure_modes: list[str]
    overall_pass: bool  # mean ≥ 3.5 AND no dimension < 2.5
```
Model: `claude-sonnet-4-6`. Temperature: 0. **Different family from generators.**

**Prompt must include 3–4 anchor examples** to stabilize scoring. Include at least
one negative anchor (a bad conversation) so the judge doesn't default to high scores.

**Pass threshold:** `mean(scores) >= 3.5 AND min(score) >= 2.5`.

**Dimension rationale:**
- `naturalness`: reads like a real user talking to a real assistant.
- `tool_correctness`: right endpoint, valid arguments, right intent match.
- `chain_coherence`: later calls use values from earlier outputs (not hallucinated).
- `task_completion`: original request resolved and confirmed in final message.
**Principles:** P2 (temperature 0, structured output, different model family = modular evaluation), P4 (judge sees only structurally-valid convs — validation pipeline gates this).
**Done when:**
- [x] `Judge.score(...)` returns valid `JudgeResult` with all four dimensions.
- [x] Unit test: same conversation twice → same scores (temperature 0 reproducibility).
- [x] Negative anchor example in `prompts/judge.md`.
- [x] Commit: `feat(agents): judge with 4-dimensional scoring`.

---

## F4.6 — LangGraph wiring: full single-conversation generator ☐

### Task Formulation [confirmed]
**Inputs:** F2.2, F3.x, F4.1–F4.5; `src/toolforge/generator/graph.py`,
`src/toolforge/generator/state.py`, `src/toolforge/generator/loop.py`.

**`ConversationState` TypedDict:**
```python
class ConversationState(TypedDict):
    conversation_id: str
    seed: int
    plan: TaskPlan
    sampled_chain: list[str]
    messages: list[Message]
    session_state: SessionState
    judge_result: JudgeResult | None
    repair_attempts: int
    status: Literal["running", "done", "failed", "needs_repair"]
```

**LangGraph nodes:**
```
START → plan → user_turn → assistant_turn
                               ↓ (tool_call?)
                           executor → back to user_turn
                               ↓ (done)
                           finalize → judge
                               ↓ (pass)
                              END
                               ↓ (fail)
                            repair → judge (max 2 times)
                               ↓
                              END
```

**`generate_one(seed, constraints) -> Conversation`** entrypoint.

**Invariants:**
- Graph sampler called INSIDE this loop — no hardcoded chains.
- Hard cap: 12 turns maximum to prevent infinite loops.
- Only executor mutates `SessionState`.
**Principles:** P3 (state flows through graph; executor is sole mutator), P4 (each node independently testable).
**Done when:**
- [x] `generate_one(seed=42, ...)` produces one valid conversation end-to-end.
- [x] Unit test: mocked sampler → generation crashes (proves hard dependency).
- [ ] Smoke test: `toolforge generate --n 3 --seed 42` produces 3 valid records. ← F7 (CLI wiring)
- [x] Commit: `feat(generator): single-conversation langgraph loop`.

---

# Phase 5 — Validation + Repair (≈1.5 hours)

## F5.1 — Stage-level structural validation pipeline ☑

### Task Formulation [confirmed]
**Inputs:** finished conversation + `SessionState`;
`src/toolforge/evaluation/validation.py`.
**Outputs:** `validate_conversation(conv, state) -> list[ValidationResult]`.
Five validators, all pure functions (no I/O, no LLM):

| Validator | Hard/Soft | Failure routes to |
|---|---|---|
| `validate_structure` | Hard | Discard (generator bug) |
| `validate_tool_calls` | Hard | Tier-1 repair |
| `validate_grounding` | Hard | Tier-1 repair (should rarely fire — executor prevents this) |
| `validate_completeness` | Hard | Append final assistant turn |
| `validate_constraints` | Soft | Log warning, pass through |

`ValidationResult`: `stage`, `passed`, `errors: list[str]`, `warnings: list[str]`.

**Invariants:**
- All validators pure — no side effects.
- Conversation only reaches the judge after passing all hard validators.
**Principles:** P4 (this IS P4 — modular evaluation, failures localize to a stage).
**Done when:**
- [x] All five validators implemented.
- [x] Unit test: deliberately broken conv (hallucinated ID, missing summary, malformed
  structure) → each fires the right validator.
- [x] Commit: `feat(evaluation): stage-level validation pipeline`.

---

## F5.2 — Repair loop (Tier 1 soft repair) ☑

### Task Formulation [confirmed]
**Inputs:** failing conversation, `JudgeResult` and/or `ValidationResult`s;
`src/toolforge/agents/repair.py`, `prompts/repair.md`.
**Outputs:** `RepairAgent` producing structured edit operations:
```python
class RegenerateTurn(BaseModel):
    type: Literal["regenerate_turn"]
    turn_index: int
    reason: str

class AppendTurn(BaseModel):
    type: Literal["append_turn"]
    role: str
    reason: str

EditOperation = RegenerateTurn | AppendTurn
```
Model: `claude-sonnet-4-6`. Temperature: 0. Repair runner applies edits, re-validates, re-judges.
Max 2 repair attempts. Tracks failure modes across attempts — if same failure appears
twice, give up.
**Invariants:**
- Only affected turns regenerated, not whole conversation.
- Repaired conversations re-enter validation pipeline before re-judging.
**Principles:** P4 (targeted repair on stage-level failures = payoff for the validation pipeline).
**REQUIRED by assignment:** integration test proves this works.
**Done when:**
- [x] Integration test: hallucinated-ID conversation → repair fixes it → judge re-passes.
- [x] Unit test: max-attempts guard works.
- [x] Commit: `feat(agents): tier-1 repair loop`.

---

# Phase 6 — Corpus Memory + Steering (≈1.5 hours)

## F6.1 — CorpusDiversityTracker (deterministic) ☑

### Task Formulation [confirmed]
**Inputs:** `src/toolforge/memory/corpus_stats.py`.
**Outputs:** `CorpusDiversityTracker` with:
- `tool_usage: Counter[str]`, `category_usage: Counter[str]`,
  `tool_pair_usage: Counter[tuple[str,str]]`, `chain_pattern_usage: Counter[str]`,
  `length_bucket_usage: Counter[str]`
- `should_accept(chain) -> tuple[bool, str]` — hard rejection with reason
- `sampling_weight(endpoint_id) -> float` — `1/(1+usage_count)`
- `update(chain, pattern, length_bucket)` — call after each successful conversation

**Caps (tuned for N=120):**
- `MAX_ENDPOINT_COUNT = 8`
- `MAX_CATEGORY_FRACTION = 0.15`
- `MAX_TOOL_PAIR_COUNT = 4`

**`--no-cross-conversation-steering` mode:** stub tracker whose `should_accept`
always returns `(True, "ok")` and `sampling_weight` always returns `1.0`.

**Invariants:**
- Fully deterministic — no mem0, no embeddings.
- Sampler (F2.2) consults this tracker on every call.
**Principles:** P5 (this IS P5 — diversity by design, not post-hoc), P2 (deterministic; diversity experiment reproducibility depends on this).
**Done when:**
- [x] Hard caps fire correctly in unit tests.
- [x] Sampling weights are inverse-frequency.
- [x] Stub mode short-circuits everything for `--no-cross-conversation-steering`.
- [x] Sampler wired to consult tracker.
- [x] Commit: `feat(memory): corpus diversity tracker`.

---

## F6.2 — mem0 task-embedding memory (optional — cut if behind hour 14) ☐

### Task Formulation [confirmed]
**Inputs:** `mem0ai`; planner's `initial_query` + `chain_rationale` per conversation;
`src/toolforge/memory/task_memory.py`.
**Outputs:** `TaskMemory` wrapper with `add(query, rationale)` and
`nearest(query, k=5) -> list[str]`. Before each new plan, query for K nearest
neighbors → inject into planner prompt as "avoid these archetypes."
**Invariants:**
- mem0 wrapped — never called directly from agent code.
- `--no-cross-conversation-steering` also disables this (never queried, never written).
- Determinism caveat documented in DESIGN.md §7.2.
**Failure modes:** mem0 backend misbehaves → fall back to no-op, log warning.
  The frequency tracker (F6.1) is the primary signal anyway.
**CUT CRITERIA:** if at hour 14 this isn't done, drop it. F6.1 alone is sufficient
for the diversity experiment to produce meaningful numbers.
**Done when:**
- [ ] `TaskMemory.add(...)` and `nearest(...)` work.
- [ ] Planner receives archetypes when steering is on.
- [ ] Stub mode for `--no-cross-conversation-steering`.
- [ ] Commit: `feat(memory): mem0 task embedding memory`.

---

# Phase 7 — Batch Generation + Evaluation (≈2 hours)

## F7.1 — `toolforge generate` batch loop ☑

### Task Formulation [confirmed]
**Inputs:** F4.6 single-conv generator, F6.1 tracker, F6.2 memory (optional);
`src/toolforge/generator/loop.py`, `src/toolforge/cli.py`.
**Outputs:** `toolforge generate --n N --seed S [--no-cross-conversation-steering] --out PATH`
producing JSONL. Each record includes:
```json
{
  "conversation_id": "conv_0042",
  "messages": [...],
  "tool_calls": [...],
  "tool_outputs": [...],
  "judge_scores": {"naturalness": 4, "tool_correctness": 5, "chain_coherence": 4, "task_completion": 5},
  "validation_results": [...],
  "metadata": {
    "seed": 42,
    "sampled_chain": [...],
    "pattern": "linear",
    "length_bucket": "medium",
    "repair_attempts": 0,
    "was_steered": true,
    "tools_used": [...],
    "num_turns": 7
  }
}
```
**Invariants:**
- `--n > 10` requires explicit user confirmation (cost guardrail).
- Same `(N, seed)` → byte-identical deterministic output (modulo mem0 ANN noise).
- ≥50% of output has ≥3 tool calls AND ≥2 distinct tools.
- ≥20% has a disambiguation turn before any tool call.
- Length mix: 30/40/30 short/medium/long.
**Principles:** all five, integrated.
**Done when:**
- [x] `toolforge generate --n 5 --seed 42` → 5 valid records.
- [x] `toolforge generate --n 5 --seed 42 --no-cross-conversation-steering` → 5 records with `was_steered=false`.
- [x] Hard-requirement checks verified on 20-conversation sample.
- [x] Commit: `feat(cli): toolforge generate batch loop`.

---

## F7.2 — `toolforge evaluate` + diversity metrics ☑

### Task Formulation [confirmed]
**Inputs:** JSONL run file; `src/toolforge/evaluation/metrics.py`,
`src/toolforge/evaluation/report.py`, `src/toolforge/cli.py`.
**Outputs:** `toolforge evaluate --in PATH --out PATH [--diversity]` computing:

**Quality metrics:**
- Mean + per-dimension judge scores
- Pass rate at threshold
- % multi-step (≥3 tool calls), % multi-tool (≥2 distinct tools), % disambiguation
- Length distribution

**Diversity metrics (when `--diversity`):**
1. **Tool coverage entropy** — Shannon entropy of `endpoint_id` usage distribution.
   Range: 0 (one tool always) to `log(n_endpoints)` (perfectly uniform).
2. **Distinct tool bigrams (distinct-2)** — `(unique adjacent pairs) / (total pairs)`.
   Measures chain-pattern diversity beyond individual tool diversity.
3. **Task embedding dispersion** — mean pairwise cosine distance of `initial_query`
   embeddings via `all-MiniLM-L6-v2` (local, deterministic, free).

**Why these three:** entropy captures tool frequency flattening, distinct-2 captures
chain pattern variety, embedding dispersion captures semantic task variety. Together
they measure the three things the steering mechanism is designed to affect.

**Invariants:**
- All metrics deterministic from the run file.
- Output: JSON report + human-readable Markdown side-by-side.
**Principles:** P4 (evaluation is a separate stage — re-runnable on stored runs), P5 (metrics measure what F6.1 was designed to affect).
**Done when:**
- [ ] `toolforge evaluate --in runs/mini.jsonl --diversity` produces complete report.
- [ ] Commit: `feat(evaluation): metrics and report`.

---

## F7.3 — `toolforge compare` (Run A vs Run B) ☑

### Task Formulation [confirmed]
**Inputs:** two evaluation reports from F7.2.
**Outputs:** `toolforge compare --a reports/run_a.json --b reports/run_b.json`
prints a side-by-side table of all metrics with deltas + written tradeoff summary.
**Invariants:**
- Table includes the prior from PROJECT_PLAN §10.3 alongside the actual delta.
**Done when:**
- [ ] `toolforge compare --a ... --b ...` prints table and summary.
- [ ] Commit: `feat(cli): compare two runs`.

---

## F7.4 — Run the diversity experiment ☐

### Task Formulation [confirmed]
**Inputs:** everything above.
**Outputs:** `runs/run_a.jsonl`, `runs/run_b.jsonl`, `reports/run_a.json`,
`reports/run_b.json`, `reports/comparison.md`.

```bash
# Run A — steering off
toolforge generate --n 120 --seed 42 --no-cross-conversation-steering --out runs/run_a.jsonl

# Run B — steering on
toolforge generate --n 120 --seed 42 --out runs/run_b.jsonl

toolforge evaluate --in runs/run_a.jsonl --diversity --out reports/run_a.json
toolforge evaluate --in runs/run_b.jsonl --diversity --out reports/run_b.json
toolforge compare --a reports/run_a.json --b reports/run_b.json
```

**CONFIRM WITH USER BEFORE RUNNING** — ~4,000 LLM calls, ~$12 total.

**Done when:**
- [ ] Both runs complete.
- [ ] Comparison report saved to `reports/comparison.md`.
- [ ] Numbers pasted into DESIGN.md with honest analysis (including where prior was wrong).
- [ ] Commit: `feat: diversity experiment results`.

---

# Phase 8 — Tests + Documentation (≈2 hours)

## F8.1 — End-to-end test ☐

### Task Formulation [confirmed]
**Inputs:** full pipeline; `tests/e2e/test_full_pipeline.py`.
**Outputs:** `@pytest.mark.e2e` test that:
1. Runs `toolforge build` (cache warm — 0 LLM calls).
2. Runs `toolforge generate --n 100 --seed 42`.
3. Asserts: ≥100 valid convs, mean judge score ≥ 3.5, ≥50% multi-step+multi-tool,
   ≥20% disambiguation.
**Invariants:**
- Gated behind `pytest -m e2e` — never runs on every commit.
- Uses cached LLM responses; subsequent runs nearly free.
**Done when:**
- [ ] Test passes.
- [ ] Commit: `test: e2e pipeline`.

---

## F8.2 — DESIGN.md final polish ☐

### Task Formulation [confirmed]
**Every section must carry evidence, not paraphrase of PROJECT_PLAN.**

Required sections:
1. **Architecture** — diagram (PROJECT_PLAN §2 ASCII is fine), agent roles, data flow.
2. **Agent protocol** — structured-output schemas, `ConversationState` flow.
3. **Context management** — within-conv grounding (3 layers), cross-conv steering (2 signals), actual `CHAIN_ONLY_TYPES` list pasted in.
4. **Prompt iteration log** — ≥1 documented failure: broken prompt → symptom → fix → lesson. Required by assignment.
5. **Diversity & quality analysis** — numbers from F7.4, side-by-side table, analysis of tradeoff including where the prior (PROJECT_PLAN §10.3) was wrong.
6. **Where this breaks** — concrete failure cases observed, not hand-waves.
7. **At scale** — PROJECT_PLAN §16 fleshed out.
8. **Normalization decisions table** — real ToolBench examples from F1.3 report.

**Done when:**
- [ ] All eight sections present and evidence-rich.
- [ ] No section paraphrases PROJECT_PLAN.
- [ ] Commit: `docs: design.md final`.

---

## F8.3 — README.md final ☐

### Task Formulation [confirmed]
**Outputs:** README with Quickstart walking the grader through:
install → build → generate run A → generate run B → evaluate → compare.
Short "what is this" intro + link to DESIGN.md.
**Done when:**
- [ ] Grader on a clean machine can reproduce the diversity experiment from README alone.
- [ ] Commit: `docs: readme one-command reproduction`.

---

# Cut Order (if behind at hour 12)

Drop one at a time, re-evaluate after each cut:

1. **F2.2 parallel + branch_merge patterns** → linear chains only (saves ~0.5h).
2. **F5.2 Tier-2 hard repair** → soft repair only (saves ~0.5h).
3. **F6.2 mem0 task-embedding memory** → frequency-based steering only; diversity
   experiment still runs with F6.1 alone (saves ~1h).
4. **F1.4 stratification target** → reduce to 200 endpoints / 75 tools (saves ~0.5h build/debug).

**Never cut:** F1.1 (provenance models), F3.3 (executor grounding), F5.1 (validation
pipeline), F5.2 integration test (the repair test is a hard requirement), F7.4 (the
diversity experiment), F8.1 (E2E test), F8.2 (DESIGN.md depth).
