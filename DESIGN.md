# DESIGN.md — toolforge Lab Notebook

> This is a lab notebook, not a spec rewrite. Evidence over assertions.
> Every section is updated in real time as the system is built.

---

## §1 Architecture & Decisions

### §1.1 System Overview

### §1.2 Component Communication Protocol

### §1.3 Model Choices & Self-Preference Bias Mitigation

### §1.4 Python Version Decision

Original plan pinned `requires-python = ">=3.11,<3.12"` to avoid `mem0ai`/`chromadb`/`qdrant-client`
wheel gaps on 3.12+. Dropped the `<3.12` upper bound during F0.1 because the development machine
runs Python 3.12 and the upper-bound restriction blocked local development entirely. The grader's
machine note is addressed in README.md; if `mem0ai` causes source-build failures on any interpreter,
the fix is to pin `mem0ai` more tightly, not restrict the interpreter version.

---

## §2 Tool Registry

### §2.1 ToolBench Inconsistencies Observed (with real examples)

Three raw ToolBench files were inspected before writing any normalization code:

| File | What it revealed |
|------|-----------------|
| `Entertainment/trendy_cricket_player.json` | Minimal tool — no params on any endpoint, confirmed top-level `api_list` key as the canonical endpoint container. Established baseline: even trivially simple tools parse correctly. |
| `Travel/flighthive_explore_the_skies.json` | Confirmed key naming conventions (`required_parameters`, `optional_parameters`), but endpoints still had no parameters. Showed that many ToolBench tools expose list/search endpoints with zero required inputs. |
| `Transportation/deutsche_bahn.json` | First tool with rich parameters. Revealed concrete inconsistencies: type strings are uppercase (`STRING`, `NUMBER`), `ENUM` used as a type with no enum values in the field, `TIME` as a non-standard type string, defaults that are empty strings or `""`, and descriptions present on all params. This file drove the normalization rules below. |

### §2.2 Normalization Decisions Table

Nine rules, applied in order by the normalizer. Each rule tag is recorded in
`ParamProvenance.normalization_rules_applied` when it fires.

| # | Rule tag | Trigger | Action | Example from ToolBench |
|---|----------|---------|--------|----------------------|
| 1 | `type-lowercased` | `raw_type_string` is not lowercase | Lowercase the type string | `STRING` → `string`, `NUMBER` → `number` |
| 2 | `unknown-type-fallback` | Lowered type not in `{string, number, integer, boolean, array, object}` | Map to `string`, record original | `TIME` → `string`, `ENUM` → `string` |
| 3 | `enum-no-values` | Type is `ENUM` but no enum values provided | Type becomes `string`, no enum field | deutsche_bahn `ENUM` params with no values list |
| 4 | `enum-split-from-default` | `default` field contains comma-separated candidate values | Parse into `enum` tuple, clear default | `default: "RE,S,ICE"` → `enum: ("RE","S","ICE")` |
| 5 | `empty-default-dropped` | `default` is `""` (empty string) | Set default to `None` | deutsche_bahn empty-string defaults |
| 6 | `null-default-dropped` | `default` is JSON `null` or Python `None` | Set default to `None` (no-op, but recorded) | Ubiquitous across ToolBench |
| 7 | `synthesized-description` | Parameter or endpoint has missing/empty description | Synthesize from `"{tool_name} {endpoint_name}"` | Tools with blank `description` fields |
| 8 | `method-fallback-unknown` | HTTP method not in `{GET,POST,PUT,DELETE,PATCH}` | Map to `UNKNOWN` | Endpoints with missing or non-standard methods |
| 9 | `schema-empty-ignored` | Endpoint's `schema` field is `""` or `{}` | Set `response_schema = ()`, leave `mock_policy = None` (F1.6 decides later) | deutsche_bahn "Search trips" has `schema: {}` → `response_schema = ()`; "Autocomplete" has a real schema object → parsed into `ResponseField` tuples |

### §2.3 Subset Selection Strategy

### §2.4 Semantic Typing: Vocab Design & LLM Call Design

---

## §3 Tool Graph & Chain Sampler

### §3.1 Graph Schema Decisions

### §3.2 Edge Type Justification

### §3.3 Sampler Algorithm & Tradeoffs

---

## §4 Offline Execution Model

### §4.1 SessionState Design

### §4.2 Grounding Enforcement (Layer 3)

### §4.3 Mock Responder Strategy

---

## §5 Multi-Agent System

### §5.1 Agent Roles & Communication

### §5.2 LangGraph Orchestration Design

### §5.3 Disambiguation Mechanism

---

## §6 Evaluation Pipeline

### §6.1 Validator Design (Deterministic Stage)

### §6.2 Judge Dimensions & Justification

### §6.3 Repair Strategy

### §6.4 Prompt Iteration Log

> At least one documented failure required. Record failures in real time.

---

## §7 Diversity & Quality Analysis

### §7.1 Diversity Metrics Chosen & Justification

### §7.2 Run A vs Run B Results (numeric, 3 decimal places)

### §7.3 Diversity–Quality Tradeoff Analysis

### §7.4 Non-Determinism Caveat (mem0 ANN)

---

## §8 Limitations & Honest Failure Cases

### §8.1 Known Failure Modes Observed During Runs

### §8.2 What I Would Do Next
