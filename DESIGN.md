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

**Distinct raw type strings found across the 3-category sample (730 tools, 12,599 parameters):**

```
ARRAY  BINARY  BOOLEAN  CREDENTIALS  DATE (YYYY-MM-DD)  DATEPICKER  ENUM
FILE  GEOPOINT (latitude, longitude)  JSON  LIST  MAP  NUMBER  OBJECT
SELECT  STRING  TIME (24-hour HH:MM)  string
```

18 distinct strings for what are logically 6–7 semantic types. The exotic ones
(`GEOPOINT (latitude, longitude)`, `DATEPICKER`, `CREDENTIALS`, `DATE (YYYY-MM-DD)`,
`TIME (24-hour HH:MM)`, `SELECT`, `MAP`) all fall through to `unknown-type-fallback → string`.
This is not defensive coding — it fired 402 times on real data. Removing it would cause
roughly 3% of parameters to fail Pydantic validation. Note also the single lowercase
`string` entry alongside uppercase `STRING`: both exist in the corpus, proving that
`type-lowercased` is similarly non-optional (fired 11,998 times).

### §2.2 Normalization Decisions Table

Ten rules, applied in order by the normalizer. Each rule tag is recorded in
`ParamProvenance.normalization_rules_applied` when it fires.

| # | Rule tag | Trigger | Action | Example from ToolBench |
|---|----------|---------|--------|----------------------|
| 1 | `type-lowercased` | `raw_type_string` is not lowercase | Lowercase the type string | `STRING` → `string`, `NUMBER` → `number` |
| 2 | `unknown-type-fallback` | Lowered type not in `{string, number, integer, boolean, array, object}` | Map to `string`, record original | `TIME` → `string`, `ENUM` → `string` |
| 3 | `enum-no-values` | Type is `ENUM` but no enum values provided | Type becomes `string`, no enum field | deutsche_bahn `ENUM` params with no values list |
| 4 | `enum-split-from-default` | `default` field contains comma-separated candidate values | Parse into `enum` tuple, clear default | `default: "RE,S,ICE"` → `enum: ("RE","S","ICE")` |
| 4b | `complex-default-stringified` | `default` is a dict or list | JSON-serialize to string | Found at runtime: `default: {"areatype": "drivetime", ...}` in geo-search tool |
| 5 | `empty-default-dropped` | `default` is `""` (empty string) | Set default to `None` | deutsche_bahn empty-string defaults |
| 6 | `null-default-dropped` | `default` is JSON `null` or Python `None` | Set default to `None` (no-op, but recorded) | Ubiquitous across ToolBench |
| 7 | `synthesized-description` | Parameter or endpoint has missing/empty description | Synthesize from `"{tool_name} {endpoint_name}"` | Tools with blank `description` fields |
| 8 | `method-fallback-unknown` | HTTP method not in `{GET,POST,PUT,DELETE,PATCH}` | Map to `UNKNOWN` | Endpoints with missing or non-standard methods |
| 9 | `schema-empty-ignored` | Endpoint's `schema` field is `""` or `{}` | Set `response_schema = ()`, leave `mock_policy = None` (F1.6 decides later) | deutsche_bahn "Search trips" has `schema: {}` → `response_schema = ()`; "Autocomplete" has a real schema object → parsed into `ResponseField` tuples |

**Rule fire counts — 3-category sample (Financial + Sports + Travel, 730 tools, 12,599 params):**

| Rule tag | Count | Notes |
|----------|-------|-------|
| `type-lowercased` | 11,998 | Effectively every parameter; ToolBench is all-caps by convention |
| `synthesized-description` | 3,990 | ~32% of params lack any description text |
| `empty-default-dropped` | 3,755 | Empty-string defaults are the norm, not the exception |
| `schema-empty-ignored` | 3,469 | Majority of endpoints have no machine-readable response schema |
| `unknown-type-fallback` | 402 | 3.2% of params — GEOPOINT, DATEPICKER, CREDENTIALS, etc. |
| `enum-split-from-default` | 175 | Useful signal: enum options hidden in the default field |
| `complex-default-stringified` | 5 | Rule 4b — discovered at runtime, not anticipated in the pre-code spec |

Rule 4b (`complex-default-stringified`) was not in the original 9-rule design. It was added
after the first full corpus run failed with a Pydantic `ValidationError` because a geo-search
tool stored a dict as its default value (`{"areatype": "drivetime", "units": "minutes", ...}`).
The fix was to JSON-serialize complex defaults to strings — consistent with Rule 2's
philosophy that exotic values are preserved as strings rather than discarded.

### §2.3 Subset Selection Strategy

**Result: 60 tools, 500 endpoints across 8 categories (seed=42).**

Pre-filter drops:
- 910 tools dropped entirely — all their endpoints had zero required AND zero optional parameters (nothing for the executor to bind arguments to)
- 2,058 individual zero-param endpoints pruned from tools that otherwise survived

Key design decision: **10-endpoint cap per tool.** Without the cap, a single large Sports tool consumed the entire 62-endpoint category budget, leaving only 1 tool for the category. Capping at 10 forces the budget to spread across more tools, achieving ≥7 tools per category.

| Category | Tools selected | Endpoints | Budget |
|---|---|---|---|
| Advertising | 9 | 63 | 63 |
| Business | 7 | 63 | 63 |
| Communication | 7 | 63 | 63 |
| Data | 7 | 63 | 63 |
| Devices | 9 | 62 | 62 |
| Financial | 7 | 62 | 62 |
| Sports | 7 | 62 | 62 |
| Travel | 7 | 62 | 62 |
| **TOTAL** | **60** | **500** | **500** |

Budget split: `500 // 8 = 62`, first 4 categories get +1 remainder. Minimum 3 tools per category enforced (achieved 7+ in practice). Deterministic given `(tools, target_endpoints=500, seed=42)`.

### §2.4 Semantic Typing: Vocab Design (Empirical Basis)

Before writing any F1.5 code, we ran a vocabulary analysis over the filtered subset produced
by the normalizer and `select_subset` (60 tools, 494 endpoints, seed=42). The analysis
collected all 1,887 parameters across those endpoints — 949 required, 938 optional — and
recorded the raw name, normalized family, description, and required/optional status for each.

**What the numbers show.** The 1,887 parameters resolve to 558 distinct raw names, but the
raw name space is extremely noisy. The same logical concept appears as `accessToken`,
`access_token`, `accountId`, `account_id`, `userId`, `user_id`, and so on. Normalizing
camelCase and kebab-case into `_`-separated lowercase families reduces 558 raw names to a
much smaller set of meaningful clusters. The normalized `access_token` family, for instance,
collapses `accessToken` and `access_token` into a single family of 24 occurrences. This
confirms that F1.5's semantic vocabulary must be defined over normalized families, not raw
strings — matching on raw strings would fragment each concept across multiple entries and
make cross-tool chaining impossible to express cleanly.

Of the 1,887 parameters, 494 (~26%) are reference-like by the analysis heuristics: their
normalized name contains a reference token (`id`, `uuid`, `key`, `slug`, `token`, `code`,
`handle`) or their description contains a phrase such as "identifier", "ID of", or "unique".
That density — roughly one parameter in four — is high enough to support multi-step chaining
without artificially engineering the tool graph. Each reference-like parameter is a potential
consumer endpoint waiting to be paired with a producer that returns the same entity.

**Observed naming patterns and what they imply.** Raw name `id` is the single most common
parameter (61 occurrences), but it is useless as a vocabulary entry on its own: its
descriptions range from "Tenant ID to delete" to "User id" to "This is the app ID". A
semantic type called `id` would be trivially satisfied by any prior tool call that returned
any identifier, which degrades grounding to near-random. The vocabulary therefore avoids
generic names and instead uses entity-specific families. The entity stem analysis extracted
stems by stripping trailing `_id`, `_uuid`, `_key`, and similar suffixes; the stems that
appear with meaningful frequency and domain coverage include **account** (17 occurrences),
**user** (11), **hotel** (10), **customer** (9), **client** (9), **venue** (8), and
**product** (8). Additional stems observed in the chainability candidates and raw examples —
**operation**, **market**, **conversation**, **channel**, **device**, **project**, **tenant**,
**property**, **media**, **booking**, **flight**, and **checkout** — are sparser in this
subset but represent realistic multi-step scenarios and are included in the seed vocabulary
on that basis. Together these form the `CHAIN_ONLY` type tier: values that are often obtained
from prior tool calls and are strong candidates for chained value flow.

The analysis also surfaces a second tier of concepts that users can supply directly. The top
raw parameter names include **page** (37 occurrences, 16 required), **search** (32), and
**currency** (31), and the reference-like family analysis confirms that **league** (14
reference-like occurrences) and **team** (13) are often described as IDs ("The id of the
league", "The id of the team") yet in realistic user conversations a person simply names the
league or team they care about rather than looking up an opaque identifier first. Similarly,
**season** (26 occurrences, 10 required), **country_code** (9 reference-like), **date** (12),
and **email** (present in the raw examples, described as "Email ID of the user") are values
a user brings to the conversation, not values they retrieve via a prior API call. These form
the `USER_PROVIDED` tier: values the conversation can plausibly source directly from user
intent rather than from a prior tool output.

**Two explicit classification decisions.** First, `access_token` is classified as
`CHAIN_ONLY` — the executor will not accept a manually typed value. The analysis finds 24 occurrences in the normalized family, all 19 of the raw
`accessToken` occurrences are required, and the descriptions are explicit: "OAuth2 Access
Token from `getAccessToken` method." OAuth tokens are machine-generated strings that users
cannot reasonably dictate, so any conversation needing one must include an earlier auth or
login endpoint. Treating `access_token` as `USER_PROVIDED` would allow the LLM to hallucinate
token values, breaking grounding at the most security-sensitive parameter type in the corpus.

Second, `league`, `season`, and `team` are kept in `USER_PROVIDED` despite some APIs labelling
them as identifiers. The data shows that `team` is optional in 14 of its 16 occurrences and
`league` is optional in 12 of its 21 occurrences. More importantly, in natural speech a user
says "get me stats for Manchester City in the Premier League" — they are not querying a lookup
table first. Forcing a mandatory chain step to resolve a team or league name into an ID would
make the generated conversations feel artificial and would incorrectly inflate chain length
metrics. The classification honours what users actually do, not what API designers called the
parameter.

**Post-corpus corrections (applied after full 500-endpoint run).** Four accepted new types were force-nulled via `NULL_OVERRIDE_TYPES` in the post-processor: `public_key` and `private_key` (static cryptographic config values present in IoT/blockchain APIs — not produced by a prior tool call and dangerous to expose as graph edges), `otp` (one-time passwords arrive out-of-band via SMS/email, not as tool outputs, so graph edges through `otp` would be unreachable in practice), and `ticket` (too generic — the LLM applied it to event tickets, support tickets, and booking confirmations interchangeably, making any CHAINS_TO edge using it semantically meaningless). Three accepted new types were reclassified from CHAIN_ONLY to `USER_PROVIDED`: `country_name` (count=17; users say "United States" directly, not via a lookup API — parallel to `city_name` already in USER_PROVIDED), `company_name` (count=4; users type company names from intent, not from prior API responses), and `city_code` (count=5; analogous to `country_code` which was already classified USER_PROVIDED). These corrections reduce the accepted CHAIN_ONLY new-type set from 40 to 33 effective types, and USER_PROVIDED grows from 19 to 22 seed entries.

**Cache key tradeoff.** The cache key for each LLM call is `(endpoint_id, model, prompt_version)`;
the parameter list itself is excluded. This avoids cache churn from minor normalisation tweaks,
but means that if normalization rules change substantially — adding or dropping parameters from
an endpoint — cached annotations become stale. The mitigation is to bump `prompt_version`
manually whenever normalization rules change in a way that would affect parameter identity.

---

## §3 Tool Graph & Chain Sampler

### §3.1 Graph Schema Decisions

### §3.2 Edge Type Justification

**Graph builder invariant — no self-loop CHAINS_TO edges.**
During the F1.5 pilot review, `getOrderById` was observed both consuming and producing
`order_id`: the response field `id` was typed `order_id` (plausible from context) while
the required parameter `orderId` also maps to `order_id`. This creates a self-loop where
endpoint A would appear to CHAINS_TO itself. Self-loops are semantically meaningless for
chaining — an endpoint cannot usefully consume its own output as an input — and including
them would inflate the CHAINS_TO edge count and distort sampler probabilities.
**F2.1 must filter `CHAINS_TO` edges where source == target endpoint.**

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
