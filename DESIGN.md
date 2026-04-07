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

### §2.2 Normalization Decisions Table

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
