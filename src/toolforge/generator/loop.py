"""Batch generation loop — F7.1.

Public entry points:
  generate_one(seed, constraints, generator)  — single conversation (F4.6, unchanged)
  generate_batch(n, seed, artifacts_dir, ...)  — full batch; builds all deps from disk

Design:
  generate_batch() loads artifacts → builds all agents → calls _run_batch().
  _run_batch() is a pure inner loop that takes an already-built generator +
  tracker; unit tests can inject mocks here without touching the disk.

  Tracker.update() is called with graph node IDs (the "ep:" prefixed form
  the tracker and sampler both use internally).  conv.sampled_chain contains
  bare endpoint IDs (without "ep:") — so _run_batch re-adds the prefix before
  calling update().  This is the only place where the conversion is done.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from toolforge.execution.session import make_session, session_to_dict, SessionState, ToolOutput
from toolforge.generator.graph import ConversationGenerator
from toolforge.generator.state import Conversation, ConversationState
from toolforge.graph.sampler import ChainConstraints
from toolforge.memory.corpus_stats import (
    CorpusDiversityTracker,
    NoOpDiversityTracker,
)

log = structlog.get_logger(__name__)

# Default chain constraints for batch generation.
# Length 2–5 + min 2 distinct tools satisfies the ≥50% / ≥3-tool-calls requirement.
_DEFAULT_CONSTRAINTS = ChainConstraints(length=(2, 5), min_distinct_tools=2)


# ---------------------------------------------------------------------------
# Single-conversation entry point (F4.6 — unchanged)
# ---------------------------------------------------------------------------

def generate_one(
    seed: int,
    constraints: ChainConstraints,
    generator: ConversationGenerator,
    *,
    conversation_id: str | None = None,
) -> Conversation:
    """Run the LangGraph loop for one seed and return an immutable Conversation.

    Args:
        seed: RNG seed.  Same seed + same generator state → same Conversation.
        constraints: Chain-sampling constraints (length, categories, etc.).
        generator: Pre-built ConversationGenerator holding all agents and deps.
        conversation_id: Optional override for the conversation ID.
            Defaults to "conv-{seed}".

    Returns:
        Conversation with all messages, session_summary, and judge_result.

    Raises:
        RuntimeError: If the ChainSampler returns a truncated/failed result.
    """
    cid = conversation_id or f"conv-{seed}"

    initial: ConversationState = {
        "conversation_id": cid,
        "seed": seed,
        "constraints": constraints,
        # Fields set by plan_node:
        "sampled_chain": [],
        "chain_endpoints": [],
        "all_endpoints": generator._all_endpoints,
        "plan": None,
        # Live conversation:
        "messages": [],
        "session_state": make_session(cid, seed),
        "turn_count": 0,
        "chain_index": 0,
        "last_assistant_turn": None,
        # Results:
        "judge_result": None,
        "repair_attempts": 0,
        "status": "running",
    }

    final_state: ConversationState = generator.run(initial)  # type: ignore[assignment]
    return _state_to_conversation(final_state)


def _state_to_conversation(state: ConversationState) -> Conversation:
    """Convert a completed ConversationState to an immutable Conversation record."""
    return Conversation(
        conversation_id=state["conversation_id"],
        seed=state["seed"],
        sampled_chain=state["sampled_chain"],
        messages=state["messages"],
        session_summary=session_to_dict(state["session_state"]),
        judge_result=state["judge_result"],
        status=state["status"],
        repair_attempts=state.get("repair_attempts", 0),  # type: ignore[attr-defined]
    )


# ---------------------------------------------------------------------------
# Batch entry point (F7.1)
# ---------------------------------------------------------------------------

def generate_batch(
    n: int,
    seed: int,
    artifacts_dir: Path,
    cache_dir: Path,
    *,
    was_steered: bool = True,
    constraints: ChainConstraints | None = None,
) -> list[dict[str, Any]]:
    """Build all dependencies from artifacts and run *n* conversations.

    Args:
        n: Number of conversations to generate.
        seed: Base RNG seed.  Conversation i uses seed ``seed + i``.
        artifacts_dir: Directory produced by ``toolforge build`` (contains
            ``registry.json`` and ``graph.pkl``).
        cache_dir: LLM cache root (typically ``settings.cache_dir``).
        was_steered: If True, use CorpusDiversityTracker (Run B).
            If False, use NoOpDiversityTracker (Run A baseline).
        constraints: Optional chain constraints override.  Defaults to
            ``ChainConstraints(length=(2, 5), min_distinct_tools=2)``.

    Returns:
        List of JSON-safe dicts, one per conversation.  Each record includes
        conversation_id, messages, tool_calls, tool_outputs, judge_scores,
        validation_results, and a metadata sub-dict.
    """
    from toolforge.registry.models import Tool, Endpoint
    from toolforge.graph.build import load_graph
    from toolforge.graph.sampler import ChainSampler
    from toolforge.execution.mock_responder import MockResponder
    from toolforge.execution.executor import Executor
    from toolforge.agents.llm_client import LLMClient
    from toolforge.agents.planner import Planner
    from toolforge.agents.user_sim import UserSimulator
    from toolforge.agents.assistant import Assistant
    from toolforge.agents.judge import Judge
    from toolforge.agents.repair import RepairAgent
    from toolforge.memory.corpus_stats import build_endpoint_metadata

    # --- Load registry -------------------------------------------------------
    registry_json = json.loads((artifacts_dir / "registry.json").read_text("utf-8"))
    tools = [Tool.model_validate(t) for t in registry_json]
    endpoint_registry: dict[str, Endpoint] = {
        ep.id: ep
        for tool in tools
        for ep in tool.endpoints
    }
    all_endpoints: list[Endpoint] = list(endpoint_registry.values())
    log.info("generate_batch.registry_loaded", endpoints=len(all_endpoints))

    # --- Load graph + build sampler ------------------------------------------
    graph = load_graph(artifacts_dir)
    sampler = ChainSampler(graph)

    # --- Build tracker -------------------------------------------------------
    endpoint_meta = build_endpoint_metadata(graph)
    tracker: CorpusDiversityTracker | NoOpDiversityTracker
    if was_steered:
        tracker = CorpusDiversityTracker(endpoint_meta)
    else:
        tracker = NoOpDiversityTracker()
    log.info("generate_batch.tracker", steering=was_steered)

    # --- LLM clients ---------------------------------------------------------
    llm_cache = cache_dir / "llm"
    haiku_client = LLMClient(
        model="claude-haiku-4-5-20251001",
        temperature=0.7,
        cache_dir=llm_cache,
    )
    sonnet_client = LLMClient(
        model="claude-sonnet-4-6",
        temperature=0.0,
        cache_dir=llm_cache,
    )

    # --- Agents + executor ---------------------------------------------------
    planner = Planner(haiku_client)
    user_sim = UserSimulator(haiku_client)
    assistant = Assistant(haiku_client)
    judge = Judge(sonnet_client)
    repair_agent = RepairAgent(sonnet_client)
    responder = MockResponder()
    executor = Executor(endpoint_registry, responder)

    # --- Assemble generator --------------------------------------------------
    generator = ConversationGenerator(
        sampler=sampler,
        registry=endpoint_registry,
        all_endpoints=all_endpoints,
        executor=executor,
        planner=planner,
        user_sim=user_sim,
        assistant=assistant,
        judge=judge,
        repair_agent=repair_agent,
        tracker=tracker,
    )

    resolved_constraints = constraints if constraints is not None else _DEFAULT_CONSTRAINTS

    return _run_batch(
        n=n,
        seed=seed,
        generator=generator,
        tracker=tracker,
        constraints=resolved_constraints,
        was_steered=was_steered,
    )


# ---------------------------------------------------------------------------
# Inner loop (testable with a mock generator)
# ---------------------------------------------------------------------------

def _run_batch(
    n: int,
    seed: int,
    generator: ConversationGenerator,
    tracker: CorpusDiversityTracker | NoOpDiversityTracker,
    constraints: ChainConstraints,
    *,
    was_steered: bool,
) -> list[dict[str, Any]]:
    """Generate *n* conversations and return serialised records.

    Separated from generate_batch() so unit tests can inject a mock generator
    without touching the filesystem.

    For each conversation:
    1. generate_one(seed+i) → Conversation
    2. validate_conversation() → list[ValidationResult]
    3. tracker.update(node_ids, pattern, length_bucket)
    4. Serialise → record dict

    Tracker.update() receives graph node IDs (the "ep:..." form).  Since
    conv.sampled_chain stores bare IDs, the "ep:" prefix is re-added here.
    """
    from toolforge.evaluation.validators import validate_conversation

    records: list[dict[str, Any]] = []

    for i in range(n):
        conv_seed = seed + i
        conv_id = f"conv-{conv_seed:06d}"
        try:
            conv = generate_one(
                conv_seed,
                constraints,
                generator,
                conversation_id=conv_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "generate_batch.skip",
                i=i,
                conv_seed=conv_seed,
                error=str(exc),
            )
            continue

        # Reconstruct SessionState for validation (validators need ToolOutput objects).
        session_state = _session_from_summary(conv.session_summary)
        validation_results = validate_conversation(conv, session_state)

        # Compute metadata fields.
        pattern = "linear"
        length_bucket = _length_bucket(len(conv.sampled_chain))

        # Update tracker with graph node IDs (conv.sampled_chain uses bare IDs).
        node_ids = [f"ep:{ep}" for ep in conv.sampled_chain]
        tracker.update(node_ids, pattern, length_bucket)

        record = _conv_to_record(
            conv=conv,
            validation_results=validation_results,
            was_steered=was_steered,
        )
        records.append(record)

        judge_pass = conv.judge_result.overall_pass if conv.judge_result else False
        log.info(
            "generate_batch.conversation_done",
            i=i + 1,
            n=n,
            conversation_id=conv_id,
            status=conv.status,
            judge_pass=judge_pass,
        )

    log.info("generate_batch.complete", produced=len(records), requested=n)
    return records


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def _length_bucket(chain_len: int) -> str:
    """Coarse bucket label for chain length."""
    if chain_len <= 2:
        return "short"
    if chain_len <= 4:
        return "medium"
    return "long"


def _session_from_summary(summary: dict[str, Any]) -> SessionState:
    """Reconstruct a SessionState from session_to_dict() output.

    Used to feed the validators, which need ToolOutput objects.  Only the
    fields consumed by validators (tool_outputs, conversation_id, seed) are
    fully populated; others are left empty.
    """
    tool_outputs = [
        ToolOutput(
            endpoint_id=o["endpoint_id"],
            arguments=o.get("arguments", {}),
            response=o.get("response"),
            error=o.get("error"),
            timestamp=o.get("timestamp", "turn-0"),
        )
        for o in summary.get("tool_outputs", [])
    ]
    return SessionState(
        conversation_id=summary.get("conversation_id", ""),
        seed=summary.get("seed", 0),
        available_values_by_type={},
        resolved_entities={},
        created_entities=[],
        tool_outputs=tool_outputs,
        private_user_knowledge={},
    )


def _conv_to_record(
    conv: Conversation,
    validation_results: list[Any],
    was_steered: bool,
) -> dict[str, Any]:
    """Serialise a Conversation + validation results to a JSON-safe record dict.

    Output schema (all fields required by F7.1 spec):
        conversation_id, messages, tool_calls, tool_outputs,
        judge_scores, validation_results, metadata
    """
    session_tool_outputs: list[dict[str, Any]] = conv.session_summary.get("tool_outputs", [])

    # tool_calls: input side of each tool interaction (endpoint + args).
    tool_calls = [
        {"endpoint_id": o["endpoint_id"], "arguments": o.get("arguments", {})}
        for o in session_tool_outputs
    ]

    # judge_scores: flattened from JudgeResult or empty dict if not scored.
    if conv.judge_result is not None:
        jr = conv.judge_result
        judge_scores: dict[str, Any] = {
            "naturalness": jr.naturalness.score,
            "tool_correctness": jr.tool_correctness.score,
            "chain_coherence": jr.chain_coherence.score,
            "task_completion": jr.task_completion.score,
            "mean": jr.mean_score(),
            "overall_pass": jr.overall_pass,
        }
    else:
        judge_scores = {}

    # validation_results: list of stage dicts.
    val_list = [
        {
            "stage": vr.stage,
            "passed": vr.passed,
            "is_hard": vr.is_hard,
            "errors": vr.errors,
            "warnings": vr.warnings,
        }
        for vr in validation_results
    ]

    # tools_used: unique tool names (middle component of "Category/Tool/endpoint").
    tools_used: list[str] = sorted({
        ep_id.split("/")[1]
        for ep_id in conv.sampled_chain
        if ep_id.count("/") >= 2
    })

    repair_attempts = getattr(conv, "repair_attempts", 0)

    metadata: dict[str, Any] = {
        "seed": conv.seed,
        "sampled_chain": conv.sampled_chain,
        "pattern": "linear",
        "length_bucket": _length_bucket(len(conv.sampled_chain)),
        "repair_attempts": repair_attempts,
        "was_steered": was_steered,
        "tools_used": tools_used,
        "num_turns": len(conv.messages),
    }

    # Serialise messages (may be dict or Message objects).
    messages = [
        (m if isinstance(m, dict) else {"role": m.role, "content": m.content})
        for m in conv.messages
    ]

    return {
        "conversation_id": conv.conversation_id,
        "messages": messages,
        "tool_calls": tool_calls,
        "tool_outputs": session_tool_outputs,
        "judge_scores": judge_scores,
        "validation_results": val_list,
        "metadata": metadata,
    }
