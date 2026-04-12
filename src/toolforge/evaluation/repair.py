"""Repair runner — F5.2.

Applies RepairAgent edit operations, re-validates, and re-judges until the
conversation passes or the attempt budget is exhausted.

run_repair() is the sole public entry point.  It is called by:
  - ConversationGenerator._repair_node() (LangGraph integration)
  - tests/integration/test_repair_loop.py (integration test)

Design notes
------------
- run_repair() owns the max-attempts cap (default 2).  The LangGraph router
  (_route_after_judge) does NOT check attempt counts — it only checks status.
- The judge is called only after hard validators pass (avoids wasted LLM calls
  on structurally-broken conversations).
- Failure-signature deduplication: if the same set of hard-validator errors
  appears on two consecutive attempts, the conversation is unrecoverable and
  we abort early rather than looping indefinitely on the same failure.
- `state` is the ORIGINAL SessionState from the completed conversation.  It is
  NOT re-executed during repair — validators read its tool_outputs as-is.
  This is correct: repaired messages are validated against the original
  execution record.

Principle: P4 (targeted repair on stage-level failures = payoff for the
validation pipeline).
"""

from __future__ import annotations

import structlog

from toolforge.agents.judge import Judge, JudgeResult
from toolforge.agents.repair import RepairAgent, RepairOperation
from toolforge.evaluation.validators import ValidationResult, validate_conversation
from toolforge.execution.session import SessionState
from toolforge.generator.state import Conversation

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_repair(
    conv: Conversation,
    state: SessionState,
    repair_agent: RepairAgent,
    judge: Judge,
    *,
    _prior_sigs: frozenset[str] | None = None,
    _attempt: int = 1,
    max_attempts: int = 2,
) -> tuple[Conversation, JudgeResult | None, int]:
    """Run up to max_attempts repair cycles on a failing conversation.

    Args:
        conv:          The conversation that failed (validation or judging).
        state:         The original SessionState from generation (not re-executed).
        repair_agent:  RepairAgent (claude-sonnet-4-6, temperature=0).
        judge:         Judge agent used to re-score after repair.
        _prior_sigs:   Internal — set of failure signatures from previous attempts.
        _attempt:      Internal — current attempt number (1-based).
        max_attempts:  Maximum number of repair attempts (default 2).

    Returns:
        (repaired_conv, judge_result, attempts_used)

        - repaired_conv: the best conversation reached (may be the original if
          all repair attempts failed or returned discard).
        - judge_result:  the JudgeResult from the final scoring, or None if
          judging was never reached (hard failures persisted).
        - attempts_used: number of repair attempts consumed (0 means no repairs
          were needed — the conversation already passed).
    """
    if _attempt > max_attempts:
        return (conv, None, max_attempts)

    prior_sigs = _prior_sigs or frozenset()

    # Step 1: validate the conversation.
    results = validate_conversation(conv, state)
    hard_failures = [r for r in results if r.is_hard and not r.passed]

    # Step 2: if all hard validators pass, judge first.
    #   - Passes → return immediately (no repair needed / repair succeeded).
    #   - Fails  → fall through to repair.
    judge_result: JudgeResult | None = None
    if not hard_failures:
        judge_result = judge.score(conv.messages, conv.sampled_chain)
        log.info(
            "repair.judged",
            conversation_id=conv.conversation_id,
            mean_score=judge_result.mean_score(),
            overall_pass=judge_result.overall_pass,
            attempt=_attempt,
        )
        if judge_result.overall_pass:
            # _attempt - 1 because no repair was consumed on this pass
            # (first call with _attempt=1 → attempts_used=0 means "no repairs needed").
            return (conv, judge_result, _attempt - 1)
        # Judge failed but validation passed — proceed to repair below.

    # Step 3: check attempt budget.
    if _attempt > max_attempts:
        return (conv, judge_result, max_attempts)

    # Step 4: compute failure signature and check for repeat.
    sig = _failure_signature(results)
    if sig and sig in prior_sigs:
        log.warning(
            "repair.repeated_failure",
            conversation_id=conv.conversation_id,
            signature=sig[:120],
            attempt=_attempt,
        )
        return (conv, judge_result, _attempt)

    # Step 5: ask the repair agent for an edit operation.
    op = repair_agent.suggest(conv, results, attempt=_attempt)

    if op.type == "discard":
        log.info(
            "repair.discard",
            conversation_id=conv.conversation_id,
            reason=op.reason,
            attempt=_attempt,
        )
        return (conv, judge_result, _attempt)

    # Step 6: apply the edit.
    new_messages = _apply_operation(list(conv.messages), op)
    new_conv = Conversation(
        conversation_id=conv.conversation_id,
        seed=conv.seed,
        sampled_chain=conv.sampled_chain,
        messages=new_messages,
        session_summary=conv.session_summary,
        judge_result=None,
        status="done",
    )

    # Step 7: recurse — re-validate, re-judge, or repair again.
    return run_repair(
        new_conv, state, repair_agent, judge,
        _prior_sigs=prior_sigs | ({sig} if sig else frozenset()),
        _attempt=_attempt + 1,
        max_attempts=max_attempts,
    )


# ---------------------------------------------------------------------------
# Helpers (exported for unit tests)
# ---------------------------------------------------------------------------

def _apply_operation(
    messages: list[dict],
    op: RepairOperation,
) -> list[dict]:
    """Apply a RepairOperation to a message list.

    Pure function — returns a new list; the input is not mutated.

    regenerate_turn: replace messages[turn_index]["content"] with op.content.
                     Out-of-bounds index is a no-op (defensive).
    append_turn:     append {"role": op.role, "content": op.content}.
    discard:         callers should not reach this; returns messages unchanged.
    """
    msgs = list(messages)  # shallow copy — contents are immutable dicts
    if op.type == "regenerate_turn":
        if 0 <= op.turn_index < len(msgs):
            original = msgs[op.turn_index]
            msgs[op.turn_index] = {
                "role": original["role"] if isinstance(original, dict) else original.role,  # type: ignore[union-attr]
                "content": op.content,
            }
    elif op.type == "append_turn":
        msgs.append({"role": op.role, "content": op.content})
    return msgs


def _failure_signature(results: list[ValidationResult]) -> str:
    """Deterministic string summarising all hard-validator errors.

    Two calls produce the same signature iff the same hard validators fired
    with the same error messages (order-independent).  Used to detect repeat
    failures across repair attempts.

    Returns an empty string when all hard validators pass.
    """
    parts: list[str] = []
    for r in results:
        if r.is_hard and not r.passed:
            parts.extend(r.errors)
    return "|".join(sorted(parts))
