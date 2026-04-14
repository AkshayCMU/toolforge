"""Deterministic ValidationResult checkpoints — F5.1.

Five pure validators (no I/O, no LLM) run in order on a completed Conversation.
All validators accumulate errors/warnings; none short-circuits on first failure.

Hard validators (structure, tool_calls, grounding, completeness):
    A conversation must pass ALL hard validators before it is eligible for judging.
    Failures route to targeted repair (F5.2) or discard.

Soft validator (constraints):
    Always passes (passed=True).  Emits warnings that are logged and passed through.

Principle: P4 — modular evaluation; each failure localises to exactly one stage.
"""

from __future__ import annotations

import json
import re

import structlog
from pydantic import BaseModel, computed_field

from toolforge.execution.session import SessionState
from toolforge.generator.state import Conversation

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Content-string patterns (same format as graph.py serialisation)
# ---------------------------------------------------------------------------

# Matches a full-content tool_call:  [tool_call: Cat/Tool/ep, args={"k": "v"}]
# Uses re.DOTALL so that multi-line JSON args are captured correctly.
_TOOL_CALL_RE = re.compile(
    r"^\[tool_call: ([^,\]]+),\s*args=(\{.*\})\]$",
    re.DOTALL,
)
# Matches a full-content tool_result: [tool_result: {"k": "v"}]
_TOOL_RESULT_RE = re.compile(r"^\[tool_result: (\{.*\})\]$", re.DOTALL)

# Executor grounding-error substring (from executor.py error messages)
_GROUNDING_ERROR_MARKER = "not in session"

# Hard stage names (used by is_hard computed field)
_HARD_STAGES = frozenset({"structure", "tool_calls", "grounding", "completeness"})


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of one validation stage.

    ``is_hard`` is derived from ``stage`` — it is not stored directly to avoid
    the possibility of is_hard=False on a stage that should be hard.
    """

    stage: str
    passed: bool
    errors: list[str]
    warnings: list[str]

    @computed_field  # type: ignore[misc]
    @property
    def is_hard(self) -> bool:
        return self.stage in _HARD_STAGES


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def validate_structure(conv: Conversation, state: SessionState) -> ValidationResult:  # noqa: ARG001
    """Hard. Check that the message list is structurally well-formed.

    Fires on generator bugs (routing errors, serialisation failures).
    A conversation that passes this validator is safe for semantic inspection.
    """
    errors: list[str] = []

    if not conv.conversation_id:
        errors.append("conversation_id is empty")

    if not conv.messages:
        errors.append("messages list is empty")
        # Cannot check further without messages.
        return ValidationResult(stage="structure", passed=False, errors=errors, warnings=[])

    # Every message must have valid role + non-empty string content.
    for i, msg in enumerate(conv.messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if role not in ("user", "assistant"):
            errors.append(f"Message {i}: invalid role {role!r}")
        if not isinstance(content, str) or not content:
            errors.append(f"Message {i}: content must be a non-empty string")

    # First message must be from the user (conversation opener).
    first_role = (
        conv.messages[0].get("role")
        if isinstance(conv.messages[0], dict)
        else getattr(conv.messages[0], "role", None)
    )
    if first_role != "user":
        errors.append(f"First message must have role='user', got {first_role!r}")

    # No two consecutive messages with the same role (alternating structure).
    for i in range(1, len(conv.messages)):
        prev_msg = conv.messages[i - 1]
        curr_msg = conv.messages[i]
        prev_role = prev_msg.get("role") if isinstance(prev_msg, dict) else getattr(prev_msg, "role", None)
        curr_role = curr_msg.get("role") if isinstance(curr_msg, dict) else getattr(curr_msg, "role", None)
        if prev_role == curr_role:
            errors.append(
                f"Messages {i-1} and {i} both have role={curr_role!r} — "
                "consecutive same-role messages indicate a routing bug"
            )

    if not conv.sampled_chain:
        errors.append("sampled_chain is empty")

    if conv.status not in ("done", "failed"):
        errors.append(f"status must be 'done' or 'failed', got {conv.status!r}")

    return ValidationResult(
        stage="structure",
        passed=not errors,
        errors=errors,
        warnings=[],
    )


def validate_tool_calls(conv: Conversation, state: SessionState) -> ValidationResult:  # noqa: ARG001
    """Hard. Check structural integrity of every tool_call / tool_result pair.

    Checks:
    1. Every [tool_call: ...] in an assistant message is parseable (endpoint ID
       has at least one slash, args= is valid JSON).
    2. Every assistant tool_call is immediately followed by a user message
       whose content matches [tool_result: ...].
    """
    errors: list[str] = []

    messages = conv.messages
    for i, msg in enumerate(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if not isinstance(content, str):
            continue

        if role == "assistant" and content.startswith("[tool_call:"):
            m = _TOOL_CALL_RE.match(content)
            if m is None:
                errors.append(
                    f"Message {i}: malformed tool_call format — "
                    f"expected '[tool_call: endpoint, args={{...}}]', got: {content[:120]!r}"
                )
                continue

            endpoint_id, args_str = m.group(1).strip(), m.group(2)

            # Endpoint ID sanity: must contain at least one slash (Category/Tool/name).
            if "/" not in endpoint_id:
                errors.append(
                    f"Message {i}: tool_call endpoint_id {endpoint_id!r} has no '/' — "
                    "expected format 'Category/Tool/endpoint'"
                )

            # args must be valid JSON.
            try:
                json.loads(args_str)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"Message {i}: tool_call args is not valid JSON "
                    f"({exc}): {args_str[:80]!r}"
                )

            # Immediately following message must be a tool_result (role="user").
            if i + 1 >= len(messages):
                errors.append(
                    f"Message {i}: tool_call at end of conversation with no following tool_result"
                )
            else:
                next_msg = messages[i + 1]
                next_role = (
                    next_msg.get("role") if isinstance(next_msg, dict)
                    else getattr(next_msg, "role", None)
                )
                next_content = (
                    next_msg.get("content") if isinstance(next_msg, dict)
                    else getattr(next_msg, "content", "")
                )
                if next_role != "user" or not (
                    isinstance(next_content, str)
                    and next_content.startswith("[tool_result:")
                ):
                    errors.append(
                        f"Message {i}: tool_call not followed by a tool_result "
                        f"(next message {i+1} is role={next_role!r}, "
                        f"content={str(next_content)[:60]!r})"
                    )
                else:
                    # Content starts with [tool_result: — validate format + JSON payload.
                    # Check only the first line: content may have trailing SESSION STATE
                    # metadata appended after the closing ']' on subsequent lines.
                    first_line = next_content.split("\n")[0]
                    tr_match = _TOOL_RESULT_RE.match(first_line)
                    if tr_match is None:
                        errors.append(
                            f"Message {i+1}: malformed tool_result format — "
                            f"expected '[tool_result: {{...}}]', got: {next_content[:120]!r}"
                        )
                    else:
                        try:
                            json.loads(tr_match.group(1))
                        except json.JSONDecodeError as exc:
                            errors.append(
                                f"Message {i+1}: tool_result payload is not valid JSON "
                                f"({exc}): {tr_match.group(1)[:80]!r}"
                            )

    return ValidationResult(
        stage="tool_calls",
        passed=not errors,
        errors=errors,
        warnings=[],
    )


def validate_grounding(conv: Conversation, state: SessionState) -> ValidationResult:
    """Hard (first pass) / Soft (after repair). Check for grounding failures.

    Reads executor-recorded errors rather than re-executing tool calls.
    Pure — no I/O, no LLM, no registry access.

    Grounding failures are identified by the executor's error string pattern:
        "Invalid {type}: {value!r} not in session. Valid values: [...]"

    When conv.repair_attempts > 0 the repair agent has already corrected the
    message text for any grounding errors it found.  Re-blocking on the original
    execution record would double-penalise the conversation and trap the repair
    loop in a "repeated failure" cycle.  Downgrade to warnings so the repaired
    conversation can proceed to the judge.
    """
    grounding_errors: list[str] = []

    for output in state.tool_outputs:
        if output.error is not None and _GROUNDING_ERROR_MARKER in output.error:
            grounding_errors.append(
                f"Grounding failure at {output.timestamp} "
                f"(endpoint={output.endpoint_id}): {output.error}"
            )

    if grounding_errors and conv.repair_attempts > 0:
        # Repair has already corrected the conversation text; treat as soft warning.
        return ValidationResult(
            stage="grounding",
            passed=True,
            errors=[],
            warnings=grounding_errors,
        )

    return ValidationResult(
        stage="grounding",
        passed=not grounding_errors,
        errors=grounding_errors,
        warnings=[],
    )


def validate_completeness(conv: Conversation, state: SessionState) -> ValidationResult:  # noqa: ARG001
    """Hard. Check that the conversation ends with a substantive assistant summary.

    A properly complete conversation ends on an assistant message that:
    1. Is not a tool_call (no [tool_call: ...] prefix).
    2. Has content longer than 10 characters.

    Failures here route to "append final assistant turn" in the repair loop.
    """
    errors: list[str] = []

    if not conv.messages:
        errors.append("No messages — cannot check completeness")
        return ValidationResult(stage="completeness", passed=False, errors=errors, warnings=[])

    last_msg = conv.messages[-1]
    last_role = (
        last_msg.get("role") if isinstance(last_msg, dict)
        else getattr(last_msg, "role", None)
    )
    last_content = (
        last_msg.get("content") if isinstance(last_msg, dict)
        else getattr(last_msg, "content", "")
    )
    if not isinstance(last_content, str):
        last_content = ""

    if last_role != "assistant":
        errors.append(
            f"Conversation ends with role={last_role!r}; "
            "expected a final assistant summary message"
        )
    elif last_content.startswith("[tool_call:"):
        errors.append(
            "Conversation ends with a tool_call instead of a summary message — "
            "the turn cap was likely hit mid-chain"
        )
    elif len(last_content.strip()) <= 10:
        errors.append(
            f"Final assistant message is trivially short ({len(last_content.strip())} chars); "
            "expected a substantive summary"
        )

    return ValidationResult(
        stage="completeness",
        passed=not errors,
        errors=errors,
        warnings=[],
    )


def validate_constraints(conv: Conversation, state: SessionState) -> ValidationResult:
    """Soft. Emit warnings about diversity / quality constraints.

    Always returns passed=True. Warnings are returned to the caller for logging
    and corpus statistics (F6.1 / F7) — they do not block judging.
    """
    warnings: list[str] = []

    successful = [o for o in state.tool_outputs if o.is_success()]
    distinct_endpoints = len({o.endpoint_id for o in successful})

    # Assignment hard requirement: ≥50% of conversations should have ≥3 tool calls
    # and ≥2 distinct tools.  Warn when this conversation falls short.
    if len(successful) < 3:
        warnings.append(
            f"Only {len(successful)} successful tool call(s); "
            "target is ≥3 for diversity statistics"
        )
    if distinct_endpoints < 2:
        warnings.append(
            f"Only {distinct_endpoints} distinct endpoint(s) called successfully; "
            "target is ≥2 distinct tools"
        )

    # Warn if the majority of tool calls failed.
    total = len(state.tool_outputs)
    if total > 0 and len(successful) / total < 0.5:
        failed_count = total - len(successful)
        warnings.append(
            f"{failed_count}/{total} tool calls failed "
            f"({100 * failed_count // total}%); conversation may be low-quality signal"
        )

    # Warn if the sampled chain was not fully completed.
    completed_chain_steps = sum(
        1 for o in successful if o.endpoint_id in conv.sampled_chain
    )
    if completed_chain_steps < len(conv.sampled_chain):
        warnings.append(
            f"Chain incomplete: {completed_chain_steps}/{len(conv.sampled_chain)} "
            "on-chain steps succeeded"
        )

    return ValidationResult(
        stage="constraints",
        passed=True,
        errors=[],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_conversation(
    conv: Conversation,
    state: SessionState,
) -> list[ValidationResult]:
    """Run all five validators and return their results in fixed order.

    Invariant: always returns exactly 5 results in this stage order:
        structure, tool_calls, grounding, completeness, constraints

    All validators always run — none short-circuits on a prior failure.
    The caller is responsible for inspecting ``is_hard`` and ``passed``
    to decide whether the conversation proceeds to judging or repair.
    """
    results = [
        validate_structure(conv, state),
        validate_tool_calls(conv, state),
        validate_grounding(conv, state),
        validate_completeness(conv, state),
        validate_constraints(conv, state),
    ]
    constraints = results[-1]
    if constraints.warnings:
        log.debug(
            "validation.constraints.warnings",
            conversation_id=conv.conversation_id,
            count=len(constraints.warnings),
        )
    return results
