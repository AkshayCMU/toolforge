"""RepairAgent — F5.2.

Produces a single RepairOperation given a failing conversation and its
validation results.  Follows the flat-Pydantic-model pattern established
by AssistantTurn (see DESIGN.md §5.4) to avoid discriminated-union JSON
Schema reliability issues under Anthropic tool-use forcing.

Deviation from FEATURES.md spec: the spec's EditOperation schema has only
a `reason` field.  This implementation adds a `content` field to
RepairOperation so the repair runner can apply edits without calling other
agents (self-contained repair).  Documented in DESIGN.md §5.4.

Model: claude-sonnet-4-6, temperature=0 (set at LLMClient construction time).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, model_validator

from toolforge.agents.base import Agent

if TYPE_CHECKING:
    from toolforge.agents.llm_client import LLMClient
    from toolforge.evaluation.validators import ValidationResult
    from toolforge.generator.state import Conversation

log = structlog.get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "repair.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class RepairOperation(BaseModel):
    """Single structured edit operation produced by the RepairAgent.

    Flat model — same pattern as AssistantTurn.  A @model_validator enforces
    field-presence per type, preserving the logical invariants of a
    discriminated union without its JSON Schema reliability penalty.

    Types:
        regenerate_turn — replace the content of an existing turn at turn_index.
        append_turn     — append a new message at the end with the given role.
        discard         — conversation is unrecoverable; no edit applied.
    """

    type: str  # "regenerate_turn" | "append_turn" | "discard"
    turn_index: int = -1       # regenerate_turn: 0-based index; -1 = not applicable
    role: str = ""             # append_turn: "user" | "assistant"; "" otherwise
    content: str = ""          # new turn content; "" for discard
    reason: str                # always required — explains the failure and fix

    @model_validator(mode="after")
    def _validate_fields(self) -> "RepairOperation":
        if self.type == "regenerate_turn":
            if self.turn_index < 0:
                raise ValueError(
                    "regenerate_turn requires turn_index >= 0, "
                    f"got {self.turn_index}"
                )
            if not self.content:
                raise ValueError("regenerate_turn requires non-empty content")
        elif self.type == "append_turn":
            if self.role not in ("user", "assistant"):
                raise ValueError(
                    f"append_turn role must be 'user' or 'assistant', got {self.role!r}"
                )
            if not self.content:
                raise ValueError("append_turn requires non-empty content")
        elif self.type != "discard":
            raise ValueError(
                f"type must be 'regenerate_turn', 'append_turn', or 'discard', "
                f"got {self.type!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class RepairAgent(Agent):
    """Suggests one RepairOperation for a failing conversation.

    Usage::

        client = LLMClient(model="claude-sonnet-4-6", temperature=0.0)
        agent = RepairAgent(client)
        op = agent.suggest(conv, validation_results, attempt=1)
    """

    name = "repair"
    system_prompt = _SYSTEM_PROMPT

    def suggest(
        self,
        conv: "Conversation",
        validation_results: "list[ValidationResult]",
        attempt: int = 1,
    ) -> RepairOperation:
        """Return one RepairOperation for the failing conversation.

        Args:
            conv: The Conversation that failed validation or judging.
            validation_results: Results from validate_conversation().
            attempt: Current repair attempt (1 or 2).  Included in the prompt
                     so the agent knows whether this is the last chance.
        """
        user_prompt = _build_user_prompt(conv, validation_results, attempt)
        result = self._client.call(
            self.system_prompt,
            user_prompt,
            RepairOperation,
            prompt_version="v1",
            agent_name=self.name,
        )
        log.info(
            "repair.suggested",
            conversation_id=conv.conversation_id,
            op_type=result.type,
            turn_index=result.turn_index,
            reason=result.reason,
            attempt=attempt,
        )
        return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_user_prompt(
    conv: "Conversation",
    validation_results: "list[ValidationResult]",
    attempt: int,
) -> str:
    """Serialise the failing conversation and validation results into a repair prompt."""
    lines: list[str] = []

    lines.append(f"## Repair attempt {attempt}/2")
    lines.append("")

    # Failed validation stages.
    failed = [r for r in validation_results if not r.passed]
    if failed:
        lines.append("## Failed validation stages")
        for r in failed:
            lines.append(f"- Stage: **{r.stage}** ({'hard' if r.is_hard else 'soft'})")
            for err in r.errors:
                lines.append(f"  Error: {err}")
    else:
        lines.append("## Validation")
        lines.append("All structural validators passed — conversation failed the quality judge.")

    lines.append("")
    lines.append("## Conversation (0-indexed)")
    for i, msg in enumerate(conv.messages):
        role = msg["role"].upper() if isinstance(msg, dict) else msg.role.upper()  # type: ignore[union-attr]
        content = msg["content"] if isinstance(msg, dict) else msg.content  # type: ignore[union-attr]
        lines.append(f"[{i}] {role}: {content}")

    lines.append("")
    lines.append("## Sampled chain (expected API call sequence)")
    lines.append(", ".join(conv.sampled_chain))

    return "\n".join(lines)
