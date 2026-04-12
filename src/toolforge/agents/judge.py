"""Judge agent — F4.5.

Scores a completed conversation on four dimensions using a different model
family from the generators (claude-sonnet-4-6) to avoid self-preference bias.

Design decisions:
  - JudgeResult / DimensionScore are Pydantic models, structured output only.
  - Temperature 0 for reproducibility (same conversation → same scores).
  - overall_pass computed post-parse from scores, not by the LLM.
  - Prompt is loaded from prompts/judge.md; anchors are baked into the prompt
    so the judge has calibration examples every call (4 anchors, including a
    negative one).
  - Conversation is serialised to a compact text format rather than raw JSON
    to reduce token count (P2: no LLM for serialisation logic).

Pass threshold (per FEATURES.md F4.5):
    mean(naturalness, tool_correctness, chain_coherence, task_completion) >= 3.5
    AND min(all four) >= 2.5
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, model_validator

from toolforge.agents.base import Agent
from toolforge.agents.llm_client import LLMClient
from toolforge.agents.user_sim import Message

log = structlog.get_logger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"
_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")

_PASS_MEAN = 3.5
_PASS_MIN = 2.5


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class DimensionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class JudgeResult(BaseModel):
    """Structured output of the Judge agent.

    overall_pass is computed from scores; the LLM does NOT set it.
    It is included in the schema (tool-use forcing sends the schema to the
    model) but the model_validator overwrites whatever the model wrote.
    """

    naturalness: DimensionScore
    tool_correctness: DimensionScore
    chain_coherence: DimensionScore
    task_completion: DimensionScore
    failure_modes: list[str] = Field(default_factory=list)
    overall_pass: bool = False  # always overwritten by validator

    @model_validator(mode="after")
    def _compute_overall_pass(self) -> "JudgeResult":
        scores = [
            self.naturalness.score,
            self.tool_correctness.score,
            self.chain_coherence.score,
            self.task_completion.score,
        ]
        mean = sum(scores) / len(scores)
        min_score = min(scores)
        object.__setattr__(self, "overall_pass", mean >= _PASS_MEAN and min_score >= _PASS_MIN)
        return self

    def mean_score(self) -> float:
        scores = [
            self.naturalness.score,
            self.tool_correctness.score,
            self.chain_coherence.score,
            self.task_completion.score,
        ]
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Judge agent
# ---------------------------------------------------------------------------

class Judge(Agent):
    """Scores a completed conversation on four dimensions.

    Usage::

        client = LLMClient(model="claude-sonnet-4-6", temperature=0.0)
        judge = Judge(client)
        result: JudgeResult = judge.score(conversation, available_endpoints)
    """

    name = "judge"
    system_prompt = _SYSTEM_PROMPT

    def score(
        self,
        conversation: list[Message],
        available_endpoints: list[str],
    ) -> JudgeResult:
        """Score a completed conversation.

        Args:
            conversation: Full message history (user/assistant turns, may
                include tool_call and tool_result turns serialised as content).
            available_endpoints: Endpoint IDs that were legitimately available
                to the assistant during this conversation.
        """
        user_prompt = _build_user_prompt(conversation, available_endpoints)
        result = self._client.call(
            self.system_prompt,
            user_prompt,
            JudgeResult,
            prompt_version="v1",
            agent_name=self.name,
        )
        log.info(
            "judge.scored",
            naturalness=result.naturalness.score,
            tool_correctness=result.tool_correctness.score,
            chain_coherence=result.chain_coherence.score,
            task_completion=result.task_completion.score,
            mean=result.mean_score(),
            overall_pass=result.overall_pass,
        )
        return result


# ---------------------------------------------------------------------------
# Prompt construction (pure Python, P2)
# ---------------------------------------------------------------------------

def _build_user_prompt(
    conversation: list[Message],
    available_endpoints: list[str],
) -> str:
    """Serialise the conversation and endpoint list into a compact prompt."""
    conv_lines: list[str] = []
    for msg in conversation:
        role = msg["role"].upper()
        content = msg["content"]
        conv_lines.append(f"{role}: {content}")

    payload: dict[str, Any] = {
        "conversation": conv_lines,
        "available_endpoints": available_endpoints,
    }
    return (
        "Please score the following conversation:\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )
