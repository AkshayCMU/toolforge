"""User simulator agent — F4.3.

Produces free-text user turns in a synthetic conversation. The simulator:
  - Plays the role defined in TaskPlan.user_persona
  - Withholds private_user_knowledge until the assistant directly asks
  - Produces a short closing message when the task is resolved
  - Never references API names or endpoint details

Uses LLMClient.call_text() — free-text output, not structured.
FEATURES.md principle note: "P2 (free text is correct here — structuring
defeats the purpose)." Tool-use forcing changes generation distribution
and degrades conversational naturalness.

Model: claude-haiku-4-5-20251001, temperature=0.7.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from typing_extensions import TypedDict

from toolforge.agents.base import Agent
from toolforge.agents.llm_client import LLMClient
from toolforge.agents.planner import TaskPlan

_PROMPT_PATH = Path(__file__).parent / "prompts" / "user_simulator.md"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text(encoding="utf-8")


class Message(TypedDict):
    """Minimal conversation turn used across F4.2–F4.4.

    F4.6 (LangGraph wiring) will formalise a richer Message type that also
    carries tool_call_id and tool_result fields for executor turns. For now,
    only user/assistant text turns are needed by the agents in this slice.
    """
    role: Literal["user", "assistant"]
    content: str


class UserSimulator(Agent):
    """Simulates a human user turn given a TaskPlan and conversation history.

    Usage::

        client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.7)
        sim = UserSimulator(client)
        turn: str = sim.respond(plan, history=[])
    """

    name = "user_simulator"

    def respond(
        self,
        plan: TaskPlan,
        history: list[Message],
    ) -> str:
        """Return the next user turn as free text.

        Args:
            plan: The TaskPlan produced by the Planner.
            history: Conversation so far. The returned string is NOT appended
                     here — the caller (F4.6) owns the history list.
        """
        system_prompt = self._build_system_prompt(plan)
        user_prompt = self._build_user_prompt(history)
        return self._client.call_text(
            system_prompt,
            user_prompt,
            prompt_version="v1",
            agent_name=self.name,
        )

    # ------------------------------------------------------------------
    # Prompt construction (pure Python, P2)
    # ------------------------------------------------------------------

    def _build_system_prompt(self, plan: TaskPlan) -> str:
        private_knowledge_text = _format_private_knowledge(plan.private_user_knowledge)
        return _PROMPT_TEMPLATE.format(
            persona=plan.user_persona,
            expected_outcome=plan.expected_final_outcome,
            private_knowledge=private_knowledge_text or "(none — you have shared everything)",
        )

    def _build_user_prompt(self, history: list[Message]) -> str:
        if not history:
            return "Start the conversation by sending your initial request to the assistant."
        lines: list[str] = ["## Conversation so far", ""]
        for msg in history:
            role_label = "You" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {msg['content']}")
        lines.append("")
        lines.append("What do you say next?")
        return "\n".join(lines)


def _format_private_knowledge(knowledge: dict[str, Any]) -> str:
    if not knowledge:
        return ""
    items = [f"  - {k}: {v}" for k, v in knowledge.items()]
    return "\n".join(items)
