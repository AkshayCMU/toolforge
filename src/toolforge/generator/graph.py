"""LangGraph conversation generator — F4.6.

ConversationGenerator wires the five agents and the Executor into a
LangGraph StateGraph.  Dependencies are injected at construction time;
the compiled graph is built once and reused across all generate_one() calls.

Node layout
-----------
    START → plan → user_turn → assistant_turn
                                    ├─(tool_call)──────────────────► executor
                                    ├─(message, chain_index≥len)──► finalize → judge → END
                                    └─(message, chain_index<len)──► user_turn  (loop)
                   executor ─────────────────────────────────────► assistant_turn
                   (executor also routes to finalize when turn_count≥12)

Routing deviation from FEATURES.md spec
-----------------------------------------
FEATURES.md shows "executor → user_turn".  This implementation uses
"executor → assistant_turn" because:

1. Tool results are serialised as role="user" messages (matching judge anchor
   format in test_judge.py).  After executor appends a role="user" tool_result,
   routing to user_turn would invoke the UserSimulator on a conversation whose
   last message is already role="user" — producing two consecutive user messages.

2. The shipped assistant.md prompt expects to see the tool result and
   synthesise ("If the task is complete, output a `message` turn confirming
   success") — this requires the assistant to act immediately after the result.

This deviation is documented in DESIGN.md §5.

chain_index progress
---------------------
chain_index advances in executor_node ONLY when:
  - the call succeeds (tool_output.is_success()), AND
  - turn.endpoint == sampled_chain[chain_index]  (on-chain, in-order).

Distractor calls and out-of-order calls are logged as warnings and do NOT
advance chain_index.  This prevents premature finalization.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from toolforge.agents.assistant import Assistant, select_distractors
from toolforge.agents.judge import Judge
from toolforge.agents.planner import Planner
from toolforge.agents.repair import RepairAgent
from toolforge.agents.user_sim import UserSimulator
from toolforge.execution.executor import Executor
from toolforge.execution.session import make_session, session_to_dict
from toolforge.generator.state import Conversation, ConversationState, _to_registry_id
from toolforge.graph.sampler import ChainSampler
from toolforge.registry.models import Endpoint

log = structlog.get_logger(__name__)


class ConversationGenerator:
    """Compiled LangGraph loop for single-conversation generation.

    Usage::

        gen = ConversationGenerator(sampler, registry, all_endpoints,
                                    executor, planner, user_sim, assistant, judge)
        final_state = gen.run(initial_state)
    """

    def __init__(
        self,
        sampler: ChainSampler,
        registry: dict[str, Endpoint],
        all_endpoints: list[Endpoint],
        executor: Executor,
        planner: Planner,
        user_sim: UserSimulator,
        assistant: Assistant,
        judge: Judge,
        *,
        repair_agent: RepairAgent | None = None,
    ) -> None:
        self._sampler = sampler
        self._registry = registry
        self._all_endpoints = all_endpoints
        self._executor = executor
        self._planner = planner
        self._user_sim = user_sim
        self._assistant = assistant
        self._judge = judge
        self._repair_agent = repair_agent
        self._graph = self._build_graph()

    def run(self, initial_state: ConversationState) -> ConversationState:
        """Run the graph from initial_state to completion and return the final state."""
        return self._graph.invoke(initial_state)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):  # type: ignore[return-value]
        workflow: StateGraph = StateGraph(ConversationState)

        workflow.add_node("plan", self._plan_node)
        workflow.add_node("user_turn", self._user_turn_node)
        workflow.add_node("assistant_turn", self._assistant_turn_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("judge", self._judge_node)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "user_turn")
        workflow.add_edge("user_turn", "assistant_turn")
        workflow.add_conditional_edges(
            "assistant_turn",
            _route_after_assistant,
            {"executor": "executor", "user_turn": "user_turn", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "executor",
            _route_after_executor,
            {"assistant_turn": "assistant_turn", "finalize": "finalize"},
        )
        workflow.add_edge("finalize", "judge")

        if self._repair_agent is not None:
            workflow.add_node("repair", self._repair_node)
            workflow.add_conditional_edges(
                "judge",
                _route_after_judge,
                {END: END, "repair": "repair"},
            )
            workflow.add_edge("repair", END)
        else:
            # No repair agent — unconditionally end after judge.
            workflow.add_edge("judge", END)

        return workflow.compile()

    # ------------------------------------------------------------------
    # Node functions (closures over self)
    # ------------------------------------------------------------------

    def _plan_node(self, state: ConversationState) -> dict[str, Any]:
        """Sample chain, look up Endpoints, call Planner, initialise SessionState."""
        result = self._sampler.sample(state["constraints"], seed=state["seed"])
        if result.truncated:
            raise RuntimeError(
                f"ChainSampler failed: {result.failure_reason.value}. "
                "No hardcoded-chain fallback exists — fix constraints or graph."
            )

        chain_ids = [_to_registry_id(nid) for nid in result.endpoint_ids]
        chain_endpoints = [self._registry[ep_id] for ep_id in chain_ids]

        plan = self._planner.plan(
            chain=chain_ids,
            persona_seed=state["seed"],
            chain_endpoints=chain_endpoints,
        )

        session = make_session(state["conversation_id"], state["seed"])
        session.private_user_knowledge = dict(plan.private_user_knowledge)

        log.info(
            "generator.plan_node",
            conversation_id=state["conversation_id"],
            chain=chain_ids,
            has_private_knowledge=bool(plan.private_user_knowledge),
        )
        return {
            "sampled_chain": chain_ids,
            "chain_endpoints": chain_endpoints,
            "plan": plan,
            "session_state": session,
            "messages": [],
            "turn_count": 0,
            "chain_index": 0,
            "last_assistant_turn": None,
        }

    def _user_turn_node(self, state: ConversationState) -> dict[str, Any]:
        """Invoke UserSimulator to produce the next user message."""
        user_text = self._user_sim.respond(state["plan"], state["messages"])
        new_messages = list(state["messages"]) + [
            {"role": "user", "content": user_text}
        ]
        log.debug("generator.user_turn", turn_count=state["turn_count"] + 1)
        return {
            "messages": new_messages,
            "turn_count": state["turn_count"] + 1,
        }

    def _assistant_turn_node(self, state: ConversationState) -> dict[str, Any]:
        """Invoke Assistant to produce the next assistant action."""
        distractors = select_distractors(
            chain_endpoints=state["chain_endpoints"],
            all_endpoints=state["all_endpoints"],
            seed=state["seed"] + state["turn_count"],
            n=3,
        )
        turn = self._assistant.act(
            history=state["messages"],
            session_state=state["session_state"],
            chain_endpoints=state["chain_endpoints"],
            distractors=distractors,
        )

        if turn.type == "message":
            content = turn.content
        else:
            content = f"[tool_call: {turn.endpoint}, args={json.dumps(turn.arguments, ensure_ascii=False)}]"

        new_messages = list(state["messages"]) + [
            {"role": "assistant", "content": content}
        ]
        log.debug("generator.assistant_turn", turn_type=turn.type,
                  endpoint=turn.endpoint if turn.type == "tool_call" else None)
        return {
            "messages": new_messages,
            "last_assistant_turn": turn,
        }

    def _executor_node(self, state: ConversationState) -> dict[str, Any]:
        """Execute the tool call, append the result, advance chain_index if on-chain."""
        turn = state["last_assistant_turn"]
        assert turn is not None and turn.type == "tool_call", (
            "executor_node reached with no tool_call turn — routing bug"
        )

        # Executor mutates session_state in-place (appends to tool_outputs,
        # updates available_values_by_type).
        output = self._executor.execute(turn.endpoint, turn.arguments, state["session_state"])

        result_payload: dict[str, Any] = output.response if output.response is not None else {"error": output.error}
        result_content = f"[tool_result: {json.dumps(result_payload, ensure_ascii=False)}]"
        new_messages = list(state["messages"]) + [
            {"role": "user", "content": result_content}
        ]

        # Advance chain_index only for on-chain, in-order, successful calls.
        current_index = state["chain_index"]
        sampled = state["sampled_chain"]
        expected = sampled[current_index] if current_index < len(sampled) else None

        if output.is_success() and turn.endpoint == expected:
            new_index = current_index + 1
            log.debug("generator.chain_advance", endpoint=turn.endpoint,
                      chain_index=new_index, chain_len=len(sampled))
        elif output.is_success() and turn.endpoint != expected:
            new_index = current_index
            log.warning(
                "generator.off_chain_call",
                endpoint_id=turn.endpoint,
                expected=expected,
                chain_index=current_index,
            )
        else:
            # Execution failed — no progress.
            new_index = current_index
            log.debug("generator.executor_failure", endpoint=turn.endpoint,
                      error=output.error)

        return {
            "messages": new_messages,
            "chain_index": new_index,
        }

    def _finalize_node(self, state: ConversationState) -> dict[str, Any]:  # noqa: ARG002
        """Mark the conversation ready for judging."""
        log.info(
            "generator.finalize",
            conversation_id=state["conversation_id"],
            turn_count=state["turn_count"],
            chain_index=state["chain_index"],
            chain_len=len(state["sampled_chain"]),
        )
        return {"status": "done"}

    def _judge_node(self, state: ConversationState) -> dict[str, Any]:
        """Score the completed conversation."""
        result = self._judge.score(state["messages"], state["sampled_chain"])
        status = "done" if result.overall_pass else "failed"
        log.info(
            "generator.judge",
            conversation_id=state["conversation_id"],
            mean_score=result.mean_score(),
            overall_pass=result.overall_pass,
            status=status,
        )
        return {
            "judge_result": result,
            "status": status,
        }

    def _repair_node(self, state: ConversationState) -> dict[str, Any]:
        """Apply the F5.2 repair loop to a failed conversation.

        Validates, calls RepairAgent, applies edits, re-validates, and
        re-judges — all inside run_repair().  The graph routes here only
        when self._repair_agent is not None and status == "failed".
        """
        from toolforge.evaluation.repair import run_repair  # local to avoid circular import at module level

        assert self._repair_agent is not None  # guard — only wired when set

        conv = Conversation(
            conversation_id=state["conversation_id"],
            seed=state["seed"],
            sampled_chain=state["sampled_chain"],
            messages=state["messages"],
            session_summary=session_to_dict(state["session_state"]),
            judge_result=state["judge_result"],
            status=state["status"],
        )

        repaired_conv, judge_result, attempts_used = run_repair(
            conv=conv,
            state=state["session_state"],
            repair_agent=self._repair_agent,
            judge=self._judge,
        )

        new_status = "done" if (judge_result is not None and judge_result.overall_pass) else "failed"
        log.info(
            "generator.repair",
            conversation_id=state["conversation_id"],
            attempts_used=attempts_used,
            new_status=new_status,
        )
        return {
            "messages": repaired_conv.messages,
            "judge_result": judge_result,
            "repair_attempts": state["repair_attempts"] + attempts_used,
            "status": new_status,
        }


# ---------------------------------------------------------------------------
# Routing functions (pure functions — independently testable)
# ---------------------------------------------------------------------------

def _route_after_assistant(state: ConversationState) -> str:
    """Decide next node after assistant_turn_node.

    Priority:
    1. Hard turn cap (12) → finalize regardless of turn type.
    2. tool_call → executor.
    3. message + chain complete (chain_index ≥ len(sampled_chain)) → finalize.
    4. message + chain incomplete → user_turn (clarification exchange).
    """
    if state["turn_count"] >= 12:
        return "finalize"
    turn = state["last_assistant_turn"]
    if turn is not None and turn.type == "tool_call":
        return "executor"
    if state["chain_index"] >= len(state["sampled_chain"]):
        return "finalize"
    return "user_turn"


def _route_after_executor(state: ConversationState) -> str:
    """Decide next node after executor_node.

    Routes to assistant_turn so the assistant can synthesise the result
    or call the next chain step.  Routes to finalize only on the hard cap.
    """
    if state["turn_count"] >= 12:
        return "finalize"
    return "assistant_turn"


def _route_after_judge(state: ConversationState) -> str:
    """Route to repair if the conversation failed; otherwise END.

    run_repair() owns the max-attempts cap — this router only checks status.
    This function is only registered as a conditional edge when repair_agent
    is not None (see _build_graph), so returning "repair" is always safe.
    """
    if state["status"] == "failed":
        return "repair"
    return END
