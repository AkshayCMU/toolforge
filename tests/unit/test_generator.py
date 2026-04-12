"""Unit tests for F4.6 — LangGraph conversation generator.

Coverage:
  - _to_registry_id normalization helper (conditional strip)
  - All node functions (called directly on ConversationGenerator instances)
  - Routing functions
  - Sampler dependency proof (mocked sampler → RuntimeError; planner not called)

No live LLM calls.  All agents are either MagicMocked or use dry_run+cache.
The Executor tests use a real Executor with a minimal in-process registry to
exercise the chain_index logic without mocking the executor internals.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from toolforge.agents.assistant import AssistantTurn
from toolforge.agents.judge import DimensionScore, JudgeResult
from toolforge.agents.planner import TaskPlan
from toolforge.execution.executor import Executor
from toolforge.execution.mock_responder import MockResponder
from toolforge.execution.session import SessionState, ToolOutput, make_session
from toolforge.generator.graph import (
    ConversationGenerator,
    _route_after_assistant,
    _route_after_executor,
)
from toolforge.generator.loop import _state_to_conversation, generate_one
from toolforge.generator.state import ConversationState, _to_registry_id
from toolforge.graph.sampler import ChainConstraints, ChainResult, FailureReason
from toolforge.registry.models import Endpoint, Parameter, ParamProvenance, ResponseField


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _prov() -> ParamProvenance:
    return ParamProvenance(raw_required_field="required_parameters", raw_type_string="STRING")


def _ep(ep_id: str, params: tuple[Parameter, ...] = (),
        resp: tuple[ResponseField, ...] = ()) -> Endpoint:
    return Endpoint(
        id=ep_id, name=ep_id.split("/")[-1], description=f"Endpoint {ep_id}",
        parameters=params, response_schema=resp,
        mock_policy="schema",
    )


def _param(name: str, required: bool = True) -> Parameter:
    return Parameter(name=name, type="string", description=f"param {name}",
                     required=required, provenance=_prov())


def _resp_field(path: str) -> ResponseField:
    return ResponseField(path=path, type="string")


def _task_plan(**kwargs) -> TaskPlan:
    defaults = dict(
        user_persona="test user",
        initial_query="do the thing",
        clarification_points=[],
        expected_final_outcome="done",
        chain_rationale="test chain",
        private_user_knowledge={},
    )
    defaults.update(kwargs)
    return TaskPlan(**defaults)


def _judge_result(pass_: bool = True) -> JudgeResult:
    score = 5 if pass_ else 1
    return JudgeResult(
        naturalness=DimensionScore(score=score, rationale="ok"),
        tool_correctness=DimensionScore(score=score, rationale="ok"),
        chain_coherence=DimensionScore(score=score, rationale="ok"),
        task_completion=DimensionScore(score=score, rationale="ok"),
        failure_modes=[],
        overall_pass=pass_,
    )


def _make_gen(
    sampler=None,
    registry: dict | None = None,
    all_endpoints: list | None = None,
    executor=None,
    planner=None,
    user_sim=None,
    assistant=None,
    judge=None,
) -> ConversationGenerator:
    """Build a ConversationGenerator with mock defaults for unspecified deps."""
    return ConversationGenerator(
        sampler=sampler or MagicMock(),
        registry=registry or {},
        all_endpoints=all_endpoints or [],
        executor=executor or MagicMock(),
        planner=planner or MagicMock(),
        user_sim=user_sim or MagicMock(),
        assistant=assistant or MagicMock(),
        judge=judge or MagicMock(),
    )


def _base_state(**overrides) -> ConversationState:
    """Construct a minimal ConversationState for testing."""
    ep1 = _ep("Cat/Tool/ep1", params=(_param("q"),), resp=(_resp_field("result"),))
    ep2 = _ep("Cat/Tool/ep2", params=(_param("q"),), resp=(_resp_field("result"),))
    session = make_session("conv-test", 42)
    state: dict = dict(
        conversation_id="conv-test",
        seed=42,
        constraints=ChainConstraints(length=2),
        sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
        chain_endpoints=[ep1, ep2],
        all_endpoints=[ep1, ep2],
        plan=_task_plan(),
        messages=[],
        session_state=session,
        turn_count=0,
        chain_index=0,
        last_assistant_turn=None,
        judge_result=None,
        repair_attempts=0,
        status="running",
    )
    state.update(overrides)
    return state  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# 1. Endpoint-ID normalisation helper
# ---------------------------------------------------------------------------

def test_to_registry_id_strips_ep_prefix() -> None:
    assert _to_registry_id("ep:Travel/Hotels/createBooking") == "Travel/Hotels/createBooking"


def test_to_registry_id_noop_without_prefix() -> None:
    assert _to_registry_id("Travel/Hotels/createBooking") == "Travel/Hotels/createBooking"


def test_to_registry_id_partial_prefix_untouched() -> None:
    """Only exact "ep:" prefix is stripped; other prefixes are preserved."""
    assert _to_registry_id("epx:foo") == "epx:foo"
    assert _to_registry_id(":foo") == ":foo"


# ---------------------------------------------------------------------------
# 2. plan_node
# ---------------------------------------------------------------------------

class TestPlanNode:
    def _setup(self):
        ep1 = _ep("Cat/Tool/ep1", params=(_param("q"),))
        ep2 = _ep("Cat/Tool/ep2", params=(_param("q"),))
        registry = {"Cat/Tool/ep1": ep1, "Cat/Tool/ep2": ep2}

        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = ChainResult(
            endpoint_ids=["ep:Cat/Tool/ep1", "ep:Cat/Tool/ep2"],
            truncated=False,
            failure_reason=FailureReason.NONE,
            seed_endpoint_id="ep:Cat/Tool/ep1",
        )
        mock_planner = MagicMock()
        mock_planner.plan.return_value = _task_plan(private_user_knowledge={"city": "Paris"})

        gen = _make_gen(sampler=mock_sampler, registry=registry,
                        all_endpoints=[ep1, ep2], planner=mock_planner)
        state = _base_state(sampled_chain=[], chain_endpoints=[], messages=[],
                            turn_count=5, chain_index=99)
        return gen, state, mock_sampler, mock_planner

    def test_strips_ep_prefix_from_sampled_chain(self) -> None:
        gen, state, _, _ = self._setup()
        result = gen._plan_node(state)
        assert result["sampled_chain"] == ["Cat/Tool/ep1", "Cat/Tool/ep2"]
        assert all("ep:" not in ep_id for ep_id in result["sampled_chain"])

    def test_passes_endpoint_objects_to_planner(self) -> None:
        gen, state, _, mock_planner = self._setup()
        gen._plan_node(state)
        call_kwargs = mock_planner.plan.call_args
        chain_endpoints = call_kwargs.kwargs.get("chain_endpoints") or call_kwargs[1].get("chain_endpoints")
        assert chain_endpoints is not None
        assert all(isinstance(ep, Endpoint) for ep in chain_endpoints)

    def test_initializes_chain_index_zero(self) -> None:
        gen, state, _, _ = self._setup()
        result = gen._plan_node(state)
        assert result["chain_index"] == 0

    def test_initializes_turn_count_zero(self) -> None:
        gen, state, _, _ = self._setup()
        result = gen._plan_node(state)
        assert result["turn_count"] == 0

    def test_populates_private_user_knowledge_in_session(self) -> None:
        gen, state, _, _ = self._setup()
        result = gen._plan_node(state)
        assert result["session_state"].private_user_knowledge == {"city": "Paris"}

    def test_truncated_sampler_raises_without_fallback(self) -> None:
        mock_sampler = MagicMock()
        mock_sampler.sample.return_value = ChainResult(
            endpoint_ids=[], truncated=True, failure_reason=FailureReason.UNSATISFIABLE
        )
        mock_planner = MagicMock()
        gen = _make_gen(sampler=mock_sampler, planner=mock_planner)

        with pytest.raises(RuntimeError, match="ChainSampler failed"):
            gen._plan_node(_base_state())

        mock_planner.plan.assert_not_called()


# ---------------------------------------------------------------------------
# 3. user_turn_node
# ---------------------------------------------------------------------------

class TestUserTurnNode:
    def test_appends_user_role_message(self) -> None:
        mock_sim = MagicMock()
        mock_sim.respond.return_value = "I need a hotel please."
        gen = _make_gen(user_sim=mock_sim)
        state = _base_state(messages=[], turn_count=0)

        result = gen._user_turn_node(state)

        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "I need a hotel please."

    def test_increments_turn_count(self) -> None:
        mock_sim = MagicMock()
        mock_sim.respond.return_value = "hello"
        gen = _make_gen(user_sim=mock_sim)
        state = _base_state(messages=[], turn_count=3)

        result = gen._user_turn_node(state)
        assert result["turn_count"] == 4

    def test_preserves_prior_messages(self) -> None:
        mock_sim = MagicMock()
        mock_sim.respond.return_value = "next message"
        gen = _make_gen(user_sim=mock_sim)
        prior = [{"role": "user", "content": "first"}, {"role": "assistant", "content": "reply"}]
        state = _base_state(messages=prior, turn_count=1)

        result = gen._user_turn_node(state)
        assert len(result["messages"]) == 3
        assert result["messages"][0]["content"] == "first"


# ---------------------------------------------------------------------------
# 4. assistant_turn_node
# ---------------------------------------------------------------------------

class TestAssistantTurnNode:
    def test_message_turn_serialised_correctly(self) -> None:
        mock_asst = MagicMock()
        mock_asst.act.return_value = AssistantTurn(type="message", content="Sure, let me check.")
        gen = _make_gen(assistant=mock_asst)
        state = _base_state(messages=[{"role": "user", "content": "hi"}])

        result = gen._assistant_turn_node(state)

        last = result["messages"][-1]
        assert last["role"] == "assistant"
        assert last["content"] == "Sure, let me check."
        assert result["last_assistant_turn"].type == "message"

    def test_tool_call_turn_serialised_correctly(self) -> None:
        mock_asst = MagicMock()
        mock_asst.act.return_value = AssistantTurn(
            type="tool_call", endpoint="Cat/Tool/ep1", arguments={"q": "Paris"}
        )
        gen = _make_gen(assistant=mock_asst)
        state = _base_state(messages=[{"role": "user", "content": "go"}])

        result = gen._assistant_turn_node(state)

        last = result["messages"][-1]
        assert last["role"] == "assistant"
        assert last["content"].startswith("[tool_call: Cat/Tool/ep1,")
        assert result["last_assistant_turn"].type == "tool_call"

    def test_last_assistant_turn_set_in_result(self) -> None:
        mock_asst = MagicMock()
        turn = AssistantTurn(type="message", content="done")
        mock_asst.act.return_value = turn
        gen = _make_gen(assistant=mock_asst)

        result = gen._assistant_turn_node(_base_state())
        assert result["last_assistant_turn"] is turn


# ---------------------------------------------------------------------------
# 5. executor_node — chain_index logic
# ---------------------------------------------------------------------------

class TestExecutorNode:
    """Uses a real Executor with a minimal registry + MockResponder."""

    def _make_executor_and_registry(self):
        ep1 = _ep("Cat/Tool/ep1", params=(_param("q"),), resp=(_resp_field("result"),))
        ep2 = _ep("Cat/Tool/ep2", params=(_param("q"),), resp=(_resp_field("result"),))
        ep_distractor = _ep("Other/Tool/distractor", params=(_param("q"),),
                            resp=(_resp_field("x"),))
        registry = {
            "Cat/Tool/ep1": ep1,
            "Cat/Tool/ep2": ep2,
            "Other/Tool/distractor": ep_distractor,
        }
        executor = Executor(registry, MockResponder())
        return executor, registry, ep1, ep2, ep_distractor

    def test_appends_tool_result_as_user_message(self) -> None:
        executor, registry, _, _, _ = self._make_executor_and_registry()
        gen = _make_gen(executor=executor, registry=registry)
        turn = AssistantTurn(type="tool_call", endpoint="Cat/Tool/ep1",
                             arguments={"q": "test"})
        state = _base_state(
            sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
            chain_index=0,
            last_assistant_turn=turn,
            messages=[],
        )

        result = gen._executor_node(state)

        assert result["messages"][-1]["role"] == "user"
        assert result["messages"][-1]["content"].startswith("[tool_result:")

    def test_increments_chain_index_on_on_chain_success(self) -> None:
        executor, registry, _, _, _ = self._make_executor_and_registry()
        gen = _make_gen(executor=executor, registry=registry)
        turn = AssistantTurn(type="tool_call", endpoint="Cat/Tool/ep1",
                             arguments={"q": "test"})
        state = _base_state(
            sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
            chain_index=0,
            last_assistant_turn=turn,
            messages=[],
        )

        result = gen._executor_node(state)
        assert result["chain_index"] == 1

    def test_no_increment_on_executor_failure(self) -> None:
        """Grounding failure on ep2 (missing required param) → chain_index unchanged."""
        executor, registry, _, _, _ = self._make_executor_and_registry()
        gen = _make_gen(executor=executor, registry=registry)
        # Omit required "q" → structural validation fails
        turn = AssistantTurn(type="tool_call", endpoint="Cat/Tool/ep1",
                             arguments={})
        state = _base_state(
            sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
            chain_index=0,
            last_assistant_turn=turn,
            messages=[],
        )

        result = gen._executor_node(state)
        assert result["chain_index"] == 0

    def test_no_increment_on_distractor_call(self) -> None:
        """Successful call to a distractor endpoint must NOT advance chain_index.
        Completing the distractor must not trigger chain completion."""
        executor, registry, _, _, _ = self._make_executor_and_registry()
        gen = _make_gen(executor=executor, registry=registry)
        # Call the distractor (not in sampled_chain) successfully
        turn = AssistantTurn(type="tool_call", endpoint="Other/Tool/distractor",
                             arguments={"q": "anything"})
        state = _base_state(
            sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
            chain_index=0,
            last_assistant_turn=turn,
            messages=[],
        )

        result = gen._executor_node(state)

        # chain_index must NOT advance
        assert result["chain_index"] == 0
        # chain is NOT complete (0 < 2)
        assert result["chain_index"] < len(state["sampled_chain"])

    def test_no_increment_on_out_of_order_call(self) -> None:
        """Successful call to sampled_chain[1] when chain_index=0 must not advance.
        The expected next step is sampled_chain[0]; ep2 is out-of-order."""
        executor, registry, _, _, _ = self._make_executor_and_registry()
        gen = _make_gen(executor=executor, registry=registry)
        # ep2 = sampled_chain[1], but chain_index=0 expects ep1
        turn = AssistantTurn(type="tool_call", endpoint="Cat/Tool/ep2",
                             arguments={"q": "test"})
        state = _base_state(
            sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
            chain_index=0,
            last_assistant_turn=turn,
            messages=[],
        )

        result = gen._executor_node(state)

        assert result["chain_index"] == 0  # NOT advanced despite success


# ---------------------------------------------------------------------------
# 6. Routing functions
# ---------------------------------------------------------------------------

class TestRoutingAfterAssistant:
    def test_tool_call_routes_to_executor(self) -> None:
        turn = AssistantTurn(type="tool_call", endpoint="Cat/Tool/ep1", arguments={})
        state = _base_state(turn_count=0, chain_index=0, last_assistant_turn=turn)
        assert _route_after_assistant(state) == "executor"

    def test_message_chain_complete_routes_to_finalize(self) -> None:
        turn = AssistantTurn(type="message", content="All done!")
        state = _base_state(
            turn_count=3,
            sampled_chain=["Cat/Tool/ep1"],
            chain_index=1,      # 1 >= len(["Cat/Tool/ep1"]) → complete
            last_assistant_turn=turn,
        )
        assert _route_after_assistant(state) == "finalize"

    def test_message_chain_incomplete_routes_to_user_turn(self) -> None:
        turn = AssistantTurn(type="message", content="Which city?")
        state = _base_state(
            turn_count=1,
            sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
            chain_index=0,      # 0 < 2 → not complete
            last_assistant_turn=turn,
        )
        assert _route_after_assistant(state) == "user_turn"

    def test_hard_turn_cap_overrides_tool_call(self) -> None:
        turn = AssistantTurn(type="tool_call", endpoint="Cat/Tool/ep1", arguments={})
        state = _base_state(turn_count=12, chain_index=0, last_assistant_turn=turn)
        assert _route_after_assistant(state) == "finalize"

    def test_hard_turn_cap_overrides_message(self) -> None:
        turn = AssistantTurn(type="message", content="still going")
        state = _base_state(turn_count=12, chain_index=0, last_assistant_turn=turn,
                            sampled_chain=["a", "b"])
        assert _route_after_assistant(state) == "finalize"


class TestRoutingAfterExecutor:
    def test_routes_to_assistant_turn_normally(self) -> None:
        state = _base_state(turn_count=3)
        assert _route_after_executor(state) == "assistant_turn"

    def test_hard_turn_cap_routes_to_finalize(self) -> None:
        state = _base_state(turn_count=12)
        assert _route_after_executor(state) == "finalize"


# ---------------------------------------------------------------------------
# 7. Sampler dependency proof
# ---------------------------------------------------------------------------

def test_sampler_dependency_enforced() -> None:
    """Truncated sampler → RuntimeError. Planner must NOT have been called.
    Proves the generator has no hardcoded-chain fallback."""
    mock_sampler = MagicMock()
    mock_sampler.sample.return_value = ChainResult(
        endpoint_ids=[], truncated=True, failure_reason=FailureReason.UNSATISFIABLE
    )
    mock_planner = MagicMock()
    gen = _make_gen(sampler=mock_sampler, planner=mock_planner)

    with pytest.raises(RuntimeError, match="ChainSampler failed"):
        generate_one(
            seed=42,
            constraints=ChainConstraints(length=2),
            generator=gen,
        )

    mock_sampler.sample.assert_called_once()   # sampler was called
    mock_planner.plan.assert_not_called()       # no hardcoded-chain fallback reached planner


# ---------------------------------------------------------------------------
# 8. _state_to_conversation helper
# ---------------------------------------------------------------------------

def test_state_to_conversation_no_ep_prefix() -> None:
    """sampled_chain in the output must contain bare IDs (no 'ep:' prefix)."""
    state = _base_state(
        sampled_chain=["Cat/Tool/ep1", "Cat/Tool/ep2"],
        judge_result=_judge_result(pass_=True),
        status="done",
    )
    conv = _state_to_conversation(state)
    assert all("ep:" not in ep_id for ep_id in conv.sampled_chain)


def test_state_to_conversation_includes_session_summary() -> None:
    state = _base_state(judge_result=None, status="failed")
    conv = _state_to_conversation(state)
    assert "tool_outputs" in conv.session_summary
    assert "conversation_id" in conv.session_summary
