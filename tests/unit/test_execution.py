"""Unit tests for the Phase 3 execution layer (F3.1, F3.2, F3.3)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from toolforge.execution.executor import Executor
from toolforge.execution.mock_responder import MockResponder, _is_creation_endpoint
from toolforge.execution.session import (
    SessionState,
    ToolOutput,
    make_session,
    session_to_dict,
)
from toolforge.registry.models import (
    Endpoint,
    Parameter,
    ParamProvenance,
    ResponseField,
    Tool,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _prov() -> ParamProvenance:
    return ParamProvenance(raw_required_field="required_parameters", raw_type_string="STRING")


def _param(
    name: str,
    required: bool = True,
    semantic_type: str | None = None,
    type_: str = "string",
) -> Parameter:
    return Parameter(
        name=name,
        type=type_,
        description=f"Param {name}",
        required=required,
        semantic_type=semantic_type,
        provenance=_prov(),
    )


def _field(path: str, semantic_type: str | None = None, type_: str = "string") -> ResponseField:
    return ResponseField(path=path, type=type_, semantic_type=semantic_type)


def _endpoint(
    ep_id: str,
    name: str = "",
    params: tuple[Parameter, ...] = (),
    fields: tuple[ResponseField, ...] = (),
    mock_policy: str | None = "schema",
) -> Endpoint:
    return Endpoint(
        id=ep_id,
        name=name or ep_id.split("/")[-1],
        description="test endpoint",
        parameters=params,
        response_schema=fields,
        mock_policy=mock_policy,
    )


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def responder() -> MockResponder:
    return MockResponder()


@pytest.fixture()
def fresh_state() -> SessionState:
    return make_session("conv-test", seed=42)


# A simple 3-endpoint registry for F3.3 tests:
#   ep1 "createBooking" — requires city_name (USER_PROVIDED), produces booking_id (CHAIN_ONLY)
#   ep2 "getBookingDetails" — requires booking_id (CHAIN_ONLY), produces some string fields
#   ep3 "updateBooking" — requires booking_id (CHAIN_ONLY), produces a boolean

EP1 = _endpoint(
    "Travel/Hotels/createBooking",
    name="createBooking",
    params=(_param("city_name", required=True, semantic_type="city_name"),),
    fields=(_field("booking_id", semantic_type="booking_id"),),
    mock_policy="schema",
)

EP2 = _endpoint(
    "Travel/Hotels/fetchBookingDetails",
    name="fetchBookingDetails",
    params=(_param("booking_id", required=True, semantic_type="booking_id"),),
    fields=(
        _field("hotel_name"),
        _field("check_in_date"),
    ),
    mock_policy="schema",
)

EP3 = _endpoint(
    "Travel/Hotels/updateBooking",
    name="updateBooking",
    params=(
        _param("booking_id", required=True, semantic_type="booking_id"),
        _param("notes", required=False),
    ),
    fields=(_field("success", type_="boolean"),),
    mock_policy="llm",
)

# Registry used by executor tests.
REGISTRY: dict[str, Endpoint] = {ep.id: ep for ep in (EP1, EP2, EP3)}

# A tiny chain_only_types set for tests — just booking_id.
CHAIN_ONLY_TEST: frozenset[str] = frozenset({"booking_id"})


@pytest.fixture()
def executor() -> Executor:
    return Executor(REGISTRY, MockResponder(), chain_only_types=CHAIN_ONLY_TEST)


# ===========================================================================
# F3.1 — SessionState + ToolOutput
# ===========================================================================

class TestSessionState:
    def test_construction(self, fresh_state: SessionState):
        """make_session() initialises all fields correctly."""
        s = fresh_state
        assert s.conversation_id == "conv-test"
        assert s.seed == 42
        assert s.available_values_by_type == {}
        assert s.resolved_entities == {}
        assert s.created_entities == []
        assert s.tool_outputs == []
        assert s.private_user_knowledge == {}

    def test_state_append(self, fresh_state: SessionState):
        """Manually appending a ToolOutput grows tool_outputs."""
        output = ToolOutput(
            endpoint_id="ep1",
            arguments={},
            response={"x": 1},
            error=None,
            timestamp="turn-0",
        )
        fresh_state.tool_outputs.append(output)
        assert len(fresh_state.tool_outputs) == 1
        assert fresh_state.tool_outputs[0].endpoint_id == "ep1"


class TestToolOutput:
    def test_to_dict_json_serializable(self):
        """to_dict() must round-trip through json.dumps without error."""
        output = ToolOutput(
            endpoint_id="Travel/Hotels/searchHotels",
            arguments={"city_name": "Paris"},
            response={"hotel_id": "abc-123", "_mock_policy": "schema"},
            error=None,
            timestamp="turn-0",
        )
        serialised = json.dumps(output.to_dict())
        assert isinstance(serialised, str)
        restored = json.loads(serialised)
        assert restored["endpoint_id"] == "Travel/Hotels/searchHotels"
        assert restored["timestamp"] == "turn-0"

    def test_error_output_json_serializable(self):
        """Error ToolOutputs (response=None) also round-trip cleanly."""
        output = ToolOutput(
            endpoint_id="ep1",
            arguments={"booking_id": "fake"},
            response=None,
            error="Invalid booking_id: 'fake' not in session. Valid values: []",
            timestamp="turn-0",
        )
        serialised = json.dumps(output.to_dict())
        restored = json.loads(serialised)
        assert restored["response"] is None
        assert "not in session" in restored["error"]

    def test_is_success(self):
        ok = ToolOutput("ep1", {}, {"x": 1}, None, "turn-0")
        err = ToolOutput("ep1", {}, None, "bad", "turn-0")
        assert ok.is_success()
        assert not err.is_success()

    def test_to_dict_does_not_contain_resolved_entities(self):
        """to_dict() serialises only ToolOutput fields; no SessionState bleed."""
        output = ToolOutput("ep1", {}, {}, None, "turn-0")
        d = output.to_dict()
        assert "resolved_entities" not in d
        assert set(d.keys()) == {"endpoint_id", "arguments", "response", "error", "timestamp"}


class TestSessionToDict:
    def test_tuple_keys_converted(self):
        """session_to_dict() converts resolved_entities tuple keys to strings."""
        state = make_session("c1", 1)
        state.resolved_entities[("booking_id", "bk-001")] = {"hotel": "Ritz"}
        d = session_to_dict(state)
        assert "booking_id:bk-001" in d["resolved_entities"]
        assert d["resolved_entities"]["booking_id:bk-001"] == {"hotel": "Ritz"}

    def test_session_to_dict_json_serializable(self):
        """Full session dict round-trips through json.dumps."""
        state = make_session("c2", 99)
        state.resolved_entities[("user_id", "u-99")] = {"name": "Alice"}
        state.tool_outputs.append(
            ToolOutput("ep1", {}, {"x": 1}, None, "turn-0")
        )
        serialised = json.dumps(session_to_dict(state))
        assert isinstance(serialised, str)


# ===========================================================================
# F3.2 — MockResponder
# ===========================================================================

class TestMockResponder:
    def test_deterministic_across_fresh_sessions(self, responder: MockResponder):
        """Same seed → same response across two independent fresh sessions."""
        ep = _endpoint(
            "Cat/Tool/searchItems",
            name="searchItems",
            params=(_param("query", semantic_type="search_query"),),
            fields=(_field("name"), _field("price", type_="number")),
            mock_policy="schema",
        )
        s1 = make_session("c1", seed=42)
        s2 = make_session("c2", seed=42)
        r1 = responder.respond(ep, {"query": "shoes"}, s1)
        r2 = responder.respond(ep, {"query": "shoes"}, s2)
        # Remove _mock_policy for comparison (it's always the same anyway).
        r1.pop("_mock_policy")
        r2.pop("_mock_policy")
        assert r1 == r2

    def test_registers_chain_only_values_on_creation(self, responder: MockResponder, fresh_state: SessionState):
        """Creation endpoint: booking_id produced → registered in state pool."""
        ep = _endpoint(
            "Travel/Hotels/createBooking",
            name="createBooking",
            fields=(_field("booking_id", semantic_type="booking_id"),),
        )
        responder.respond(ep, {}, fresh_state)
        assert "booking_id" in fresh_state.available_values_by_type
        assert len(fresh_state.available_values_by_type["booking_id"]) == 1

    def test_reuses_most_recent_pool_value(self, responder: MockResponder, fresh_state: SessionState):
        """Non-creation endpoint: reuses pool[-1] when pool has multiple entries."""
        fresh_state.available_values_by_type["booking_id"] = ["bk-001", "bk-002"]
        ep = _endpoint(
            "Travel/Hotels/lookupDetails",
            name="lookupDetails",  # no creation verb substring
            fields=(_field("booking_id", semantic_type="booking_id"),),
        )
        response = responder.respond(ep, {}, fresh_state)
        assert response["booking_id"] == "bk-002"
        # Pool should be unchanged (no new value added).
        assert fresh_state.available_values_by_type["booking_id"] == ["bk-001", "bk-002"]

    def test_generates_fresh_when_pool_empty(self, responder: MockResponder, fresh_state: SessionState):
        """Non-creation endpoint with empty pool: generates fresh, appends to pool."""
        ep = _endpoint(
            "Travel/Hotels/fetchBookingDetails",
            name="fetchBookingDetails",
            fields=(_field("booking_id", semantic_type="booking_id"),),
        )
        response = responder.respond(ep, {}, fresh_state)
        pool = fresh_state.available_values_by_type.get("booking_id", [])
        assert len(pool) == 1
        assert response["booking_id"] == pool[0]

    def test_all_mock_policies_produce_response_with_policy_key(self, responder: MockResponder):
        """static, schema, and llm mock_policy endpoints all produce a response."""
        fields = (_field("result"),)
        for policy in ("static", "schema", "llm"):
            ep = _endpoint(f"Cat/Tool/ep_{policy}", fields=fields, mock_policy=policy)
            state = make_session(f"conv-{policy}", seed=7)
            response = responder.respond(ep, {}, state)
            assert isinstance(response, dict)
            assert response["_mock_policy"] == policy

    def test_path_flattening_last_segment(self, responder: MockResponder, fresh_state: SessionState):
        """response_schema paths are flattened to last segment as key."""
        ep = _endpoint(
            "Cat/Tool/ep",
            fields=(
                _field("results[].hotel_id"),
                _field("data.user.email"),
            ),
        )
        response = responder.respond(ep, {}, fresh_state)
        assert "hotel_id" in response
        assert "email" in response

    def test_path_collision_uses_full_path(self, responder: MockResponder, fresh_state: SessionState):
        """Two paths with same last segment → both use full path as key."""
        ep = _endpoint(
            "Cat/Tool/ep",
            fields=(
                _field("a.id"),
                _field("b.id"),
            ),
        )
        response = responder.respond(ep, {}, fresh_state)
        assert "a.id" in response
        assert "b.id" in response
        assert "id" not in response  # bare 'id' must not appear


class TestCreationHeuristic:
    def test_creation_verbs_detected(self):
        for name in ("createBooking", "submitRequest", "placeOrder",
                     "registerAccount", "addUser", "insertRecord"):
            assert _is_creation_endpoint(name), f"Expected creation: {name}"

    def test_lookup_verbs_not_creation(self):
        # Names with no creation-verb substring.
        for name in ("searchHotels", "fetchDetails", "retrieveUser", "queryItems"):
            assert not _is_creation_endpoint(name), f"Expected non-creation: {name}"

    def test_snake_case_creation(self):
        assert _is_creation_endpoint("create_booking")
        assert _is_creation_endpoint("submit_request")
        assert not _is_creation_endpoint("fetch_details")

    def test_empty_name(self):
        assert not _is_creation_endpoint("")

    def test_creation_verb_across_naming_styles(self):
        """Creation verbs detected in camelCase, PascalCase, snake_case, ALL_CAPS."""
        # camelCase
        assert _is_creation_endpoint("createBooking")
        assert _is_creation_endpoint("submitOrder")
        # PascalCase (previously broken: first-word split yielded empty string)
        assert _is_creation_endpoint("CreateBooking")
        assert _is_creation_endpoint("SubmitOrder")
        # snake_case
        assert _is_creation_endpoint("create_booking")
        assert _is_creation_endpoint("place_order")
        # SCREAMING_SNAKE
        assert _is_creation_endpoint("CREATE_BOOKING")
        assert _is_creation_endpoint("PLACE_ORDER")
        # ALL_CAPS no underscore (previously broken)
        assert _is_creation_endpoint("CREATEBOOKING")
        # Non-creation across styles
        assert not _is_creation_endpoint("fetchDetails")
        assert not _is_creation_endpoint("FetchDetails")
        assert not _is_creation_endpoint("fetch_details")
        assert not _is_creation_endpoint("FETCH_DETAILS")


# ===========================================================================
# F3.3 — Executor with grounding invariant
# ===========================================================================

class TestExecutor:
    def test_unknown_endpoint_returns_error(self, executor: Executor, fresh_state: SessionState):
        """Unknown endpoint_id -> structured error appended to state."""
        output = executor.execute("NonExistent/ep", {}, fresh_state)
        assert output.error is not None
        assert "Unknown endpoint" in output.error
        assert len(fresh_state.tool_outputs) == 1
        assert fresh_state.tool_outputs[0].response is None

    def test_missing_required_param_returns_error(self, executor: Executor, fresh_state: SessionState):
        """Missing required param -> structural error appended to state."""
        output = executor.execute(EP1.id, {}, fresh_state)
        assert output.error is not None
        assert "Missing required parameter" in output.error
        assert "city_name" in output.error
        assert len(fresh_state.tool_outputs) == 1

    def test_required_param_none_value_is_missing(self, executor: Executor, fresh_state: SessionState):
        """Required param present but value=None -> treated as missing."""
        output = executor.execute(EP1.id, {"city_name": None}, fresh_state)
        assert output.error is not None
        assert "Missing required parameter" in output.error
        assert "city_name" in output.error

    def test_grounding_rejects_hallucinated_chain_only_id(self, executor: Executor, fresh_state: SessionState):
        """CHAIN_ONLY param with value not in pool -> grounding error appended to state."""
        output = executor.execute(EP2.id, {"booking_id": "fake-bk-999"}, fresh_state)
        assert output.error is not None
        assert "not in session" in output.error
        assert "[]" in output.error  # valid values listed (empty pool)
        assert len(fresh_state.tool_outputs) == 1

    def test_first_call_user_provided_params_succeeds(self, executor: Executor, fresh_state: SessionState):
        """Mandatory: fresh state + USER_PROVIDED-only params → first call succeeds.

        city_name is NOT in CHAIN_ONLY_TEST, so no grounding check is applied.
        This test guards against the 'CHAIN_ONLY_TYPES too strict' failure mode
        from FEATURES.md F3.3.
        """
        output = executor.execute(EP1.id, {"city_name": "Paris"}, fresh_state)
        assert output.error is None, f"First call failed: {output.error}"
        assert output.response is not None
        assert len(fresh_state.tool_outputs) == 1
        assert fresh_state.tool_outputs[0].timestamp == "turn-0"

    def test_chain_only_in_pool_accepted(self, executor: Executor, fresh_state: SessionState):
        """CHAIN_ONLY param whose value IS in the pool → grounding passes."""
        fresh_state.available_values_by_type["booking_id"] = ["bk-real-001"]
        output = executor.execute(EP2.id, {"booking_id": "bk-real-001"}, fresh_state)
        assert output.error is None, f"Unexpected grounding error: {output.error}"
        assert len(fresh_state.tool_outputs) == 1

    def test_three_step_sequence_updates_state(self, executor: Executor, fresh_state: SessionState):
        """End-to-end: ep1 produces booking_id → ep2 + ep3 consume it correctly.

        After ep1: available_values_by_type["booking_id"] has one entry.
        After ep2 + ep3: three ToolOutputs in state, booking_id pool unchanged size.
        """
        # Step 1: create booking (USER_PROVIDED param only).
        out1 = executor.execute(EP1.id, {"city_name": "London"}, fresh_state)
        assert out1.error is None, f"Step 1 failed: {out1.error}"
        assert out1.timestamp == "turn-0"
        pool = fresh_state.available_values_by_type.get("booking_id", [])
        assert len(pool) == 1
        booking_id = pool[0]

        # Step 2: get booking details using the produced booking_id.
        out2 = executor.execute(EP2.id, {"booking_id": booking_id}, fresh_state)
        assert out2.error is None, f"Step 2 failed: {out2.error}"
        assert out2.timestamp == "turn-1"

        # Step 3: update booking.
        out3 = executor.execute(EP3.id, {"booking_id": booking_id}, fresh_state)
        assert out3.error is None, f"Step 3 failed: {out3.error}"
        assert out3.timestamp == "turn-2"

        assert len(fresh_state.tool_outputs) == 3

    def test_optional_chain_only_absent_skips_grounding(self, executor: Executor, fresh_state: SessionState):
        """Optional CHAIN_ONLY param absent from arguments → no grounding check."""
        # EP3 has booking_id (required) and notes (optional, no semantic_type).
        # Provide booking_id from pool; omit notes — should pass.
        fresh_state.available_values_by_type["booking_id"] = ["bk-123"]
        output = executor.execute(EP3.id, {"booking_id": "bk-123"}, fresh_state)
        assert output.error is None

    def test_failures_appended_to_history(self, executor: Executor, fresh_state: SessionState):
        """Failed calls ARE appended to state.tool_outputs with response=None."""
        output = executor.execute("Unknown/ep", {}, fresh_state)
        assert len(fresh_state.tool_outputs) == 1
        assert fresh_state.tool_outputs[0].response is None
        assert fresh_state.tool_outputs[0].error is not None

    def test_repeated_failures_advance_timestamp(self, executor: Executor, fresh_state: SessionState):
        """Each failure increments the turn index: turn-0, turn-1, turn-2."""
        out0 = executor.execute("bad/ep", {}, fresh_state)
        out1 = executor.execute("bad/ep", {}, fresh_state)
        out2 = executor.execute("bad/ep", {}, fresh_state)
        assert out0.timestamp == "turn-0"
        assert out1.timestamp == "turn-1"
        assert out2.timestamp == "turn-2"
        assert len(fresh_state.tool_outputs) == 3

    def test_mixed_success_and_failure_timestamps(self, executor: Executor, fresh_state: SessionState):
        """Timestamps advance monotonically across a mix of successes and failures."""
        out0 = executor.execute(EP1.id, {"city_name": "Rome"}, fresh_state)
        assert out0.timestamp == "turn-0"
        assert out0.error is None

        out1 = executor.execute("Bad/ep", {}, fresh_state)
        assert out1.timestamp == "turn-1"
        assert out1.error is not None

        booking_id = fresh_state.available_values_by_type["booking_id"][0]
        out2 = executor.execute(EP2.id, {"booking_id": booking_id}, fresh_state)
        assert out2.timestamp == "turn-2"
        assert out2.error is None
        assert len(fresh_state.tool_outputs) == 3
