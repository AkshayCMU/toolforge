"""Unit tests for toolforge.registry.models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolforge.registry.models import (
    Endpoint,
    Parameter,
    ParamProvenance,
    ResponseField,
    Tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _provenance(**kwargs) -> ParamProvenance:
    defaults = dict(
        raw_required_field="required_parameters",
        raw_type_string="STRING",
    )
    defaults.update(kwargs)
    return ParamProvenance(**defaults)


def _parameter(**kwargs) -> Parameter:
    defaults = dict(
        name="city",
        type="string",
        description="City name",
        required=True,
        provenance=_provenance(),
    )
    defaults.update(kwargs)
    return Parameter(**defaults)


def _endpoint(**kwargs) -> Endpoint:
    defaults = dict(
        id="Travel/Hotels/searchHotels",
        name="searchHotels",
        description="Search hotels by city",
    )
    defaults.update(kwargs)
    return Endpoint(**defaults)


def _tool(**kwargs) -> Tool:
    defaults = dict(
        name="Hotels",
        category="Travel",
        description="Hotel booking API",
        file_stem="hotels",
    )
    defaults.update(kwargs)
    return Tool(**defaults)


# ---------------------------------------------------------------------------
# ParamProvenance
# ---------------------------------------------------------------------------

class TestParamProvenance:
    def test_minimal_construction(self):
        p = _provenance()
        assert p.raw_required_field == "required_parameters"
        assert p.raw_type_string == "STRING"
        assert p.synthesized_description is False
        assert p.normalization_rules_applied == []

    def test_normalization_rules_default_is_empty_list(self):
        p = _provenance()
        assert isinstance(p.normalization_rules_applied, list)
        assert len(p.normalization_rules_applied) == 0

    def test_normalization_rules_populated(self):
        p = _provenance(normalization_rules_applied=["unknown-type-fallback", "null-default-dropped"])
        assert p.normalization_rules_applied == ["unknown-type-fallback", "null-default-dropped"]

    def test_frozen_rejects_mutation(self):
        p = _provenance()
        with pytest.raises(ValidationError):
            p.raw_type_string = "NUMBER"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------

class TestParameter:
    def test_provenance_is_required(self):
        """Parameter must reject construction without a provenance record."""
        with pytest.raises(ValidationError, match="provenance"):
            Parameter(
                name="city",
                type="string",
                description="City name",
                required=True,
                # provenance intentionally omitted
            )

    def test_minimal_construction(self):
        p = _parameter()
        assert p.name == "city"
        assert p.semantic_type is None
        assert p.enum is None
        assert p.default is None

    def test_enum_as_tuple(self):
        p = _parameter(enum=("USD", "EUR", "GBP"))
        assert p.enum == ("USD", "EUR", "GBP")

    def test_frozen_rejects_mutation(self):
        p = _parameter()
        with pytest.raises(ValidationError):
            p.name = "other"  # type: ignore[misc]

    def test_round_trip_json(self):
        p = _parameter(
            enum=("asc", "desc"),
            semantic_type="city_name",
            provenance=_provenance(
                normalization_rules_applied=["enum-split"],
                synthesized_description=True,
            ),
        )
        dumped = p.model_dump_json()
        restored = Parameter.model_validate_json(dumped)
        assert restored == p


# ---------------------------------------------------------------------------
# ResponseField
# ---------------------------------------------------------------------------

class TestResponseField:
    def test_minimal_construction(self):
        rf = ResponseField(path="results[].id", type="string")
        assert rf.description == ""
        assert rf.semantic_type is None

    def test_no_mock_policy_field(self):
        """mock_policy must NOT exist on ResponseField — endpoint-level concern only."""
        rf = ResponseField(path="results[].id", type="string")
        assert not hasattr(rf, "mock_policy")

    def test_round_trip_json(self):
        rf = ResponseField(path="results[].price", type="number", semantic_type="price")
        restored = ResponseField.model_validate_json(rf.model_dump_json())
        assert restored == rf


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class TestEndpoint:
    def test_id_format(self):
        ep = _endpoint()
        assert ep.id == "Travel/Hotels/searchHotels"

    def test_default_method_is_get(self):
        ep = _endpoint()
        assert ep.method == "GET"

    def test_valid_methods(self):
        for method in ("GET", "POST", "PUT", "DELETE", "PATCH", "UNKNOWN"):
            ep = _endpoint(method=method)
            assert ep.method == method

    def test_invalid_method_rejected(self):
        with pytest.raises(ValidationError):
            _endpoint(method="HEAD")

    def test_mock_policy_on_endpoint(self):
        for policy in ("static", "schema", "llm"):
            ep = _endpoint(mock_policy=policy)
            assert ep.mock_policy == policy

    def test_mock_policy_invalid_rejected(self):
        with pytest.raises(ValidationError):
            _endpoint(mock_policy="unknown")

    def test_parameters_default_empty_tuple(self):
        ep = _endpoint()
        assert ep.parameters == ()

    def test_parameters_stored_correctly(self):
        p = _parameter()
        ep = _endpoint(parameters=(p,))
        assert len(ep.parameters) == 1
        assert ep.parameters[0] == p

    def test_round_trip_json(self):
        p = _parameter(
            provenance=_provenance(normalization_rules_applied=["unknown-type-fallback"]),
        )
        rf = ResponseField(path="booking_id", type="string", semantic_type="booking_id")
        ep = _endpoint(
            method="POST",
            parameters=(p,),
            response_schema=(rf,),
            mock_policy="llm",
        )
        restored = Endpoint.model_validate_json(ep.model_dump_json())
        assert restored == ep


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class TestTool:
    def test_minimal_construction(self):
        t = _tool()
        assert t.name == "Hotels"
        assert t.category == "Travel"
        assert t.endpoints == ()

    def test_with_endpoints(self):
        ep = _endpoint()
        t = _tool(endpoints=(ep,))
        assert len(t.endpoints) == 1

    def test_round_trip_json(self):
        p = _parameter()
        ep = _endpoint(parameters=(p,))
        t = _tool(endpoints=(ep,))
        restored = Tool.model_validate_json(t.model_dump_json())
        assert restored == t

    def test_nested_round_trip_preserves_provenance(self):
        """Full nesting: Tool → Endpoint → Parameter → ParamProvenance survives JSON."""
        prov = _provenance(
            raw_type_string="NUMBER",
            normalization_rules_applied=["enum-split", "null-default-dropped"],
            synthesized_description=True,
        )
        p = _parameter(type="number", provenance=prov)
        ep = _endpoint(parameters=(p,), mock_policy="static")
        t = _tool(endpoints=(ep,))

        restored = Tool.model_validate_json(t.model_dump_json())
        restored_prov = restored.endpoints[0].parameters[0].provenance
        assert restored_prov.raw_type_string == "NUMBER"
        assert restored_prov.normalization_rules_applied == ["enum-split", "null-default-dropped"]
        assert restored_prov.synthesized_description is True
