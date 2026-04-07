"""Frozen Pydantic v2 data models for the tool registry.

Every normalized value traces back to its raw source via ParamProvenance (P1).
All models are frozen — treat instances as immutable value objects.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ParamProvenance(BaseModel):
    """Audit trail for a single normalized parameter.

    Records exactly which raw JSON key was used, the original type string,
    whether the description was synthesized, and every normalization rule
    that was applied. This list is the primary evidence for the
    normalization-decisions table in DESIGN.md §2.2.
    """

    model_config = ConfigDict(frozen=True)

    raw_required_field: str
    """JSON key where the parameter was found: 'required_parameters' | 'params' | 'parameters'."""

    raw_type_string: str
    """Original type string before normalization, e.g. 'STRING', 'Number', 'ENUM'."""

    synthesized_description: bool = False
    """True when the description was generated from tool/endpoint name rather than sourced."""

    normalization_rules_applied: list[str] = Field(default_factory=list)
    """Ordered list of rule tags applied, e.g. ['unknown-type-fallback', 'null-default-dropped'].
    Appended to by the normalizer as each rule fires. Non-empty whenever normalization changed
    the raw value.
    """


class Parameter(BaseModel):
    """One parameter of an endpoint, fully normalized with provenance."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: str
    """Normalized type: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object'."""

    description: str
    required: bool
    default: str | int | float | bool | None = None
    enum: tuple[str, ...] | None = None
    """Enum values parsed from a comma-separated string or a raw list."""

    semantic_type: str | None = None
    """Populated by the semantic typing pass (F1.5). None until then."""

    provenance: ParamProvenance
    """Non-optional — every Parameter must have a full provenance record."""


class ResponseField(BaseModel):
    """One field in an endpoint's response schema.

    mock_policy is intentionally absent — it is an endpoint-level concern (§6.2).
    """

    model_config = ConfigDict(frozen=True)

    path: str
    """Dot-notation path within the response JSON, e.g. 'results[].hotel_id'."""

    type: str
    """One of: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object' | 'null'."""

    description: str = ""
    semantic_type: str | None = None
    """Populated by the semantic typing pass (F1.5). None until then."""


class Endpoint(BaseModel):
    """One callable API endpoint within a Tool."""

    model_config = ConfigDict(frozen=True)

    id: str
    """Globally unique identifier: '{category}/{tool_name}/{endpoint_name}'."""

    name: str
    description: str

    method: Literal["GET", "POST", "PUT", "DELETE", "PATCH", "UNKNOWN"] = "GET"
    """HTTP method. Normalizer maps anything unrecognised to 'UNKNOWN' and records
    'method-fallback-unknown' in the affected parameter's provenance."""

    parameters: tuple[Parameter, ...] = ()
    response_schema: tuple[ResponseField, ...] = ()
    """Populated by the response schema inference pass (F1.6). Empty until then."""

    mock_policy: Literal["static", "schema", "llm"] | None = None
    """Set by F1.6: 'static' when a response example exists, 'schema' for schema-derived
    mocks without an example, 'llm' for fully LLM-inferred schemas."""


class Tool(BaseModel):
    """A named API with one or more endpoints, belonging to one ToolBench category."""

    model_config = ConfigDict(frozen=True)

    name: str
    category: str
    description: str
    endpoints: tuple[Endpoint, ...] = ()
