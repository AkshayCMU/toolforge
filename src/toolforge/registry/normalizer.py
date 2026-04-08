"""Raw dict → Tool/Endpoint/Parameter normalizer (F1.3).

Applies the 9 normalization rules from DESIGN.md §2.2 and records every
rule that fires in ``ParamProvenance.normalization_rules_applied``.

No LLM calls anywhere in this module.  Pure deterministic Python.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass, field
from typing import Iterator

import structlog

from toolforge.registry.models import (
    Endpoint,
    ParamProvenance,
    Parameter,
    ResponseField,
    Tool,
)

log = structlog.get_logger(__name__)

# Canonical type strings after normalization (rule 1 + 2).
_KNOWN_TYPES = frozenset({"string", "number", "integer", "boolean", "array", "object"})

# Canonical HTTP methods (rule 8).
_KNOWN_METHODS = frozenset({"GET", "POST", "PUT", "DELETE", "PATCH"})

# Keys to search for required parameters, in priority order (rule provenance).
_REQUIRED_KEYS = ("required_parameters", "params", "parameters")

# Keys to search for optional parameters.
_OPTIONAL_KEYS = ("optional_parameters",)


@dataclass
class NormalizationReport:
    """Aggregate stats from a full normalization run."""

    total_seen: int = 0
    total_kept: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)
    per_category_counts: dict[str, int] = field(default_factory=dict)
    rule_counts: dict[str, int] = field(default_factory=dict)
    distinct_raw_type_strings: set[str] = field(default_factory=set)

    def record_drop(self, reason: str) -> None:
        self.drop_reasons[reason] = self.drop_reasons.get(reason, 0) + 1

    def record_rule(self, rule: str) -> None:
        self.rule_counts[rule] = self.rule_counts.get(rule, 0) + 1


# ---------------------------------------------------------------------------
# Parameter normalization
# ---------------------------------------------------------------------------

def _normalize_param(
    raw: dict,
    source_key: str,
    is_required: bool,
    tool_name: str,
    endpoint_name: str,
    report: NormalizationReport,
) -> Parameter | None:
    """Normalize one raw parameter dict into a ``Parameter``, or None to skip."""

    name = raw.get("name", "").strip()
    if not name:
        return None

    # --- Rule 9 (at param level): skip empty schema dicts ---
    # An entire param that is just {} with no name is already caught above.
    # This guards against {"name": "", ...} which yields an empty name.

    raw_type = str(raw.get("type", "string"))
    rules: list[str] = []
    report.distinct_raw_type_strings.add(raw_type)

    # --- Rule 1: type-lowercased ---
    lowered = raw_type.lower()
    if lowered != raw_type:
        rules.append("type-lowercased")
        report.record_rule("type-lowercased")

    # --- Rule 3: enum-no-values (before type fallback, since ENUM triggers both) ---
    enum_values: tuple[str, ...] | None = None
    if lowered == "enum":
        # Check if there are actual enum values provided
        raw_enum = raw.get("enum")
        if not raw_enum:
            rules.append("enum-no-values")
            report.record_rule("enum-no-values")
            # Falls through to unknown-type-fallback below since "enum" not in _KNOWN_TYPES
        else:
            if isinstance(raw_enum, str):
                enum_values = tuple(v.strip() for v in raw_enum.split(",") if v.strip())
            elif isinstance(raw_enum, list):
                enum_values = tuple(str(v).strip() for v in raw_enum if str(v).strip())

    # --- Rule 2: unknown-type-fallback ---
    final_type = lowered
    if final_type not in _KNOWN_TYPES:
        rules.append("unknown-type-fallback")
        report.record_rule("unknown-type-fallback")
        final_type = "string"

    # --- Rule 4: enum-split-from-default ---
    default = raw.get("default")
    if (
        enum_values is None
        and isinstance(default, str)
        and "," in default
        and len(default.split(",")) >= 2
    ):
        candidates = tuple(v.strip() for v in default.split(",") if v.strip())
        if len(candidates) >= 2:
            enum_values = candidates
            default = None
            rules.append("enum-split-from-default")
            report.record_rule("enum-split-from-default")

    # --- Rule 4b: complex-default-stringified ---
    if isinstance(default, (dict, list)):
        default = _json.dumps(default, ensure_ascii=False)
        rules.append("complex-default-stringified")
        report.record_rule("complex-default-stringified")

    # --- Rule 5: empty-default-dropped ---
    if isinstance(default, str) and default == "":
        default = None
        rules.append("empty-default-dropped")
        report.record_rule("empty-default-dropped")

    # --- Rule 6: null-default-dropped ---
    if default is None:
        # Only record the rule if the raw value was explicitly null
        # (not already None from rule 4/5).
        raw_default = raw.get("default")
        if raw_default is None and "default" in raw:
            rules.append("null-default-dropped")
            report.record_rule("null-default-dropped")

    # --- Rule 7: synthesized-description ---
    description = str(raw.get("description", "")).strip()
    synthesized = False
    if not description:
        description = f"{tool_name} {endpoint_name}"
        synthesized = True
        rules.append("synthesized-description")
        report.record_rule("synthesized-description")

    provenance = ParamProvenance(
        raw_required_field=source_key,
        raw_type_string=raw_type,
        synthesized_description=synthesized,
        normalization_rules_applied=rules,
    )

    return Parameter(
        name=name,
        type=final_type,
        description=description,
        required=is_required,
        default=default,
        enum=enum_values,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# Endpoint normalization
# ---------------------------------------------------------------------------

def _find_params(
    raw_endpoint: dict,
    keys: tuple[str, ...],
    is_required: bool,
) -> list[tuple[str, dict]]:
    """Return ``(source_key, param_dict)`` pairs for each param found."""
    for key in keys:
        val = raw_endpoint.get(key)
        if isinstance(val, list) and val:
            return [(key, p) for p in val if isinstance(p, dict)]
    return []


def _normalize_endpoint(
    raw: dict,
    category: str,
    tool_name: str,
    report: NormalizationReport,
) -> Endpoint | None:
    """Normalize one raw endpoint dict into an ``Endpoint``."""
    name = str(raw.get("name", "")).strip()
    if not name:
        return None

    endpoint_id = f"{category}/{tool_name}/{name}"

    # --- Rule 8: method-fallback-unknown ---
    raw_method = str(raw.get("method", "GET")).upper().strip()
    method = raw_method if raw_method in _KNOWN_METHODS else "UNKNOWN"
    if method == "UNKNOWN" and raw_method != "UNKNOWN":
        report.record_rule("method-fallback-unknown")

    # --- Rule 7 at endpoint level ---
    description = str(raw.get("description", "")).strip()
    if not description:
        description = f"{tool_name} {name}"
        report.record_rule("synthesized-description")

    # --- Collect parameters ---
    params: list[Parameter] = []

    for source_key, raw_param in _find_params(raw, _REQUIRED_KEYS, is_required=True):
        p = _normalize_param(raw_param, source_key, True, tool_name, name, report)
        if p is not None:
            params.append(p)

    for source_key, raw_param in _find_params(raw, _OPTIONAL_KEYS, is_required=False):
        p = _normalize_param(raw_param, source_key, False, tool_name, name, report)
        if p is not None:
            params.append(p)

    # --- Rule 9: schema-empty-ignored ---
    response_schema: tuple[ResponseField, ...] = ()
    raw_schema = raw.get("schema")
    if raw_schema and isinstance(raw_schema, dict) and raw_schema != {}:
        # Non-empty schema — parse into ResponseFields.
        response_schema = _parse_response_schema(raw_schema)
    else:
        # Empty string, empty dict, or missing → leave empty, F1.6 decides later.
        if raw_schema is not None:
            report.record_rule("schema-empty-ignored")

    return Endpoint(
        id=endpoint_id,
        name=name,
        description=description,
        method=method,  # type: ignore[arg-type]
        parameters=tuple(params),
        response_schema=response_schema,
    )


def _parse_response_schema(schema: dict, prefix: str = "") -> tuple[ResponseField, ...]:
    """Recursively walk a schema dict and extract ResponseField objects."""
    fields: list[ResponseField] = []

    if "properties" in schema and isinstance(schema["properties"], dict):
        for key, val in schema["properties"].items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(val, dict):
                field_type = str(val.get("type", "object")).lower()
                desc = str(val.get("description", ""))
                fields.append(ResponseField(path=path, type=field_type, description=desc))
                # Recurse into nested objects
                if field_type == "object":
                    fields.extend(_parse_response_schema(val, prefix=path))
                elif field_type == "array" and "items" in val:
                    items = val["items"]
                    if isinstance(items, dict):
                        fields.extend(_parse_response_schema(items, prefix=f"{path}[]"))
    elif "items" in schema and isinstance(schema["items"], dict):
        fields.extend(_parse_response_schema(schema["items"], prefix=f"{prefix}[]"))

    return tuple(fields)


# ---------------------------------------------------------------------------
# Tool normalization (top-level)
# ---------------------------------------------------------------------------

def normalize_tool(
    raw: dict,
    category: str,
    report: NormalizationReport,
) -> Tool | None:
    """Normalize one raw tool dict into a ``Tool``, or None to drop.

    Drop reasons are recorded in *report*.
    """
    tool_name = str(raw.get("tool_name", "")).strip()
    if not tool_name:
        report.record_drop("missing-tool-name")
        return None

    api_list = raw.get("api_list")
    if not isinstance(api_list, list) or not api_list:
        report.record_drop("no-api-list")
        return None

    description = str(raw.get("tool_description", "")).strip()
    if not description:
        description = tool_name
        report.record_rule("synthesized-description")

    endpoints: list[Endpoint] = []
    for raw_ep in api_list:
        if not isinstance(raw_ep, dict):
            continue
        ep = _normalize_endpoint(raw_ep, category, tool_name, report)
        if ep is not None:
            endpoints.append(ep)

    if not endpoints:
        report.record_drop("zero-endpoints-after-normalization")
        return None

    return Tool(
        name=tool_name,
        category=category,
        description=description,
        endpoints=tuple(endpoints),
    )


def normalize_corpus(
    raw_iter: Iterator[tuple[str, str, dict]],
) -> tuple[list[Tool], NormalizationReport]:
    """Normalize a full iterator of ``(category, tool_name, raw_dict)`` tuples.

    Returns the kept tools and a report with aggregate stats.
    """
    report = NormalizationReport()
    tools: list[Tool] = []

    for category, _file_stem, raw in raw_iter:
        report.total_seen += 1
        tool = normalize_tool(raw, category, report)
        if tool is not None:
            report.total_kept += 1
            report.per_category_counts[category] = (
                report.per_category_counts.get(category, 0) + 1
            )
            tools.append(tool)

    return tools, report
