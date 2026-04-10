"""Response schema inference for F1.6.

Two deterministic paths (no LLM calls in this module yet):
  static  — response example found in example_dir; walk the schema tree.
  schema  — endpoint already has response_schema from the normalizer; preserve it.

LLM fallback (mock_policy='llm') is not implemented here; endpoints that need
it are returned unchanged so the caller can count and confirm before proceeding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

import anthropic

from toolforge.registry.models import Endpoint, ResponseField, Tool

log = structlog.get_logger(__name__)

# Map Python type-name strings used in response_examples to normalized types.
_EXAMPLE_TYPE_MAP: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "nonetype": "null",
    "none": "null",
    "list": "array",
    "dict": "object",
}


# ---------------------------------------------------------------------------
# Example-file loader
# ---------------------------------------------------------------------------

def _load_example_index(tool: Tool, example_dir: Path) -> dict[str, Any] | None:
    """Return a mapping of endpoint_name -> schema dict for *tool*, or None.

    Returns None when the file doesn't exist or is malformed.
    """
    path = example_dir / tool.category / f"{tool.file_stem}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("schema_infer.bad_example_file", path=str(path), error=str(exc))
        return None

    api_list = raw.get("api_list")
    if not isinstance(api_list, list):
        return None

    index: dict[str, Any] = {}
    for entry in api_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        schema = entry.get("schema")
        if name and schema is not None:
            index[name] = schema
    return index


# ---------------------------------------------------------------------------
# Schema tree flattener
# ---------------------------------------------------------------------------

def _type_str(raw: str) -> str:
    return _EXAMPLE_TYPE_MAP.get(raw.lower(), "string")


def _flatten(obj: Any, prefix: str = "") -> list[ResponseField]:
    """Recursively flatten a response_examples schema into ResponseField list."""
    fields: list[ResponseField] = []

    if isinstance(obj, dict):
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(val, str):
                fields.append(ResponseField(path=path, type=_type_str(val)))
            elif isinstance(val, list):
                fields.append(ResponseField(path=path, type="array"))
                if val and isinstance(val[0], dict):
                    fields.extend(_flatten(val[0], prefix=f"{path}[]"))
                elif val and isinstance(val[0], str):
                    fields.append(ResponseField(path=f"{path}[]", type=_type_str(val[0])))
            elif isinstance(val, dict):
                fields.append(ResponseField(path=path, type="object"))
                fields.extend(_flatten(val, prefix=path))
    elif isinstance(obj, list):
        if obj and isinstance(obj[0], dict):
            fields.extend(_flatten(obj[0], prefix=f"{prefix}[]" if prefix else "[]"))
    return fields


# ---------------------------------------------------------------------------
# Per-endpoint inference
# ---------------------------------------------------------------------------

def infer_schema(
    tool: Tool,
    endpoint: Endpoint,
    example_dir: Path,
    client: anthropic.Anthropic,
    cache_dir: Path,
) -> Endpoint:
    """Return *endpoint* with response_schema and mock_policy populated.

    Priority:
      1. Static: example file exists and contains a matching endpoint entry.
      2. Schema: endpoint already has response_schema from the normalizer.
      3. LLM fallback: not implemented -- returns endpoint unchanged (mock_policy stays None).
    """
    example_index = _load_example_index(tool, example_dir)

    if example_index is not None:
        schema = example_index.get(endpoint.name)
        if schema is not None:
            fields = _flatten(schema)
            if fields:
                log.debug(
                    "schema_infer.static",
                    endpoint_id=endpoint.id,
                    field_count=len(fields),
                )
                return endpoint.model_copy(
                    update={
                        "response_schema": tuple(fields),
                        "mock_policy": "static",
                    }
                )
            log.debug("schema_infer.static_empty_schema", endpoint_id=endpoint.id)
        else:
            log.debug(
                "schema_infer.static_no_ep_match",
                endpoint_id=endpoint.id,
                tool_file_stem=tool.file_stem,
            )

    if endpoint.response_schema:
        log.debug("schema_infer.schema", endpoint_id=endpoint.id)
        return endpoint.model_copy(update={"mock_policy": "schema"})

    log.debug("schema_infer.needs_llm", endpoint_id=endpoint.id)
    return endpoint


# ---------------------------------------------------------------------------
# Corpus-level inference
# ---------------------------------------------------------------------------

def infer_corpus(
    tools: list[Tool],
    example_dir: Path,
    client: anthropic.Anthropic,
    cache_dir: Path,
) -> list[Tool]:
    """Run schema inference over every endpoint in *tools*.

    Returns a new list of Tool objects with updated endpoints.
    LLM-fallback endpoints are left unchanged (mock_policy=None).
    """
    updated_tools: list[Tool] = []

    for tool in tools:
        updated_eps = [
            infer_schema(tool, ep, example_dir, client, cache_dir)
            for ep in tool.endpoints
        ]
        updated_tools.append(
            tool.model_copy(update={"endpoints": tuple(updated_eps)})
        )

    static_count = sum(
        1 for t in updated_tools for ep in t.endpoints if ep.mock_policy == "static"
    )
    schema_count = sum(
        1 for t in updated_tools for ep in t.endpoints if ep.mock_policy == "schema"
    )
    llm_needed = sum(
        1 for t in updated_tools for ep in t.endpoints if ep.mock_policy is None
    )
    log.info(
        "schema_infer.corpus_done",
        static=static_count,
        schema=schema_count,
        llm_needed=llm_needed,
    )
    return updated_tools
