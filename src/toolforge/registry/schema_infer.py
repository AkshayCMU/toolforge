"""Response schema inference for F1.6.

Three-tier strategy:
  static  — response example found in example_dir; walk the schema tree.
  schema  — endpoint already has response_schema from the normalizer; preserve it.
  llm     — no example, no existing schema; one structured LLM call, cached.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import anthropic
import structlog
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from toolforge.registry.models import Endpoint, ResponseField, Tool

log = structlog.get_logger(__name__)

MODEL = "claude-haiku-4-5-20251001"
PROMPT_VERSION = "v1"

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

_SYSTEM_PROMPT = (
    "You are an API documentation assistant. "
    "Given an API endpoint name, description, and parameters, produce a realistic "
    "JSON response schema for a successful call. "
    "Rules: "
    "Use dot-notation paths for nested fields (e.g. 'data.id', 'user.email'). "
    "For array responses use [] notation for items (e.g. 'results[].id', 'results[].name'). "
    "Keep schemas compact but realistic: 5-20 fields is typical. "
    "Valid types: string, integer, number, boolean, array, object, null. "
    "Do not include error or status wrapper schemas unless the endpoint clearly warrants it. "
    "Echo the endpoint_id exactly as given."
)


# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class InferredField(BaseModel):
    path: str
    type: Literal["string", "integer", "number", "boolean", "array", "object", "null"]


class EndpointSchemaResult(BaseModel):
    endpoint_id: str
    fields: list[InferredField]


_TOOL_DEF: dict = {
    "name": "return_response_schema",
    "description": "Return a plausible response schema for the given API endpoint.",
    "input_schema": EndpointSchemaResult.model_json_schema(),
}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(endpoint: Endpoint, model: str, prompt_version: str) -> str:
    payload = json.dumps(
        {"model": model, "prompt_version": prompt_version, "endpoint_id": endpoint.id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_cache(path: Path) -> EndpointSchemaResult | None:
    if not path.exists():
        return None
    try:
        return EndpointSchemaResult.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        log.warning("schema_infer.cache_corrupt", path=str(path), error=str(exc))
        path.unlink(missing_ok=True)
        return None


def _save_cache(path: Path, result: EndpointSchemaResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _user_prompt(tool: Tool, endpoint: Endpoint) -> str:
    lines = [
        f"Endpoint ID: {endpoint.id}",
        f"Category: {tool.category}",
        f"Endpoint name: {endpoint.name}",
        f"Description: {endpoint.description}",
        "",
        "Parameters:",
    ]
    if endpoint.parameters:
        for p in endpoint.parameters:
            req = "required" if p.required else "optional"
            lines.append(f"  - {p.name!r} ({p.type}, {req}): {p.description}")
    else:
        lines.append("  (none)")
    lines += [
        "",
        "Return a return_response_schema tool call with a realistic response schema.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM call (with retry)
# ---------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_llm(
    client: anthropic.Anthropic,
    tool: Tool,
    endpoint: Endpoint,
) -> EndpointSchemaResult:
    message = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL_DEF],
        tool_choice={"type": "tool", "name": "return_response_schema"},
        messages=[{"role": "user", "content": _user_prompt(tool, endpoint)}],
    )
    tool_block = next(b for b in message.content if b.type == "tool_use")
    data = dict(tool_block.input)
    data.setdefault("fields", [])
    return EndpointSchemaResult.model_validate(data)


# ---------------------------------------------------------------------------
# Example-file loader
# ---------------------------------------------------------------------------

def _load_example_index(tool: Tool, example_dir: Path) -> dict[str, Any] | None:
    """Return a mapping of endpoint_name -> schema dict for *tool*, or None."""
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
# Schema tree flattener (response_examples custom format)
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
) -> tuple[Endpoint, str]:
    """Return (updated_endpoint, path_taken) where path_taken is 'static'|'schema'|'llm'.

    Priority:
      1. Static: example file exists and contains a matching endpoint entry.
      2. Schema: endpoint already has response_schema from the normalizer.
      3. LLM fallback: one cached structured LLM call.
    """
    example_index = _load_example_index(tool, example_dir)

    if example_index is not None:
        raw_schema = example_index.get(endpoint.name)
        if raw_schema is not None:
            fields = _flatten(raw_schema)
            if fields:
                log.debug(
                    "schema_infer.static",
                    endpoint_id=endpoint.id,
                    field_count=len(fields),
                )
                return (
                    endpoint.model_copy(
                        update={"response_schema": tuple(fields), "mock_policy": "static"}
                    ),
                    "static",
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
        return endpoint.model_copy(update={"mock_policy": "schema"}), "schema"

    # LLM fallback
    key = _cache_key(endpoint, MODEL, PROMPT_VERSION)
    cache_path = cache_dir / "llm_schema" / f"{key}.json"
    cached = _load_cache(cache_path)
    cache_hit = cached is not None

    if cached is None:
        log.info("schema_infer.llm_call", endpoint_id=endpoint.id)
        cached = _call_llm(client, tool, endpoint)
        _save_cache(cache_path, cached)

    if cached.fields:
        rf_tuple = tuple(
            ResponseField(path=f.path, type=f.type) for f in cached.fields
        )
        log.debug(
            "schema_infer.llm",
            endpoint_id=endpoint.id,
            field_count=len(rf_tuple),
            cache_hit=cache_hit,
        )
        return (
            endpoint.model_copy(
                update={"response_schema": rf_tuple, "mock_policy": "llm"}
            ),
            "llm",
        )

    # LLM returned empty fields — log and leave mock_policy=None
    log.warning("schema_infer.llm_empty_response", endpoint_id=endpoint.id)
    return endpoint, "empty"


# ---------------------------------------------------------------------------
# Corpus-level inference
# ---------------------------------------------------------------------------

def infer_corpus(
    tools: list[Tool],
    example_dir: Path,
    client: anthropic.Anthropic,
    cache_dir: Path,
) -> tuple[list[Tool], dict[str, int]]:
    """Run schema inference over every endpoint in *tools*.

    Returns (updated_tools, stats) where stats has keys:
      static, schema, llm, empty, llm_calls, cache_hits
    """
    stats: dict[str, int] = {
        "static": 0, "schema": 0, "llm": 0, "empty": 0,
        "llm_calls": 0, "cache_hits": 0,
    }
    updated_tools: list[Tool] = []

    for tool in tools:
        updated_eps: list[Endpoint] = []
        for ep in tool.endpoints:
            key = _cache_key(ep, MODEL, PROMPT_VERSION)
            cache_path = cache_dir / "llm_schema" / f"{key}.json"
            was_cached = cache_path.exists()

            updated_ep, path_taken = infer_schema(tool, ep, example_dir, client, cache_dir)
            updated_eps.append(updated_ep)
            stats[path_taken] = stats.get(path_taken, 0) + 1

            if path_taken == "llm":
                if was_cached:
                    stats["cache_hits"] += 1
                else:
                    stats["llm_calls"] += 1

        updated_tools.append(tool.model_copy(update={"endpoints": tuple(updated_eps)}))

    log.info("schema_infer.corpus_done", **stats)
    return updated_tools, stats
