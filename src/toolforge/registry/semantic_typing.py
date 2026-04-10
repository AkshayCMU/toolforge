"""LLM-backed semantic typing pass (cached) — F1.5.

One LLM call per endpoint, structured output via tool use, content-addressed cache.
Post-processor re-derives is_new_type from the seed vocabulary; new types accepted
only if they appear ≥3 times across the full corpus.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Literal

import anthropic
import structlog
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from toolforge.registry.models import Endpoint, Tool
from toolforge.registry.semantic_vocab import ALL_VOCAB, CHAIN_ONLY_VOCAB, NULL_OVERRIDE_TYPES, USER_PROVIDED_VOCAB

log = structlog.get_logger(__name__)

PROMPT_VERSION = "v3"
MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = (
    "You are a semantic typing assistant for API tool parameters and response fields. "
    "Your output is used to build a tool-chaining graph: parameters are consumers "
    "(a CHAIN_ONLY parameter means this endpoint requires a value produced by a prior "
    "tool call) and response fields are producers (a typed response field means this "
    "endpoint produces a value that a downstream tool can consume via a CHAINS_TO edge). "
    "Classify each parameter and response field with a taxonomy tier and a semantic type "
    "from the following seed vocabulary. "
    "CHAIN_ONLY — opaque identifiers or tokens a user cannot dictate; these must come "
    "from a prior tool output: hotel_id, booking_id, user_id, order_id, product_id, "
    "account_id, customer_id, client_id, venue_id, operation_id, market_id, "
    "conversation_id, channel_id, device_id, project_id, tenant_id, flight_id, "
    "property_id, media_id, checkout_id, buy_id, deposit_id, dest_id, access_token. "
    "USER_PROVIDED — values a user can state directly from their own knowledge or intent: "
    "search_query, city_name, country_code, date, datetime, currency_code, page_number, "
    "limit, email, phone_number, url, price, latitude, longitude, league_id, season, "
    "team_name, language_code, postal_code. "
    "When typing response fields, only assign a CHAIN_ONLY type. "
    "Response fields are producers for downstream tool inputs — USER_PROVIDED types must "
    "never appear as producer outputs. "
    "If a response field does not contain an entity identifier that a downstream tool "
    "would need as a CHAIN_ONLY parameter, assign null. "
    "For response fields, prefer proposing a precise new type over collapsing to the "
    "nearest wrong seed match. "
    "Set both tier and semantic_type to null for generic control or formatting parameters "
    "(format, locale, jsonp, callback) and for ambiguous pagination parameters unless "
    "assigning page_number or limit specifically. "
    "Note that api_key, apikey, app_key, and similar static developer credentials are "
    "NOT access_token — they are static configuration values; assign null for them. "
    "The bare parameter name 'id' with no entity prefix should receive null unless the "
    "description unambiguously identifies the entity type. "
    "Artifact or file-handle parameters used to stop, retrieve, or continue an existing "
    "operation — such as a recording file path, a scheduled hangup handle, or a call "
    "reference — may be classified as CHAIN_ONLY with a precise new type when they "
    "clearly refer to machine-generated workflow artifacts that a user cannot supply. "
    "Propose a new type only when highly confident the concept is not covered by the "
    "seed vocabulary and will recur across many tools; use snake_case and set "
    "is_new_type to true. "
    "Echo endpoint_id, param_name, normalized_family, and field_path exactly as given."
)

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ParameterTyping(BaseModel):
    param_name: str
    normalized_family: str
    tier: Literal["CHAIN_ONLY", "USER_PROVIDED"] | None
    semantic_type: str | None
    is_new_type: bool  # advisory; post-processor overwrites from vocab


class ResponseFieldTyping(BaseModel):
    field_path: str
    semantic_type: str | None
    is_new_type: bool  # advisory; post-processor overwrites from vocab


class EndpointTypingResult(BaseModel):
    endpoint_id: str
    parameter_types: list[ParameterTyping]
    response_field_types: list[ResponseFieldTyping]
    new_types_proposed: list[str]  # post-processor re-derives this


# ---------------------------------------------------------------------------
# Name normalisation (mirrors analyze_vocab.py logic)
# ---------------------------------------------------------------------------

def _normalize_name(raw: str) -> str:
    """camelCase / kebab-case / snake_case → lowercase snake_case."""
    s = re.sub(r"([A-Z])", r"_\1", raw).lower()
    tokens = re.split(r"[_\-]+", s)
    cleaned = []
    for t in tokens:
        t = t.translate(_PUNCT_TABLE).strip()
        if t:
            cleaned.append("id" if t == "ids" else t)
    return "_".join(cleaned) if cleaned else raw.lower()


# Field families too generic to safely apply the mismatch rule.
_GENERIC_FIELD_FAMILIES: frozenset[str] = frozenset(
    {"id", "value", "result", "data", "item", "object", "info", "response"}
)


def _field_leaf_family(field_path: str) -> str:
    """Normalized entity family of the leaf token in a response field path.

    Examples:
      petId              → pet_id
      hotel_id           → hotel_id
      results[].hotel_id → hotel_id   (leaf only, not the array wrapper)
      CallUUID           → call_uuid
      category.id        → id         (generic — caller checks _GENERIC_FIELD_FAMILIES)
    """
    tokens = re.split(r"[.\[\]]+", field_path)
    leaf = next((t for t in reversed(tokens) if t.strip()), field_path)
    return _normalize_name(leaf)


def _entity_stem(semantic_type: str) -> str:
    """Strip trailing _id / _uuid / _token to get the bare entity stem."""
    for suffix in ("_id", "_uuid", "_token"):
        if semantic_type.endswith(suffix) and len(semantic_type) > len(suffix):
            return semantic_type[: -len(suffix)]
    return semantic_type


def _clearly_different(field_family: str, semantic_type: str) -> bool:
    """True when field_family and semantic_type refer to clearly different entities.

    'Clearly different' means neither stem is a prefix or suffix of the other,
    so partial matches (customer_id vs customer_order_id) are treated as related.
    """
    f_stem = _entity_stem(field_family)
    s_stem = _entity_stem(semantic_type)
    if f_stem == s_stem:
        return False
    if f_stem.startswith(s_stem) or f_stem.endswith(s_stem):
        return False
    if s_stem.startswith(f_stem) or s_stem.endswith(f_stem):
        return False
    return True


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _cache_key(endpoint: Endpoint, model: str, prompt_version: str) -> str:
    """SHA-256 of the exact prompt payload — everything the LLM actually sees."""
    payload = {
        "model": model,
        "prompt_version": prompt_version,
        "endpoint_id": endpoint.id,
        "params": [
            {
                "param_name": p.name,
                "normalized_family": _normalize_name(p.name),
                "description": p.description,
                "required": p.required,
            }
            for p in sorted(endpoint.parameters, key=lambda p: p.name)
        ],
        "response_fields": [
            {
                "field_path": f.path,
                "description": f.description,
            }
            for f in sorted(endpoint.response_schema, key=lambda f: f.path)
        ],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(path: Path) -> EndpointTypingResult | None:
    if not path.exists():
        return None
    try:
        return EndpointTypingResult.model_validate(
            json.loads(path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        log.warning("typing.cache_corrupt", path=str(path), error=str(exc))
        path.unlink(missing_ok=True)
        return None


def _save_cache(path: Path, result: EndpointTypingResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _user_prompt(endpoint: Endpoint) -> str:
    lines = [f"Endpoint ID: {endpoint.id}", "", "Parameters:"]
    if endpoint.parameters:
        for p in endpoint.parameters:
            norm = _normalize_name(p.name)
            req = "required" if p.required else "optional"
            lines.append(
                f"  - param_name: {p.name!r}, normalized_family: {norm!r}, "
                f"description: {p.description!r}, {req}"
            )
    else:
        lines.append("  (none)")

    lines += ["", "Response fields:"]
    if endpoint.response_schema:
        for f in endpoint.response_schema:
            lines.append(
                f"  - field_path: {f.path!r}, description: {f.description!r}"
            )
    else:
        lines.append("  (none)")

    lines += [
        "",
        "Return an annotate_endpoint tool call. "
        "Echo endpoint_id, each param_name, normalized_family, and field_path exactly as shown above.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _postprocess(result: EndpointTypingResult) -> EndpointTypingResult:
    """Re-derive is_new_type from vocab; fix tier contradictions; rebuild new_types_proposed."""
    new_params: list[ParameterTyping] = []
    for pt in result.parameter_types:
        # NULL_OVERRIDE: force-null types that are too generic or arrive out-of-band.
        if pt.semantic_type in NULL_OVERRIDE_TYPES:
            log.info(
                "typing.null_override",
                endpoint_id=result.endpoint_id,
                param=pt.param_name,
                semantic_type=pt.semantic_type,
            )
            pt = pt.model_copy(update={"semantic_type": None, "tier": None, "is_new_type": False})

        is_new = pt.semantic_type is not None and pt.semantic_type not in ALL_VOCAB
        tier = pt.tier
        # Parameter-level family-mismatch rule: null out seed CHAIN_ONLY types whose
        # entity stem is clearly different from the parameter's normalized name family.
        # Fires only for seed types (not newly proposed types) and only when the param
        # name itself is not a generic token (id, value, etc.).
        # Rationale: LLMs tend to approximate to the nearest seed entry when no exact
        # match exists, e.g. SchedHangupId → operation_id. This rule reverts such
        # force-fits to null, letting the ≥3-occurrence filter handle genuinely new types.
        # e.g. SchedHangupId (family=sched_hangup_id) → operation_id: sched_hangup ≠
        #      operation → null. call_uuid is a new type (not in seed) → rule skips it.
        if (
            pt.semantic_type is not None
            and pt.semantic_type in CHAIN_ONLY_VOCAB
            and not is_new
        ):
            param_family = _normalize_name(pt.param_name)
            if (
                param_family not in _GENERIC_FIELD_FAMILIES
                and _clearly_different(param_family, pt.semantic_type)
            ):
                log.info(
                    "typing.param_family_mismatch_nulled",
                    endpoint_id=result.endpoint_id,
                    param=pt.param_name,
                    param_family=param_family,
                    semantic_type=pt.semantic_type,
                )
                pt = pt.model_copy(update={"semantic_type": None, "tier": None, "is_new_type": False})
                is_new = False
                tier = None

        # Correct tier/semantic_type contradictions
        if pt.semantic_type in CHAIN_ONLY_VOCAB and tier == "USER_PROVIDED":
            log.warning(
                "typing.tier_contradiction",
                endpoint_id=result.endpoint_id,
                param=pt.param_name,
                semantic_type=pt.semantic_type,
                corrected_to="CHAIN_ONLY",
            )
            tier = "CHAIN_ONLY"
        elif pt.semantic_type in USER_PROVIDED_VOCAB and tier == "CHAIN_ONLY":
            log.warning(
                "typing.tier_contradiction",
                endpoint_id=result.endpoint_id,
                param=pt.param_name,
                semantic_type=pt.semantic_type,
                corrected_to="USER_PROVIDED",
            )
            tier = "USER_PROVIDED"
        # Null-tier cleanup: a null semantic_type must not carry a tier label.
        # Covers LLM outputs where tier is set but semantic_type was never assigned.
        if pt.semantic_type is None:
            tier = None
        new_params.append(pt.model_copy(update={"is_new_type": is_new, "tier": tier}))

    new_fields: list[ResponseFieldTyping] = []
    for ft in result.response_field_types:
        # NULL_OVERRIDE: force-null types that are too generic or arrive out-of-band.
        if ft.semantic_type in NULL_OVERRIDE_TYPES:
            log.info(
                "typing.null_override",
                endpoint_id=result.endpoint_id,
                field_path=ft.field_path,
                semantic_type=ft.semantic_type,
            )
            ft = ft.model_copy(update={"semantic_type": None, "is_new_type": False})

        is_new = ft.semantic_type is not None and ft.semantic_type not in ALL_VOCAB
        # Change A: response fields must never carry USER_PROVIDED types — they are
        # producers for downstream chain inputs, not user-intent values.
        if ft.semantic_type in USER_PROVIDED_VOCAB:
            log.info(
                "typing.response_field_user_provided_nulled",
                endpoint_id=result.endpoint_id,
                field_path=ft.field_path,
                semantic_type=ft.semantic_type,
            )
            ft = ft.model_copy(update={"semantic_type": None, "is_new_type": False})
            is_new = False
        # Family-mismatch rule: null out CHAIN_ONLY seed types whose entity stem is
        # clearly different from the leaf field family. Fires only when:
        #   - semantic_type is a seed CHAIN_ONLY type (not a newly proposed type)
        #   - the leaf field family is not a generic token (id, data, result, …)
        #   - neither stem is a prefix/suffix of the other (so related names pass)
        # e.g. petId (family=pet_id) → product_id: pet ≠ product, no overlap → null
        # e.g. results[].hotel_id (family=hotel_id) → hotel_id: hotel==hotel → keep
        # e.g. SchedHangupId is a parameter handled by the param-level rule above
        elif ft.semantic_type in CHAIN_ONLY_VOCAB and not is_new:
            field_family = _field_leaf_family(ft.field_path)
            if (
                field_family not in _GENERIC_FIELD_FAMILIES
                and _clearly_different(field_family, ft.semantic_type)
            ):
                log.info(
                    "typing.response_field_family_mismatch_nulled",
                    endpoint_id=result.endpoint_id,
                    field_path=ft.field_path,
                    field_family=field_family,
                    semantic_type=ft.semantic_type,
                )
                ft = ft.model_copy(update={"semantic_type": None, "is_new_type": False})
                is_new = False
        new_fields.append(ft.model_copy(update={"is_new_type": is_new}))

    # Re-derive new_types_proposed from corrected entries
    new_types: set[str] = set()
    for pt in new_params:
        if pt.semantic_type and pt.is_new_type:
            new_types.add(pt.semantic_type)
    for ft in new_fields:
        if ft.semantic_type and ft.is_new_type:
            new_types.add(ft.semantic_type)

    return result.model_copy(
        update={
            "parameter_types": new_params,
            "response_field_types": new_fields,
            "new_types_proposed": sorted(new_types),
        }
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

_TOOL_DEF: dict = {
    "name": "annotate_endpoint",
    "description": "Return semantic type annotations for all parameters and response fields of the given endpoint.",
    "input_schema": EndpointTypingResult.model_json_schema(),
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _call_llm(
    client: anthropic.Anthropic,
    model: str,
    user_prompt_text: str,
) -> EndpointTypingResult:
    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[_TOOL_DEF],
        tool_choice={"type": "tool", "name": "annotate_endpoint"},
        messages=[{"role": "user", "content": user_prompt_text}],
    )
    tool_block = next(b for b in message.content if b.type == "tool_use")
    data = dict(tool_block.input)
    # Defensive fill: LLM occasionally omits list fields for endpoints with no
    # response schema or trivial parameter sets.  Backfill with empty lists so
    # Pydantic validation doesn't hard-fail and burn all three tenacity retries.
    data.setdefault("parameter_types", [])
    data.setdefault("response_field_types", [])
    data.setdefault("new_types_proposed", [])
    return EndpointTypingResult.model_validate(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def type_endpoint(
    endpoint: Endpoint,
    client: anthropic.Anthropic,
    cache_dir: Path,
    model: str = MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[EndpointTypingResult, bool]:
    """Type one endpoint. Returns (result, cache_hit).

    Cache stores raw LLM output. Post-processor always runs on both cache hits and
    misses — post-processor rule changes never require cache invalidation.
    """
    key = _cache_key(endpoint, model, prompt_version)
    cache_path = cache_dir / "llm" / f"{key}.json"

    cached = _load_cache(cache_path)
    if cached is not None:
        log.debug("typing.cache_hit", endpoint_id=endpoint.id, key=key[:8])
        return _postprocess(cached), True

    log.info("typing.llm_call", endpoint_id=endpoint.id)
    raw = _call_llm(client, model, _user_prompt(endpoint))
    _save_cache(cache_path, raw)
    return _postprocess(raw), False


def type_corpus(
    tools: list[Tool],
    client: anthropic.Anthropic,
    cache_dir: Path,
    model: str = MODEL,
    prompt_version: str = PROMPT_VERSION,
) -> tuple[list[Tool], dict[str, int]]:
    """Type every endpoint in *tools*. Returns updated tools + accepted new types.

    New types are accepted only if they appear ≥3 times across all endpoints.
    Updates Parameter.semantic_type and ResponseField.semantic_type in place
    (via model_copy — models remain frozen).
    """
    all_results: list[EndpointTypingResult] = []
    calls = 0
    hits = 0

    for tool in tools:
        for ep in tool.endpoints:
            result, cache_hit = type_endpoint(ep, client, cache_dir, model, prompt_version)
            all_results.append(result)
            if cache_hit:
                hits += 1
            else:
                calls += 1

    log.info(
        "typing.corpus_done",
        llm_calls=calls,
        cache_hits=hits,
        total=calls + hits,
    )

    # Consolidate new types: accept ≥3 occurrences
    new_type_counts: Counter[str] = Counter()
    for r in all_results:
        for t in r.new_types_proposed:
            new_type_counts[t] += 1
    accepted_new: dict[str, int] = {t: c for t, c in new_type_counts.items() if c >= 3}
    if accepted_new:
        log.info("typing.new_types_accepted", types=accepted_new)

    accepted_vocab = ALL_VOCAB | frozenset(accepted_new)

    # Build lookup: endpoint_id → result
    result_map = {r.endpoint_id: r for r in all_results}

    # Annotate tools (frozen models → model_copy throughout)
    updated_tools: list[Tool] = []
    for tool in tools:
        updated_eps = []
        for ep in tool.endpoints:
            result = result_map.get(ep.id)
            if result is None:
                updated_eps.append(ep)
                continue

            # Map param typings by name
            param_map = {pt.param_name: pt for pt in result.parameter_types}
            new_params = []
            for p in ep.parameters:
                pt = param_map.get(p.name)
                if pt and pt.semantic_type and pt.semantic_type in accepted_vocab:
                    new_params.append(p.model_copy(update={"semantic_type": pt.semantic_type}))
                else:
                    new_params.append(p)

            # Map field typings by path
            field_map = {ft.field_path: ft for ft in result.response_field_types}
            new_fields = []
            for f in ep.response_schema:
                ft = field_map.get(f.path)
                if ft and ft.semantic_type and ft.semantic_type in accepted_vocab:
                    new_fields.append(f.model_copy(update={"semantic_type": ft.semantic_type}))
                else:
                    new_fields.append(f)

            updated_eps.append(
                ep.model_copy(
                    update={
                        "parameters": tuple(new_params),
                        "response_schema": tuple(new_fields),
                    }
                )
            )
        updated_tools.append(tool.model_copy(update={"endpoints": tuple(updated_eps)}))

    return updated_tools, accepted_new
