"""Schema-aware mock response generator — F3.2.

Three tiers (static, schema, llm) all use the same schema-driven Faker generation
at mock time. The distinction between tiers is provenance (how the response_schema
was derived at build time), not runtime logic:

  static  — schema walked from a real response example file (F1.6 tier 1)
  schema  — schema preserved from the normaliser (F1.6 tier 2)
  llm     — schema inferred by Haiku in F1.6 (tier 3)

All three arrive here with a populated Endpoint.response_schema tuple; no LLM
call is made at mock time. The mock_policy value is preserved in the response
dict under the key "_mock_policy" for logging and test assertions.

Faker rules are co-located here rather than in a separate faker_rules.py —
single-module scope, no unnecessary file proliferation.
"""

from __future__ import annotations

import re
from typing import Any

import structlog
from faker import Faker

from toolforge.execution.session import SessionState
from toolforge.registry.models import Endpoint
from toolforge.registry.semantic_vocab import CHAIN_ONLY_VOCAB, USER_PROVIDED_VOCAB

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Creation-verb heuristic
# ---------------------------------------------------------------------------

_CREATION_VERBS: frozenset[str] = frozenset({
    "create", "book", "order", "register", "add", "post",
    "submit", "make", "new", "place", "reserve", "insert",
    "write", "set", "put",
})


def _is_creation_endpoint(name: str) -> bool:
    """Return True if the endpoint name suggests it creates a new entity.

    Heuristic: the first camelCase word (or snake_case word) is a creation verb.

    Examples:
      createBooking → 'create' → True
      bookHotel     → 'book'   → True
      searchHotels  → 'search' → False
      getBooking    → 'get'    → False
    """
    if not name:
        return False
    if "_" in name:
        first_word = name.split("_")[0].lower()
    else:
        parts = re.split(r"(?=[A-Z])", name)
        first_word = (parts[0] if parts else name).lower()
    return first_word in _CREATION_VERBS


# ---------------------------------------------------------------------------
# Path flattening
# ---------------------------------------------------------------------------

def _last_path_segment(path: str) -> str:
    """Extract the last segment from a dot/bracket-notation path.

    Examples:
      'booking_id'         → 'booking_id'
      'results[].hotel_id' → 'hotel_id'
      'data.user.email'    → 'email'
      'data.id'            → 'id'
    """
    segments = [s for s in re.split(r"[.\[\]]+", path) if s]
    return segments[-1] if segments else path


def _build_flat_response(fields: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build a flat response dict from (path, value) pairs.

    Uses the last path segment as the key. On collision (two paths share the
    same last segment), falls back to the full path for both to avoid silent
    data loss.

    Example collision: ('a.id', 1), ('b.id', 2) → {'a.id': 1, 'b.id': 2}
    No collision:      ('results[].hotel_id', 'h1') → {'hotel_id': 'h1'}
    """
    last_segment_counts: dict[str, int] = {}
    for path, _ in fields:
        seg = _last_path_segment(path)
        last_segment_counts[seg] = last_segment_counts.get(seg, 0) + 1

    result: dict[str, Any] = {}
    for path, value in fields:
        seg = _last_path_segment(path)
        key = seg if last_segment_counts[seg] == 1 else path
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# Faker helpers
# ---------------------------------------------------------------------------

def _make_faker(seed: int) -> Faker:
    """Create a Faker instance seeded for this specific call."""
    faker = Faker()
    faker.seed_instance(seed)
    return faker


# Explicit USER_PROVIDED semantic_type → Faker method mapping.
# Types not listed fall back to faker.word().
_USER_PROVIDED_FAKER: dict[str, str] = {
    "city_name": "city",
    "country_code": "country_code",
    "country_name": "country",
    "company_name": "company",
    "email": "email",
    "phone_number": "phone_number",
    "url": "url",
    "date": "date",
    "datetime": "iso8601",
    "currency_code": "currency_code",
    "postal_code": "postcode",
    "search_query": "sentence",
}


def _generate_string(faker: Faker, semantic_type: str | None) -> str:
    """Generate a string value appropriate for the given semantic type."""
    if semantic_type and semantic_type in USER_PROVIDED_VOCAB:
        method = _USER_PROVIDED_FAKER.get(semantic_type)
        if method:
            try:
                return str(getattr(faker, method)())
            except Exception:
                pass
    return faker.word()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class MockResponder:
    """Generates deterministic mock responses from Endpoint.response_schema.

    All three mock_policy tiers use the same Faker-based generation.
    CHAIN_ONLY fields register produced values into SessionState so downstream
    executor calls can validate grounding.

    Usage::

        responder = MockResponder()
        response = responder.respond(endpoint, arguments, state)
        # state.available_values_by_type may now contain new entries
    """

    def respond(
        self,
        endpoint: Endpoint,
        arguments: dict[str, Any],
        state: SessionState,
    ) -> dict[str, Any]:
        """Generate a mock response dict for *endpoint*.

        Side-effects on *state*:
        - Appends freshly generated CHAIN_ONLY values to
          state.available_values_by_type[semantic_type].

        The response always includes a "_mock_policy" key for tier provenance.
        """
        # Each call in a session gets a unique but reproducible seed.
        faker = _make_faker(state.seed + len(state.tool_outputs))
        is_creation = _is_creation_endpoint(endpoint.name)

        fields: list[tuple[str, Any]] = []
        for rf in endpoint.response_schema:
            value = self._generate_field(
                field_type=rf.type,
                semantic_type=rf.semantic_type,
                faker=faker,
                state=state,
                is_creation=is_creation,
            )
            fields.append((rf.path, value))

        response = _build_flat_response(fields)
        response["_mock_policy"] = endpoint.mock_policy or "unknown"
        return response

    # ------------------------------------------------------------------
    # Field generation
    # ------------------------------------------------------------------

    def _generate_field(
        self,
        field_type: str,
        semantic_type: str | None,
        faker: Faker,
        state: SessionState,
        is_creation: bool,
    ) -> Any:
        """Generate one response field value.

        CHAIN_ONLY fields obey the reuse-vs-fresh rule:
        - Creation verb AND semantic type is CHAIN_ONLY → generate fresh, append.
        - Pool empty (regardless of verb) → generate fresh, append.
        - Otherwise → reuse pool[-1] (most-recently produced value, lookup semantics).
        """
        if semantic_type and semantic_type in CHAIN_ONLY_VOCAB:
            pool = state.available_values_by_type.setdefault(semantic_type, [])
            if is_creation or not pool:
                fresh = str(faker.uuid4())
                pool.append(fresh)
                log.debug(
                    "responder.chain_only.fresh",
                    semantic_type=semantic_type,
                    value=fresh,
                    reason="creation" if is_creation else "empty_pool",
                )
                return fresh
            else:
                reused = pool[-1]
                log.debug(
                    "responder.chain_only.reuse",
                    semantic_type=semantic_type,
                    value=reused,
                )
                return reused

        # Non-CHAIN_ONLY: type-based generation.
        if field_type == "string":
            return _generate_string(faker, semantic_type)
        elif field_type == "integer":
            return faker.random_int(min=1, max=9999)
        elif field_type == "number":
            return round(faker.pyfloat(min_value=0, max_value=9999, right_digits=2), 2)
        elif field_type == "boolean":
            return faker.boolean()
        elif field_type == "array":
            return [faker.word(), faker.word()]
        elif field_type == "object":
            return {"value": faker.word()}
        elif field_type == "null":
            return None
        else:
            return faker.word()
