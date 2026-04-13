"""LLM client wrapper with structured output and content-addressed cache — F4.1.

Every LLM call in toolforge goes through this module. No agent subclass
calls anthropic.Anthropic() directly.

Two call variants:
  call()      — structured output via Pydantic (tool-use forcing)
  call_text() — free-text response (no schema enforcement)

Both variants are:
  - Cached on disk at .cache/llm/{key[:2]}/{key}.json (content-addressed)
  - Retried on transient errors only (APIConnectionError, RateLimitError,
    InternalServerError, ValidationError)
  - Logged via structlog after every live call

dry_run=True mode:
  - Cache hit  → return as normal
  - Cache miss → raise CacheMissError (used in unit tests to assert no live calls)

Cache payload (for debuggability):
  {"key": str, "model": str, "prompt_version": str, "schema": str, "response": Any}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import anthropic
import structlog
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Sentinel used as the "schema" key in cache entries for free-text calls.
_TEXT_SENTINEL = "__text__"


class CacheMissError(RuntimeError):
    """Raised in dry_run=True mode when no cached response exists for the key."""


class LLMClient:
    """Caching, retrying LLM client.

    Usage::

        client = LLMClient(model="claude-haiku-4-5-20251001", temperature=0.7)
        result: MySchema = client.call(system, user, MySchema)
        text: str = client.call_text(system, user)
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        cache_dir: Path = Path(".cache/llm"),
        dry_run: bool = False,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._cache_dir = Path(cache_dir)
        self._dry_run = dry_run
        self._max_tokens = max_tokens
        self._anthropic: anthropic.Anthropic | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        *,
        prompt_version: str = "v1",
        agent_name: str | None = None,
    ) -> T:
        """Structured-output call. Returns a validated instance of output_schema.

        Uses Anthropic tool-use forcing to guarantee the response conforms to
        the Pydantic schema. The model is given the schema as a single tool
        and forced to call it.

        agent_name is optional and included in the llm_client.usage log event
        for per-agent token attribution. It does NOT affect the cache key.
        """
        key = self._cache_key(
            system_prompt, user_prompt, output_schema.__name__,
            output_schema.model_json_schema(), prompt_version,
        )
        cached = self._load_cache(key)
        if cached is not None:
            log.debug("llm_client.cache_hit", model=self._model, schema=output_schema.__name__)
            log.info("llm_client.usage", model=self._model,
                     prompt_tokens=0, completion_tokens=0, cached=True,
                     agent_name=agent_name)
            return output_schema.model_validate(cached["response"])

        if self._dry_run:
            raise CacheMissError(
                f"dry_run=True: no cached response for key={key[:8]}... "
                f"(model={self._model}, schema={output_schema.__name__})"
            )

        log.info(
            "llm_client.live_call",
            model=self._model,
            schema=output_schema.__name__,
            agent_name=agent_name,
            key_prefix=key[:8],
        )
        result = self._live_structured_call(
            system_prompt, user_prompt, output_schema, agent_name=agent_name
        )
        self._save_cache(key, {
            "key": key,
            "model": self._model,
            "prompt_version": prompt_version,
            "schema": output_schema.__name__,
            "response": result.model_dump(mode="json"),
        })
        return result

    def call_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        prompt_version: str = "v1",
        agent_name: str | None = None,
    ) -> str:
        """Free-text call. Returns the assistant's first text content block.

        Uses a __text__ sentinel in the cache key so free-text and
        structured calls never collide even with identical prompts.

        agent_name is optional and included in the llm_client.usage log event
        for per-agent token attribution. It does NOT affect the cache key.
        """
        key = self._cache_key(
            system_prompt, user_prompt, _TEXT_SENTINEL, None, prompt_version,
        )
        cached = self._load_cache(key)
        if cached is not None:
            log.debug("llm_client.cache_hit", model=self._model, schema=_TEXT_SENTINEL)
            log.info("llm_client.usage", model=self._model,
                     prompt_tokens=0, completion_tokens=0, cached=True,
                     agent_name=agent_name)
            return cached["response"]

        if self._dry_run:
            raise CacheMissError(
                f"dry_run=True: no cached response for key={key[:8]}... "
                f"(model={self._model}, schema={_TEXT_SENTINEL})"
            )

        log.info(
            "llm_client.live_call",
            model=self._model,
            schema=_TEXT_SENTINEL,
            agent_name=agent_name,
            key_prefix=key[:8],
        )
        text = self._live_text_call(system_prompt, user_prompt, agent_name=agent_name)
        self._save_cache(key, {
            "key": key,
            "model": self._model,
            "prompt_version": prompt_version,
            "schema": _TEXT_SENTINEL,
            "response": text,
        })
        return text

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema_json: dict | None,
        prompt_version: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt_version": prompt_version,
            "system": system_prompt,
            "user": user_prompt,
            "schema": schema_name,
        }
        if schema_json is not None:
            payload["schema_json"] = schema_json
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / key[:2] / f"{key}.json"

    def _load_cache(self, key: str) -> dict | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log.warning("llm_client.cache_corrupt", key=key[:8])
            return None

    def _save_cache(self, key: str, payload: dict) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log.debug("llm_client.cache_write", key=key[:8])

    # ------------------------------------------------------------------
    # Live call helpers (retried)
    # ------------------------------------------------------------------

    def _get_client(self) -> anthropic.Anthropic:
        if self._anthropic is None:
            # Read the key via pydantic-settings (which loads .env) so that
            # the key does not need to be exported as an OS env var separately.
            from toolforge.config import get_settings
            self._anthropic = anthropic.Anthropic(
                api_key=get_settings().anthropic_api_key
            )
        return self._anthropic

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
            ValidationError,
        )),
        reraise=True,
    )
    def _live_structured_call(
        self,
        system_prompt: str,
        user_prompt: str,
        output_schema: type[T],
        agent_name: str | None = None,
    ) -> T:
        """Make a live structured-output call using tool-use forcing.

        Returns a validated instance of output_schema.
        ValidationError is in the retry set — near-valid JSON from the model
        often succeeds on a second attempt.
        """
        tool_def = {
            "name": "output",
            "description": f"Return a structured {output_schema.__name__} object.",
            "input_schema": output_schema.model_json_schema(),
        }
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[tool_def],
            tool_choice={"type": "tool", "name": "output"},
        )
        log.info(
            "llm_client.usage",
            model=self._model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            cached=False,
            agent_name=agent_name,
        )
        # Extract the tool_use content block's input dict and validate.
        for block in response.content:
            if block.type == "tool_use" and block.name == "output":
                return output_schema.model_validate(block.input)
        raise RuntimeError(
            f"Anthropic response contained no tool_use block for schema "
            f"{output_schema.__name__!r}. Content types: "
            f"{[b.type for b in response.content]}"
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        )),
        reraise=True,
    )
    def _live_text_call(
        self, system_prompt: str, user_prompt: str, agent_name: str | None = None
    ) -> str:
        """Make a live free-text call."""
        client = self._get_client()
        response = client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        log.info(
            "llm_client.usage",
            model=self._model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            cached=False,
            agent_name=agent_name,
        )
        for block in response.content:
            if hasattr(block, "text"):
                return block.text
        return ""
