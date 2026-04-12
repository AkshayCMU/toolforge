"""Unit tests for F4.1 — LLMClient and Agent ABC.

All tests operate with dry_run=True or mocked Anthropic calls.
No live LLM calls are made. Tests that exercise the live path mock
anthropic.Anthropic to avoid network I/O and API keys.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import anthropic
import pytest
import structlog.testing
from pydantic import BaseModel

from toolforge.agents.base import Agent
from toolforge.agents.llm_client import CacheMissError, LLMClient


# ---------------------------------------------------------------------------
# Test schemas
# ---------------------------------------------------------------------------

class _SchemaA(BaseModel):
    value: str
    count: int


class _SchemaB(BaseModel):
    name: str


# ---------------------------------------------------------------------------
# Mock response builders
# ---------------------------------------------------------------------------

def _structured_response(data: dict) -> MagicMock:
    """Build a mock Anthropic response for a tool_use structured call."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = "output"
    block.input = data
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 50
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    return resp


def _text_response(text: str) -> MagicMock:
    """Build a mock Anthropic response for a free-text call."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    usage = MagicMock()
    usage.input_tokens = 80
    usage.output_tokens = 30
    resp = MagicMock()
    resp.content = [block]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Cache seed helper (shared with agent tests)
# ---------------------------------------------------------------------------

def seed_cache(
    client: LLMClient,
    system: str,
    user: str,
    schema: type[BaseModel] | None,
    data: dict | str,
    prompt_version: str = "v1",
) -> None:
    """Write a pre-built response into the client's cache directory."""
    if schema is not None:
        key = client._cache_key(
            system, user, schema.__name__, schema.model_json_schema(), prompt_version
        )
        payload: dict[str, Any] = {
            "key": key,
            "model": client._model,
            "prompt_version": prompt_version,
            "schema": schema.__name__,
            "response": data,
        }
    else:
        key = client._cache_key(system, user, "__text__", None, prompt_version)
        payload = {
            "key": key,
            "model": client._model,
            "prompt_version": prompt_version,
            "schema": "__text__",
            "response": data,
        }
    path = client._cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# F4.1 — LLMClient structured call
# ---------------------------------------------------------------------------

def test_structured_call_populates_cache(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = _structured_response(
        {"value": "hello", "count": 3}
    )
    client._anthropic = mock_anthropic

    result = client.call("sys", "usr", _SchemaA)

    assert isinstance(result, _SchemaA)
    assert result.value == "hello"
    assert result.count == 3

    # Cache file must exist
    key = client._cache_key("sys", "usr", "_SchemaA", _SchemaA.model_json_schema(), "v1")
    cache_path = client._cache_path(key)
    assert cache_path.exists()
    payload = json.loads(cache_path.read_text())
    assert payload["key"] == key
    assert payload["model"] == "claude-test"
    assert payload["prompt_version"] == "v1"
    assert payload["schema"] == "_SchemaA"
    assert "response" in payload


def test_second_structured_call_hits_cache(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = _structured_response(
        {"value": "x", "count": 1}
    )
    client._anthropic = mock_anthropic

    client.call("sys", "usr", _SchemaA)
    client.call("sys", "usr", _SchemaA)  # second call — must hit cache

    assert mock_anthropic.messages.create.call_count == 1


def test_text_call_cached_separately(tmp_path: Path) -> None:
    """call_text and call with same prompts use different cache keys."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = [
        _structured_response({"value": "a", "count": 0}),
        _text_response("hello text"),
    ]
    client._anthropic = mock_anthropic

    client.call("sys", "usr", _SchemaA)
    client.call_text("sys", "usr")

    # Both calls were live (different keys → two cache misses)
    assert mock_anthropic.messages.create.call_count == 2

    # Verify the text cache key uses __text__ sentinel
    text_key = client._cache_key("sys", "usr", "__text__", None, "v1")
    struct_key = client._cache_key("sys", "usr", "_SchemaA", _SchemaA.model_json_schema(), "v1")
    assert text_key != struct_key


def test_dry_run_cache_hit_returns_model(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path, dry_run=True)
    seed_cache(client, "sys", "usr", _SchemaA, {"value": "cached", "count": 7})

    result = client.call("sys", "usr", _SchemaA)
    assert isinstance(result, _SchemaA)
    assert result.value == "cached"
    assert result.count == 7


def test_dry_run_cache_miss_raises(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path, dry_run=True)
    with pytest.raises(CacheMissError, match="dry_run=True"):
        client.call("sys", "usr", _SchemaA)


def test_dry_run_text_cache_hit(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path, dry_run=True)
    seed_cache(client, "sys", "usr", None, "pre-cached text")

    result = client.call_text("sys", "usr")
    assert result == "pre-cached text"


def test_dry_run_text_cache_miss_raises(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path, dry_run=True)
    with pytest.raises(CacheMissError, match="dry_run=True"):
        client.call_text("sys", "usr")


def test_cache_key_changes_on_schema_change(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    key_a = client._cache_key("sys", "usr", "_SchemaA", _SchemaA.model_json_schema(), "v1")
    key_b = client._cache_key("sys", "usr", "_SchemaB", _SchemaB.model_json_schema(), "v1")
    assert key_a != key_b


def test_cache_key_changes_on_prompt_version(tmp_path: Path) -> None:
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    key_v1 = client._cache_key("sys", "usr", "_SchemaA", _SchemaA.model_json_schema(), "v1")
    key_v2 = client._cache_key("sys", "usr", "_SchemaA", _SchemaA.model_json_schema(), "v2")
    assert key_v1 != key_v2


def test_transient_error_retried(tmp_path: Path) -> None:
    """APIConnectionError on first attempt → retry succeeds on second."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = [
        anthropic.APIConnectionError(request=MagicMock()),
        _structured_response({"value": "retry-ok", "count": 0}),
    ]
    client._anthropic = mock_anthropic

    with patch("time.sleep"):  # suppress tenacity wait
        result = client.call("sys", "usr", _SchemaA)

    assert result.value == "retry-ok"
    assert mock_anthropic.messages.create.call_count == 2


def test_auth_error_not_retried(tmp_path: Path) -> None:
    """AuthenticationError is a config error — must NOT be retried."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.side_effect = anthropic.AuthenticationError(
        message="Bad key", response=MagicMock(), body={}
    )
    client._anthropic = mock_anthropic

    with pytest.raises(anthropic.AuthenticationError):
        client.call("sys", "usr", _SchemaA)

    assert mock_anthropic.messages.create.call_count == 1


def test_token_usage_logged(tmp_path: Path) -> None:
    """Live call emits a structlog event with prompt/completion tokens."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = _structured_response(
        {"value": "log-test", "count": 0}
    )
    client._anthropic = mock_anthropic

    with structlog.testing.capture_logs() as logs:
        client.call("sys", "usr", _SchemaA)

    usage_events = [e for e in logs if e.get("event") == "llm_client.usage"]
    assert len(usage_events) >= 1
    ev = usage_events[0]
    assert ev["prompt_tokens"] == 100
    assert ev["completion_tokens"] == 50
    assert ev["cached"] is False


def test_cached_call_logs_usage_with_zero_tokens(tmp_path: Path) -> None:
    """Cache hit emits a usage event with prompt/completion = 0 and cached=True."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path, dry_run=True)
    seed_cache(client, "sys", "usr", _SchemaA, {"value": "c", "count": 0})

    with structlog.testing.capture_logs() as logs:
        client.call("sys", "usr", _SchemaA)

    usage_events = [e for e in logs if e.get("event") == "llm_client.usage"]
    assert len(usage_events) >= 1
    ev = usage_events[0]
    assert ev["prompt_tokens"] == 0
    assert ev["completion_tokens"] == 0
    assert ev["cached"] is True


# ---------------------------------------------------------------------------
# F4.1 — Agent ABC
# ---------------------------------------------------------------------------

def test_agent_abc_cannot_be_instantiated_directly(tmp_path: Path) -> None:
    """Agent has an abstract member — instantiating it with a client still raises TypeError."""
    client = LLMClient(model="test", cache_dir=tmp_path)
    with pytest.raises(TypeError, match="abstract"):
        Agent(client)  # type: ignore[abstract]


def test_agent_subclass_stores_client(tmp_path: Path) -> None:
    """A concrete subclass with name defined stores the client correctly."""

    class ConcreteAgent(Agent):
        name = "concrete"  # satisfies the abstract property

    client = LLMClient(model="test", cache_dir=tmp_path)
    agent = ConcreteAgent(client)
    assert agent._client is client


# ---------------------------------------------------------------------------
# F4.1 — agent_name in usage log
# ---------------------------------------------------------------------------

def test_agent_name_appears_in_usage_log_live_call(tmp_path: Path) -> None:
    """Live structured call includes agent_name in the llm_client.usage event."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = _structured_response(
        {"value": "x", "count": 1}
    )
    client._anthropic = mock_anthropic

    with structlog.testing.capture_logs() as logs:
        client.call("sys", "usr", _SchemaA, agent_name="planner")

    usage_events = [e for e in logs if e.get("event") == "llm_client.usage"]
    assert any(e.get("agent_name") == "planner" for e in usage_events)


def test_agent_name_appears_in_usage_log_cache_hit(tmp_path: Path) -> None:
    """Cache-hit path includes agent_name in the llm_client.usage event."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path, dry_run=True)
    seed_cache(client, "sys", "usr", _SchemaA, {"value": "c", "count": 0})

    with structlog.testing.capture_logs() as logs:
        client.call("sys", "usr", _SchemaA, agent_name="judge")

    usage_events = [e for e in logs if e.get("event") == "llm_client.usage"]
    assert any(e.get("agent_name") == "judge" for e in usage_events)


def test_agent_name_none_does_not_break_logging(tmp_path: Path) -> None:
    """Omitting agent_name (None) still produces a valid usage log event."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path, dry_run=True)
    seed_cache(client, "sys", "usr", _SchemaA, {"value": "c", "count": 0})

    with structlog.testing.capture_logs() as logs:
        client.call("sys", "usr", _SchemaA)  # no agent_name

    usage_events = [e for e in logs if e.get("event") == "llm_client.usage"]
    assert len(usage_events) >= 1
    # agent_name key is present and is None
    assert usage_events[0].get("agent_name") is None


def test_agent_name_does_not_affect_cache_key(tmp_path: Path) -> None:
    """agent_name must NOT change the cache key — different names hit the same cache entry."""
    client = LLMClient(model="claude-test", cache_dir=tmp_path)
    mock_anthropic = MagicMock()
    mock_anthropic.messages.create.return_value = _structured_response(
        {"value": "x", "count": 1}
    )
    client._anthropic = mock_anthropic

    client.call("sys", "usr", _SchemaA, agent_name="planner")
    # Second call with a different agent_name must hit the cache (one live call total).
    client.call("sys", "usr", _SchemaA, agent_name="judge")

    assert mock_anthropic.messages.create.call_count == 1
