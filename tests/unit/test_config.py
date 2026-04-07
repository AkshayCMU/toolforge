"""Unit tests for toolforge.config."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

import toolforge.config as config_module
from toolforge.config import Settings, get_settings


def _clear_cache() -> None:
    """Clear the lru_cache on get_settings between tests."""
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Missing required key
# ---------------------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings without anthropic_api_key raises ValidationError."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="anthropic_api_key"):
        Settings(_env_file=None)  # type: ignore[call-arg]  # nocheck


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------


def test_default_paths_are_relative() -> None:
    """Default path fields are relative (resolved relative to CWD)."""
    s = Settings(anthropic_api_key="sk-ant-test")  # type: ignore[call-arg]  # nocheck
    assert s.cache_dir == Path(".cache/")
    assert s.artifacts_dir == Path("artifacts/")
    assert s.runs_dir == Path("runs/")
    assert s.reports_dir == Path("reports/")
    assert s.toolbench_data_dir == Path("../toolbench_raw/data/data/toolenv/tools")


def test_paths_can_be_overridden(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path fields can be overridden via environment variables."""
    monkeypatch.setenv("CACHE_DIR", "/tmp/mycache")
    s = Settings(anthropic_api_key="sk-ant-test")  # type: ignore[call-arg]  # nocheck
    assert s.cache_dir == Path("/tmp/mycache")


# ---------------------------------------------------------------------------
# Log level validator
# ---------------------------------------------------------------------------


def test_invalid_log_level_raises() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(anthropic_api_key="sk-ant-test", log_level="VERBOSE")  # type: ignore[call-arg]  # nocheck


def test_log_level_case_insensitive() -> None:
    s = Settings(anthropic_api_key="sk-ant-test", log_level="debug")  # type: ignore[call-arg]  # nocheck
    assert s.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# get_settings cache
# ---------------------------------------------------------------------------


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() returns the same object on repeated calls."""
    _clear_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    _clear_cache()
