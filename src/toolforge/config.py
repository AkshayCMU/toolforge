"""Settings and structlog configuration for toolforge."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path

import structlog
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Required
    anthropic_api_key: str = Field(..., description="Anthropic API key (sk-ant-...).")

    # Paths — all resolved relative to CWD at access time
    toolbench_data_dir: Path = Field(
        default=Path("../toolbench_raw/data/data/toolenv/tools"),
        description="Root of raw ToolBench tool JSON files.",
    )
    cache_dir: Path = Field(default=Path(".cache/"), description="LLM response cache.")
    artifacts_dir: Path = Field(
        default=Path("artifacts/"), description="Build artifacts output dir."
    )
    runs_dir: Path = Field(
        default=Path("runs/"), description="Generated dataset output dir."
    )
    reports_dir: Path = Field(
        default=Path("reports/"), description="Evaluation report output dir."
    )

    log_level: str = Field(default="INFO", description="Logging level.")

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()  # type: ignore[call-arg]


def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog: JSON in non-TTY, human-readable in TTY."""
    if settings is None:
        settings = get_settings()

    log_level = getattr(logging, settings.log_level, logging.INFO)
    is_tty = sys.stderr.isatty()

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if is_tty:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Keep stdlib logging quiet unless DEBUG
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stderr,
    )
