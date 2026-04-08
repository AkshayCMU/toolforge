"""Raw ToolBench JSON walker (F1.2).

Pure I/O generator — no normalization, no LLM.  Yields one dict per tool
file so the normalizer can decide what to keep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import structlog

log = structlog.get_logger(__name__)

# Directory names the walker should never descend into.
_SKIP_DIRS = frozenset({"__MACOSX", ".DS_Store"})


def walk_toolbench(
    root: Path,
) -> Iterator[tuple[str, str, dict]]:
    """Walk *root* and yield ``(category, tool_name, raw_json_dict)``.

    *root* is expected to point at the ``tools/`` directory that contains
    one sub-directory per category, each containing ``*.json`` tool files.

    Guarantees
    ----------
    * Pure generator — never loads all 16 k files at once.
    * Malformed JSON → log warning, skip the file.
    * Wrong-shape JSON (not a tool def) is still yielded — the normalizer
      decides whether to keep or drop it.
    * ``__MACOSX/`` and hidden directories are silently skipped.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"ToolBench data directory not found: {root}"
        )

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir():
            continue
        if category_dir.name in _SKIP_DIRS or category_dir.name.startswith("."):
            log.debug("walker.skip_dir", path=str(category_dir))
            continue

        category = category_dir.name

        for tool_file in sorted(category_dir.iterdir()):
            if not tool_file.is_file() or tool_file.suffix != ".json":
                continue
            if tool_file.name.startswith("._"):
                log.debug("walker.skip_macosx_resource", path=str(tool_file))
                continue

            tool_name = tool_file.stem

            try:
                raw = json.loads(tool_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning(
                    "walker.bad_json",
                    path=str(tool_file),
                    error=str(exc),
                )
                continue

            yield category, tool_name, raw
