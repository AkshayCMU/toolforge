"""Tool call executor with grounding enforcement — F3.3.

execute() is the hot path: pure Python, no LLM calls.

Five-step contract:
  1. Endpoint lookup — unknown endpoint_id → structured error
  2. Structural validation — required params present
  3. Semantic grounding check — CHAIN_ONLY args must exist in session pool
  4. Generate mock response via MockResponder
  5. Build ToolOutput, append to state, return

On any error (steps 1–3): returns a ToolOutput with error set and does NOT
append to state.tool_outputs. The failed call is not part of the session log.

Grounding invariant (P3 layer 3):
  Every argument whose Parameter.semantic_type is in chain_only_types must
  appear in state.available_values_by_type[semantic_type]. Hallucinated IDs
  are rejected with a structured error that lists valid values so the
  Assistant agent can self-correct.

The CHAIN_ONLY set defaults to semantic_vocab.CHAIN_ONLY_VOCAB (the seed
vocabulary from F1.5). Inject a smaller set in tests to isolate behaviour.
"""

from __future__ import annotations

from typing import Any

import structlog

from toolforge.execution.mock_responder import MockResponder
from toolforge.execution.session import SessionState, ToolOutput
from toolforge.registry.models import Endpoint
from toolforge.registry.semantic_vocab import CHAIN_ONLY_VOCAB

log = structlog.get_logger(__name__)


class Executor:
    """Stateless executor — all mutable state lives in SessionState.

    Owns the P3 grounding invariant (layer 3 of three).
    """

    def __init__(
        self,
        endpoint_registry: dict[str, Endpoint],
        responder: MockResponder,
        chain_only_types: frozenset[str] = CHAIN_ONLY_VOCAB,
    ) -> None:
        self._registry = endpoint_registry
        self._responder = responder
        self._chain_only = chain_only_types

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        endpoint_id: str,
        arguments: dict[str, Any],
        state: SessionState,
    ) -> ToolOutput:
        """Execute a tool call and return a ToolOutput.

        Errors return a ToolOutput without appending to state; the caller
        can log or repair the failed call. Successes are always appended.
        """
        # Step 1: Endpoint lookup.
        endpoint = self._registry.get(endpoint_id)
        if endpoint is None:
            sample = sorted(self._registry)[:5]
            return self._error(
                endpoint_id, arguments, state,
                f"Unknown endpoint: {endpoint_id!r}. Known (sample): {sample}",
            )

        # Step 2: Structural validation — all required params present.
        for param in endpoint.parameters:
            if param.required and param.name not in arguments:
                return self._error(
                    endpoint_id, arguments, state,
                    f"Missing required parameter: {param.name!r}",
                )

        # Step 3: Semantic grounding check for CHAIN_ONLY params.
        for param in endpoint.parameters:
            if not param.semantic_type:
                continue
            if param.semantic_type not in self._chain_only:
                continue
            value = arguments.get(param.name)
            if value is None:
                # Absent (optional) or explicitly None — skip grounding check.
                continue
            pool = state.available_values_by_type.get(param.semantic_type, [])
            if value not in pool:
                return self._error(
                    endpoint_id, arguments, state,
                    f"Invalid {param.semantic_type}: {value!r} not in session. "
                    f"Valid values: {pool}",
                )

        # Step 4: Generate mock response (may register values into state).
        response = self._responder.respond(endpoint, arguments, state)

        # Step 5: Build output, append to state, return.
        timestamp = f"turn-{len(state.tool_outputs)}"
        output = ToolOutput(
            endpoint_id=endpoint_id,
            arguments=arguments,
            response=response,
            error=None,
            timestamp=timestamp,
        )
        state.tool_outputs.append(output)
        log.debug("executor.success", endpoint_id=endpoint_id, turn=timestamp)
        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error(
        endpoint_id: str,
        arguments: dict[str, Any],
        state: SessionState,
        message: str,
    ) -> ToolOutput:
        """Build an error ToolOutput. Does NOT append to state."""
        timestamp = f"turn-{len(state.tool_outputs)}"
        log.debug("executor.error", endpoint_id=endpoint_id, error=message)
        return ToolOutput(
            endpoint_id=endpoint_id,
            arguments=arguments,
            response=None,
            error=message,
            timestamp=timestamp,
        )
