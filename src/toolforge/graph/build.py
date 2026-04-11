"""NetworkX graph builder from the enriched registry — F2.1.

Produces a deterministic MultiDiGraph with five edge types:
  BELONGS_TO   Endpoint → Tool
  IN_CATEGORY  Tool → Category
  CONSUMES     Endpoint → SemanticType  (via parameter semantic_type)
  PRODUCES     Endpoint → SemanticType  (via response field semantic_type)
  CHAINS_TO    Endpoint → Endpoint      (A produces T, B consumes T, T in chain_only_types)

Determinism guarantee: same registry + chain_only_types → byte-identical graph pickle.
All node/edge insertion follows sorted order; no bare set/dict iteration.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import structlog

from toolforge.registry.models import Tool

log = structlog.get_logger(__name__)

# Pickle protocol pinned for reproducibility within CPython 3.11.
_PICKLE_PROTOCOL = 5


# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------

def _cat_id(category: str) -> str:
    return f"cat:{category}"


def _tool_id(category: str, tool_name: str) -> str:
    return f"tool:{category}/{tool_name}"


def _ep_id(endpoint_id: str) -> str:
    return f"ep:{endpoint_id}"


def _st_id(semantic_type: str) -> str:
    return f"st:{semantic_type}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(tools: list[Tool], chain_only_types: list[str]) -> nx.MultiDiGraph:
    """Build a deterministic MultiDiGraph from the enriched registry.

    Args:
        tools: Enriched Tool list from Phase 1 (semantic types populated).
        chain_only_types: List of opaque ID types that can only come from a
            prior tool output (from artifacts/chain_only_types.json).

    Returns:
        A MultiDiGraph with BELONGS_TO, IN_CATEGORY, CONSUMES, PRODUCES,
        and CHAINS_TO edges. All node/edge insertion is deterministic.
    """
    chain_only_set: frozenset[str] = frozenset(chain_only_types)
    G: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Pass 1: add category, tool, and endpoint nodes (sorted insertion).
    # ------------------------------------------------------------------
    sorted_tools = sorted(tools, key=lambda t: (t.category, t.name))

    categories_seen: set[str] = set()
    for tool in sorted_tools:
        # Category node (idempotent)
        if tool.category not in categories_seen:
            G.add_node(_cat_id(tool.category), node_type="category", name=tool.category)
            categories_seen.add(tool.category)

        # Tool node
        G.add_node(
            _tool_id(tool.category, tool.name),
            node_type="tool",
            name=tool.name,
            category=tool.category,
        )

        # Endpoint nodes (sorted by id within each tool)
        for ep in sorted(tool.endpoints, key=lambda e: e.id):
            has_typed_param = any(p.semantic_type for p in ep.parameters)
            has_typed_field = any(f.semantic_type for f in ep.response_schema)
            terminal = not has_typed_param and not has_typed_field

            G.add_node(
                _ep_id(ep.id),
                node_type="endpoint",
                id=ep.id,
                tool=tool.name,
                category=tool.category,
                terminal=terminal,
            )

    # ------------------------------------------------------------------
    # Pass 2: add SemanticType nodes (collect all, then insert sorted).
    # ------------------------------------------------------------------
    all_semantic_types: set[str] = set()
    for tool in sorted_tools:
        for ep in tool.endpoints:
            for p in ep.parameters:
                if p.semantic_type:
                    all_semantic_types.add(p.semantic_type)
            for f in ep.response_schema:
                if f.semantic_type:
                    all_semantic_types.add(f.semantic_type)

    for st in sorted(all_semantic_types):
        G.add_node(_st_id(st), node_type="semantic_type", name=st)

    # ------------------------------------------------------------------
    # Pass 3: structural edges — BELONGS_TO and IN_CATEGORY (sorted).
    # ------------------------------------------------------------------
    for tool in sorted_tools:
        tool_node = _tool_id(tool.category, tool.name)
        cat_node = _cat_id(tool.category)
        G.add_edge(tool_node, cat_node, edge_type="IN_CATEGORY")

        for ep in sorted(tool.endpoints, key=lambda e: e.id):
            G.add_edge(_ep_id(ep.id), tool_node, edge_type="BELONGS_TO")

    # ------------------------------------------------------------------
    # Pass 4: semantic edges — CONSUMES and PRODUCES (sorted).
    # ------------------------------------------------------------------
    for tool in sorted_tools:
        for ep in sorted(tool.endpoints, key=lambda e: e.id):
            ep_node = _ep_id(ep.id)

            for p in sorted(ep.parameters, key=lambda x: x.name):
                if p.semantic_type:
                    G.add_edge(ep_node, _st_id(p.semantic_type), edge_type="CONSUMES")

            for f in sorted(ep.response_schema, key=lambda x: x.path):
                if f.semantic_type:
                    G.add_edge(ep_node, _st_id(f.semantic_type), edge_type="PRODUCES")

    # ------------------------------------------------------------------
    # Pass 5: CHAINS_TO edges — materialize (A produces T, B consumes T,
    # T in chain_only_set). Both endpoint loops are sorted.
    # ------------------------------------------------------------------
    # Build producer/consumer indexes for efficient derivation.
    # produced_by[ep_id] = sorted set of chain-only types produced
    # consumed_by[ep_id] = sorted set of all typed params
    produced_by: dict[str, list[str]] = {}
    consumed_by: dict[str, list[str]] = {}

    all_endpoints_sorted: list[Any] = []
    for tool in sorted_tools:
        for ep in sorted(tool.endpoints, key=lambda e: e.id):
            all_endpoints_sorted.append(ep)
            produced_by[ep.id] = sorted(
                {f.semantic_type for f in ep.response_schema
                 if f.semantic_type and f.semantic_type in chain_only_set}
            )
            consumed_by[ep.id] = sorted(
                {p.semantic_type for p in ep.parameters if p.semantic_type}
            )

    for ep_a in all_endpoints_sorted:
        produced = set(produced_by[ep_a.id])
        if not produced:
            continue
        for ep_b in all_endpoints_sorted:
            if ep_b.id == ep_a.id:
                continue
            consumed = set(consumed_by[ep_b.id])
            shared = produced & consumed
            if not shared:
                continue
            # One edge per bridging type (sorted for determinism).
            for via_type in sorted(shared):
                G.add_edge(
                    _ep_id(ep_a.id),
                    _ep_id(ep_b.id),
                    edge_type="CHAINS_TO",
                    via_type=via_type,
                )

    log.info(
        "graph.built",
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        endpoints=sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "endpoint"),
        chains_to=sum(
            1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "CHAINS_TO"
        ),
    )
    return G


def save_graph(graph: nx.MultiDiGraph, artifacts_dir: Path) -> None:
    """Pickle graph to artifacts_dir/graph.pkl and write graph_report.json."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = artifacts_dir / "graph.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(graph, f, protocol=_PICKLE_PROTOCOL)
    log.info("graph.saved", path=str(pkl_path))

    # Build edge-type counts.
    edge_type_counts: dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        et = data.get("edge_type", "unknown")
        edge_type_counts[et] = edge_type_counts.get(et, 0) + 1

    node_type_counts: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        nt = data.get("node_type", "unknown")
        node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

    # Registry hash — read from graph.pkl bytes for a stable fingerprint.
    pkl_bytes = pkl_path.read_bytes()
    graph_hash = hashlib.sha256(pkl_bytes).hexdigest()[:16]

    report = {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "node_type_counts": dict(sorted(node_type_counts.items())),
        "edge_type_counts": dict(sorted(edge_type_counts.items())),
        "pickle_protocol": _PICKLE_PROTOCOL,
        "graph_hash": graph_hash,
    }
    report_path = artifacts_dir / "graph_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("graph.report_saved", path=str(report_path))


def load_graph(artifacts_dir: Path) -> nx.MultiDiGraph:
    """Load the pickled graph from artifacts_dir/graph.pkl."""
    pkl_path = artifacts_dir / "graph.pkl"
    with open(pkl_path, "rb") as f:
        graph: nx.MultiDiGraph = pickle.load(f)
    log.info("graph.loaded", nodes=graph.number_of_nodes(), edges=graph.number_of_edges())
    return graph
