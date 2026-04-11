"""Unit tests for toolforge.graph.build (F2.1)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import networkx as nx
import pytest

from toolforge.graph.build import build_graph, load_graph, save_graph
from toolforge.registry.models import (
    Endpoint,
    Parameter,
    ParamProvenance,
    ResponseField,
    Tool,
)


# ---------------------------------------------------------------------------
# Helpers — mirrors test_models.py pattern
# ---------------------------------------------------------------------------

def _prov(**kw) -> ParamProvenance:
    defaults = dict(raw_required_field="required_parameters", raw_type_string="STRING")
    defaults.update(kw)
    return ParamProvenance(**defaults)


def _param(name: str = "q", semantic_type: str | None = None) -> Parameter:
    return Parameter(
        name=name,
        type="string",
        description=f"Param {name}",
        required=True,
        semantic_type=semantic_type,
        provenance=_prov(),
    )


def _field(path: str = "id", semantic_type: str | None = None) -> ResponseField:
    return ResponseField(path=path, type="string", semantic_type=semantic_type)


def _ep(
    ep_id: str,
    params: tuple[Parameter, ...] = (),
    fields: tuple[ResponseField, ...] = (),
) -> Endpoint:
    return Endpoint(id=ep_id, name=ep_id.split("/")[-1], description="d",
                    parameters=params, response_schema=fields)


def _tool(name: str, category: str, endpoints: tuple[Endpoint, ...]) -> Tool:
    return Tool(name=name, category=category, description="d",
                file_stem=name.lower(), endpoints=endpoints)


# ---------------------------------------------------------------------------
# Minimal registry used by most tests:
# 2 categories, 3 tools, 5 endpoints, 2 semantic types
# ep_A produces booking_id (CHAIN_ONLY) → ep_B consumes it (CHAINS_TO edge)
# ---------------------------------------------------------------------------

CHAIN_ONLY = ["booking_id"]

EP_A = _ep("Cat1/ToolA/searchA",
            fields=(_field("booking_id", "booking_id"),))
EP_B = _ep("Cat1/ToolA/confirmB",
            params=(_param("bid", "booking_id"),))
EP_C = _ep("Cat1/ToolB/detailC",
            params=(_param("q", "city_name"),),          # city_name NOT in chain_only
            fields=(_field("price", "price"),))
EP_D = _ep("Cat2/ToolC/listD")                           # no semantic types → terminal
EP_E = _ep("Cat2/ToolC/fetchE",
            params=(_param("uid", "user_id"),))          # user_id NOT in chain_only

TOOL_A = _tool("ToolA", "Cat1", (EP_A, EP_B))
TOOL_B = _tool("ToolB", "Cat1", (EP_C,))
TOOL_C = _tool("ToolC", "Cat2", (EP_D, EP_E))

ALL_TOOLS = [TOOL_A, TOOL_B, TOOL_C]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphDeterminism:
    def test_identical_graphs_same_input(self):
        g1 = build_graph(ALL_TOOLS, CHAIN_ONLY)
        g2 = build_graph(ALL_TOOLS, CHAIN_ONLY)

        assert sorted(g1.nodes()) == sorted(g2.nodes())

        def edge_keys(g: nx.MultiDiGraph) -> list[tuple]:
            return sorted(
                (u, v, d.get("edge_type", ""), d.get("via_type", ""))
                for u, v, d in g.edges(data=True)
            )

        assert edge_keys(g1) == edge_keys(g2)


class TestNodeCounts:
    def test_counts_by_type(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        counts = {}
        for _, d in g.nodes(data=True):
            nt = d["node_type"]
            counts[nt] = counts.get(nt, 0) + 1

        assert counts["category"] == 2      # Cat1, Cat2
        assert counts["tool"] == 3          # ToolA, ToolB, ToolC
        assert counts["endpoint"] == 5      # EP_A … EP_E

        # Semantic types: booking_id, city_name, price, user_id
        assert counts["semantic_type"] == 4


class TestStructuralEdges:
    def test_belongs_to_edges(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        belongs_to = [
            (u, v) for u, v, d in g.edges(data=True)
            if d.get("edge_type") == "BELONGS_TO"
        ]
        assert len(belongs_to) == 5  # one per endpoint

    def test_in_category_edges(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        in_cat = [
            (u, v) for u, v, d in g.edges(data=True)
            if d.get("edge_type") == "IN_CATEGORY"
        ]
        assert len(in_cat) == 3  # one per tool

    def test_belongs_to_targets_correct_tools(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        ep_node = "ep:Cat1/ToolA/searchA"
        targets = [
            v for u, v, d in g.out_edges(ep_node, data=True)
            if d.get("edge_type") == "BELONGS_TO"
        ]
        assert targets == ["tool:Cat1/ToolA"]


class TestSemanticEdges:
    def test_consumes_edge(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        consumes = [
            (u, v) for u, v, d in g.out_edges("ep:Cat1/ToolA/confirmB", data=True)
            if d.get("edge_type") == "CONSUMES"
        ]
        assert ("ep:Cat1/ToolA/confirmB", "st:booking_id") in consumes

    def test_produces_edge(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        produces = [
            (u, v) for u, v, d in g.out_edges("ep:Cat1/ToolA/searchA", data=True)
            if d.get("edge_type") == "PRODUCES"
        ]
        assert ("ep:Cat1/ToolA/searchA", "st:booking_id") in produces

    def test_null_semantic_type_no_edge(self):
        """Endpoints with semantic_type=None must not create CONSUMES/PRODUCES edges."""
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        ep_node = "ep:Cat2/ToolC/listD"  # terminal, no semantic types
        semantic_edges = [
            d for _, _, d in g.out_edges(ep_node, data=True)
            if d.get("edge_type") in ("CONSUMES", "PRODUCES")
        ]
        assert semantic_edges == []


class TestChainsToEdges:
    def test_chains_to_created_for_chain_only_type(self):
        """ep_A produces booking_id (CHAIN_ONLY) → ep_B consumes it: CHAINS_TO edge."""
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        chains = [
            (u, v, d) for u, v, d in g.edges(data=True)
            if d.get("edge_type") == "CHAINS_TO"
        ]
        edge_triples = [(u, v, d["via_type"]) for u, v, d in chains]
        assert ("ep:Cat1/ToolA/searchA", "ep:Cat1/ToolA/confirmB", "booking_id") in edge_triples

    def test_chains_to_not_for_user_provided_type(self):
        """city_name is NOT in chain_only_types → no CHAINS_TO edge for it."""
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        # EP_C produces 'price' and consumes 'city_name'. Neither is CHAIN_ONLY.
        # No endpoint produces city_name via response_schema in our fixture.
        chains_via_city = [
            d for _, _, d in g.edges(data=True)
            if d.get("edge_type") == "CHAINS_TO" and d.get("via_type") == "city_name"
        ]
        assert chains_via_city == []

    def test_chains_to_not_for_null_semantic(self):
        """Response fields with semantic_type=None must not contribute CHAINS_TO edges."""
        null_ep = _ep("Cat1/ToolA/nullEp",
                      fields=(_field("result", None),))
        null_tool = _tool("ToolA", "Cat1", (null_ep, EP_B))
        g = build_graph([null_tool, TOOL_B, TOOL_C], CHAIN_ONLY)
        # null_ep produces nothing typed → no CHAINS_TO from it
        chains_from_null = [
            (u, v) for u, v, d in g.edges(data=True)
            if u == "ep:Cat1/ToolA/nullEp" and d.get("edge_type") == "CHAINS_TO"
        ]
        assert chains_from_null == []

    def test_no_co_category_edges(self):
        """CO_CATEGORY must not appear — design decision confirmed."""
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        co_cat = [
            d for _, _, d in g.edges(data=True)
            if d.get("edge_type") == "CO_CATEGORY"
        ]
        assert co_cat == []


class TestTerminalFlag:
    def test_terminal_endpoint_flagged(self):
        """EP_D has no typed params and no typed response fields → terminal=True."""
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        assert g.nodes["ep:Cat2/ToolC/listD"]["terminal"] is True

    def test_non_terminal_endpoint_not_flagged(self):
        """EP_A has a typed response field → terminal=False."""
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        assert g.nodes["ep:Cat1/ToolA/searchA"]["terminal"] is False


class TestLoadRoundtrip:
    def test_save_load_preserves_graph(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = Path(tmpdir)
            save_graph(g, artifacts)
            g2 = load_graph(artifacts)

        assert g2.number_of_nodes() == g.number_of_nodes()
        assert g2.number_of_edges() == g.number_of_edges()

        # Spot-check node attributes survive round-trip.
        assert g2.nodes["ep:Cat1/ToolA/searchA"]["terminal"] is False
        assert g2.nodes["ep:Cat2/ToolC/listD"]["terminal"] is True

    def test_graph_report_json_written(self):
        g = build_graph(ALL_TOOLS, CHAIN_ONLY)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = Path(tmpdir)
            save_graph(g, artifacts)
            report_path = artifacts / "graph_report.json"
            assert report_path.exists()

            import json
            report = json.loads(report_path.read_text())
            assert report["total_nodes"] == g.number_of_nodes()
            assert report["total_edges"] == g.number_of_edges()
            assert "CHAINS_TO" in report["edge_type_counts"]
