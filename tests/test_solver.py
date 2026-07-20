from __future__ import annotations

import networkx as nx
import pytest

from crisp_route import CrispRouter, RouteContext, RouteWeights


def test_risk_aware_route_avoids_fragile_shortcut() -> None:
    graph = nx.DiGraph()
    graph.add_edge("s", "a", mean_cost=2.0, std_cost=3.0, reliability=0.90)
    graph.add_edge("a", "t", mean_cost=2.0, std_cost=3.0, reliability=0.90)
    graph.add_edge("s", "b", mean_cost=3.0, std_cost=0.1, reliability=0.999)
    graph.add_edge("b", "t", mean_cost=3.0, std_cost=0.1, reliability=0.999)

    result = CrispRouter().route(
        graph,
        "s",
        "t",
        RouteContext(
            weights=RouteWeights(risk=1.0, externality=0.0, reliability=1.0, instability=0.0),
            regime_score=0.0,
        ),
    )
    assert result.path == ("s", "b", "t")


def test_externality_avoids_saturated_corridor() -> None:
    graph = nx.DiGraph()
    graph.add_edge(
        "s",
        "a",
        mean_cost=1.0,
        std_cost=0.0,
        reliability=1.0,
        free_flow_cost=1.0,
        flow=99.0,
        capacity=100.0,
    )
    graph.add_edge(
        "a",
        "t",
        mean_cost=1.0,
        std_cost=0.0,
        reliability=1.0,
        free_flow_cost=1.0,
        flow=99.0,
        capacity=100.0,
    )
    graph.add_edge(
        "s",
        "b",
        mean_cost=1.2,
        std_cost=0.0,
        reliability=1.0,
        free_flow_cost=1.2,
        flow=10.0,
        capacity=100.0,
    )
    graph.add_edge(
        "b",
        "t",
        mean_cost=1.2,
        std_cost=0.0,
        reliability=1.0,
        free_flow_cost=1.2,
        flow=10.0,
        capacity=100.0,
    )

    result = CrispRouter().route(
        graph,
        "s",
        "t",
        RouteContext(
            demand=20.0,
            weights=RouteWeights(risk=0.0, externality=2.0, reliability=0.0, instability=0.0),
            regime_score=0.0,
        ),
    )
    assert result.path == ("s", "b", "t")


def test_instability_penalty_can_preserve_previous_path() -> None:
    graph = nx.DiGraph()
    graph.add_edge("s", "a", mean_cost=2.0, reliability=1.0)
    graph.add_edge("a", "t", mean_cost=2.0, reliability=1.0)
    graph.add_edge("s", "b", mean_cost=1.9, reliability=1.0)
    graph.add_edge("b", "t", mean_cost=1.9, reliability=1.0)

    result = CrispRouter().route(
        graph,
        "s",
        "t",
        RouteContext(
            previous_path=("s", "a", "t"),
            weights=RouteWeights(risk=0.0, externality=0.0, reliability=0.0, instability=1.0),
            regime_score=0.0,
        ),
    )
    assert result.path == ("s", "a", "t")


def test_missing_path_raises() -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(["s", "t"])
    with pytest.raises(nx.NetworkXNoPath):
        CrispRouter().route(graph, "s", "t", RouteContext(regime_score=0.0))
