from __future__ import annotations

import networkx as nx

from crisp_route.regime import RegimeDetector


def make_graph(flow: float, std: float, reliability: float) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        mean_cost=10.0,
        std_cost=std,
        flow=flow,
        capacity=100.0,
        reliability=reliability,
    )
    return graph


def test_detector_reacts_to_stress() -> None:
    detector = RegimeDetector(decay=0.8, warmup=2)
    detector.update(make_graph(30.0, 0.5, 0.999))
    detector.update(make_graph(31.0, 0.5, 0.999))
    detector.update(make_graph(32.0, 0.6, 0.998))
    stress = detector.update(make_graph(95.0, 4.0, 0.90))
    assert stress > 0.0
