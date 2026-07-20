"""Baseline routers used in examples and benchmarks."""

from __future__ import annotations

import math
from typing import Hashable

import networkx as nx

from .externality import marginal_externality
from .risk import gaussian_cvar_excess

Node = Hashable


def expected_shortest_path(graph: nx.DiGraph, source: Node, target: Node) -> tuple[Node, ...]:
    """Dijkstra route minimizing expected edge cost."""

    path = nx.shortest_path(
        graph,
        source,
        target,
        weight=lambda _u, _v, data: max(
            float(data.get("mean_cost", data.get("weight", 1.0))), 0.0
        ),
    )
    return tuple(path)


def risk_adjusted_path(
    graph: nx.DiGraph,
    source: Node,
    target: Node,
    *,
    risk_weight: float = 0.75,
    alpha: float = 0.95,
) -> tuple[Node, ...]:
    """Scalar Dijkstra baseline using edge-wise Gaussian CVaR excess."""

    def weight(_u: Node, _v: Node, data: dict[str, object]) -> float:
        mean = max(float(data.get("mean_cost", data.get("weight", 1.0))), 0.0)
        std = max(float(data.get("std_cost", 0.0)), 0.0)
        return mean + risk_weight * gaussian_cvar_excess(std, alpha)

    return tuple(nx.shortest_path(graph, source, target, weight=weight))


def congestion_adjusted_path(
    graph: nx.DiGraph,
    source: Node,
    target: Node,
    *,
    demand: float = 1.0,
    externality_weight: float = 0.35,
) -> tuple[Node, ...]:
    """Scalar Dijkstra baseline with marginal congestion externality."""

    def weight(_u: Node, _v: Node, data: dict[str, object]) -> float:
        mean = max(float(data.get("mean_cost", data.get("weight", 1.0))), 0.0)
        capacity = float(data.get("capacity", math.inf))
        if not math.isfinite(capacity):
            return mean
        externality = marginal_externality(
            max(float(data.get("free_flow_cost", mean)), 0.0),
            max(float(data.get("flow", 0.0)), 0.0),
            capacity,
            demand,
            max(float(data.get("bpr_alpha", 0.15)), 0.0),
            max(float(data.get("bpr_beta", 4.0)), 1.0),
        )
        return mean + externality_weight * externality

    return tuple(nx.shortest_path(graph, source, target, weight=weight))
