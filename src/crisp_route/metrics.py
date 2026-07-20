"""Path evaluation helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Hashable

import networkx as nx

from .externality import marginal_externality
from .risk import gaussian_cvar_excess

Node = Hashable


def path_edges(path: Sequence[Node]) -> tuple[tuple[Node, Node], ...]:
    return tuple(zip(path[:-1], path[1:]))


def evaluate_path(
    graph: nx.DiGraph,
    path: Sequence[Node],
    *,
    demand: float = 1.0,
    alpha: float = 0.95,
) -> dict[str, float]:
    expected = 0.0
    variance = 0.0
    externality = 0.0
    unreliability = 0.0
    for u, v in path_edges(path):
        data = graph[u][v]
        mean = max(float(data.get("mean_cost", data.get("weight", 1.0))), 0.0)
        std = max(float(data.get("std_cost", 0.0)), 0.0)
        reliability = min(max(float(data.get("reliability", 1.0)), 1e-12), 1.0)
        expected += mean
        variance += std * std
        unreliability += -math.log(reliability)
        capacity = float(data.get("capacity", math.inf))
        if math.isfinite(capacity):
            externality += marginal_externality(
                max(float(data.get("free_flow_cost", mean)), 0.0),
                max(float(data.get("flow", 0.0)), 0.0),
                capacity,
                demand,
                max(float(data.get("bpr_alpha", 0.15)), 0.0),
                max(float(data.get("bpr_beta", 4.0)), 1.0),
            )
    return {
        "expected_cost": expected,
        "std_cost": math.sqrt(variance),
        "cvar": expected + gaussian_cvar_excess(math.sqrt(variance), alpha),
        "tail_risk_excess": gaussian_cvar_excess(math.sqrt(variance), alpha),
        "externality": externality,
        "unreliability": unreliability,
        "hops": float(max(len(path) - 1, 0)),
    }
