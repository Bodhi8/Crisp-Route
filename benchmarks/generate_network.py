"""Synthetic dynamic network generator for CRISP-Route benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from crisp_route.externality import bpr_cost


@dataclass(frozen=True)
class Regime:
    name: str
    congestion_multiplier: float
    uncertainty_multiplier: float
    reliability_shift: float
    corridor_probability: float


REGIMES = {
    "normal": Regime("normal", 1.00, 1.00, 0.000, 0.08),
    "congestion": Regime("congestion", 1.22, 1.35, 0.003, 0.35),
    "disruption": Regime("disruption", 1.12, 2.10, 0.035, 0.25),
    "recovery": Regime("recovery", 1.04, 1.20, 0.006, 0.12),
}


def build_base_grid(size: int = 8, seed: int = 7) -> nx.DiGraph:
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    for row in range(size):
        for col in range(size):
            graph.add_node((row, col), x=col, y=row)

    for row in range(size):
        for col in range(size):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + dr, col + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                free_flow = float(rng.uniform(0.85, 1.35))
                capacity = float(rng.uniform(70.0, 145.0))
                utilization = float(rng.uniform(0.28, 0.76))
                flow = capacity * utilization
                mean_cost = bpr_cost(free_flow, flow, capacity)
                graph.add_edge(
                    (row, col),
                    (nr, nc),
                    free_flow_cost=free_flow,
                    capacity=capacity,
                    base_flow=flow,
                    flow=flow,
                    mean_cost=mean_cost,
                    std_cost=mean_cost * float(rng.uniform(0.05, 0.16)),
                    base_reliability=float(rng.uniform(0.992, 0.9995)),
                    reliability=float(rng.uniform(0.992, 0.9995)),
                )
    return graph


def graph_for_episode(
    base_graph: nx.DiGraph,
    episode: int,
    regime_name: str,
    seed: int = 11,
) -> nx.DiGraph:
    regime = REGIMES[regime_name]
    rng = np.random.default_rng(seed + episode * 7919)
    graph = base_graph.copy()
    size = int(math.sqrt(graph.number_of_nodes()))

    # A moving corridor creates correlated, path-level shocks rather than
    # independent edge noise.
    corridor_row = (episode * 3 + 2) % size
    corridor_col = (episode * 5 + 1) % size

    for u, v, data in graph.edges(data=True):
        base_flow = float(data["base_flow"])
        capacity = float(data["capacity"])
        free_flow = float(data["free_flow_cost"])
        base_reliability = float(data["base_reliability"])

        in_corridor = (
            u[0] == corridor_row
            or v[0] == corridor_row
            or u[1] == corridor_col
            or v[1] == corridor_col
        )
        corridor_shock = 1.0
        if in_corridor and rng.random() < regime.corridor_probability:
            corridor_shock += float(rng.uniform(0.18, 0.48))

        local_noise = float(rng.lognormal(mean=0.0, sigma=0.055))
        flow = min(
            capacity * 1.35,
            base_flow * regime.congestion_multiplier * corridor_shock * local_noise,
        )
        mean_cost = bpr_cost(free_flow, flow, capacity)
        uncertainty = float(rng.uniform(0.07, 0.18)) * regime.uncertainty_multiplier
        if in_corridor:
            uncertainty *= 1.0 + 0.45 * (corridor_shock - 1.0)

        reliability = base_reliability - regime.reliability_shift
        if in_corridor and corridor_shock > 1.0:
            reliability -= float(rng.uniform(0.004, 0.020))
        reliability = float(np.clip(reliability, 0.82, 0.9999))

        data["flow"] = flow
        data["mean_cost"] = mean_cost
        data["std_cost"] = mean_cost * uncertainty
        data["reliability"] = reliability

    return graph
