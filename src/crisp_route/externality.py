"""Counterfactual congestion and system-impact calculations."""

from __future__ import annotations

import math


def bpr_cost(
    free_flow_cost: float,
    flow: float,
    capacity: float,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float:
    """Compute the Bureau of Public Roads volume-delay function."""

    if free_flow_cost < 0:
        raise ValueError("free_flow_cost must be non-negative")
    if flow < 0:
        raise ValueError("flow must be non-negative")
    if capacity <= 0:
        return math.inf
    return free_flow_cost * (1.0 + alpha * (flow / capacity) ** beta)


def marginal_externality(
    free_flow_cost: float,
    flow: float,
    capacity: float,
    demand: float,
    alpha: float = 0.15,
    beta: float = 4.0,
) -> float:
    """Estimate per-unit external cost caused by adding ``demand`` to an edge.

    The calculation compares the counterfactual increase in total system cost
    against the direct cost paid by the newly routed demand. It is zero on an
    uncongested edge and grows nonlinearly as utilization approaches capacity.
    """

    if demand <= 0:
        raise ValueError("demand must be positive")
    before = flow * bpr_cost(free_flow_cost, flow, capacity, alpha, beta)
    after_flow = flow + demand
    after_unit_cost = bpr_cost(free_flow_cost, after_flow, capacity, alpha, beta)
    after = after_flow * after_unit_cost
    incremental_system_cost_per_unit = (after - before) / demand
    direct_cost_per_unit = after_unit_cost
    return max(0.0, incremental_system_cost_per_unit - direct_cost_per_unit)
