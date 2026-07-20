from __future__ import annotations

from crisp_route.externality import bpr_cost, marginal_externality


def test_bpr_cost_grows_with_flow() -> None:
    assert bpr_cost(10.0, 90.0, 100.0) > bpr_cost(10.0, 10.0, 100.0)


def test_externality_is_larger_near_capacity() -> None:
    low = marginal_externality(10.0, 20.0, 100.0, 5.0)
    high = marginal_externality(10.0, 95.0, 100.0, 5.0)
    assert high > low >= 0.0
