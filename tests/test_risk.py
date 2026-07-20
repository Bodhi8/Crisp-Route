from __future__ import annotations

import pytest

from crisp_route.risk import empirical_cvar, gaussian_cvar_excess


def test_gaussian_cvar_excess_zero_std() -> None:
    assert gaussian_cvar_excess(0.0, 0.95) == 0.0


def test_gaussian_cvar_excess_increases_with_std() -> None:
    assert gaussian_cvar_excess(2.0, 0.95) > gaussian_cvar_excess(1.0, 0.95)


def test_empirical_cvar_focuses_on_upper_tail() -> None:
    values = [1, 1, 1, 1, 10]
    assert empirical_cvar(values, 0.8) == pytest.approx(10.0)
