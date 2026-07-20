"""Risk utilities used by CRISP-Route."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Iterable

import numpy as np


def gaussian_cvar_excess(std: float, alpha: float = 0.95) -> float:
    """Return Gaussian CVaR minus the mean.

    For ``X ~ Normal(mu, sigma)``, upper-tail CVaR is
    ``mu + sigma * phi(Phi^-1(alpha)) / (1 - alpha)``.
    """

    if std < 0:
        raise ValueError("std must be non-negative")
    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be between 0.5 and 1.0")
    if std == 0:
        return 0.0
    z = NormalDist().inv_cdf(alpha)
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return std * phi / (1.0 - alpha)


def empirical_cvar(values: Iterable[float], alpha: float = 0.95) -> float:
    """Return upper-tail empirical CVaR with linear quantile interpolation."""

    if not 0.5 < alpha < 1.0:
        raise ValueError("alpha must be between 0.5 and 1.0")
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        raise ValueError("values must not be empty")
    threshold = float(np.quantile(array, alpha))
    tail = array[array >= threshold]
    return float(tail.mean())
