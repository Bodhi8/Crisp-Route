"""Online regime-stress detection for dynamic graphs."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class RegimeDetector:
    """EWMA detector over graph-level congestion, uncertainty, and reliability.

    The detector intentionally remains lightweight. It is not a forecasting
    model; it turns a stream of graph snapshots into a non-negative stress score
    used to adapt routing preferences.
    """

    decay: float = 0.92
    warmup: int = 4
    _mean: np.ndarray | None = None
    _var: np.ndarray | None = None
    _count: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.decay < 1:
            raise ValueError("decay must be between 0 and 1")
        if self.warmup < 1:
            raise ValueError("warmup must be positive")

    @staticmethod
    def summarize(graph: nx.Graph) -> np.ndarray:
        utilizations: list[float] = []
        coefficients_of_variation: list[float] = []
        failure_probabilities: list[float] = []

        for _, _, data in graph.edges(data=True):
            mean = max(float(data.get("mean_cost", data.get("weight", 1.0))), 1e-12)
            std = max(float(data.get("std_cost", 0.0)), 0.0)
            flow = max(float(data.get("flow", 0.0)), 0.0)
            capacity = float(data.get("capacity", np.inf))
            reliability = min(max(float(data.get("reliability", 1.0)), 1e-12), 1.0)
            utilization = 0.0 if not np.isfinite(capacity) else flow / max(capacity, 1e-12)
            utilizations.append(utilization)
            coefficients_of_variation.append(std / mean)
            failure_probabilities.append(1.0 - reliability)

        if not utilizations:
            return np.zeros(3, dtype=float)

        return np.array(
            [
                float(np.mean(utilizations)),
                float(np.mean(coefficients_of_variation)),
                float(np.mean(failure_probabilities)),
            ],
            dtype=float,
        )

    def update(self, graph: nx.Graph) -> float:
        observation = self.summarize(graph)
        self._count += 1

        if self._mean is None or self._var is None:
            self._mean = observation.copy()
            self._var = np.full_like(observation, 1e-6)
            return 0.0

        baseline_mean = self._mean.copy()
        baseline_var = self._var.copy()
        delta = observation - baseline_mean

        self._mean = self.decay * self._mean + (1.0 - self.decay) * observation
        self._var = self.decay * self._var + (1.0 - self.decay) * delta * delta

        if self._count <= self.warmup:
            return 0.0

        z = delta / np.sqrt(baseline_var + 1e-9)
        positive_stress = np.maximum(z, 0.0)
        return float(np.sqrt(np.mean(positive_stress * positive_stress)))
