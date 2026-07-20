"""Data models for CRISP-Route."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence

Node = Hashable


@dataclass(frozen=True)
class RouteWeights:
    """Weights used to select one route from the bounded Pareto frontier.

    All terms are non-negative. ``risk`` multiplies the path CVaR excess above
    its expected cost rather than the full CVaR, which avoids double-counting
    the expected component.
    """

    risk: float = 0.75
    externality: float = 0.35
    reliability: float = 1.0
    instability: float = 0.20

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class RouteContext:
    """Runtime context for a routing decision."""

    demand: float = 1.0
    cvar_alpha: float = 0.95
    weights: RouteWeights = field(default_factory=RouteWeights)
    regime_sensitivity: float = 1.0
    regime_score: float | None = None
    previous_path: Sequence[Node] | None = None
    epsilon: float = 0.02
    max_labels_per_node: int = 48
    max_target_labels: int = 12
    search_slack: float = 0.20

    def __post_init__(self) -> None:
        if self.demand <= 0:
            raise ValueError("demand must be positive")
        if not 0.5 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must be between 0.5 and 1.0")
        if self.regime_sensitivity < 0:
            raise ValueError("regime_sensitivity must be non-negative")
        if not 0 <= self.epsilon < 1:
            raise ValueError("epsilon must be in [0, 1)")
        if self.max_labels_per_node < 1:
            raise ValueError("max_labels_per_node must be positive")
        if self.max_target_labels < 1:
            raise ValueError("max_target_labels must be positive")
        if self.search_slack < 0:
            raise ValueError("search_slack must be non-negative")


@dataclass(frozen=True)
class RouteComponents:
    """Auditable decomposition of a route objective."""

    expected_cost: float
    tail_risk_excess: float
    externality: float
    unreliability: float
    instability: float

    def as_dict(self) -> dict[str, float]:
        return {
            "expected_cost": self.expected_cost,
            "tail_risk_excess": self.tail_risk_excess,
            "externality": self.externality,
            "unreliability": self.unreliability,
            "instability": self.instability,
        }


@dataclass(frozen=True)
class ParetoRoute:
    """A non-dominated route candidate."""

    path: tuple[Node, ...]
    score: float
    components: RouteComponents


@dataclass(frozen=True)
class RouteResult:
    """Result returned by :class:`crisp_route.CrispRouter`."""

    path: tuple[Node, ...]
    score: float
    components: RouteComponents
    regime_score: float
    effective_weights: RouteWeights
    pareto_routes: tuple[ParetoRoute, ...]
    metadata: Mapping[str, float | int | str]
