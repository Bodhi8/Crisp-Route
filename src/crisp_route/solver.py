"""Bounded Pareto label-setting solver for CRISP-Route."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Hashable, Iterable

import networkx as nx

from .externality import marginal_externality
from .models import (
    ParetoRoute,
    RouteComponents,
    RouteContext,
    RouteResult,
    RouteWeights,
)
from .regime import RegimeDetector
from .risk import gaussian_cvar_excess

Node = Hashable


@dataclass(frozen=True)
class _Label:
    node: Node
    path: tuple[Node, ...]
    expected: float
    variance: float
    externality: float
    unreliability: float
    instability: float


class CrispRouter:
    """Counterfactual Regime-Informed Stochastic Pathfinding.

    CRISP-Route maintains an epsilon-bounded Pareto set over expected cost,
    tail risk, marginal externality, unreliability, and route instability.
    A context-dependent scalarization selects the operating route while the
    non-dominated alternatives remain available for inspection.
    """

    def __init__(self, regime_detector: RegimeDetector | None = None) -> None:
        self.regime_detector = regime_detector or RegimeDetector()

    @staticmethod
    def _previous_edges(path: Iterable[Node] | None) -> set[tuple[Node, Node]]:
        if path is None:
            return set()
        sequence = tuple(path)
        return set(zip(sequence[:-1], sequence[1:]))

    @staticmethod
    def _effective_weights(
        base: RouteWeights, regime_score: float, sensitivity: float
    ) -> RouteWeights:
        multiplier = 1.0 + sensitivity * max(regime_score, 0.0)
        return RouteWeights(
            risk=base.risk * multiplier,
            externality=base.externality * (1.0 + 0.5 * sensitivity * regime_score),
            reliability=base.reliability * multiplier,
            instability=base.instability,
        )

    @staticmethod
    def _components(label: _Label, alpha: float) -> RouteComponents:
        return RouteComponents(
            expected_cost=label.expected,
            tail_risk_excess=gaussian_cvar_excess(math.sqrt(label.variance), alpha),
            externality=label.externality,
            unreliability=label.unreliability,
            instability=label.instability,
        )

    @staticmethod
    def _score(components: RouteComponents, weights: RouteWeights) -> float:
        return (
            components.expected_cost
            + weights.risk * components.tail_risk_excess
            + weights.externality * components.externality
            + weights.reliability * components.unreliability
            + weights.instability * components.instability
        )

    @classmethod
    def _vector(cls, label: _Label, alpha: float) -> tuple[float, ...]:
        c = cls._components(label, alpha)
        return (
            c.expected_cost,
            c.tail_risk_excess,
            c.externality,
            c.unreliability,
            c.instability,
        )

    @classmethod
    def _dominates(cls, a: _Label, b: _Label, epsilon: float, alpha: float) -> bool:
        av = cls._vector(a, alpha)
        bv = cls._vector(b, alpha)
        no_worse = all(x <= y * (1.0 + epsilon) + 1e-12 for x, y in zip(av, bv))
        strictly_better = any(x < y * (1.0 - epsilon) - 1e-12 for x, y in zip(av, bv))
        return no_worse and strictly_better

    @staticmethod
    def _edge_increment(
        u: Node,
        v: Node,
        data: dict[str, object],
        context: RouteContext,
        previous_edges: set[tuple[Node, Node]],
    ) -> tuple[float, float, float, float, float]:
        expected = max(float(data.get("mean_cost", data.get("weight", 1.0))), 0.0)
        std = max(float(data.get("std_cost", 0.0)), 0.0)
        reliability = min(max(float(data.get("reliability", 1.0)), 1e-12), 1.0)
        unreliability = -math.log(reliability)

        free_flow = max(float(data.get("free_flow_cost", expected)), 0.0)
        flow = max(float(data.get("flow", 0.0)), 0.0)
        capacity = float(data.get("capacity", math.inf))
        bpr_alpha = max(float(data.get("bpr_alpha", 0.15)), 0.0)
        bpr_beta = max(float(data.get("bpr_beta", 4.0)), 1.0)
        externality = 0.0
        if math.isfinite(capacity):
            externality = marginal_externality(
                free_flow,
                flow,
                capacity,
                context.demand,
                bpr_alpha,
                bpr_beta,
            )

        if not previous_edges:
            instability = 0.0
        else:
            instability = 0.0 if (u, v) in previous_edges else 1.0 / len(previous_edges)

        return expected, std * std, externality, unreliability, instability

    def route(
        self,
        graph: nx.DiGraph,
        source: Node,
        target: Node,
        context: RouteContext | None = None,
        *,
        update_regime: bool = True,
    ) -> RouteResult:
        """Compute a bounded-Pareto risk-aware route.

        Edge attributes:
        ``mean_cost`` (or ``weight``), ``std_cost``, ``reliability``,
        ``free_flow_cost``, ``flow``, ``capacity``, ``bpr_alpha``, and
        ``bpr_beta``. Missing uncertainty and congestion attributes safely
        default to deterministic, uncongested behavior.
        """

        if source not in graph:
            raise nx.NodeNotFound(f"source {source!r} is not in graph")
        if target not in graph:
            raise nx.NodeNotFound(f"target {target!r} is not in graph")
        if source == target:
            zero = RouteComponents(0.0, 0.0, 0.0, 0.0, 0.0)
            weights = (context or RouteContext()).weights
            return RouteResult(
                path=(source,),
                score=0.0,
                components=zero,
                regime_score=0.0,
                effective_weights=weights,
                pareto_routes=(ParetoRoute((source,), 0.0, zero),),
                metadata={"expanded_labels": 0, "generated_labels": 1, "algorithm": "CRISP-Route"},
            )

        context = context or RouteContext()
        if context.regime_score is not None:
            regime_score = max(context.regime_score, 0.0)
        elif update_regime:
            regime_score = self.regime_detector.update(graph)
        else:
            regime_score = 0.0

        weights = self._effective_weights(
            context.weights, regime_score, context.regime_sensitivity
        )
        previous_edges = self._previous_edges(context.previous_path)

        start = _Label(source, (source,), 0.0, 0.0, 0.0, 0.0, 0.0)
        labels_by_node: dict[Node, list[_Label]] = {source: [start]}
        counter = itertools.count()
        queue: list[tuple[float, int, _Label]] = [(0.0, next(counter), start)]
        target_labels: list[_Label] = []
        best_target_score = math.inf
        expanded = 0
        generated = 1

        while queue:
            priority, _, label = heapq.heappop(queue)
            node_labels = labels_by_node.get(label.node, [])
            if label not in node_labels:
                continue
            if best_target_score < math.inf and priority > best_target_score * (1.0 + context.search_slack):
                break

            if label.node == target:
                target_labels.append(label)
                best_target_score = min(best_target_score, priority)
                if len(target_labels) >= context.max_target_labels:
                    break
                continue

            expanded += 1
            for neighbor, data in graph[label.node].items():
                if neighbor in label.path:
                    continue
                edge_data = dict(data)
                inc = self._edge_increment(label.node, neighbor, edge_data, context, previous_edges)
                candidate = _Label(
                    node=neighbor,
                    path=label.path + (neighbor,),
                    expected=label.expected + inc[0],
                    variance=label.variance + inc[1],
                    externality=label.externality + inc[2],
                    unreliability=label.unreliability + inc[3],
                    instability=label.instability + inc[4],
                )
                generated += 1

                existing = labels_by_node.setdefault(neighbor, [])
                if any(
                    self._dominates(old, candidate, context.epsilon, context.cvar_alpha)
                    for old in existing
                ):
                    continue

                existing[:] = [
                    old
                    for old in existing
                    if not self._dominates(candidate, old, context.epsilon, context.cvar_alpha)
                ]
                existing.append(candidate)

                if len(existing) > context.max_labels_per_node:
                    existing.sort(
                        key=lambda item: self._score(
                            self._components(item, context.cvar_alpha), weights
                        )
                    )
                    del existing[context.max_labels_per_node :]
                    if candidate not in existing:
                        continue

                components = self._components(candidate, context.cvar_alpha)
                score = self._score(components, weights)
                heapq.heappush(queue, (score, next(counter), candidate))

        if not target_labels:
            raise nx.NetworkXNoPath(f"No path between {source!r} and {target!r}")

        candidates: list[ParetoRoute] = []
        for label in target_labels:
            components = self._components(label, context.cvar_alpha)
            candidates.append(
                ParetoRoute(
                    path=label.path,
                    score=self._score(components, weights),
                    components=components,
                )
            )
        candidates.sort(key=lambda route: route.score)
        best = candidates[0]

        return RouteResult(
            path=best.path,
            score=best.score,
            components=best.components,
            regime_score=regime_score,
            effective_weights=weights,
            pareto_routes=tuple(candidates),
            metadata={
                "expanded_labels": expanded,
                "generated_labels": generated,
                "algorithm": "CRISP-Route",
                "epsilon": context.epsilon,
            },
        )
