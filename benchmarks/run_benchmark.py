"""Reproducible benchmark comparing CRISP-Route with scalar baselines."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from crisp_route import (  # noqa: E402
    CrispRouter,
    RouteContext,
    RouteWeights,
    congestion_adjusted_path,
    evaluate_path,
    expected_shortest_path,
    risk_adjusted_path,
)
from crisp_route.metrics import path_edges  # noqa: E402

from generate_network import build_base_grid, graph_for_episode  # noqa: E402


def churn(previous: tuple[object, ...] | None, current: tuple[object, ...]) -> float:
    if previous is None:
        return 0.0
    a = set(path_edges(previous))
    b = set(path_edges(current))
    union = a | b
    return 0.0 if not union else 1.0 - len(a & b) / len(union)


def simulate_path_costs(
    graph: nx.DiGraph,
    path: tuple[object, ...],
    rng: np.random.Generator,
    draws: int,
) -> np.ndarray:
    costs = np.zeros(draws, dtype=float)
    for u, v in path_edges(path):
        data = graph[u][v]
        mean = float(data["mean_cost"])
        std = float(data["std_cost"])
        reliability = float(data["reliability"])
        edge_cost = np.maximum(rng.normal(mean, std, size=draws), 0.01)
        failures = rng.random(draws) > reliability
        # Failure cost is intentionally finite: a detour/recovery penalty, not
        # an impossible edge. This allows all algorithms to be compared on the
        # same realized scenarios.
        edge_cost += failures * (4.0 * mean + 2.0)
        costs += edge_cost
    return costs


def upper_cvar(values: np.ndarray, alpha: float = 0.95) -> float:
    threshold = np.quantile(values, alpha)
    return float(values[values >= threshold].mean())


def route_all(
    graph: nx.DiGraph,
    source: object,
    target: object,
    crisp_router: CrispRouter,
    previous_paths: dict[str, tuple[object, ...] | None],
) -> tuple[dict[str, tuple[object, ...]], dict[str, float], float]:
    paths: dict[str, tuple[object, ...]] = {}
    runtimes: dict[str, float] = {}

    start = time.perf_counter()
    paths["Expected-Dijkstra"] = expected_shortest_path(graph, source, target)
    runtimes["Expected-Dijkstra"] = time.perf_counter() - start

    start = time.perf_counter()
    paths["Risk-Dijkstra"] = risk_adjusted_path(
        graph, source, target, risk_weight=0.80, alpha=0.95
    )
    runtimes["Risk-Dijkstra"] = time.perf_counter() - start

    start = time.perf_counter()
    paths["Congestion-Dijkstra"] = congestion_adjusted_path(
        graph, source, target, demand=8.0, externality_weight=0.50
    )
    runtimes["Congestion-Dijkstra"] = time.perf_counter() - start

    context = RouteContext(
        demand=8.0,
        cvar_alpha=0.95,
        weights=RouteWeights(risk=0.78, externality=0.48, reliability=1.25, instability=0.45),
        previous_path=previous_paths["CRISP-Route"],
        regime_sensitivity=0.32,
        epsilon=0.04,
        max_labels_per_node=28,
        max_target_labels=8,
        search_slack=0.18,
    )
    start = time.perf_counter()
    result = crisp_router.route(graph, source, target, context)
    runtimes["CRISP-Route"] = time.perf_counter() - start
    paths["CRISP-Route"] = result.path
    return paths, runtimes, result.regime_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    base = build_base_grid(size=8, seed=7)
    source, target = (0, 0), (7, 7)
    crisp_router = CrispRouter()
    algorithms = [
        "Expected-Dijkstra",
        "Risk-Dijkstra",
        "Congestion-Dijkstra",
        "CRISP-Route",
    ]
    previous_paths: dict[str, tuple[object, ...] | None] = {name: None for name in algorithms}
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []

    regime_sequence = ["normal"] * 10 + ["congestion"] * 10 + ["disruption"] * 10 + ["recovery"] * 10
    regime_sequence = regime_sequence[: args.episodes]

    for episode, regime in enumerate(regime_sequence):
        graph = graph_for_episode(base, episode, regime)
        paths, runtimes, regime_score = route_all(
            graph, source, target, crisp_router, previous_paths
        )

        for algorithm, path in paths.items():
            realized = simulate_path_costs(graph, path, rng, args.draws)
            metrics = evaluate_path(graph, path, demand=8.0, alpha=0.95)
            row: dict[str, object] = {
                "episode": episode,
                "regime": regime,
                "algorithm": algorithm,
                "mean_realized_cost": float(realized.mean()),
                "p95_realized_cost": float(np.quantile(realized, 0.95)),
                "cvar95_realized_cost": upper_cvar(realized, 0.95),
                "externality": metrics["externality"],
                "unreliability": metrics["unreliability"],
                "hops": metrics["hops"],
                "route_churn": churn(previous_paths[algorithm], path),
                "runtime_ms": runtimes[algorithm] * 1000.0,
                "regime_score": regime_score,
                "path": " -> ".join(map(str, path)),
            }
            rows.append(row)
            previous_paths[algorithm] = path

    results = pd.DataFrame(rows)
    output_dir = ROOT / "benchmarks" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "episode_results.csv", index=False)

    summary = (
        results.groupby("algorithm", as_index=False)
        .agg(
            mean_realized_cost=("mean_realized_cost", "mean"),
            cvar95_realized_cost=("cvar95_realized_cost", "mean"),
            externality=("externality", "mean"),
            route_churn=("route_churn", "mean"),
            runtime_ms=("runtime_ms", "median"),
        )
        .sort_values("cvar95_realized_cost")
    )
    summary.to_csv(output_dir / "summary.csv", index=False)

    regime_summary = (
        results.groupby(["regime", "algorithm"], as_index=False)
        .agg(
            mean_realized_cost=("mean_realized_cost", "mean"),
            cvar95_realized_cost=("cvar95_realized_cost", "mean"),
            externality=("externality", "mean"),
            route_churn=("route_churn", "mean"),
        )
    )
    regime_summary.to_csv(output_dir / "regime_summary.csv", index=False)

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
