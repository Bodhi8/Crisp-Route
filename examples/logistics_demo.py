"""Small end-to-end demonstration of CRISP-Route."""

from __future__ import annotations

import networkx as nx

from crisp_route import CrispRouter, RouteContext, RouteWeights, evaluate_path


def build_demo_graph() -> nx.DiGraph:
    graph = nx.DiGraph()

    # Fast but fragile and nearly saturated corridor.
    graph.add_edge(
        "warehouse",
        "hub_fast",
        mean_cost=4.0,
        std_cost=2.4,
        reliability=0.94,
        free_flow_cost=3.2,
        flow=95.0,
        capacity=100.0,
    )
    graph.add_edge(
        "hub_fast",
        "customer",
        mean_cost=4.0,
        std_cost=2.2,
        reliability=0.95,
        free_flow_cost=3.2,
        flow=96.0,
        capacity=100.0,
    )

    # Slightly longer but predictable and under capacity.
    graph.add_edge(
        "warehouse",
        "hub_safe",
        mean_cost=5.2,
        std_cost=0.5,
        reliability=0.995,
        free_flow_cost=5.0,
        flow=42.0,
        capacity=120.0,
    )
    graph.add_edge(
        "hub_safe",
        "customer",
        mean_cost=5.1,
        std_cost=0.4,
        reliability=0.996,
        free_flow_cost=4.9,
        flow=38.0,
        capacity=120.0,
    )

    return graph


def main() -> None:
    graph = build_demo_graph()
    router = CrispRouter()
    context = RouteContext(
        demand=8.0,
        weights=RouteWeights(risk=0.9, externality=0.5, reliability=1.2, instability=0.1),
        regime_score=1.4,
    )
    result = router.route(graph, "warehouse", "customer", context)

    print("Selected path:", " -> ".join(result.path))
    print("Objective score:", round(result.score, 4))
    print("Regime score:", round(result.regime_score, 4))
    print("Components:")
    for name, value in result.components.as_dict().items():
        print(f"  {name:>18}: {value:.4f}")

    print("\nRaw route metrics:")
    for name, value in evaluate_path(graph, result.path, demand=context.demand).items():
        print(f"  {name:>18}: {value:.4f}")

    print("\nPareto alternatives:")
    for candidate in result.pareto_routes:
        print(f"  {candidate.path} score={candidate.score:.4f}")


if __name__ == "__main__":
    main()
