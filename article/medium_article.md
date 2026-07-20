# Beyond Shortest Path: Building CRISP-Route for Routing Under Risk, Congestion, and Regime Change

## An open-source, explainable routing algorithm that asks not only which path is cheapest, but which path remains defensible when the network is uncertain and the route itself changes the system

**Brian Curry**  
Independent Researcher, Vector1  
GitHub: https://github.com/Bodhi8/crisp-route

Routing is often introduced as a clean graph problem. A network contains nodes and edges, every edge receives a cost, and an algorithm returns the path with the smallest accumulated weight. That abstraction is extraordinarily useful, but it becomes fragile when the edge weight is treated as if it were a complete description of the decision. A road can be fast on average but unreliable in its upper tail. A network link can have low latency before a new flow is assigned to it but become congested because of the routing decision itself. A logistics corridor can appear attractive at 9:00 a.m. and become operationally dangerous ten minutes later. Even when two routes have nearly identical costs, repeatedly switching between them can create dispatch churn, cache invalidation, control instability, and human confusion.

The classical shortest path is therefore not wrong. It is incomplete for environments in which cost is uncertain, capacity is endogenous, operating conditions shift by regime, and routing changes have their own implementation cost. CRISP-Route—Counterfactual Regime-Informed Stochastic Pathfinding—is an experimental attempt to make those missing dimensions explicit while retaining a graph-search core that remains inspectable. The full Python implementation, tests, benchmark generator, figures, and documentation are open source in the accompanying GitHub repository.

This article explains the algorithm as a research and engineering artifact rather than a branding exercise. The central contribution is not a new acronym wrapped around Dijkstra’s algorithm. CRISP-Route preserves a bounded Pareto frontier over expected cost, tail risk, marginal network externality, unreliability, and route instability. It then applies regime-dependent preferences to select one operating route while returning the non-dominated alternatives and a complete decomposition of why the chosen path won.

**INSERT IMAGE 1 HERE**

*Caption: The CRISP-Route decision pipeline. A dynamic graph snapshot feeds both a regime-stress detector and a counterfactual edge-impact calculation. A bounded Pareto search combines those signals with the previous route and returns an auditable decision.*

*Alt text: Flow diagram showing a graph snapshot entering regime detection and counterfactual impact modules, which feed a Pareto label search alongside the previous route, producing a selected route and objective decomposition.*

Image file: `docs/assets/crisp_route_architecture.png`

## Why another routing algorithm is justified

Research on routing under uncertainty is already substantial, and any credible new project has to begin by acknowledging that fact. Robust and distributionally robust shortest-path methods have developed into a distinct literature spanning static, recourse, and dynamic formulations. A 2025 survey by Carlo Filippi, Francesca Maggioni, and Maria Grazia Speranza reviewed 29 papers and identified continuing open directions in robust shortest-path research. Real-time shortest-path models have also been studied with interval-valued dynamic edge costs, while recent learning-based systems increasingly combine graph neural networks, temporal prediction, and combinatorial optimization.

That recent work demonstrates how active the field has become. A 2026 preprint on risk-aware smart-logistics routing uses a spatiotemporal graph neural network to forecast congestion risk and then incorporates those forecasts into path planning. Work accepted at ICLR 2026 addresses neural multi-objective routing on multigraphs. Other research has explored knowledge-guided machine learning for low-latency, near-shortest routing policies. These systems are important, but they also leave room for a complementary design point: a model-light algorithm whose operating logic is explicitly decomposable, whose multi-objective behavior does not require a trained neural policy, and whose route decision includes both the consequences of adding flow and the cost of changing the route.

CRISP-Route occupies that design space. It is not presented as a proof that neural routing is unnecessary, nor as a claim that no prior work has combined any of its ingredients. The defensible claim is narrower: the project integrates counterfactual marginal congestion, path-level tail-risk approximation, online regime adaptation, reliability, and route stability inside a bounded Pareto label-setting implementation, then exposes every term to the user. The resulting system is useful as an open research baseline, an educational implementation, and a foundation for more rigorous extensions.

## The decision problem

At time t, let the network be a directed graph G_t = (V, E). Each edge e contains more than one weight. In the initial implementation, the edge state includes an expected traversal cost m_e, a cost standard deviation s_e, a reliability probability r_e, a current flow f_e, a capacity c_e, and a free-flow cost t0_e. These attributes can describe roads, packet links, logistics lanes, service dependencies, cloud inference routes, or any other directed network in which movement has uncertain cost and finite capacity.

For a path P, CRISP-Route constructs five quantities. The first is expected cost. The second is upper-tail risk, expressed as Conditional Value at Risk above the mean. The third is the marginal externality created when the new demand is added to the path. The fourth is path unreliability. The fifth is instability relative to the route used in the preceding decision period.

The operating objective is written in plain-text form below so it can be pasted directly into Medium without depending on LaTeX rendering:

```text
J_t(P) = ExpectedCost(P)
       + lambda_t × TailRiskExcess(P)
       + mu_t     × MarginalExternality(P, demand)
       + kappa_t  × Unreliability(P)
       + nu       × RouteInstability(P, previous_path)
```

The weights lambda_t, mu_t, and kappa_t can change with the detected network regime. The stability weight nu is fixed by default, although the API permits any non-negative value. The distinction matters because CRISP-Route does not merely recompute a different static weighted sum at each edge. It maintains multiple non-dominated labels during graph search and delays final scalar selection until path alternatives have been preserved within a bounded approximation.

## Tail risk instead of average cost alone

Expected cost is necessary, but averages can conceal exactly the events that make routing operationally expensive. A path with an expected travel time of 20 minutes and a narrow distribution is not equivalent to a path with the same mean and a long disruption tail. CRISP-Route uses upper-tail Conditional Value at Risk, or CVaR, because it summarizes the average loss within the worst fraction of outcomes rather than reporting only a percentile threshold.

The first release uses an independent Gaussian approximation. If edge costs have means m_e and standard deviations s_e, the path mean and standard deviation are calculated as:

```text
PathMean(P) = sum of m_e over edges in P
PathStd(P)  = square root of the sum of s_e squared over edges in P
```

For confidence level alpha, the Gaussian CVaR excess above the mean is:

```text
TailRiskExcess(P) = PathStd(P)
                  × phi(Phi_inverse(alpha))
                  / (1 - alpha)
```

Here phi is the standard normal density and Phi_inverse is the standard normal quantile. The implementation stores only the excess above the mean because expected cost already appears as a separate objective term. Adding the full CVaR would count the mean twice.

This formulation is intentionally transparent, but it is also an approximation. Real edge costs can be skewed, heavy-tailed, and correlated. Correlation is especially important because adjacent links often fail or congest together. The repository therefore treats the Gaussian model as a baseline to be replaced or augmented by scenario-based path risk, copulas, distributionally robust ambiguity sets, or learned joint forecasts in later versions. The current code makes that limitation visible rather than burying it inside a trained model.

## Counterfactual routing: what happens because this path is chosen?

Most routing objectives evaluate the network as observed. CRISP-Route adds a second question: how does the edge change after the new demand is assigned to it? That question is counterfactual in a practical, engineering sense. The algorithm compares the system cost before adding the routed demand with the system cost after adding it.

The default implementation uses the Bureau of Public Roads volume-delay function, a widely used transportation relationship between flow, capacity, and travel time:

```text
tau_e(f) = free_flow_cost_e × [1 + a × (f / capacity_e)^b]
```

With the conventional default parameters a = 0.15 and b = 4, cost rises nonlinearly as flow approaches and exceeds capacity. For a new demand quantity q, CRISP-Route calculates the incremental system cost per unit and subtracts the direct cost paid by the newly routed unit:

```text
MarginalExternality_e(q)
    = {[(f_e + q) × tau_e(f_e + q) - f_e × tau_e(f_e)] / q}
      - tau_e(f_e + q)
```

The non-negative difference estimates the burden imposed on the flow already using the edge. On an uncongested link, the term is close to zero. Near capacity, it can become large enough to make a slightly longer alternative socially and operationally preferable.

This mechanism is one of the most important distinctions in the project. A conventional route can look inexpensive because it measures only the traveler’s direct predicted cost. A counterfactual route also recognizes that assigning one more truck, packet stream, model request, or service dependency may degrade performance for everyone else. The externality callback can eventually be generalized beyond BPR. In a compute network, it could represent queueing delay. In an energy system, it could represent congestion or loss. In an LLM gateway, it could represent marginal latency, carbon intensity, or capacity depletion.

## Reliability as an additive path quantity

If edge reliabilities are treated as independent probabilities, path reliability is their product. Products are awkward inside additive graph search, so CRISP-Route transforms reliability with a negative logarithm:

```text
Unreliability(P) = sum of -log(r_e) over edges in P
```

A perfectly reliable edge contributes zero. Less reliable edges contribute a positive penalty, and the total remains equivalent to the negative log of the path survival probability under the independence assumption. As with the Gaussian cost model, independence is a simplifying baseline rather than a universal truth. Shared infrastructure and correlated failures require richer treatment, but the additive transformation provides a clean and interpretable first implementation.

## Route stability is an objective, not an afterthought

Dynamic routing systems can thrash. When two paths have nearly identical scores, small changes in predicted cost can cause the selected route to alternate repeatedly. In a navigation application, that can produce confusing instructions. In logistics, it can repeatedly alter dispatch plans. In packet routing, it can create control-plane instability. In model routing, it can invalidate warm caches and complicate capacity allocation.

CRISP-Route therefore penalizes edges that were not present in the previous path. The normalized instability term is:

```text
RouteInstability(P, previous_path)
    = number of edges in P that were not in previous_path
      divided by the number of edges in previous_path
```

The objective does not force the old route to remain. It creates hysteresis. A new path must improve the risk-adjusted, externality-aware objective enough to justify the operational cost of changing. When the network moves into a genuinely different regime, the other terms can still dominate and trigger a switch.

This feature also separates CRISP-Route from a purely myopic solver. The previous decision becomes a state variable. That does not turn the algorithm into a full Markov decision process, but it acknowledges that routing decisions occur in sequence and that sequential consistency has value.

## Regime detection changes how the algorithm values risk

A fixed risk preference can be too conservative in normal conditions and too weak during disruption. CRISP-Route includes a lightweight online detector that summarizes every graph snapshot with three statistics: mean capacity utilization, mean coefficient of variation, and mean edge-failure probability. An exponentially weighted moving baseline tracks normal behavior. Positive standardized deviations are combined into a non-negative regime-stress score.

The score does not predict the future. It measures how unusual the current network state is relative to its recent history. Risk and reliability weights increase with stress:

```text
risk_weight_t = base_risk_weight
              × (1 + sensitivity × regime_score_t)

reliability_weight_t = base_reliability_weight
                     × (1 + sensitivity × regime_score_t)
```

The externality weight increases more slowly in the reference implementation, while the stability penalty remains fixed. This design means that a path accepted during normal operation can be rejected during a high-variance or low-reliability regime even when its expected cost has not changed dramatically.

A more advanced release could replace this detector with Bayesian change-point detection, a hidden Markov model, a state-space model, or a learned regime encoder. The advantage of the current EWMA design is that its behavior can be inspected in a few lines of code and reproduced without training data.

## Why a bounded Pareto search is necessary

The simplest way to combine several objectives is to convert every edge into one scalar weight and run Dijkstra’s algorithm. That approach is fast, but it can discard routes that are temporarily unattractive under one weight configuration yet important under another. It can also behave poorly when path risk is not edge-additive, as occurs when path standard deviation is the square root of accumulated variance.

CRISP-Route instead attaches a label to every partial path. A label records expected cost, variance, marginal externality, unreliability, instability, the current node, and the path itself. When a candidate label reaches a node, it is compared with the surviving labels at that node. A label is discarded when another label is no worse within an epsilon tolerance on every dimension and materially better on at least one.

Exact multi-objective pathfinding can generate an exponential number of non-dominated labels. CRISP-Route deliberately avoids claiming exact Pareto completeness. Every node retains at most K labels, and epsilon dominance compresses near-equivalent alternatives. When the cap is exceeded, labels are ranked by the current regime-adjusted operating objective and the least useful candidates are removed.

The resulting procedure can be summarized as follows:

```text
1. Place a zero-cost label at the source.
2. Pop the surviving label with the smallest current operating score.
3. Extend it over each outgoing edge that does not create a cycle.
4. Update expected cost, variance, externality, reliability, and instability.
5. Discard the new label if it is epsilon-dominated at the destination node.
6. Remove destination labels that the new label dominates.
7. Enforce the per-node label cap.
8. Continue until enough target labels are found or the search bound is reached.
9. Return the lowest-score target path, its decomposition, and Pareto alternatives.
```

With a fixed cap K, practical behavior is approximately O(K E log(K V)) on ordinary sparse graphs. That expression should not be mistaken for a proof of exact worst-case complexity. The cap makes the method tractable by accepting approximation. The repository documents this trade-off directly.

## The Python API

The public interface uses NetworkX directed graphs and dataclasses for routing context. A minimal call looks like this:

```python
from crisp_route import CrispRouter, RouteContext, RouteWeights

router = CrispRouter()

result = router.route(
    graph,
    source="warehouse",
    target="customer",
    context=RouteContext(
        demand=8.0,
        cvar_alpha=0.95,
        previous_path=("warehouse", "hub_2", "customer"),
        weights=RouteWeights(
            risk=0.78,
            externality=0.48,
            reliability=1.25,
            instability=0.45,
        ),
    ),
)

print(result.path)
print(result.score)
print(result.components.as_dict())
print(result.pareto_routes)
```

The result does not simply report a path. It reports the effective regime-adjusted weights, the regime score, the number of generated and expanded labels, the cost decomposition, and a tuple of non-dominated target routes. This is critical for auditability. An operator can see whether a path won because it had a lower mean, a smaller tail, less congestion impact, higher reliability, or greater continuity with the previous decision.

## A benchmark designed to test behavior, not manufacture a victory

The repository includes a reproducible synthetic benchmark on an 8 by 8 directed grid. Edge free-flow costs, capacities, baseline flows, uncertainties, and reliabilities are generated from a fixed seed. Forty sequential episodes are divided into normal, congestion, disruption, and recovery regimes. A moving corridor creates correlated shocks so the experiment is not merely independent noise on isolated edges.

Four algorithms are compared. Expected-Dijkstra minimizes expected edge cost. Risk-Dijkstra adds an edge-wise Gaussian tail-risk adjustment. Congestion-Dijkstra adds counterfactual marginal externality to a scalar edge weight. CRISP-Route uses the full bounded Pareto method with regime adaptation and route stability. For every selected path, the benchmark simulates 500 realized cost scenarios with stochastic traversal cost and finite failure penalties.

The committed seed produced the following aggregate results:

```text
CRISP-Route
  mean realized cost:       15.7349
  mean realized CVaR95:     23.5981
  mean externality:          0.8382
  mean route churn:          0.0803
  median runtime:           21.31 ms

Expected-Dijkstra
  mean realized cost:       15.7863
  mean realized CVaR95:     23.9004
  mean externality:          0.9905
  mean route churn:          0.1810
  median runtime:            0.12 ms

Risk-Dijkstra
  mean realized cost:       15.8546
  mean realized CVaR95:     23.7973
  mean externality:          1.3807
  mean route churn:          0.6286
  median runtime:            0.22 ms

Congestion-Dijkstra
  mean realized cost:       15.7790
  mean realized CVaR95:     23.8137
  mean externality:          0.8509
  mean route churn:          0.0676
  median runtime:            0.24 ms
```

Relative to Expected-Dijkstra, CRISP-Route reduced mean realized CVaR95 by approximately 1.27 percent, reduced measured marginal externality by approximately 15.38 percent, and reduced route churn by approximately 55.62 percent in this synthetic experiment. Its average realized cost was also about 0.33 percent lower. Those differences are modest in absolute terms, which is exactly why the result is more credible than a spectacular but poorly controlled claim. The algorithm’s largest cost was computational: median query time rose from roughly 0.12 milliseconds for Expected-Dijkstra to 21.31 milliseconds for CRISP-Route.

**INSERT IMAGE 2 HERE**

*Caption: Mean realized route cost versus mean realized CVaR95 across 40 dynamic episodes. In this fixed-seed synthetic benchmark, CRISP-Route achieved the lowest value on both axes, but required substantially more computation.*

*Alt text: Scatter plot comparing CRISP-Route, Expected-Dijkstra, Risk-Dijkstra, and Congestion-Dijkstra by mean realized cost and CVaR95. CRISP-Route appears in the lower-left position.*

Image file: `docs/assets/benchmark_tradeoff.png`

The result should not be interpreted as a state-of-the-art claim. It is one synthetic graph family, one parameter configuration, one demand level, one failure model, and one random seed. The benchmark demonstrates that the components interact as intended. A publication-grade evaluation would require multiple graph families, larger networks, real datasets, parameter sweeps, statistical confidence intervals, stronger exact and approximate baselines, correlated scenario models, and ablation studies that remove one CRISP component at a time.

The regime-level results are also informative. CRISP-Route had the lowest average CVaR95 in normal and recovery episodes and improved materially over Expected-Dijkstra during disruption, although Congestion-Dijkstra slightly outperformed it on disruption CVaR95 in the committed run. That outcome is useful rather than embarrassing: a multi-objective method is not guaranteed to dominate a specialized baseline on every metric and every regime. Its purpose is to produce a balanced operating decision across several objectives.

**INSERT IMAGE 3 HERE**

*Caption: Mean realized CVaR95 by operating regime. The comparison reveals where specialized scalar baselines remain competitive and where regime-aware multi-objective routing adds value.*

*Alt text: Grouped bar chart showing CVaR95 for four routing algorithms across normal, congestion, disruption, and recovery regimes.*

Image file: `docs/assets/benchmark_by_regime.png`

## What is genuinely new here, and what is not

Dijkstra’s algorithm is not new. A* is not new. CVaR is not new. Robust shortest paths are not new. Multi-objective pathfinding, congestion-aware routing, route hysteresis, and online change detection all have established histories. The project would be intellectually weak if it presented any of those ingredients as an isolated invention.

The innovative unit is the architecture and its open implementation. CRISP-Route treats marginal system impact, regime-conditioned risk preference, reliability, and route continuity as first-class path dimensions inside a bounded Pareto search. It returns the decomposition and alternative routes rather than hiding the choice in a single opaque score. The algorithm is deliberately model-light and reproducible, making it suitable as a baseline against which more sophisticated learned or distributionally robust systems can be evaluated.

Academic novelty requires a much more exhaustive literature review and formal comparison than a Medium article can provide. The repository therefore describes version 0.1.0 as experimental research software, not as a theorem-backed breakthrough. That restraint is part of the project’s design philosophy. Open-source algorithm work becomes more useful when its assumptions, trade-offs, and failures are easy to inspect.

## Where the research should go next

The most important extension is correlated scenario risk. The current path variance assumes independent edges, which understates risk when a weather event, infrastructure failure, or demand shock affects an entire corridor. A scenario matrix could preserve joint outcomes across edges and compute empirical path CVaR directly. Distributionally robust ambiguity sets could then protect against sampling error and regime shifts.

A second extension is time dependence. Traversing an edge changes the arrival time at the next node, so its cost distribution may depend on when the edge is entered. That requires labels to carry time and potentially resource constraints. Incremental graph updates are also important because a production router should not recompute every label from scratch after a small edge change.

A third extension is formal analysis. The current label cap and epsilon rule are engineering controls. Future work should characterize approximation error, derive conditions under which the selected route is near-optimal for the scalarized objective, and study how regime-weight adaptation affects stability. An ablation benchmark should quantify the separate contribution of tail risk, externality, reliability, regime adaptation, and hysteresis.

Finally, the project can become a bridge between optimization and learning rather than an argument for one side. A neural or statistical model can forecast edge distributions, correlations, or regime probabilities, while CRISP-Route remains the auditable decision layer. The learning system predicts. The graph algorithm decides. The decomposition explains.

## Reproducing the project

The repository contains the installable package, unit tests, a GitHub Actions workflow, an end-to-end logistics example, algorithm notes, benchmark scripts, generated CSV results, and all figures used in this article. The benchmark can be reproduced with two commands:

```bash
PYTHONPATH=src python benchmarks/run_benchmark.py --episodes 40 --draws 500
PYTHONPATH=src python benchmarks/create_assets.py
```

The core test suite is run with:

```bash
PYTHONPATH=src pytest
```

The code is released under the MIT License. The intended repository location is:

https://github.com/Bodhi8/crisp-route

## Conclusion

The shortest path is a powerful abstraction, but real routing decisions rarely depend on expected cost alone. They depend on what happens in the tail, how reliable the route is, whether adding demand worsens the system, whether the network has entered an abnormal regime, and whether a small theoretical improvement is worth the disruption of changing course.

CRISP-Route turns those concerns into explicit path dimensions. Its bounded Pareto search is more expensive than Dijkstra’s algorithm and less theoretically complete than an unrestricted exact multi-objective solver. That middle ground is intentional. The algorithm aims to be understandable enough to audit, flexible enough to extend, and rigorous enough to benchmark without pretending that a synthetic demonstration settles the research question.

The larger idea is straightforward: routing should not merely find a cheap path through the network it observes. It should evaluate the network that its own decision helps create.

## References

1. Dijkstra, E. W. “A Note on Two Problems in Connexion with Graphs.” *Numerische Mathematik*, 1959. Background and implementation reference: https://networkx.org/documentation/stable/reference/algorithms/shortest_paths/dijkstra.html

2. Hart, P. E., Nilsson, N. J., and Raphael, B. “A Formal Basis for the Heuristic Determination of Minimum Cost Paths.” *IEEE Transactions on Systems Science and Cybernetics*, 1968.

3. Rockafellar, R. T., and Uryasev, S. “Conditional Value-at-Risk for General Loss Distributions.” *Journal of Banking & Finance*, 2002. https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf

4. United States Bureau of Public Roads. *Traffic Assignment Manual for Application with a Large, High Speed Computer.* U.S. Department of Commerce, 1964. https://books.google.com/books?id=AvNUR_O_JEcC

5. Filippi, C., Maggioni, F., and Speranza, M. G. “Robust and Distributionally Robust Shortest Path Problems: A Survey.” *Computers & Operations Research*, 182, 107096, 2025. https://doi.org/10.1016/j.cor.2025.107096

6. Xu, B. et al. “A Comparison of Three Real-Time Shortest Path Models in Dynamic Interval Graph.” *Mathematics*, 13(1), 134, 2025. https://doi.org/10.3390/math13010134

7. Xue, Z., Zhao, S., Qi, Y., Zeng, X., and Yu, Z. “Resilient Routing: Risk-Aware Dynamic Routing in Smart Logistics via Spatiotemporal Graph Learning.” arXiv preprint, 2026. https://arxiv.org/abs/2601.13632

8. Rydin, F., Lischka, A., Wu, J., Haghir Chehreghani, M., and Kulcsár, B. “Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs.” ICLR 2026. https://arxiv.org/abs/2506.22095

9. Chen, Y.-F., Lin, S., and Arora, A. “Knowledge-Guided Machine Learning for Stabilizing Near-Shortest Path Routing.” 2025. https://arxiv.org/abs/2509.06640
