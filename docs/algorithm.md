# CRISP-Route Algorithm Notes

## 1. Dynamic routing model

At decision time `t`, the network is a directed graph `G_t = (V, E)`. Every edge `e` may expose:

- expected traversal cost `m_e`;
- standard deviation `s_e`;
- reliability `r_e`;
- current flow `f_e`;
- capacity `c_e`;
- free-flow cost `t0_e`.

A path is not represented by one accumulated number during search. A label stores five dimensions:

```text
L(P) = [M(P), T(P), X(P,q), U(P), S(P,P_prev)]
```

where `M` is expected cost, `T` is CVaR excess, `X` is counterfactual marginal externality for routed demand `q`, `U` is additive unreliability, and `S` is route instability relative to the previous path.

## 2. Tail risk

Under the initial independent Gaussian approximation:

```text
M(P) = sum_e m_e
sigma(P) = sqrt(sum_e s_e^2)
T(P) = sigma(P) × phi(Phi^-1(alpha)) / (1 - alpha)
```

`T(P)` is CVaR minus the mean, so expected cost is not counted twice.

## 3. Counterfactual edge impact

For the BPR volume-delay function:

```text
tau_e(f) = t0_e × [1 + a × (f / c_e)^b]
```

CRISP-Route estimates the external cost per unit produced by adding demand `q`:

```text
X_e(q) = {[(f_e + q)tau_e(f_e + q) - f_e tau_e(f_e)] / q} - tau_e(f_e + q)
```

The first term is the counterfactual increase in total system cost per routed unit. The second is the direct cost paid by that unit. Their non-negative difference approximates the burden imposed on existing flow.

## 4. Reliability

Independent edge reliabilities multiply along a path. The negative log converts the product into an additive quantity:

```text
U(P) = sum_e -log(r_e)
```

This value is zero for perfectly reliable edges and rises as path survival probability falls.

## 5. Route stability

When a prior path exists, every newly introduced edge receives a normalized penalty:

```text
S(P, P_prev) = count(edges in P but not in P_prev) / max(1, |P_prev|)
```

This is an operational prior, not a claim that old routes are intrinsically better. It prevents repeated switching when small edge fluctuations do not justify implementation cost.

## 6. Regime adaptation

The online detector summarizes each graph snapshot by mean utilization, mean coefficient of variation, and mean edge-failure probability. Positive standardized deviations from an EWMA baseline produce a non-negative stress score `rho_t`. Risk and reliability weights then increase with stress:

```text
lambda_t = lambda_0 × (1 + sensitivity × rho_t)
kappa_t  = kappa_0  × (1 + sensitivity × rho_t)
```

The externality weight increases more slowly, while the stability weight remains fixed by default.

## 7. Operating objective

The selected route minimizes:

```text
J_t(P) = M(P)
       + lambda_t T(P)
       + mu_t X(P,q)
       + kappa_t U(P)
       + nu S(P,P_prev)
```

Because the search retains non-dominated labels, the user can inspect alternatives rather than receiving only an opaque scalar optimum.

## 8. Bounded Pareto label-setting

A label `A` epsilon-dominates `B` when it is no worse within the configured tolerance on all dimensions and materially better on at least one. Dominated labels are discarded. Each node retains at most `K` labels, ranked by the current operating score when the cap is exceeded.

```text
push zero label at source
while queue is not empty:
    pop lowest-score surviving label
    if target, retain as candidate
    otherwise, extend label over every unvisited outgoing edge
    compute expected, variance, externality, reliability, and stability increments
    discard candidate if epsilon-dominated at destination node
    remove destination labels dominated by candidate
    enforce label cap K
    push surviving candidate
return the lowest operating-score target label plus non-dominated alternatives
```

## 9. Complexity and approximation

An exact multi-objective frontier can be exponential in path length and graph size. With a fixed label cap `K`, CRISP-Route behaves approximately like `O(K E log(K V))` in ordinary sparse graphs, but the implementation does not claim a universal worst-case bound or exact Pareto completeness. The method is deliberately positioned between a single-weight Dijkstra query and an unrestricted exact multi-objective solver.
