"""Create article-ready benchmark figures from saved results."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.4, 3.9, 2.0, 1.1, "Dynamic graph\nsnapshot"),
        (3.0, 4.2, 2.2, 1.1, "Regime-stress\ndetector"),
        (3.0, 2.3, 2.2, 1.1, "Counterfactual\nedge impact"),
        (6.0, 3.25, 2.4, 1.35, "Bounded Pareto\nlabel search"),
        (9.2, 3.25, 2.2, 1.35, "Selected route +\naudit decomposition"),
        (6.0, 1.0, 2.4, 1.05, "Previous route\n(stability prior)"),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4,
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    arrows = [
        ((2.4, 4.45), (3.0, 4.75)),
        ((2.4, 4.25), (3.0, 2.85)),
        ((5.2, 4.75), (6.0, 4.15)),
        ((5.2, 2.85), (6.0, 3.55)),
        ((8.4, 3.92), (9.2, 3.92)),
        ((7.2, 2.05), (7.2, 3.25)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.3))

    ax.text(6.0, 5.65, "CRISP-Route decision pipeline", ha="center", va="center", fontsize=17)
    ax.text(
        6.0,
        0.35,
        "Expected cost, CVaR excess, marginal externality, reliability, and route instability remain separately observable.",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(ASSETS / "crisp_route_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def tradeoff(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(summary["mean_realized_cost"], summary["cvar95_realized_cost"], s=130)
    for _, row in summary.iterrows():
        ax.annotate(
            row["algorithm"],
            (row["mean_realized_cost"], row["cvar95_realized_cost"]),
            xytext=(7, 6),
            textcoords="offset points",
        )
    ax.set_xlabel("Mean realized route cost")
    ax.set_ylabel("Mean realized CVaR95")
    ax.set_title("Efficiency versus tail-risk trade-off across 40 dynamic episodes")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_tradeoff.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def by_regime(regime_summary: pd.DataFrame) -> None:
    pivot = regime_summary.pivot(index="regime", columns="algorithm", values="cvar95_realized_cost")
    order = [name for name in ["normal", "congestion", "disruption", "recovery"] if name in pivot.index]
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Operating regime")
    ax.set_ylabel("Mean realized CVaR95")
    ax.set_title("Tail-risk performance by network regime")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "benchmark_by_regime.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def churn(results: pd.DataFrame) -> None:
    episode = results.pivot(index="episode", columns="algorithm", values="route_churn")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column in episode.columns:
        ax.plot(episode.index, episode[column].rolling(4, min_periods=1).mean(), label=column)
    ax.axvline(10, linestyle="--", linewidth=1)
    ax.axvline(20, linestyle="--", linewidth=1)
    ax.axvline(30, linestyle="--", linewidth=1)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Four-episode rolling route churn")
    ax.set_title("Route stability through congestion and disruption shifts")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS / "route_churn.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    try:
        results = pd.read_csv(RESULTS / "episode_results.csv")
        summary = pd.read_csv(RESULTS / "summary.csv")
        regime_summary = pd.read_csv(RESULTS / "regime_summary.csv")
    except FileNotFoundError as exc:
        raise SystemExit("Run benchmarks/run_benchmark.py before creating assets") from exc

    architecture()
    tradeoff(summary)
    by_regime(regime_summary)
    churn(results)
    print(f"Wrote figures to {ASSETS}")


if __name__ == "__main__":
    main()
