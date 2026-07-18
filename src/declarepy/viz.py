"""Diagnosis plots: matplotlib re-implementations of the book's figure idioms.

Tranche-4 decision (TRANSLATION_ROADMAP): figures are re-implemented in
**matplotlib** — the message of each figure, not its ggplot aesthetics
(SEMANTIC_DIFFERENCES §9). plotnine is deliberately not a dependency.

Every function takes the tables produced by :func:`declarepy.diagnose` /
:func:`declarepy.diagnose_all` (``.diagnosands`` / ``.simulations``) and
returns the matplotlib ``Axes``, so figures stay verifiable: the plotted
arrays are exactly the validated diagnosand tables.

Requires matplotlib (the ``dag``/``dev`` extra).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "plot_sweep",
    "plot_power_curve",
    "plot_sampling_distribution",
    "plot_ci_caterpillar",
    "plot_grid_heatmap",
]

_COLORS = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974", "#64B5CD"]


def _ax(ax: Optional[Any]) -> Any:
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    return ax


def plot_sweep(
    diagnosands: pd.DataFrame,
    x: str,
    y: str,
    group: Optional[str] = None,
    ax: Optional[Any] = None,
    marker: str = "o",
) -> Any:
    """Line plot of one diagnosand over a redesign parameter.

    ``diagnosands`` is a :func:`declarepy.diagnose_all` table; ``x`` a swept
    parameter column; ``y`` a diagnosand column; ``group`` an optional
    grouping column (e.g. ``"estimator"``) drawn as separate lines.
    """
    ax = _ax(ax)
    if group is None:
        sub = diagnosands.sort_values(x)
        ax.plot(sub[x], sub[y], marker=marker, color=_COLORS[0])
    else:
        for i, (label, sub) in enumerate(diagnosands.groupby(group, sort=False)):
            sub = sub.sort_values(x)
            ax.plot(sub[x], sub[y], marker=marker, color=_COLORS[i % len(_COLORS)],
                    label=str(label))
        ax.legend(title=group)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def plot_power_curve(
    diagnosands: pd.DataFrame,
    x: str,
    group: Optional[str] = None,
    target: float = 0.80,
    ax: Optional[Any] = None,
) -> Any:
    """The book's workhorse: power over a design parameter, 80% line drawn."""
    ax = plot_sweep(diagnosands, x=x, y="power", group=group, ax=ax)
    ax.axhline(target, color="gray", linestyle="--", linewidth=1,
               label=f"{target:.0%} target")
    ax.set_ylim(0, 1)
    ax.set_ylabel("statistical power")
    return ax


def plot_sampling_distribution(
    simulations: pd.DataFrame,
    estimand: Optional[float] = None,
    bins: int = 40,
    ax: Optional[Any] = None,
) -> Any:
    """Histogram of the simulated estimates with the estimand marked.

    ``simulations`` is a Diagnosis's raw frame (or any frame with an
    ``estimate`` column). ``estimand=None`` uses the frame's mean estimand.
    """
    ax = _ax(ax)
    est = simulations["estimate"].to_numpy(dtype=float)
    ax.hist(est, bins=bins, color=_COLORS[0], edgecolor="white", alpha=0.9)
    if estimand is None and "estimand" in simulations:
        with np.errstate(invalid="ignore"):
            estimand_val = float(np.nanmean(simulations["estimand"].to_numpy(dtype=float)))
        estimand = None if np.isnan(estimand_val) else estimand_val
    if estimand is not None:
        ax.axvline(estimand, color=_COLORS[1], linestyle="--", linewidth=2,
                   label=f"estimand = {estimand:.3g}")
        ax.legend()
    ax.set_xlabel("estimate")
    ax.set_ylabel("simulations")
    return ax


def plot_ci_caterpillar(
    simulations: pd.DataFrame,
    n: int = 100,
    sort: bool = True,
    ax: Optional[Any] = None,
) -> Any:
    """The classic coverage picture: n simulated CIs, misses highlighted.

    Draws the first ``n`` simulations' confidence intervals (sorted by
    estimate when ``sort``), colored by whether they cover the estimand.
    """
    ax = _ax(ax)
    sub = simulations.dropna(subset=["conf_low", "conf_high", "estimand"]).head(n).copy()
    if sort:
        sub = sub.sort_values("estimate").reset_index(drop=True)
    covers = (sub["conf_low"] <= sub["estimand"]) & (sub["estimand"] <= sub["conf_high"])
    ys = np.arange(len(sub))
    for i, (lo, hi, est, ok) in enumerate(
        zip(sub["conf_low"], sub["conf_high"], sub["estimate"], covers)
    ):
        color = _COLORS[0] if ok else _COLORS[1]
        ax.plot([lo, hi], [i, i], color=color, linewidth=1.2, alpha=0.85)
        ax.plot([est], [i], marker="o", color=color, markersize=2.5)
    truth = float(sub["estimand"].mean())
    ax.axvline(truth, color="black", linewidth=1, linestyle="--",
               label=f"estimand ≈ {truth:.3g}")
    miss_share = float((~covers).mean()) if len(sub) else float("nan")
    ax.set_yticks([])
    ax.set_xlabel("estimate with 95% CI")
    ax.set_title(f"{len(sub)} simulated intervals — {miss_share:.0%} miss the estimand")
    ax.legend(loc="lower right")
    return ax


def plot_grid_heatmap(
    diagnosands: pd.DataFrame,
    x: str,
    y: str,
    value: str = "power",
    ax: Optional[Any] = None,
    fmt: str = "{:.2f}",
) -> Any:
    """Two-parameter redesign grid as an annotated heatmap."""
    ax = _ax(ax)
    pivot = diagnosands.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    for r in range(pivot.shape[0]):
        for c in range(pivot.shape[1]):
            v = pivot.iloc[r, c]
            ax.text(c, r, fmt.format(v), ha="center", va="center",
                    color="white" if v < pivot.to_numpy().mean() else "black",
                    fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.figure.colorbar(im, ax=ax, label=value)
    return ax
