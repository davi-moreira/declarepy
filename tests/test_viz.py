"""Smoke tests for declarepy.viz: figures build and plot the real tables."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import declarepy as dp
from declarepy import viz
from declarepy.library import declaration_10_1, declaration_11_1


@pytest.fixture(scope="module")
def diag() -> dp.Diagnosis:
    return dp.diagnose(declaration_10_1(), sims=200, seed=464)


class TestViz:
    def test_sampling_distribution(self, diag: dp.Diagnosis) -> None:
        ax = viz.plot_sampling_distribution(diag.simulations)
        assert ax.get_xlabel() == "estimate"

    def test_ci_caterpillar_miss_share_matches_coverage(self, diag: dp.Diagnosis) -> None:
        ax = viz.plot_ci_caterpillar(diag.simulations, n=200, sort=False)
        # The title's miss share is 1 − coverage of the same simulations.
        cov = float(diag.diagnosands["coverage"].iloc[0])
        assert f"{1 - cov:.0%}" in ax.get_title()

    def test_power_curve_plots_diagnosand_values(self) -> None:
        grid = dp.redesign(declaration_11_1, N=[100, 400])
        table = dp.diagnose_all(grid, sims=100, seed=464).diagnosands
        ax = viz.plot_power_curve(table, x="N")
        line = ax.get_lines()[0]
        assert np.allclose(line.get_ydata(), table.sort_values("N")["power"])

    def test_grid_heatmap(self) -> None:
        grid = dp.redesign(declaration_10_1, N=[50, 100], effect=[0.2, 0.5])
        table = dp.diagnose_all(grid, sims=100, seed=464).diagnosands
        ax = viz.plot_grid_heatmap(table, x="N", y="effect")
        assert ax.get_xlabel() == "N" and ax.get_ylabel() == "effect"

    def test_sweep_grouped(self) -> None:
        grid = dp.redesign(declaration_10_1, N=[50, 100])
        table = dp.diagnose_all(grid, sims=100, seed=464).diagnosands
        table = table.assign(kind="a")
        ax = viz.plot_sweep(table, x="N", y="power", group="kind")
        assert ax.get_legend() is not None
