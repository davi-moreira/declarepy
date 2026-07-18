"""T1 acceptance, part 1: the course notebooks could run on declarepy.

Each test reconstructs a course notebook's seeded flow (seed 464) using
declarepy building blocks in place of the inline helpers, and asserts the
notebook's own hard-coded numbers — including nb09's ``est == 2.0372`` — so
any behavioral drift in the package breaks these tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import declarepy as dp
from .conftest import make_world

SEED = 464


class TestNb04ModelInquiry:
    def test_true_ate_and_assignment_counts(self) -> None:
        rng = np.random.default_rng(SEED)
        world = make_world(rng=rng)
        true_ate = (world["Y1"] - world["Y0"]).mean()
        assert true_ate == 2.0
        assert world["Y0"].head(3).tolist() == [54.2, 73.1, 61.9]
        world["Z"] = rng.integers(0, 2, len(world))
        assert int(world["Z"].sum()) == 52


class TestNb09AnswerStrategy:
    def test_dim_reproduces_2_0372(self) -> None:
        rng = np.random.default_rng(SEED)
        world = make_world(n=500, rng=rng)
        world["Z"] = dp.complete_ra(500, 250, rng=rng)
        assert int(world["Z"].sum()) == 250
        world["Y"] = np.where(world["Z"] == 1, world["Y1"], world["Y0"])
        est, se = dp.difference_in_means(world)
        assert round(est, 4) == 2.0372  # nb09's hard-coded self-check value
        assert round(se, 4) == 0.8650
        ci_low, ci_high = est - 1.96 * se, est + 1.96 * se
        assert ci_low <= 2.0 <= ci_high

    def test_ols_hc2_equals_dim(self) -> None:
        rng = np.random.default_rng(SEED)
        world = make_world(n=500, rng=rng)
        world["Z"] = dp.complete_ra(500, 250, rng=rng)
        world["Y"] = np.where(world["Z"] == 1, world["Y1"], world["Y0"])
        est, _ = dp.difference_in_means(world)
        fit = dp.lm_robust("Y ~ Z", world)  # HC2 default, like the notebook
        z_row = fit[fit["term"] == "Z"].iloc[0]
        assert round(float(z_row["estimate"]), 4) == round(est, 4)

    def test_baseline_adjustment_shrinks_se(self) -> None:
        rng = np.random.default_rng(SEED)
        world = make_world(n=500, rng=rng)
        world["Z"] = dp.complete_ra(500, 250, rng=rng)
        world["Y"] = np.where(world["Z"] == 1, world["Y1"], world["Y0"])
        world["baseline"] = (world["Y0"] + rng.normal(0, 5, 500)).round(1)
        bal = world.groupby("Z")["baseline"].mean().round(2)
        assert (bal[0], bal[1]) == (50.03, 50.27)  # nb09 prose values
        dim = dp.lm_robust("Y ~ Z", world)
        adj = dp.lm_robust("Y ~ Z + baseline", world)
        se_dim = float(dim.loc[dim["term"] == "Z", "std_error"].iloc[0])
        se_adj = float(adj.loc[adj["term"] == "Z", "std_error"].iloc[0])
        assert se_adj < se_dim
        assert round(float(adj.loc[adj["term"] == "Z", "estimate"].iloc[0]), 3) == 1.849

    def test_lapop_regression_range(self) -> None:
        lapop = dp.data.load("lapop_brazil")
        assert lapop.shape == (10000, 10)
        fit = dp.lm_robust("trust_police ~ govt_responsive + ideology", lapop)
        coef = float(fit.loc[fit["term"] == "govt_responsive", "estimate"].iloc[0])
        assert 0.20 < coef < 0.28  # nb09's own range assert


def two_group_design(
    rng: np.random.Generator, n: int = 100, effect: float = 2.0, noise: float = 2.0
) -> tuple[float, float, float]:
    world = make_world(n, effect, noise, rng)
    world["Z"] = dp.complete_ra(n, n // 2, rng=rng)
    world["Y"] = np.where(world["Z"] == 1, world["Y1"], world["Y0"])
    est, se = dp.difference_in_means(world)
    return est, se, effect


class TestNb11DeclareDiagnoseRedesign:
    def test_canonical_diagnosis(self) -> None:
        canonical = dp.course.diagnose(lambda r: two_group_design(r))
        # The notebook's self-checks:
        assert abs(canonical["bias"]) < 0.15
        assert 0.92 < canonical["coverage"] < 0.98
        # Exact seeded values (seed 464, reps 1000):
        assert canonical.round(3).to_dict() == {
            "bias": -0.022, "power": 0.152, "coverage": 0.947
        }

    def test_sick_designs(self) -> None:
        def design_x(rng: np.random.Generator) -> tuple[float, float, float]:
            world = make_world(100, 2.0, 2.0, rng).sort_values("Y0").reset_index(drop=True)
            z = np.zeros(100, dtype=int)
            z[50:] = 1
            world["Z"] = z
            world["Y"] = np.where(world["Z"] == 1, world["Y1"], world["Y0"])
            est, se = dp.difference_in_means(world)
            return est, se, 2.0

        dx = dp.course.diagnose(design_x)
        assert dx["bias"] > 10 and dx["coverage"] == 0.0  # the eager-volunteer disease
        dy = dp.course.diagnose(lambda r: two_group_design(r, n=24))
        assert round(dy["power"], 3) == 0.093  # underpowered
        def design_z(rng: np.random.Generator) -> tuple[float, float, float]:
            est, se, truth = two_group_design(rng)
            return est, se * 0.5, truth
        dz = dp.course.diagnose(design_z)
        assert round(dz["coverage"], 3) == 0.685  # overconfident

    def test_redesign_grid_and_fix_comparison(self) -> None:
        powers = {
            n: dp.course.diagnose(
                lambda r, n=n: two_group_design(r, n=n, effect=2.0, noise=2.0)
            )["power"]
            for n in (100, 400)
        }
        assert powers[400] > powers[100]
        base = dp.course.diagnose(lambda r: two_group_design(r, n=100, noise=10))["power"]
        fix_1 = dp.course.diagnose(lambda r: two_group_design(r, n=200, noise=10))["power"]
        fix_2 = dp.course.diagnose(lambda r: two_group_design(r, n=100, noise=2))["power"]
        assert (round(base, 3), round(fix_1, 3), round(fix_2, 3)) == (0.121, 0.189, 0.152)
        assert fix_1 > fix_2  # 'double n' wins the equal-cost comparison


class TestNb10Inference:
    def test_sampling_distribution(self) -> None:
        small = dp.course.run_design(n=100, reps=1000)
        large = dp.course.run_design(n=400, reps=1000)
        ratio = large.est.std(ddof=1) / small.est.std(ddof=1)
        assert 0.42 < ratio < 0.58  # the notebook's √n-rule self-check
        assert abs(small.est.mean() - 2.0) < 0.2
        assert round(float(small.est.std(ddof=1)), 4) == 1.9532
        assert round(float(large.est.std(ddof=1)), 4) == 1.0205

    def test_power_grid(self) -> None:
        assert round(dp.course.power_at(100, 2.0), 3) == 0.153
        assert round(dp.course.power_at(200, 2.0), 3) == 0.284
        assert round(dp.course.power_at(100, 4.0), 3) == 0.516
        assert round(dp.course.power_at(100, 2.0, noise=0.0), 3) == 0.157
        strongest = dp.course.power_at(400, 4.0)
        weakest = dp.course.power_at(50, 0.5)
        assert strongest > weakest


class TestNb13RealData:
    def test_foos_difference_in_means(self) -> None:
        foos = dp.data.load("foos_etal")
        assert foos.shape == (8375, 5)
        r = dp.difference_in_means(foos, y="marked_register_2014", z="treat")
        n1 = int((foos["treat"] == 1).sum())
        n0 = int((foos["treat"] == 0).sum())
        assert (n1, n0) == (6104, 2271)
        assert round(r.estimate, 6) == 0.034074  # +3.4pp, the notebook's value
        assert abs(r.std_error - 0.012276) < 1e-4  # binomial vs Welch: equal to 4dp
        assert r.conf_low > 0  # the interval excludes zero

    def test_hajj_difference_in_means(self) -> None:
        hajj = dp.data.load("cliningsmith_etal")
        assert hajj.shape == (958, 8)
        r = dp.difference_in_means(hajj, y="views", z="success")
        assert round(r.estimate, 6) == 0.474834
        assert round(r.std_error, 6) == 0.162672
        assert r.estimate - 1.96 * r.std_error > 0


class TestNb07DataStrategy:
    def test_voter_file_sampling_facts(self) -> None:
        voters = dp.data.load("la_voter_file")
        assert voters.shape == (1000, 4)
        assert round(float(voters["age"].mean()), 1) == 48.8
        npp = voters.loc[voters["party"] == "NPP", "age"]
        assert abs(float(npp.mean()) - 37.0) < 0.5  # nb07's convenience-sample check

    def test_complete_ra_balance(self) -> None:
        rng = np.random.default_rng(SEED)
        baseline = rng.normal(50, 10, 100).round(1)
        z = dp.complete_ra(100, 50, rng=rng)
        assert int(z.sum()) == 50
        imbalance = abs(baseline[z == 1].mean() - baseline[z == 0].mean())
        optin = (baseline >= np.median(baseline)).astype(int)
        optin_imb = abs(baseline[optin == 1].mean() - baseline[optin == 0].mean())
        assert imbalance < optin_imb
        assert optin_imb > 10
