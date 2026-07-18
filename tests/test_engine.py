"""The declare→diagnose→redesign engine: steps, runs, diagnosis, sweeps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import declarepy as dp
from declarepy.library import declaration_10_1, declaration_11_1, declaration_18_1


def simple_design(n: int = 100, effect: float = 0.2) -> dp.Design:
    return declaration_10_1(N=n, effect=effect)


class TestDesignComposition:
    def test_step_addition_builds_design(self) -> None:
        d = dp.Model(n=10) + dp.Assignment.complete()
        assert isinstance(d, dp.Design) and len(d.steps) == 2

    def test_design_plus_design(self) -> None:
        d = dp.Design(dp.Model(n=10)) + dp.Design(dp.Assignment.complete())
        assert len(d.steps) == 2

    def test_model_requires_something(self) -> None:
        with pytest.raises(ValueError):
            dp.Model()

    def test_model_n_only_makes_id_frame(self) -> None:
        df = dp.draw_data(dp.Design(dp.Model(n=7)), rng=1)
        assert list(df.columns) == ["ID"] and len(df) == 7


class TestDrawAndRun:
    def test_draw_data_deterministic(self) -> None:
        d = simple_design()
        pd.testing.assert_frame_equal(dp.draw_data(d, rng=464), dp.draw_data(d, rng=464))

    def test_draw_data_has_pipeline_columns(self) -> None:
        df = dp.draw_data(simple_design(), rng=464)
        assert {"U", "Y0", "Y1", "Z", "Y"} <= set(df.columns)
        assert df["Z"].sum() == 50  # complete_ra default: exactly half
        # The reveal switch: Y equals the assigned potential outcome.
        assert (df["Y"] == np.where(df["Z"] == 1, df["Y1"], df["Y0"])).all()

    def test_run_design_returns_estimands_and_estimates(self) -> None:
        run = dp.run_design(simple_design(), rng=464)
        assert set(run.estimands) == {"ATE"}
        assert run.estimands["ATE"] == pytest.approx(0.2)
        assert len(run.estimates) == 1
        assert {"estimate", "std_error", "p_value", "conf_low", "conf_high"} <= set(
            run.estimates.columns
        )

    def test_sampling_filters_rows(self) -> None:
        d = dp.Design(
            dp.Model(n=1000),
            dp.Sampling.complete(n=150),
        )
        df = dp.draw_data(d, rng=1)
        assert len(df) == 150 and (df["S"] == 1).all()

    def test_multi_inquiry_multi_estimator(self) -> None:
        # declaration_10.3's shape: two outcomes, two inquiries, two estimators.
        def po(col: str, eff: float):  # type: ignore[no-untyped-def]
            return lambda df, z, rng: eff * float(z) + df["U"].to_numpy()

        d = dp.Design(
            dp.Model(n=200, build=lambda n, rng: {"U": rng.normal(size=n)}),
            dp.potential_outcomes(po("Y1", 0.2), outcome="Y1"),
            dp.potential_outcomes(po("Y2", 0.0), outcome="Y2"),
            dp.Inquiry(
                ATE1=lambda df: float((df["Y11"] - df["Y10"]).mean()),
                ATE2=lambda df: float((df["Y21"] - df["Y20"]).mean()),
            ),
            dp.Assignment.complete(),
            dp.reveal_outcomes(outcome="Y1"),
            dp.reveal_outcomes(outcome="Y2"),
            dp.Estimator.difference_in_means(y="Y1", inquiry="ATE1", label="DIM1"),
            dp.Estimator.difference_in_means(y="Y2", inquiry="ATE2", label="DIM2"),
        )
        run = dp.run_design(d, rng=464)
        assert run.estimands == {"ATE1": pytest.approx(0.2), "ATE2": pytest.approx(0.0)}
        assert list(run.estimates["estimator"]) == ["DIM1", "DIM2"]
        diag = dp.diagnose(d, sims=200, seed=464).diagnosands
        assert len(diag) == 2
        bias2 = float(diag.loc[diag["estimator"] == "DIM2", "bias"].iloc[0])
        assert abs(bias2) < 0.05


class TestDiagnose:
    def test_reproducible(self) -> None:
        d = simple_design()
        t1 = dp.diagnose(d, sims=300, seed=464).diagnosands
        t2 = dp.diagnose(d, sims=300, seed=464).diagnosands
        pd.testing.assert_frame_equal(t1, t2)

    def test_default_diagnosand_columns(self) -> None:
        diag = dp.diagnose(simple_design(), sims=200, seed=464)
        row = diag.diagnosands.iloc[0]
        for col in ["mean_estimand", "mean_estimate", "bias", "sd_estimate",
                    "rmse", "power", "coverage", "n_sims"]:
            assert col in diag.diagnosands.columns
        assert row["mean_estimand"] == pytest.approx(0.2)
        assert row["n_sims"] == 200

    def test_estimator_without_inquiry_has_nan_bias(self) -> None:
        diag = dp.diagnose(declaration_11_1(), sims=100, seed=464).diagnosands.iloc[0]
        assert np.isnan(diag["bias"]) and np.isnan(diag["coverage"])
        assert 0 <= diag["power"] <= 1

    def test_custom_diagnosands(self) -> None:
        dgs = dp.Diagnosands(
            median_estimate=lambda d: float(d["estimate"].median()),
        )
        diag = dp.diagnose(simple_design(), sims=100, seed=464, diagnosands=dgs)
        assert "median_estimate" in diag.diagnosands.columns

    def test_bootstrap_se_columns(self) -> None:
        diag = dp.diagnose(simple_design(), sims=100, seed=464, bootstrap_sims=30)
        assert "se(power)" in diag.diagnosands.columns
        assert float(diag.diagnosands["se(power)"].iloc[0]) > 0

    def test_known_truth_recovery(self) -> None:
        # Validation protocol §3: estimand recovered as reps grow.
        diag = dp.diagnose(declaration_18_1(), sims=3000, seed=7).diagnosands.iloc[0]
        assert diag["bias"] == pytest.approx(0.0, abs=0.012)
        assert diag["coverage"] == pytest.approx(0.95, abs=0.02)


class TestRedesign:
    def test_grid_labels_and_params(self) -> None:
        grid = dp.redesign(declaration_11_1, N=[100, 200, 300])
        assert grid.labels == ["design_1", "design_2", "design_3"]
        assert [p["N"] for p in grid.params] == [100, 200, 300]

    def test_cartesian_product(self) -> None:
        grid = dp.redesign(declaration_10_1, N=[100, 200], effect=[0.2, 0.5])
        assert len(grid) == 4
        assert grid.params[0] == {"N": 100, "effect": 0.2}
        assert grid.params[3] == {"N": 200, "effect": 0.5}

    def test_scalar_wrapped(self) -> None:
        grid = dp.redesign(declaration_10_1, N=100)
        assert len(grid) == 1

    def test_diagnose_all_param_columns_and_monotone_power(self) -> None:
        grid = dp.redesign(declaration_11_1, N=[100, 400, 900])
        diag = dp.diagnose_all(grid, sims=400, seed=464).diagnosands
        assert {"design", "N", "power"} <= set(diag.columns)
        powers = diag.sort_values("N")["power"].tolist()
        assert powers[0] < powers[1] < powers[2]  # power rises with N

    def test_diagnose_all_mapping(self) -> None:
        diag = dp.diagnose_all(
            {"small": simple_design(50), "large": simple_design(400)},
            sims=200,
            seed=464,
        ).diagnosands
        assert set(diag["design"]) == {"small", "large"}


class TestRevealOutcomes:
    def test_three_conditions(self) -> None:
        d = dp.Design(
            dp.Model(n=99, build=lambda n, rng: {
                "Y0": np.zeros(n), "Y1": np.ones(n), "Y2": np.full(n, 2.0)
            }),
            dp.Assignment(lambda df, rng: rng.integers(0, 3, len(df))),
            dp.reveal_outcomes(conditions=(0, 1, 2)),
        )
        df = dp.draw_data(d, rng=8)
        assert (df["Y"] == df["Z"].astype(float)).all()
