"""T2: clustered/CR2 SEs, logit + AME estimators, resample model handler.

Reference values from estimatr / glm / margins in R
(validation/reference/rgen_cluster.json, rgen_logit.json).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import declarepy as dp

REF_PATH = Path(__file__).parent.parent / "validation" / "reference"

with open(REF_PATH / "rgen_cluster.json") as f:
    CREF = json.load(f)
with open(REF_PATH / "rgen_logit.json") as f:
    LREF = json.load(f)

KEYMAP = {
    "estimate": "estimate",
    "std_error": "std.error",
    "p_value": "p.value",
    "conf_low": "conf.low",
    "conf_high": "conf.high",
    "df": "df",
}


def assert_matches(row: pd.Series, ref: dict, tol: float = 1e-8) -> None:
    for pk, rk in KEYMAP.items():
        if rk in ref:
            assert float(row[pk]) == pytest.approx(float(ref[rk]), abs=tol), pk


class TestClusteredLmRobust:
    @pytest.fixture(scope="class")
    def cl(self) -> pd.DataFrame:
        return pd.read_csv(REF_PATH / "cluster_fixture.csv")

    @pytest.mark.parametrize("se_type,key", [("CR2", "cr2_lm"), ("CR0", "cr0_lm"), ("stata", "stata_lm")])
    def test_cluster_se_types_match_estimatr(self, cl: pd.DataFrame, se_type: str, key: str) -> None:
        tidy = dp.lm_robust("Y ~ Z + X", cl, se_type=se_type, clusters="cl")
        for term in ["Z", "X"]:
            assert_matches(tidy[tidy["term"] == term].iloc[0], CREF[key][term])

    def test_cr2_within_cluster_variation(self) -> None:
        cl2 = pd.read_csv(REF_PATH / "cluster_fixture2.csv")
        tidy = dp.lm_robust("Y ~ Z + X", cl2, clusters="cl")  # CR2 default
        for term in ["Z", "X"]:
            assert_matches(tidy[tidy["term"] == term].iloc[0], CREF["cr2_lm2"][term])

    def test_clustered_dim_matches_estimatr(self, cl: pd.DataFrame) -> None:
        r = dp.difference_in_means(cl, y="Y", z="Z", clusters="cl")
        ref = CREF["dim_clustered"]["Z"]
        assert r.estimate == pytest.approx(ref["estimate"], abs=1e-8)
        assert r.std_error == pytest.approx(ref["std.error"], abs=1e-8)
        assert r.df == pytest.approx(ref["df"], abs=1e-6)
        assert r.p_value == pytest.approx(ref["p.value"], abs=1e-8)

    def test_blocked_plus_clustered_raises(self, cl: pd.DataFrame) -> None:
        cl = cl.assign(blk="b0")
        with pytest.raises(NotImplementedError):
            dp.difference_in_means(cl, y="Y", z="Z", blocks="blk", clusters="cl")

    def test_bad_cluster_se_type_raises(self, cl: pd.DataFrame) -> None:
        with pytest.raises(NotImplementedError):
            dp.lm_robust("Y ~ Z", cl, se_type="HC2", clusters="cl")


class TestLogitEstimators:
    def test_logit_matches_r_glm(self) -> None:
        foos = dp.data.load("foos_etal")
        lg = dp.glm_logit("marked_register_2014 ~ treat", foos)
        row = lg[lg["term"] == "treat"].iloc[0]
        ref = LREF["foos_logit"]["treat"]
        assert float(row["estimate"]) == pytest.approx(ref["estimate"], abs=1e-8)
        assert float(row["std_error"]) == pytest.approx(ref["std.error"], abs=1e-6)
        assert float(row["p_value"]) == pytest.approx(ref["p.value"], abs=1e-6)

    def test_ame_matches_r_margins(self) -> None:
        foos = dp.data.load("foos_etal")
        ame = dp.logit_ame("marked_register_2014 ~ treat", foos)
        row = ame[ame["term"] == "treat"].iloc[0]
        ref = LREF["foos_logit_ame"]["treat"]
        assert float(row["estimate"]) == pytest.approx(ref["estimate"], abs=1e-8)
        assert float(row["std_error"]) == pytest.approx(ref["std.error"], abs=1e-6)
        assert float(row["conf_low"]) == pytest.approx(ref["conf.low"], abs=1e-6)

    def test_ame_close_to_ols_on_binary_experiment(self) -> None:
        foos = dp.data.load("foos_etal")
        ame = dp.logit_ame("marked_register_2014 ~ treat", foos)
        dim = dp.difference_in_means(foos, y="marked_register_2014", z="treat")
        assert float(ame[ame["term"] == "treat"]["estimate"].iloc[0]) == pytest.approx(
            dim.estimate, abs=1e-4
        )


class TestGlanceAttrs:
    def test_r_squared_matches_r(self) -> None:
        foos = dp.data.load("foos_etal")
        fit = dp.lm_robust("marked_register_2014 ~ treat", foos)
        assert fit.attrs["r_squared"] == pytest.approx(LREF["foos_lm_r2"], abs=1e-10)
        assert fit.attrs["nobs"] == len(foos)


class TestResampleModel:
    def test_resample_shape_and_determinism(self) -> None:
        base = pd.DataFrame({"v": np.arange(10)})
        step = dp.Model.resample(base, n=25)
        d = dp.Design(step)
        df1 = dp.draw_data(d, rng=464)
        df2 = dp.draw_data(d, rng=464)
        assert len(df1) == 25
        pd.testing.assert_frame_equal(df1, df2)
        assert set(df1["v"]).issubset(set(base["v"]))

    def test_resample_draws_with_replacement(self) -> None:
        base = pd.DataFrame({"v": np.arange(5)})
        df = dp.draw_data(dp.Design(dp.Model.resample(base, n=100)), rng=1)
        assert df["v"].value_counts().max() > 1
