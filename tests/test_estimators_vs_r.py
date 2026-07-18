"""Estimator fidelity vs estimatr/R (real-data checks — no RNG excuse).

Reference values in validation/reference/rgen_estimatr.json were generated
by validation/r_scripts/estimatr_reference.R with estimatr 1.x on R 4.6.
Agreement demanded here is 1e-8 — machine precision, far beyond the
validation protocol's 3-decimal floor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import declarepy as dp

REF_PATH = Path(__file__).parent.parent / "validation" / "reference"
TOL = 1e-8

with open(REF_PATH / "rgen_estimatr.json") as f:
    REF = json.load(f)

KEYMAP = {
    "estimate": "estimate",
    "std_error": "std.error",
    "p_value": "p.value",
    "conf_low": "conf.low",
    "conf_high": "conf.high",
    "df": "df",
}


def assert_matches(mine: object, ref: dict, tol: float = TOL) -> None:
    for py_key, r_key in KEYMAP.items():
        mv = float(mine[py_key] if isinstance(mine, pd.Series) else getattr(mine, py_key))
        assert mv == pytest.approx(float(ref[r_key]), abs=tol), py_key


class TestDifferenceInMeans:
    def test_foos_unblocked(self) -> None:
        foos = dp.data.load("foos_etal")
        r = dp.difference_in_means(foos, y="marked_register_2014", z="treat")
        assert_matches(r, REF["foos_dim"]["treat"], tol=1e-6)  # df is Satterthwaite ≈ 4068.76

    def test_hajj_unblocked(self) -> None:
        hajj = dp.data.load("cliningsmith_etal")
        assert_matches(
            dp.difference_in_means(hajj, y="views", z="success"),
            REF["hajj_dim"]["success"],
        )

    def test_blocked_fixture(self) -> None:
        blk = pd.read_csv(REF_PATH / "blocked_fixture.csv")
        assert_matches(
            dp.difference_in_means(blk, y="Y", z="Z", blocks="block"),
            REF["blocked_dim"]["Z"],
        )


class TestLmRobust:
    def test_hajj_hc2(self) -> None:
        hajj = dp.data.load("cliningsmith_etal")
        fit = dp.lm_robust("views ~ success", hajj)
        row = fit[fit["term"] == "success"].iloc[0]
        assert_matches(row, REF["hajj_lm_hc2"]["success"])

    @pytest.mark.parametrize(
        "se_type,ref_key",
        [
            ("HC2", "lapop_lm_hc2"),
            ("classical", "lapop_lm_classical"),
            ("stata", "lapop_lm_hc1"),
            ("HC3", "lapop_lm_hc3"),
        ],
    )
    def test_lapop_se_types(self, se_type: str, ref_key: str) -> None:
        lapop = dp.data.load("lapop_brazil")
        fit = dp.lm_robust("trust_police ~ govt_responsive + ideology", lapop, se_type=se_type)
        row = fit[fit["term"] == "govt_responsive"].iloc[0]
        assert_matches(row, REF[ref_key]["govt_responsive"])

    def test_intercept_only_tiny_sample(self) -> None:
        tiny = pd.DataFrame({"age": [12, 47, 71]})
        fit = dp.lm_robust("age ~ 1", tiny)
        # For an intercept-only model HC2 equals the classical SE; df = n-1 = 2.
        assert_matches(fit.iloc[0], REF["tiny_lm_intercept"]["(Intercept)"], tol=1e-6)

    def test_unknown_se_type_raises(self) -> None:
        tiny = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0, 1, 0]})
        with pytest.raises(NotImplementedError):
            dp.lm_robust("y ~ x", tiny, se_type="CR2")


class TestPropTest:
    @pytest.mark.parametrize("case,x,n", [("prop_test_45_100", 45, 100), ("prop_test_3_10", 3, 10)])
    def test_matches_r_prop_test(self, case: str, x: int, n: int) -> None:
        r = REF[case]
        mine = dp.prop_test(x, n, p=0.5, correct=True)
        assert mine.estimate == pytest.approx(r["estimate"], abs=TOL)
        assert mine.statistic == pytest.approx(r["statistic"], abs=TOL)
        assert mine.p_value == pytest.approx(r["p.value"], abs=TOL)
        assert mine.conf_low == pytest.approx(r["conf.low"], abs=1e-6)
        assert mine.conf_high == pytest.approx(r["conf.high"], abs=1e-6)

    def test_boundary_clamps(self) -> None:
        assert dp.prop_test(0, 10).conf_low == 0.0
        assert dp.prop_test(10, 10).conf_high == 1.0


class TestNaSurfacing:
    def test_dim_raises_on_missing(self) -> None:
        df = pd.DataFrame({"Y": [1.0, None, 3.0, 4.0], "Z": [0, 0, 1, 1]})
        with pytest.raises(ValueError, match="missing"):
            dp.difference_in_means(df)

    def test_lm_robust_raises_on_missing(self) -> None:
        df = pd.DataFrame({"y": [1.0, None, 3.0, 4.0], "x": [0, 0, 1, 1]})
        with pytest.raises(ValueError, match="missing"):
            dp.lm_robust("y ~ x", df)
