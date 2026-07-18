"""T3 ch17 acceptance: validate the six chapter-17 declarations.

Compares declarepy translations of declarations 17.1-17.6 against the
book's saved diagnosis objects (validation/reference/diagnosis_17.*.json)
and the freshly R-generated references (rgen_t3_ch17_*.json) under the
VALIDATION_REPORT.md protocol:

    bias / mean_estimate / mean_estimand  ±0.02 · sd(outcome)
    sd_estimate / rmse                    ±10% relative
    power ±0.05        coverage ±0.03

Documented band choices beyond the protocol defaults:

* mean_CI_width (17.3's custom diagnosand) is on the outcome scale like
  sd_estimate, so it gets the ±10% relative band (expected delta << 1%:
  the Welch CI width is nearly deterministic given N).
* Relative bands degenerate when the reference is numerically zero
  (17.1's sd_estimate ~ 8e-16, 17.4's direct-estimator bias/rmse at
  proportion_hiding = 0, ~1e-16): both sides are pure floating-point
  noise, so references below 1e-9 are checked absolutely against a 1e-9
  noise floor instead.
* The 17.6_a behavioral helpers (_invested/_returned and their grid
  averages) are checked exactly (±1e-9) against R-computed values in
  rgen_t3_ch17_helpers.json.

Usage:  .venv/bin/python validation/validate_t3_ch17.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch17 import (
    _average_invested,
    _average_returned,
    _invested,
    _returned,
    declaration_17_1,
    declaration_17_2,
    declaration_17_3,
    declaration_17_4,
    declaration_17_5,
    declaration_17_6,
)

HERE = Path(__file__).parent
REF = HERE / "reference"

#: References smaller than this are floating-point noise: check absolutely.
NOISE_FLOOR = 1e-9

results: list[dict[str, object]] = []


def load_ref(name: str) -> dict:
    with open(REF / f"{name}.json") as f:
        return json.load(f)


def check(
    element: str,
    diagnosand: str,
    ours: float,
    ref: Optional[float],
    tol: float,
    kind: str = "abs",
) -> None:
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        return
    delta = float(ours) - float(ref)
    if kind == "rel" and abs(ref) < NOISE_FLOOR:
        ok = abs(ours) < NOISE_FLOOR and abs(ref) < NOISE_FLOOR
        band = f"noise<{NOISE_FLOOR:g}"
    elif kind == "rel":
        ok = abs(delta) <= abs(tol * ref)
        band = f"±{tol:.0%} rel"
    else:
        ok = abs(delta) <= tol
        band = f"±{tol:.4g}"
    results.append(
        {
            "element": element,
            "diagnosand": diagnosand,
            "python": round(float(ours), 6),
            "reference": round(float(ref), 6),
            "delta": round(delta, 6),
            "band": band,
            "pass": ok,
        }
    )


DEFAULT_KEYS: list[tuple[str, str]] = [
    ("mean_estimand", "mean"),
    ("mean_estimate", "mean"),
    ("bias", "mean"),
    ("sd_estimate", "rel"),
    ("rmse", "rel"),
    ("power", "power"),
    ("coverage", "coverage"),
]


def check_default_row(
    element: str, mine: pd.Series, ref_row: dict, sd_outcome: float
) -> None:
    """All seven default diagnosands of one reference row."""
    for key, style in DEFAULT_KEYS:
        if style == "mean":
            check(element, key, mine[key], ref_row.get(key), 0.02 * sd_outcome)
        elif style == "rel":
            check(element, key, mine[key], ref_row.get(key), 0.10, "rel")
        elif style == "power":
            check(element, key, mine[key], ref_row.get(key), 0.05)
        else:
            check(element, key, mine[key], ref_row.get(key), 0.03)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    args = ap.parse_args()
    sims, seed = args.sims, args.seed

    # ---- 17.6_a behavioral helpers: exact vs R ------------------------
    helpers = load_ref("rgen_t3_ch17_helpers")
    a_grid = np.asarray(helpers["a_grid"], dtype=float)
    for i, (ours_v, ref_v) in enumerate(
        zip(_average_invested(a_grid), helpers["average_invested"])
    ):
        check(f"17.6a average_invested a={a_grid[i]:.3g}", "value", ours_v, ref_v, 1e-9)
    for i, (ours_v, ref_v) in enumerate(
        zip(_average_returned(a_grid), helpers["average_returned"])
    ):
        check(f"17.6a average_returned a2={a_grid[i]:.3g}", "value", ours_v, ref_v, 1e-9)
    inv = _invested(
        np.asarray(helpers["invested_a1"]), np.asarray(helpers["invested_a2"])
    )
    for i, (ours_v, ref_v) in enumerate(zip(inv, helpers["invested_values"])):
        check(f"17.6a invested case {i + 1}", "value", ours_v, ref_v, 1e-9)
    ret = _returned(
        np.asarray(helpers["returned_x1"]), np.asarray(helpers["returned_a2"])
    )
    for i, (ours_v, ref_v) in enumerate(zip(ret, helpers["returned_values"])):
        check(f"17.6a returned case {i + 1}", "value", ours_v, ref_v, 1e-9)

    # ---- declaration_17.1: R-generated reference (degenerate audit) ----
    ref1 = load_ref("rgen_t3_ch17_17_1")["diagnosands"][0]
    d1 = declaration_17_1()
    sd_y1 = float(dp.draw_data(d1, rng=1)["Y"].std(ddof=1))
    diag1 = dp.diagnose(d1, sims=sims, seed=seed).diagnosands.iloc[0]
    check_default_row("declaration_17.1", diag1, ref1, sd_y1)

    # ---- declaration_17.2: diagnosis_17.1.json -------------------------
    ref2 = load_ref("diagnosis_17.1")["diagnosands"][0]
    d2 = declaration_17_2()
    sd_y2 = float(dp.draw_data(d2, rng=1)["Y"].std(ddof=1))
    diag2 = dp.diagnose(d2, sims=sims, seed=seed).diagnosands.iloc[0]
    check_default_row("declaration_17.2", diag2, ref2, sd_y2)

    # ---- declaration_17.3: diagnosis_17.2.json (custom diagnosands) ----
    ref3 = load_ref("diagnosis_17.2")["diagnosands"][0]
    d3 = declaration_17_3()
    sd_y3 = float(dp.draw_data(d3, rng=1)["Y_list"].std(ddof=1))
    dgs3 = dp.Diagnosands(
        bias=lambda d: float((d["estimate"] - d["estimand"]).mean()),
        mean_CI_width=lambda d: float((d["conf_high"] - d["conf_low"]).mean()),
    )
    diag3 = dp.diagnose(d3, sims=sims, seed=seed, diagnosands=dgs3).diagnosands.iloc[0]
    check("declaration_17.3", "bias", diag3["bias"], ref3["bias"], 0.02 * sd_y3)
    check(
        "declaration_17.3", "mean_CI_width",
        diag3["mean_CI_width"], ref3["mean_CI_width"], 0.10, "rel",
    )

    # ---- declaration_17.4: diagnosis_17.3.json (20-design sweep) -------
    ref4 = load_ref("diagnosis_17.3")["diagnosands"]
    ph_grid = [0.0, 0.1, 0.2, 0.3]
    n_grid = [500, 1000, 1500, 2000, 2500]
    grid4 = dp.redesign(declaration_17_4, proportion_hiding=ph_grid, N=n_grid)
    diag4 = dp.diagnose_all(grid4, sims=sims, seed=seed).diagnosands
    sd4: dict[tuple[float, int], dict[str, float]] = {}
    for params in grid4.params:
        ph, n_val = float(params["proportion_hiding"]), int(params["N"])
        draw = dp.draw_data(declaration_17_4(**params), rng=1)  # type: ignore[arg-type]
        sd4[(ph, n_val)] = {
            "list": float(draw["Y_list"].std(ddof=1)),
            "direct": float(draw["Y_direct"].std(ddof=1)),
        }
    for row in ref4:
        ph, n_val = float(row["proportion_hiding"]), int(row["N"])
        est = str(row["estimator"])
        mine4 = diag4[
            np.isclose(diag4["proportion_hiding"].astype(float), ph)
            & (diag4["N"] == n_val)
            & (diag4["estimator"] == est)
        ].iloc[0]
        check_default_row(
            f"declaration_17.4 ph={ph} N={n_val} {est}", mine4, row, sd4[(ph, n_val)][est]
        )

    # ---- declaration_17.5: diagnosis_17.4.json (conjoint AMCEs) --------
    ref5 = load_ref("diagnosis_17.4")["diagnosands"]
    d5 = declaration_17_5()
    sd_y5 = float(dp.draw_data(d5, rng=1)["choice"].std(ddof=1))
    diag5 = dp.diagnose(d5, sims=sims, seed=seed).diagnosands
    for row in ref5:
        mine5 = diag5[diag5["inquiry"] == row["inquiry"]].iloc[0]
        check_default_row(f"declaration_17.5 {row['inquiry']}", mine5, row, sd_y5)

    # ---- declaration_17.6: diagnosis_17.5.json (deceive sweep) ---------
    ref6 = load_ref("diagnosis_17.5")["diagnosands"]
    grid6 = dp.redesign(declaration_17_6, deceive=[True, False])
    diag6 = dp.diagnose_all(grid6, sims=sims, seed=seed).diagnosands
    # Outcome sds from the honest design (the trusting estimator always sees
    # the honest invested; returned's scale barely moves under deception).
    draw6 = dp.draw_data(declaration_17_6(deceive=False), rng=1)
    sd6 = {
        "trusting": float(draw6["invested"].std(ddof=1)),
        "trustworthy": float(draw6["returned"].std(ddof=1)),
    }
    for row in ref6:
        deceive = bool(row["deceive"])
        inquiry = str(row["inquiry"])
        mine6 = diag6[
            (diag6["deceive"] == deceive) & (diag6["inquiry"] == inquiry)
        ].iloc[0]
        check_default_row(
            f"declaration_17.6 deceive={deceive} {inquiry}", mine6, row, sd6[inquiry]
        )

    # ---- report --------------------------------------------------------
    table = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={sims}, seed={seed})")
    out = HERE / "t3_results_ch17.csv"
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
