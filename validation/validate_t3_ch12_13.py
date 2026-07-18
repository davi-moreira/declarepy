"""Tranche-3 acceptance: validate the chapter 12 and 13 declarations.

Runs declarepy translations of declaration_12.1 (the three-arm clustered
survey experiment, incl. the diagnosis_12.2 redesign sweep over n_villages
x citizens_per_village), declaration_13.1 and declaration_13.2, comparing
every diagnosand against the book's saved diagnosis objects
(validation/reference/diagnosis_12.1.json, diagnosis_12.2.json,
diagnosis_13.1.json) and the freshly R-generated reference for
declaration_13.1 (rgen_t3_ch12_13_13_1.json) under the
VALIDATION_REPORT.md protocol:

    bias      ±0.02 · sd(Y)      power ±0.05      coverage ±0.03
    rmse      ±10% relative      mean/sd of estimates treated like bias
    cost      exact (a deterministic function of the design parameters)

Usage:  .venv/bin/python validation/validate_t3_ch12_13.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch12 import declaration_12_1
from declarepy.library.ch13 import declaration_13_1, declaration_13_2

HERE = Path(__file__).parent
REF = HERE / "reference"

results: list[dict[str, object]] = []

#: cost diagnosand from diagnosis_12.1.R / diagnosis_12.2.R:
#: mean(10 * n_villages + 1 * n_villages * citizens_per_village) — constant.
def cost_12(n_villages: int, citizens_per_village: int) -> float:
    return float(10 * n_villages + n_villages * citizens_per_village)


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
    delta = ours - ref
    if kind == "rel":
        ok = abs(delta) <= abs(tol * ref)
        band = f"±{tol:.0%} rel"
    else:
        ok = abs(delta) <= tol
        band = f"±{tol:.4g}"
    results.append(
        {
            "element": element,
            "diagnosand": diagnosand,
            "python": round(float(ours), 5),
            "reference": round(float(ref), 5),
            "delta": round(float(delta), 5),
            "band": band,
            "pass": ok,
        }
    )


def sd_outcome(design: dp.Design, outcome: str = "Y", seed: int = 1) -> float:
    df = dp.draw_data(design, rng=seed)
    return float(df[outcome].std(ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    args = ap.parse_args()
    sims, seed = args.sims, args.seed

    # ---- declaration_12.1: bias/rmse/power/cost, two inquiries ---------
    ref12 = load_ref("diagnosis_12.1")["diagnosands"]
    d12 = declaration_12_1()
    sdY12 = sd_outcome(d12, outcome="Y_observed")  # binary outcome, sd ≈ 0.5
    diag12 = dp.diagnose(d12, sims=sims, seed=seed).diagnosands
    for row in ref12:
        mine = diag12[diag12["inquiry"] == row["inquiry"]].iloc[0]
        el = f"declaration_12.1 {row['inquiry']}"
        check(el, "bias", mine["bias"], row["bias"], 0.02 * sdY12)
        check(el, "rmse", mine["rmse"], row["rmse"], 0.10, "rel")
        check(el, "power", mine["power"], row["power"], 0.05)
        check(el, "cost", cost_12(192, 48), row["cost"], 1e-9)

    # ---- declaration_12.2: the n_villages × citizens_per_village sweep -
    ref122 = load_ref("diagnosis_12.2")["diagnosands"]
    grid = dp.redesign(
        declaration_12_1,
        n_villages=[192, 500],
        citizens_per_village=[25, 50, 75, 100],
    )
    diag122 = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    for row in ref122:
        nv, cpv = int(row["n_villages"]), int(row["citizens_per_village"])
        sub = diag122[
            (diag122["n_villages"] == nv)
            & (diag122["citizens_per_village"] == cpv)
            & (diag122["inquiry"] == row["inquiry"])
        ]
        mine = sub.iloc[0]
        el = f"declaration_12.2 v={nv} c={cpv} {row['inquiry']}"
        check(el, "bias", mine["bias"], row["bias"], 0.02 * sdY12)
        check(el, "rmse", mine["rmse"], row["rmse"], 0.10, "rel")
        check(el, "power", mine["power"], row["power"], 0.05)
        check(el, "cost", cost_12(nv, cpv), row["cost"], 1e-9)

    # ---- declaration_13.1: full default-diagnosand row (R-generated ref)
    ref131 = load_ref("rgen_t3_ch12_13_13_1")["diagnosands"][0]
    d131 = declaration_13_1()
    sdY131 = sd_outcome(d131)
    diag131 = dp.diagnose(d131, sims=sims, seed=seed).diagnosands.iloc[0]
    for key, tol, kind in [
        ("mean_estimand", 0.02 * sdY131, "abs"),
        ("mean_estimate", 0.02 * sdY131, "abs"),
        ("bias", 0.02 * sdY131, "abs"),
        ("sd_estimate", 0.10, "rel"),
        ("rmse", 0.10, "rel"),
        ("power", 0.05, "abs"),
        ("coverage", 0.03, "abs"),
    ]:
        check("declaration_13.1", key, diag131[key], ref131.get(key), tol, kind)

    # ---- declaration_13.2: DIM and OLS rows (diagnosis_13.1.rds) -------
    ref132 = load_ref("diagnosis_13.1")["diagnosands"]
    d132 = declaration_13_2()
    sdY132 = sd_outcome(d132)
    diag132 = dp.diagnose(d132, sims=sims, seed=seed).diagnosands
    for row in ref132:
        mine = diag132[diag132["estimator"] == row["estimator"]].iloc[0]
        el = f"declaration_13.2 {row['estimator']}"
        for key, tol, kind in [
            ("mean_estimand", 0.02 * sdY132, "abs"),
            ("mean_estimate", 0.02 * sdY132, "abs"),
            ("bias", 0.02 * sdY132, "abs"),
            ("sd_estimate", 0.10, "rel"),
            ("rmse", 0.10, "rel"),
            ("power", 0.05, "abs"),
            ("coverage", 0.03, "abs"),
        ]:
            check(el, key, mine[key], row.get(key), tol, kind)

    # ---- report --------------------------------------------------------
    table = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))
    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={sims}, seed={seed})")
    out = HERE / "t3_results_ch12_13.csv"
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
