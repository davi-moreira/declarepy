"""Tranche-1 acceptance: validate the five reference declarations.

Runs declarepy translations of declarations 2.1, 9.1, 10.1, 11.1 and 18.1
(plus 2.2, which shares diagnosis_2.1) and compares every diagnosand against
the book's saved diagnosis objects (validation/reference/*.json) under the
VALIDATION_REPORT.md protocol:

    bias      ±0.02 · sd(Y)      power ±0.05      coverage ±0.03
    rmse      ±10% relative      mean/sd of estimates treated like bias

Usage:  .venv/bin/python validation/validate_t1.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library import (
    declaration_2_1,
    declaration_2_2,
    declaration_9_1,
    declaration_10_1,
    declaration_11_1,
    declaration_18_1,
)

HERE = Path(__file__).parent
REF = HERE / "reference"

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
    col = outcome if outcome in df.columns else ("Y1" if "Y1" in df.columns else df.columns[-1])
    return float(df[col].std(ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    args = ap.parse_args()
    sims, seed = args.sims, args.seed

    # ---- declaration_18.1: full default-diagnosand row -----------------
    ref = load_ref("diagnosis_18.1")["diagnosands"][0]
    d18 = declaration_18_1()
    sdY = sd_outcome(d18)
    diag = dp.diagnose(d18, sims=sims, seed=seed).diagnosands.iloc[0]
    for key, tol, kind in [
        ("mean_estimand", 0.02 * sdY, "abs"),
        ("mean_estimate", 0.02 * sdY, "abs"),
        ("bias", 0.02 * sdY, "abs"),
        ("sd_estimate", 0.10, "rel"),
        ("rmse", 0.10, "rel"),
        ("power", 0.05, "abs"),
        ("coverage", 0.03, "abs"),
    ]:
        check("declaration_18.1", key, diag[key], ref.get(key), tol, kind)

    # ---- declaration_10.1: power vs the book's hand-rolled loop --------
    ref10 = load_ref("diagnosis_10.1")["data"][0]
    d10 = declaration_10_1()
    diag10 = dp.diagnose(d10, sims=sims, seed=seed).diagnosands.iloc[0]
    check("declaration_10.1", "power", diag10["power"], ref10["power"], 0.05)

    # ---- declaration_9.1: full row (lm_robust intercept, n=3) ----------
    ref9 = load_ref("diagnosis_9.1")["diagnosands"][0]
    d9 = declaration_9_1()
    sd_age = sd_outcome(d9, outcome="age")
    diag9 = dp.diagnose(d9, sims=sims, seed=seed).diagnosands.iloc[0]
    for key, tol, kind in [
        ("mean_estimand", 0.02 * sd_age, "abs"),
        ("mean_estimate", 0.05 * sd_age, "abs"),  # sd_estimate/√sims ≈ 0.3; wider mean band
        ("bias", 0.05 * sd_age, "abs"),
        ("sd_estimate", 0.10, "rel"),
        ("rmse", 0.10, "rel"),
        ("power", 0.05, "abs"),
        ("coverage", 0.03, "abs"),
    ]:
        check("declaration_9.1", key, diag9[key], ref9.get(key), tol, kind)

    # ---- declaration_11.1: redesign over N, prop.test rows -------------
    ref11 = load_ref("diagnosis_11.1")["diagnosands"]
    grid = dp.redesign(declaration_11_1, N=list(range(100, 1001, 100)))
    diag11 = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    for row in ref11:
        N = int(row["N"])
        mine = diag11[diag11["N"] == N].iloc[0]
        check(f"declaration_11.1 N={N}", "mean_estimate", mine["mean_estimate"], row["mean_estimate"], 0.01)
        check(f"declaration_11.1 N={N}", "sd_estimate", mine["sd_estimate"], row["sd_estimate"], 0.10, "rel")
        check(f"declaration_11.1 N={N}", "power", mine["power"], row["power"], 0.05)

    # ---- declaration_2.1 (+2.2): success/failure over the b grid -------
    ref2 = load_ref("diagnosis_2.1")["diagnosands"]
    program_diagnosands = dp.Diagnosands(
        success=lambda d: float(
            ((d["estimate"] > 0.3) & (d["p_value"] < 0.05) & (d["estimand"] > 0.2)).mean()
        ),
        failure=lambda d: float(
            ((d["estimate"] > 0.3) & (d["p_value"] < 0.05) & (d["estimand"] < 0.2)).mean()
        ),
    )
    b_grid = [round(x, 2) for x in np.arange(0, 3.01, 0.25)]
    for suffix, factory in [("twoarm", declaration_2_1), ("blocked", declaration_2_2)]:
        grid2 = dp.redesign(factory, b=b_grid)
        diag2 = dp.diagnose_all(
            grid2, sims=sims, seed=seed, diagnosands=program_diagnosands
        ).diagnosands
        for row in ref2:
            if not str(row["design"]).endswith(suffix):
                continue
            b = float(row["b"])
            mine = diag2[np.isclose(diag2["b"].astype(float), b)].iloc[0]
            check(f"declaration_2.x {suffix} b={b}", "success", mine["success"], row["success"], 0.05)
            check(f"declaration_2.x {suffix} b={b}", "failure", mine["failure"], row["failure"], 0.05)

    # ---- report --------------------------------------------------------
    table = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))
    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={sims}, seed={seed})")
    out = HERE / "t1_results.csv"
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
