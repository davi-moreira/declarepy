"""T3 ch16 acceptance: validate the chapter-16 declaration translations.

Compares declarepy's declarations 16.1-16.6 against the book's saved
diagnosis objects (validation/reference/diagnosis_16.*.json) and the
generated references (rgen_t3_ch16_16.4.json for the IV design, which has
no saved diagnosis, and rgen_t3_ch16_pt_posteriors.json for the exact
process-tracing posteriors), under the VALIDATION_REPORT.md protocol:

    bias / mean_estimate / mean_estimand   ±0.02 · sd(Y)
    sd_estimate / rmse                     ±10% relative
    power ±0.05          coverage ±0.03

Band widening (documented deviation): the reference diagnosands are
themselves Monte-Carlo estimates with bootstrap standard errors se(d)
recorded in the book's diagnosis objects. Wherever the protocol band is
narrower than the combined two-sided simulation noise, the effective band
is max(protocol, 3·sqrt(2)·se_ref) — 3 combined standard errors treating
our run's noise as equal to the reference's. Without this, group-conditional
diagnosands (16.1's XY cells) and heavy-tailed estimators (16.4's IV,
16.5's small-bandwidth local-linear rows) could not pass even against an
identical R re-run with a different seed. Bands widened this way are
flagged ("3se") in the band column. Structural fidelity is separately
pinned by exact fixed-input checks (process-tracing posteriors to 1e-9;
the matching, TWFE, DID_M, IV and rdrobust helpers were validated to
machine precision against R during development).

Usage:  .venv/bin/python validation/validate_t3_ch16.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch16 import (
    _PT_STRATEGIES,
    _pt_posterior,
    declaration_16_1,
    declaration_16_2,
    declaration_16_3,
    declaration_16_4,
    declaration_16_6,
)

HERE = Path(__file__).parent
REF = HERE / "reference"

results: list[dict[str, object]] = []

#: (diagnosand, protocol tolerance kind) — "sdY" scales by 0.02*sd(Y).
PROTOCOL = [
    ("mean_estimand", "sdY"),
    ("mean_estimate", "sdY"),
    ("bias", "sdY"),
    ("sd_estimate", "rel"),
    ("rmse", "rel"),
    ("power", "abs_power"),
    ("coverage", "abs_coverage"),
]


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
    se_ref: Optional[float] = None,
) -> None:
    if ref is None or (isinstance(ref, float) and math.isnan(ref)):
        return
    if isinstance(ours, float) and math.isnan(ours):
        results.append(
            {
                "element": element, "diagnosand": diagnosand,
                "python": float("nan"), "reference": round(float(ref), 5),
                "delta": float("nan"), "band": "value missing", "pass": False,
            }
        )
        return
    delta = float(ours) - float(ref)
    if kind == "rel" and ref == 0:
        # A zero reference (e.g. sd_estimate of a deterministic estimate)
        # makes a relative band degenerate; require numerical zero.
        tol_abs = 1e-9
        band = "±1e-09 (ref 0)"
    else:
        tol_abs = abs(tol * ref) if kind == "rel" else tol
        band = f"±{tol:.0%} rel" if kind == "rel" else f"±{tol:.4g}"
    if se_ref is not None and math.isfinite(se_ref):
        widened = 3.0 * math.sqrt(2.0) * se_ref
        if widened > tol_abs:
            tol_abs = widened
            band = f"±{widened:.4g} (3se)"
    ok = abs(delta) <= tol_abs
    results.append(
        {
            "element": element, "diagnosand": diagnosand,
            "python": round(float(ours), 5), "reference": round(float(ref), 5),
            "delta": round(delta, 5), "band": band, "pass": bool(ok),
        }
    )


def check_row(
    element: str,
    mine: pd.Series,
    ref_row: dict,
    sd_y: float,
) -> None:
    """Run the standard protocol checks for one diagnosands row."""
    for key, kind in PROTOCOL:
        ref_val = ref_row.get(key)
        if ref_val is None:
            continue
        se_ref = ref_row.get(f"se({key})")
        se_ref_f = float(se_ref) if se_ref is not None else None
        if kind == "sdY":
            check(element, key, mine[key], ref_val, 0.02 * sd_y, "abs", se_ref_f)
        elif kind == "rel":
            check(element, key, mine[key], ref_val, 0.10, "rel", se_ref_f)
        elif kind == "abs_power":
            check(element, key, mine[key], ref_val, 0.05, "abs", se_ref_f)
        else:
            check(element, key, mine[key], ref_val, 0.03, "abs", se_ref_f)


def sd_outcome(design: dp.Design, outcome: str = "Y", seed: int = 1) -> float:
    df = dp.draw_data(design, rng=seed)
    return float(df[outcome].std(ddof=1))


def group_diagnosands(sims: pd.DataFrame) -> dict[str, float]:
    """DeclareDesign's default diagnosands on one simulations group."""
    est = sims["estimate"].to_numpy(dtype=float)
    tru = sims["estimand"].to_numpy(dtype=float)
    return {
        "mean_estimand": float(tru.mean()),
        "mean_estimate": float(est.mean()),
        "bias": float((est - tru).mean()),
        "sd_estimate": float(np.std(est, ddof=1)) if len(est) > 1 else 0.0,
        "rmse": float(np.sqrt(((est - tru) ** 2).mean())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    args = ap.parse_args()
    sims, seed = args.sims, args.seed

    # ---- declaration_16.1a: exact process-tracing posteriors -----------
    # Deterministic fixed-input check of the whole causal-model machinery
    # (restricted nodal types, flat type prior, conditioning) against
    # CausalQueries/rdss on every observable single-case dataset.
    pt_ref = load_ref("rgen_t3_ch16_pt_posteriors")["posteriors"]
    for row in pt_ref:
        if row.get("estimate") is None:
            continue  # data pattern impossible under the model
        obs = {k: int(row[k]) for k in ("X", "M", "W", "Y")}
        mine = _pt_posterior(obs, _PT_STRATEGIES[str(row["term"])])
        check(
            f"declaration_16.1 exact X{obs['X']}M{obs['M']}W{obs['W']}Y{obs['Y']}",
            f"pt_estimate[{row['term']}]",
            mine,
            float(row["estimate"]),
            1e-9,
        )

    # ---- declaration_16.1b: Monte-Carlo diagnosis grouped by XY --------
    # The reference diagnosis used make_groups = vars(XY); estimates and
    # estimands are deterministic given the drawn causal type, so we group
    # the raw simulations by (strategy term, XY) ourselves.
    ref161 = load_ref("diagnosis_16.1")["diagnosands"]
    diag161 = dp.diagnose(declaration_16_1(), sims=sims, seed=seed)
    sims161 = diag161.simulations
    sd_y161 = 0.5  # Y is binary; sd(Y) ≈ 0.5
    for row in ref161:
        grp = sims161[
            (sims161["term"] == row["term"]) & (sims161["XY"] == row["XY"])
        ]
        element = f"declaration_16.1 {row['term']} {row['XY']}"
        if len(grp) == 0:
            check(element, "mean_estimate", float("nan"), row["mean_estimate"], 0.0)
            continue
        mine = pd.Series(group_diagnosands(grp))
        check_row(element, mine, row, sd_y161)

    # ---- declaration_16.2: exact matching --------------------------------
    ref162 = load_ref("diagnosis_16.2")["diagnosands"]
    d162 = declaration_16_2()
    sd_y162 = sd_outcome(d162)
    diag162 = dp.diagnose(d162, sims=sims, seed=seed).diagnosands
    for row in ref162:
        mine = diag162[diag162["estimator"] == row["estimator"]].iloc[0]
        check_row(f"declaration_16.2 {row['estimator']}", mine, row, sd_y162)

    # ---- declaration_16.3: staggered DID (book's later-higher POs) -----
    ref163 = load_ref("diagnosis_16.3")["diagnosands"]
    d163 = declaration_16_3()
    sd_y163 = sd_outcome(d163)
    diag163 = dp.diagnose(d163, sims=sims, seed=seed).diagnosands
    for row in ref163:
        mine = diag163[
            (diag163["estimator"] == row["estimator"])
            & (diag163["inquiry"] == row["inquiry"])
        ].iloc[0]
        check_row(
            f"declaration_16.3 {row['estimator']} {row['inquiry']}",
            mine, row, sd_y163,
        )

    # ---- diagnosis_16.4: the three PO variants of the DID design -------
    ref164d = load_ref("diagnosis_16.4")["diagnosands"]
    trend_by_design = {
        "PO_homogenous": 0.0,
        "PO_later_lower": 1.0,
        "PO_later_higher": -1.0,
    }
    for design_label, trend in trend_by_design.items():
        d = declaration_16_3(treatment_trend=trend)
        sd_y = sd_outcome(d)
        diag = dp.diagnose(d, sims=sims, seed=seed).diagnosands
        for row in ref164d:
            if row["design"] != design_label:
                continue
            mine = diag[
                (diag["estimator"] == row["estimator"])
                & (diag["inquiry"] == row["inquiry"])
            ].iloc[0]
            check_row(
                f"diagnosis_16.4 {design_label} {row['estimator']} {row['inquiry']}",
                mine, row, sd_y,
            )

    # ---- declaration_16.4: IV / LATE (generated reference) -------------
    ref16iv = load_ref("rgen_t3_ch16_16.4")["diagnosands"][0]
    d16iv = declaration_16_4()
    sd_y16iv = sd_outcome(d16iv)
    diag16iv = dp.diagnose(d16iv, sims=sims, seed=seed).diagnosands.iloc[0]
    check_row("declaration_16.4 iv_robust", diag16iv, ref16iv, sd_y16iv)

    # ---- declarations 16.5 + 16.6: RD sweep over bandwidth -------------
    ref165 = load_ref("diagnosis_16.5")["diagnosands"]
    grid = dp.redesign(
        declaration_16_6, bandwidth=[round(b, 2) for b in np.arange(0.05, 0.51, 0.05)]
    )
    diag165 = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    sd_y165 = sd_outcome(declaration_16_6())
    for row in ref165:
        bw = float(row["bandwidth"])
        mine_rows = diag165[
            np.isclose(diag165["bandwidth"].astype(float), bw)
            & (diag165["estimator"] == row["estimator"])
        ]
        mine = mine_rows.iloc[0]
        check_row(
            f"declaration_16.6 bw={bw} {row['estimator']}", mine, row, sd_y165
        )

    # ---- report --------------------------------------------------------
    table = pd.DataFrame(results)
    pd.set_option("display.width", 170)
    pd.set_option("display.max_rows", 1000)
    print(table.to_string(index=False))
    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={sims}, seed={seed})")
    out = HERE / "t3_results_ch16.csv"
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
