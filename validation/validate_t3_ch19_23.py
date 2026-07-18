"""T3 acceptance for group ch19_23: declarations 19.1-19.4 and 23.1.

Two layers of validation:

1. **Exact estimator-fixture checks** (fixed data exported from R,
   ``reference/rgen_t3_ch19_23_fixtures.json``): the private ports in
   ``declarepy.library.ch19`` of metafor's rma (REML + FE, Q-profile CI),
   bbmle's mle2 (L-BFGS-B + Hessian SEs + profile CIs) and rdss's
   best_predictor are compared number-for-number against R on the same
   inputs.

2. **Monte-Carlo diagnosand checks** against the book's saved diagnosis
   objects (``diagnosis_19.1/19.2/19a/19.3/19.4/23.1``), per
   docs/spec/VALIDATION_REPORT.md:

       bias / mean_estimate / mean_estimand  ±0.02·scale
       rmse / sd_estimate                    ±10 % relative
       power (and 19.1's 'correct')          ±0.05
       coverage                              ±0.03

   Every band is additionally floored at 3·sqrt(se_ref² + se_ours²) where
   se_ref is the reference's own bootstrap SE (stored in the JSONs) and
   se_ours ≈ se_ref·sqrt(n_sims_ref / sims_ours): both sides of the
   comparison are Monte-Carlo estimates, and the reference for 19.2/19a
   was built with only 100 simulations. This is the noise floor of the
   comparison itself, not a loosened standard — with an exact port, the
   difference is pure simulation noise and 3·SE is a 99.7 % band.

   Scale choices (documented per design below): sd(Y) of the design's
   outcome where one exists; the parameter's own magnitude for 19.2's
   structural parameters; one index step for 19.1's variable-importance
   row; MC noise alone for 19.3's tau_sq rows (no natural outcome scale,
   port verified exact on fixtures). 19.1's modal_* diagnosands round to
   one decimal, so they get a one-notch (±0.1) band.

Usage:  .venv/bin/python validation/validate_t3_ch19_23.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch19 import (
    _best_predictor,
    _binned_r_squared,
    _fit_bargaining,
    _rma_uni,
    declaration_19_1,
    declaration_19_2,
    declaration_19_3,
    declaration_19_4,
)
from declarepy.library.ch23 import declaration_23_1

HERE = Path(__file__).parent
REF = HERE / "reference"

results: list[dict[str, object]] = []


def load_ref(name: str) -> dict:
    with open(REF / f"{name}.json") as f:
        return json.load(f)


def record(element: str, diagnosand: str, ours: float, ref: float,
           band: float, label: str) -> None:
    delta = float(ours) - float(ref)
    results.append(
        {
            "element": element,
            "diagnosand": diagnosand,
            "python": round(float(ours), 5),
            "reference": round(float(ref), 5),
            "delta": round(delta, 5),
            "band": label,
            "pass": abs(delta) <= band,
        }
    )


def check_exact(element: str, diagnosand: str, ours: float, ref: object,
                tol: float) -> None:
    """Fixture check: exact numeric agreement within tol (abs)."""
    if ref is None or isinstance(ref, str):
        return
    record(element, diagnosand, ours, float(ref), tol, f"±{tol:g} exact")


BASE_ABS = {"power": 0.05, "correct": 0.05, "coverage": 0.03}
BASE_REL = {"rmse": 0.10, "sd_estimate": 0.10}
SCALED = {"bias", "mean_estimate", "mean_estimand"}


def check_mc(element: str, key: str, ours: float, ref_row: dict,
             sims_ours: int, scale: Optional[float]) -> None:
    """Monte-Carlo check with protocol base band + 3·SE noise floor."""
    ref = ref_row.get(key)
    if ref is None or isinstance(ref, str) or (isinstance(ref, float) and np.isnan(ref)):
        return
    se_ref = ref_row.get(f"se({key})") or 0.0
    if isinstance(se_ref, str):
        se_ref = 0.0
    n_ref = float(ref_row.get("n_sims") or sims_ours)
    if key in ("power", "coverage", "correct") and 0.0 < float(ref) < 1.0:
        # Binomial diagnosands: floor the reference's SE at the exact
        # binomial SE implied by its own value and sims count. The stored
        # bootstrap SE resamples the observed 0/1 draws and *understates*
        # the MC error when the proportion sits near a boundary at small
        # n_sims (diagnosis_19a's 100-sim cells: coverage 0.97 = 3 misses,
        # bootstrap SE 0.0146 vs binomial 0.0171). This is the reference's
        # own MC error, not a loosened standard; it only binds on
        # boundary-adjacent small-n_ref cells. The one row it adjudicates
        # (19a n=2 a=0.25 alpha coverage) is independently confirmed as
        # reference noise by the fresh NA-aware 500-sim R rerun
        # (rgen_t3_ch19_23_19a_sims500.json: R coverage_narm 0.955 with
        # na_ci_rate 0.042 -> NA-as-miss coverage ~0.913, matching our
        # 0.92; the book's 0.97 is a no-NA-conditional 100-sim draw --
        # see the 19a-fresh rows, all passing).
        se_ref = max(float(se_ref), float(np.sqrt(float(ref) * (1.0 - float(ref)) / n_ref)))
    se_comb = float(se_ref) * float(np.sqrt(1.0 + n_ref / sims_ours))
    if key in BASE_ABS:
        base = BASE_ABS[key]
    elif key in BASE_REL:
        base = BASE_REL[key] * abs(float(ref))
    elif key in SCALED:
        base = 0.02 * scale if scale is not None else 0.0
    elif key.startswith("modal_"):
        base = 0.10001
    else:
        base = 0.0
    band = max(base, 3.0 * se_comb)
    record(element, key, ours, float(ref), band, f"±{band:.4g}")


def row_of(table: pd.DataFrame, **kv: object) -> pd.Series:
    m = np.ones(len(table), dtype=bool)
    for k, v in kv.items():
        if isinstance(v, float):
            m &= np.isclose(table[k].astype(float), v)
        else:
            m &= (table[k] == v).to_numpy()
    sub = table[m]
    if len(sub) != 1:
        raise LookupError(f"expected 1 row for {kv}, found {len(sub)}")
    return sub.iloc[0]


def sd_col(df: pd.DataFrame, col: str = "Y") -> float:
    return float(df[col].std(ddof=1))


# =====================================================================
# Part 0 — exact fixture checks for the private estimator ports
# =====================================================================

def fixtures() -> None:
    fx = load_ref("rgen_t3_ch19_23_fixtures")

    for name in ["rma_k200", "rma_k200_tau0", "rma_k5"]:
        f = fx[name]
        yi = np.asarray(f["yi"], dtype=float)
        sei = np.asarray(f["sei"], dtype=float)
        for meth, key in [("REML", "reml"), ("FE", "fe")]:
            ours = _rma_uni(yi, sei, meth)
            ref = f[key]
            el = f"fixture {name} {meth}"
            # REML tolerance 1e-5: metafor's Fisher scoring stops at a
            # ~1e-5 relative change in tau², which propagates into mu's
            # SE/CI at that order (our optimizer runs to 1e-12). FE has no
            # iteration, so 1e-6 holds exactly.
            tol = 1e-5 if meth == "REML" else 1e-6
            pairs = [("mu", "b", tol), ("se", "se", tol), ("pval", "pval", tol),
                     ("ci_lb", "ci_lb", tol), ("ci_ub", "ci_ub", tol),
                     ("tau2", "tau2", tol), ("se_tau2", "se_tau2", tol),
                     ("tau2_ci_lb", "tau2_ci_lb", 1e-4), ("tau2_ci_ub", "tau2_ci_ub", 1e-4)]
            for a, b, t in pairs:
                check_exact(el, a, ours[a], ref.get(b), t)

    for name in ["mle_n2", "mle_n8"]:
        f = fx[name]
        df = pd.DataFrame({"y": np.asarray(f["y"], float), "Z": np.asarray(f["Z"], int)})
        tidy = _fit_bargaining(df, int(f["n_rounds"]))
        for i, term in enumerate(["k", "d", "a"]):
            row = tidy[tidy["term"] == term].iloc[0]
            el = f"fixture {name} {term}"
            check_exact(el, "estimate", float(row["estimate"]), f["estimate"][i], 1e-3)
            check_exact(el, "std_error", float(row["std_error"]), f["std_error"][i], 1e-3)
            check_exact(el, "conf_low", float(row["conf_low"]), f["conf_low"][i], 2e-3)
            check_exact(el, "conf_high", float(row["conf_high"]), f["conf_high"][i], 2e-3)

    f = fx["best_predictor"]
    X = pd.DataFrame(f["X"])
    X.columns = [f"X.{j}" for j in range(1, 11)]
    X["tau"] = np.asarray(f["tau"], float)
    for j in range(1, 11):
        r2 = _binned_r_squared(X[f"X.{j}"].to_numpy(), X["tau"].to_numpy())
        check_exact(f"fixture best_predictor X.{j}", "r_squared", r2, f["r_squared"][j - 1], 1e-9)
    check_exact("fixture best_predictor", "estimand",
                _best_predictor(X, [f"X.{j}" for j in range(1, 11)]), f["estimand"], 1e-9)


# =====================================================================
# Part 1 — diagnosis_19.1 (discovery diagnosands)
# =====================================================================

def _most_common_rounded(x: pd.Series) -> float:
    """rdss's most_common(round(x, 1)): first-appearing modal value."""
    r = np.round(np.asarray(x, dtype=float), 1)
    counts: dict[float, int] = {}
    for v in r:
        counts[float(v)] = counts.get(float(v), 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    return float(best[0])


def part_19_1(sims: int, seed: int) -> None:
    ref_rows = load_ref("diagnosis_19.1")["diagnosands"]
    design = declaration_19_1()
    sd_y = sd_col(dp.draw_data(design, rng=1))
    discovery = dp.Diagnosands(
        correct=lambda d: float((d["estimate"] == d["estimand"]).mean()),
        bias=lambda d: float((d["estimate"] - d["estimand"]).mean()),
        rmse=lambda d: float(np.sqrt(((d["estimate"] - d["estimand"]) ** 2).mean())),
        mean_estimate=lambda d: float(d["estimate"].mean()),
        modal_estimate=lambda d: _most_common_rounded(d["estimate"]),
        mean_estimand=lambda d: float(d["estimand"].mean()),
        modal_estimand=lambda d: _most_common_rounded(d["estimand"]),
    )
    t0 = time.time()
    diag = dp.diagnose(design, sims=sims, seed=seed, diagnosands=discovery).diagnosands
    print(f"[19.1] diagnosed in {time.time() - t0:.0f}s (sims={sims})")
    for ref in ref_rows:
        mine = row_of(diag, inquiry=ref["inquiry"], estimator=ref["estimator"])
        # cf's variable-importance row lives on the 1..10 index scale.
        scale = 1.0 if ref["estimator"] == "cf" else sd_y
        el = f"19.1 {ref['inquiry']}/{ref['estimator']}"
        for key in ["correct", "bias", "rmse", "mean_estimate", "modal_estimate",
                    "mean_estimand", "modal_estimand"]:
            check_mc(el, key, float(mine[key]), ref, sims, scale)


# =====================================================================
# Part 2 — diagnosis_19.2 and diagnosis_19a (structural model)
# =====================================================================

_DEFAULT_KEYS = ["mean_estimand", "mean_estimate", "bias", "sd_estimate",
                 "rmse", "power", "coverage"]


def part_19_2(sims: int, seed: int) -> None:
    ref_rows = load_ref("diagnosis_19.2")["diagnosands"]
    t0 = time.time()
    diag = dp.diagnose(declaration_19_2(), sims=sims, seed=seed).diagnosands
    print(f"[19.2] diagnosed in {time.time() - t0:.0f}s (sims={sims})")
    for ref in ref_rows:
        mine = row_of(diag, inquiry=ref["inquiry"])
        scale = abs(float(ref["mean_estimand"]))  # parameter scale (k≈2, d≈0.8, a)
        el = f"19.2 {ref['inquiry']}"
        for key in _DEFAULT_KEYS:
            check_mc(el, key, float(mine[key]), ref, sims, scale)


def part_19a(sims: int, seed: int) -> None:
    ref_rows = load_ref("diagnosis_19a")["diagnosands"]
    grid = dp.redesign(declaration_19_2, n=[2, 8], alpha=[0.25, 0.5, 0.75])
    t0 = time.time()
    diagnosis = dp.diagnose_all(grid, sims=sims, seed=seed)
    diag = diagnosis.diagnosands
    print(f"[19a] diagnosed in {time.time() - t0:.0f}s (sims={sims})")
    for ref in ref_rows:
        mine = row_of(diag, inquiry=ref["inquiry"], n=int(ref["n"]),
                      alpha=float(ref["alpha"]))
        scale = abs(float(ref["mean_estimand"]))
        el = f"19a n={ref['n']} a={ref['alpha']} {ref['inquiry']}"
        for key in _DEFAULT_KEYS:
            check_mc(el, key, float(mine[key]), ref, sims, scale)
    part_19a_fresh(diagnosis.simulations, sims)


def part_19a_fresh(sims_df: pd.DataFrame, sims: int) -> None:
    """Adjudicate 19a against the fresh 500-sim NA-aware R reference.

    The book's 100-sim reference is NA-fragile: bbmle's profile confint
    returns NA (or hard-errors) on boundary-adjacent fits and DeclareDesign
    diagnosands do not na.rm, so its power/coverage cells are conditional
    on a no-NA draw. rgen_t3_ch19_23_19a_sims500.json recomputes the sweep
    with na.rm diagnosands and NA rates; here the same NA-aware quantities
    are computed from our simulations and compared. Both sides are MC
    estimates: bands are max(protocol base, 3·combined SE).
    """
    fresh = load_ref("rgen_t3_ch19_23_19a_sims500")
    n_fresh = int(fresh["sims"])
    for ref in fresh["data"]:
        a, n, term = float(ref["alpha"]), int(ref["n"]), str(ref["term"])
        truth = {"k": 2.0, "d": 0.8, "a": a}[term]
        s = sims_df[
            np.isclose(sims_df["alpha"].astype(float), a)
            & (sims_df["n"].astype(int) == n)
            & (sims_df["term"] == term)
        ]
        est = s["estimate"].astype(float)
        ci_ok = s["conf_low"].notna() & s["conf_high"].notna()
        p_ok = s["p_value"].notna()
        ours = {
            "mean_estimate": float(est.mean()),
            "sd_estimate": float(est.std(ddof=1)),
            "bias": float((est - truth).mean()),
            "rmse": float(np.sqrt(((est - truth) ** 2).mean())),
            "power_narm": float((s.loc[p_ok, "p_value"] <= 0.05).mean()),
            "coverage_narm": float(
                ((s.loc[ci_ok, "conf_low"] <= truth) & (truth <= s.loc[ci_ok, "conf_high"])).mean()
            ),
            "na_ci_rate": float((~ci_ok).mean()),
        }
        el = f"19a-fresh n={n} a={a} {term}"
        sd_ref = float(ref["sd_estimate"])
        for key, base in [("mean_estimate", 0.02 * truth), ("bias", 0.02 * truth),
                          ("sd_estimate", 0.10 * sd_ref), ("rmse", 0.10 * float(ref["rmse"])),
                          ("power_narm", 0.05), ("coverage_narm", 0.03),
                          ("na_ci_rate", 0.02)]:
            refv = float(ref[key])
            if key in ("mean_estimate", "bias"):
                se_f = sd_ref / np.sqrt(n_fresh)
                se_o = sd_ref / np.sqrt(sims)
            elif key in ("sd_estimate", "rmse"):
                se_f = refv / np.sqrt(2 * n_fresh)
                se_o = refv / np.sqrt(2 * sims)
            else:
                p = min(max(refv, 1.0 / n_fresh), 1 - 1.0 / n_fresh)
                se_f = float(np.sqrt(p * (1 - p) / n_fresh))
                se_o = float(np.sqrt(p * (1 - p) / sims))
            band = max(base, 3.0 * float(np.hypot(se_f, se_o)))
            record(el, key, ours[key], refv, band, f"±{band:.4g}")


# =====================================================================
# Part 3 — diagnosis_19.3 (meta-analysis, tau ∈ {0, 1})
# =====================================================================

def part_19_3(sims: int, seed: int) -> None:
    ref_rows = load_ref("diagnosis_19.3")["diagnosands"]
    grid = dp.redesign(declaration_19_3, tau=[0.0, 1.0])
    t0 = time.time()
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    print(f"[19.3] diagnosed in {time.time() - t0:.0f}s (sims={sims})")
    sd_est = {
        t: sd_col(dp.draw_data(declaration_19_3(tau=t), rng=2), "estimate")
        for t in (0.0, 1.0)
    }
    for ref in ref_rows:
        tau = float(ref["tau"])
        mine = row_of(diag, inquiry=ref["inquiry"], estimator=ref["estimator"], tau=tau)
        # mu rows: scale = sd of the meta-analyzed estimates (the "outcome"
        # the estimators consume). tau_sq rows: the port is fixture-exact,
        # so the MC noise floor is the band (no natural outcome scale).
        scale = sd_est[tau] if ref["inquiry"] == "mu" else None
        el = f"19.3 tau={tau:g} {ref['inquiry']}/{ref['estimator']}"
        for key in _DEFAULT_KEYS:
            check_mc(el, key, float(mine[key]), ref, sims, scale)


# =====================================================================
# Part 4 — diagnosis_19.4 (multi-site coordination)
# =====================================================================

def part_19_4(sims: int, seed: int) -> None:
    ref_rows = load_ref("diagnosis_19.4")["diagnosands"]
    grid = dp.redesign(declaration_19_4, study_coordination=["high", "low"])
    t0 = time.time()
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    print(f"[19.4] diagnosed in {time.time() - t0:.0f}s (sims={sims})")
    # sd(Y) of the subject-level outcome: drop the site-summarizing step.
    d_full = declaration_19_4()
    d_subjects = dp.Design(*[s for s in d_full.steps if s.label != "site_lm_robust"])
    sd_y = sd_col(dp.draw_data(d_subjects, rng=3))
    for ref in ref_rows:
        mine = row_of(diag, inquiry=ref["inquiry"],
                      study_coordination=ref["study_coordination"])
        el = f"19.4 {ref['study_coordination']} {ref['inquiry']}"
        for key in _DEFAULT_KEYS:
            check_mc(el, key, float(mine[key]), ref, sims, sd_y)


# =====================================================================
# Part 5 — diagnosis_23.1 (to control or not to control)
# =====================================================================

def part_23_1(sims: int, seed: int) -> None:
    ref_rows = load_ref("diagnosis_23.1")["diagnosands"]
    designs = declaration_23_1()
    t0 = time.time()
    diag = dp.diagnose_all(designs, sims=sims, seed=seed).diagnosands
    print(f"[23.1] diagnosed in {time.time() - t0:.0f}s (sims={sims})")
    sd_y = {
        label: sd_col(dp.draw_data(d, rng=4)) for label, d in designs.items()
    }
    for ref in ref_rows:
        mine = row_of(diag, design=ref["design"], estimator=ref["estimator"])
        el = f"23.1 {ref['design']} {ref['estimator']}"
        for key in _DEFAULT_KEYS:
            check_mc(el, key, float(mine[key]), ref, sims, sd_y[str(ref["design"])])


# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="parts to skip (e.g. 19.1 19a)")
    ap.add_argument("--out", default=str(HERE / "t3_results_ch19_23.csv"),
                    help="results CSV path (default validation/t3_results_ch19_23.csv)")
    args = ap.parse_args()

    fixtures()
    parts = {
        "19.1": part_19_1,
        "19.2": part_19_2,
        "19a": part_19a,
        "19.3": part_19_3,
        "19.4": part_19_4,
        "23.1": part_23_1,
    }
    for name, fn in parts.items():
        if name in args.skip:
            print(f"[{name}] skipped")
            continue
        fn(args.sims, args.seed)

    table = pd.DataFrame(results)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={args.sims}, seed={args.seed})")
    out = Path(args.out)
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
