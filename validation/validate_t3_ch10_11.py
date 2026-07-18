"""T3 acceptance: validate the chapter 10-11 declaration library group.

Covers declarations 10.2, 10.3a/10.3b, 10.4, 10a and 11.2-11.5 (plus the
custom-diagnosand re-diagnoses of 10.1 saved as diagnosis_10.2/10.3),
comparing every diagnosand against the book's saved diagnosis objects
(validation/reference/diagnosis_*.json) or, where the book saved none,
against fresh R runs (rgen_t3_ch10_11_*.json from
validation/r_scripts/t3_ch10_11_reference.R).

Tolerances per docs/spec/VALIDATION_REPORT.md:

    bias / mean_estimate / mean_estimand  ±0.02 · sd(Y)
    power ±0.05      coverage ±0.03
    sd_estimate / rmse (and true_se)      ±10% relative

Documented bands for diagnosands on other scales:

* diagnosis_10a's ``var_estimate`` (population variance of estimates) is
  the square of a sd-scale quantity → the protocol's ±10% sd band maps to
  ±21% on the variance scale; we use ±20% relative. ``mean_var_hat`` (mean
  squared SE) is a low-noise average (bootstrap SE ≲ 1% of the value) and
  gets ±10% relative.
* diagnosis_11.3's ``cost`` is deterministic in (N, prob). The book computes
  it inside declare_diagnosands from the redesign parameters; declarepy's
  simulations frame carries no parameter columns, so we recompute
  N·2 + prob·N·20 from the sweep parameters and require exact (±1e-6)
  agreement — a plumbing check on the 273-design grid.
* the margins fixture (fixed data) checks the logit/probit AME estimators
  of declaration 11.5 against R's ``margins`` numbers: estimate ±1e-6,
  SE/CI ±0.01% relative (margins differentiates numerically; statsmodels
  analytically).

MC-noise band floor (documented band widening, per the tolerance protocol's
"do not silently widen" rule — this is the written justification):

    Every reference here is itself a 2000-sim Monte-Carlo estimate (the
    diagnosis scripts all set ``sims <- 2000``). Where the book saved
    bootstrap SEs (diagnosis_10.2/10.3/10.4/10a ran ``bootstrap_sims =
    2000``) we take ``se_ref`` from the saved ``se(<diagnosand>)`` field;
    for the rest (``bootstrap_sims = FALSE``) we set ``se_ref := se_py``,
    since the reference ran the identical design at the same sims count.
    Each stochastic check then enforces

        |python − reference| ≤ max(protocol band, Z_MC·√(se_py² + se_ref²))

    with ``Z_MC = 4`` (two-sided normal tail ≈ 6.3e-5, i.e. ≈ 0.14 expected
    false failures across all ~2200 checks). ``se_py`` is computed from our
    own simulations frame: exact for means and proportions, asymptotic
    kurtosis-adjusted for sd/variance-scale diagnosands, delta-method for
    rmse. Rows where the floor (not the protocol band) binds are visible in
    the CSV's band column as ``±4×MCSE=…``. The floor matters mainly for:

    (a) designs whose estimand is redrawn every simulation — 10.2/11.2 draw
        ATE ~ U(0, 0.5) (sd ≈ 0.144) per sim, so mean_estimate/mean_estimand
        have MC sd ≈ 0.0046 at 2000 sims and the fixed ±0.02·sd(Y) band is
        only ≈ 2–3 combined-MC sigmas wide;
    (b) declaration 11.4's grid edges, where degree-4..6 polynomial
        extrapolation makes the per-cell estimate distribution wide and
        heavy-tailed, so the ±0.02·sd(Y) band on mean_estimate/bias is far
        below the MC error of a 2000-vs-2000-sim comparison;
    (c) the small-N/prob=0.1 cells of 11.3's rmse grid, where the ±10%
        relative band is ≈ 4.3 combined sigmas — tight enough that a few of
        273 cells can fail by chance without the floor.

    Deterministic checks (11.3's cost, 11.4's mean_estimand, the margins
    fixture) never get the floor.

Usage:  .venv/bin/python validation/validate_t3_ch10_11.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch10 import (
    declaration_10_1,
    declaration_10_2,
    declaration_10_3a,
    declaration_10_3b,
    declaration_10_4,
    declaration_10a,
)
from declarepy.library.ch11 import (
    _probit_ame,
    declaration_11_2,
    declaration_11_3,
    declaration_11_4,
    declaration_11_5,
)

HERE = Path(__file__).parent
REF = HERE / "reference"

results: list[dict[str, object]] = []

#: The default-diagnosand tolerance recipe: (key, sd_scale, kind).
FULL_ROW = [
    ("mean_estimand", "sd", "abs"),
    ("mean_estimate", "sd", "abs"),
    ("bias", "sd", "abs"),
    ("sd_estimate", 0.10, "rel"),
    ("rmse", 0.10, "rel"),
    ("power", 0.05, "abs"),
    ("coverage", 0.03, "abs"),
]


#: MC-floor z-multiplier — see the module docstring's justification block.
Z_MC = 4.0


def load_ref(name: str) -> dict:
    with open(REF / f"{name}.json") as f:
        return json.load(f)


def mc_ses(
    grp: pd.DataFrame, ref_row: Optional[Mapping[str, object]] = None
) -> dict[str, float]:
    """Combined MC standard errors √(se_py² + se_ref²) per diagnosand.

    ``se_py`` from our simulations frame; ``se_ref`` from the reference's
    bootstrap ``se(...)`` field when the book saved one, else := se_py
    (identical design, same 2000-sim count). See the module docstring.
    """
    est = grp["estimate"].to_numpy(dtype=float)
    ind = grp["estimand"].to_numpy(dtype=float)
    n = len(est)
    rt_n = float(np.sqrt(n))
    se_py: dict[str, float] = {"mean_estimate": float(np.std(est, ddof=1)) / rt_n}
    if np.isfinite(ind).all():
        se_py["mean_estimand"] = float(np.std(ind, ddof=1)) / rt_n
        err = est - ind
        se_py["bias"] = float(np.std(err, ddof=1)) / rt_n
        e2 = err**2
        r = float(np.sqrt(e2.mean()))
        if r > 0:
            # delta method: se(rmse) = se(mse) / (2·rmse)
            se_py["rmse"] = float(np.std(e2, ddof=1)) / rt_n / (2.0 * r)
    c = est - est.mean()
    s2 = float(np.mean(c**2))
    m4 = float(np.mean(c**4))
    if s2 > 0:
        # asymptotic, kurtosis-adjusted: Var(s²) = (m4 − s⁴)/n
        se_var = float(np.sqrt(max(m4 - s2**2, 0.0) / n))
        se_py["var_estimate"] = se_var
        se_py["sd_estimate"] = se_var / (2.0 * float(np.sqrt(s2)))
        se_py["true_se"] = se_py["sd_estimate"]
    if grp["p_value"].notna().any():
        p = float((grp["p_value"] <= 0.05).mean())
        se_py["power"] = float(np.sqrt(p * (1.0 - p) / n))
    if grp["conf_low"].notna().any() and np.isfinite(ind).all():
        cov = float(
            (
                (grp["conf_low"] <= grp["estimand"])
                & (grp["estimand"] <= grp["conf_high"])
            ).mean()
        )
        se_py["coverage"] = float(np.sqrt(cov * (1.0 - cov) / n))
    if "std_error" in grp.columns and grp["std_error"].notna().any():
        v = grp["std_error"].to_numpy(dtype=float) ** 2
        se_py["mean_var_hat"] = float(np.std(v, ddof=1)) / rt_n
    out: dict[str, float] = {}
    for name, sp in se_py.items():
        sr = sp
        if ref_row is not None:
            raw = ref_row.get(f"se({name})")
            if raw is not None and np.isfinite(float(raw)):  # type: ignore[arg-type]
                sr = float(raw)  # type: ignore[arg-type]
        out[name] = float(np.hypot(sp, sr))
    return out


def check(
    element: str,
    diagnosand: str,
    ours: float,
    ref: Optional[float],
    tol: float,
    kind: str = "abs",
    mc_se: Optional[float] = None,
) -> None:
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        return
    delta = float(ours) - float(ref)
    if kind == "rel":
        limit = abs(tol * float(ref))
        band = f"±{tol:.2%} rel"
    else:
        limit = float(tol)
        band = f"±{tol:.4g}"
    if mc_se is not None and Z_MC * mc_se > limit:
        # Documented MC-noise floor (module docstring); visible in the CSV.
        limit = Z_MC * mc_se
        band = f"±{Z_MC:g}×MCSE={limit:.4g}"
    ok = abs(delta) <= limit
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


def full_row_checks(
    element: str,
    mine: pd.Series,
    ref: dict,
    sd_y: float,
    ses: Optional[Mapping[str, float]] = None,
) -> None:
    """The seven default diagnosands under the protocol's bands (+ MC floor)."""
    for key, tol, kind in FULL_ROW:
        t = 0.02 * sd_y if tol == "sd" else float(tol)
        check(element, key, mine[key], ref.get(key), t, kind, mc_se=(ses or {}).get(key))


def sd_outcome(design: dp.Design, outcome: str = "Y", seed: int = 1) -> float:
    df = dp.draw_data(design, rng=seed)
    return float(df[outcome].std(ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    args = ap.parse_args()
    sims, seed = args.sims, args.seed
    t0 = time.time()

    def tick(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    # ---- diagnosis_10.2 + 10.3: custom diagnosands of declaration_10.1 ----
    d101 = declaration_10_1()
    sd_y = sd_outcome(d101)
    sims_df = dp.diagnose(d101, sims=sims, seed=seed).simulations
    ref = load_ref("diagnosis_10.2")["diagnosands"][0]
    ses = mc_ses(sims_df, ref)
    check(
        "declaration_10.1 (diagnosis_10.2)",
        "power",
        float((sims_df["p_value"] <= 0.05).mean()),
        ref["power"],
        0.05,
        mc_se=ses.get("power"),
    )
    ref = load_ref("diagnosis_10.3")["diagnosands"][0]
    ses = mc_ses(sims_df, ref)
    el = "declaration_10.1 (diagnosis_10.3)"
    check(el, "bias", float((sims_df["estimate"] - sims_df["estimand"]).mean()), ref["bias"], 0.02 * sd_y, mc_se=ses.get("bias"))
    check(el, "true_se", float(sims_df["estimate"].std(ddof=1)), ref["true_se"], 0.10, "rel", mc_se=ses.get("true_se"))
    check(el, "power", float((sims_df["p_value"] <= 0.05).mean()), ref["power"], 0.05, mc_se=ses.get("power"))
    check(
        el,
        "coverage",
        float(
            (
                (sims_df["conf_low"] <= sims_df["estimand"])
                & (sims_df["estimand"] <= sims_df["conf_high"])
            ).mean()
        ),
        ref["coverage"],
        0.03,
        mc_se=ses.get("coverage"),
    )
    tick("diagnosis_10.2/10.3 (declaration_10.1) done")

    # ---- diagnosis_10.4: declaration_10.2, full default row ---------------
    d102 = declaration_10_2()
    ref = load_ref("diagnosis_10.4")["diagnosands"][0]
    d = dp.diagnose(d102, sims=sims, seed=seed)
    mine = d.diagnosands.iloc[0]
    full_row_checks(
        "declaration_10.2", mine, ref, sd_outcome(d102), ses=mc_ses(d.simulations, ref)
    )
    tick("diagnosis_10.4 (declaration_10.2) done")

    # ---- diagnosis_10.5: declarations 10.3a + 10.3b -----------------------
    refs = load_ref("diagnosis_10.5")["diagnosands"]
    d = dp.diagnose_all(
        {"design_1": declaration_10_3a(), "design_2": declaration_10_3b()},
        sims=sims,
        seed=seed,
    )
    diag = d.diagnosands
    sd_y1 = sd_outcome(declaration_10_3a(), outcome="Y1")
    for row in refs:
        sel = (diag["design"] == row["design"]) & (diag["estimator"] == row["estimator"])
        m = diag[sel].iloc[0]
        grp = d.simulations[
            (d.simulations["design"] == row["design"])
            & (d.simulations["estimator"] == row["estimator"])
        ]
        suffix = "a" if row["design"] == "design_1" else "b"
        full_row_checks(
            f"declaration_10.3{suffix} {row['estimator']}", m, row, sd_y1,
            ses=mc_ses(grp, row),
        )
    tick("diagnosis_10.5 (declarations 10.3a/b) done")

    # ---- rgen reference: declaration_10.4 ---------------------------------
    refs = load_ref("rgen_t3_ch10_11_decl_10.4")["diagnosands"]
    d104 = declaration_10_4()
    d = dp.diagnose(d104, sims=sims, seed=seed)
    diag = d.diagnosands
    sd_y = sd_outcome(d104)
    for row in refs:
        m = diag[diag["estimator"] == row["estimator"]].iloc[0]
        grp = d.simulations[d.simulations["estimator"] == row["estimator"]]
        full_row_checks(
            f"declaration_10.4 {row['estimator']}", m, row, sd_y, ses=mc_ses(grp, row)
        )
    tick("rgen decl_10.4 done")

    # ---- diagnosis_10a: var_estimate vs mean_var_hat sweep ----------------
    refs = load_ref("diagnosis_10a")["diagnosands"]
    grid = dp.redesign(
        declaration_10a,
        heteroskedasticity=[-0.4, 0.0, 0.4],
        prob_treated=[float(p) for p in np.linspace(0.1, 0.9, 7)],
    )
    dgs = dp.Diagnosands(
        # DeclareDesign's pop.var: mean((x - mean(x))^2), ddof = 0.
        var_estimate=lambda d: float(
            np.mean((d["estimate"] - d["estimate"].mean()) ** 2)
        ),
        mean_var_hat=lambda d: float((d["std_error"] ** 2).mean()),
    )
    d = dp.diagnose_all(grid, sims=sims, seed=seed, diagnosands=dgs)
    diag = d.diagnosands
    for row in refs:
        h, p = float(row["heteroskedasticity"]), float(row["prob_treated"])
        m = diag[
            np.isclose(diag["heteroskedasticity"].astype(float), h)
            & np.isclose(diag["prob_treated"].astype(float), p, atol=1e-6)
            & (diag["estimator"] == row["estimator"])
        ].iloc[0]
        grp = d.simulations[
            np.isclose(d.simulations["heteroskedasticity"].astype(float), h)
            & np.isclose(d.simulations["prob_treated"].astype(float), p, atol=1e-6)
            & (d.simulations["estimator"] == row["estimator"])
        ]
        ses = mc_ses(grp, row)
        el = f"declaration_10a h={h} p={p:.3g} {row['estimator'].split(' ')[0]}"
        check(el, "var_estimate", m["var_estimate"], row["var_estimate"], 0.20, "rel", mc_se=ses.get("var_estimate"))
        check(el, "mean_var_hat", m["mean_var_hat"], row["mean_var_hat"], 0.10, "rel", mc_se=ses.get("mean_var_hat"))
    tick("diagnosis_10a (42 rows) done")

    # ---- diagnosis_11.2: redesign over N ----------------------------------
    refs = load_ref("diagnosis_11.2")["diagnosands"]
    grid = dp.redesign(declaration_11_2, N=[100, 500, 1000])
    d = dp.diagnose_all(grid, sims=sims, seed=seed)
    diag = d.diagnosands
    sd_y = sd_outcome(declaration_11_2())
    for row in refs:
        N = int(row["N"])
        m = diag[diag["N"] == N].iloc[0]
        grp = d.simulations[d.simulations["N"] == N]
        full_row_checks(f"declaration_11.2 N={N}", m, row, sd_y, ses=mc_ses(grp, row))
    tick("diagnosis_11.2 done")

    # ---- diagnosis_11.3: cost/rmse over the 273-design grid ---------------
    refs = load_ref("diagnosis_11.3")["diagnosands"]
    grid = dp.redesign(
        declaration_11_3, N=list(range(100, 1001, 10)), prob=[0.1, 0.3, 0.5]
    )
    dgs = dp.Diagnosands(
        rmse=lambda d: float(np.sqrt(((d["estimate"] - d["estimand"]) ** 2).mean()))
    )
    d = dp.diagnose_all(grid, sims=sims, seed=seed, diagnosands=dgs)
    diag = d.diagnosands
    sims_by_cell = {
        (int(k[0]), round(float(k[1]), 6)): g
        for k, g in d.simulations.groupby(["N", "prob"], sort=False)
    }
    for row in refs:
        N, prob = int(row["N"]), float(row["prob"])
        m = diag[
            (diag["N"] == N) & np.isclose(diag["prob"].astype(float), prob)
        ].iloc[0]
        grp = sims_by_cell[(N, round(prob, 6))]
        el = f"declaration_11.3 N={N} prob={prob}"
        # cost is deterministic in the sweep parameters (see module docstring).
        check(el, "cost", 2.0 * N + prob * N * 20.0, row["cost"], 1e-6)
        check(el, "rmse", m["rmse"], row["rmse"], 0.10, "rel", mc_se=mc_ses(grp, row).get("rmse"))
    tick("diagnosis_11.3 (273 designs) done")

    # ---- diagnosis_11.4: 50 inquiries × 6 polynomial estimators -----------
    refs = load_ref("diagnosis_11.4")["diagnosands"]
    d114 = declaration_11_4()
    d = dp.diagnose(d114, sims=sims, seed=seed)
    diag = d.diagnosands
    sd_y = sd_outcome(d114)
    keyed = {
        (str(r["inquiry"]), str(r["estimator"])): r
        for r in diag.to_dict("records")
    }
    grp_by_cell = {
        (str(k[0]), str(k[1])): g
        for k, g in d.simulations.groupby(["inquiry", "estimator"], sort=False)
    }
    for row in refs:
        key = (str(row["inquiry"]), str(row["estimator"]))
        m = keyed[key]
        ses = mc_ses(grp_by_cell[key], row)
        el = f"declaration_11.4 {row['inquiry']} {row['estimator']}"
        # mean_estimand is deterministic (declare_inquiry(data = NULL)) — exact.
        check(el, "mean_estimand", m["mean_estimand"], row["mean_estimand"], 1e-6)
        check(el, "mean_estimate", m["mean_estimate"], row["mean_estimate"], 0.02 * sd_y, mc_se=ses.get("mean_estimate"))
        check(el, "bias", m["bias"], row["bias"], 0.02 * sd_y, mc_se=ses.get("bias"))
        check(el, "sd_estimate", m["sd_estimate"], row["sd_estimate"], 0.10, "rel", mc_se=ses.get("sd_estimate"))
        check(el, "rmse", m["rmse"], row["rmse"], 0.10, "rel", mc_se=ses.get("rmse"))
        # power/coverage are NA in the reference (no p-values or CIs).
    tick("diagnosis_11.4 (300 rows) done")

    # ---- diagnosis_11.5: OLS vs logit-AME vs probit-AME -------------------
    refs = load_ref("diagnosis_11.5")["diagnosands"]
    d115 = declaration_11_5()
    d = dp.diagnose(d115, sims=sims, seed=seed)
    diag = d.diagnosands
    sd_y = sd_outcome(d115)
    for row in refs:
        m = diag[diag["estimator"] == row["estimator"]].iloc[0]
        grp = d.simulations[d.simulations["estimator"] == row["estimator"]]
        full_row_checks(
            f"declaration_11.5 {row['estimator']}", m, row, sd_y, ses=mc_ses(grp, row)
        )
    tick("diagnosis_11.5 done")

    # ---- margins fixture: AME estimators on fixed data --------------------
    fix = load_ref("rgen_t3_ch10_11_probit_fixture")
    fdf = pd.DataFrame(fix["data"])
    for name, tidy in [
        ("logit", dp.logit_ame("Y ~ Z", fdf)),
        ("probit", _probit_ame("Y ~ Z", fdf)),
    ]:
        ours = tidy[tidy["term"] == "Z"].iloc[0]
        r = fix[name][0]
        el = f"margins fixture {name}"
        check(el, "estimate", ours["estimate"], r["estimate"], 1e-6)
        check(el, "std_error", ours["std_error"], r["std.error"], 1e-4, "rel")
        check(el, "conf_low", ours["conf_low"], r["conf.low"], 1e-4, "rel")
        check(el, "conf_high", ours["conf_high"], r["conf.high"], 1e-4, "rel")
    tick("margins fixture done")

    # ---- report -----------------------------------------------------------
    table = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    fails = table[~table["pass"]]
    if len(fails):
        print("\nFAILING CHECKS:")
        print(fails.to_string(index=False))
    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={sims}, seed={seed})")
    out = HERE / "t3_results_ch10_11.csv"
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
