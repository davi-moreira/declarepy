"""T3 ch18 acceptance: validate declarations 18.2–18.13 against the book.

Compares declarepy translations of the chapter-18 declarations against the
book's saved diagnosis objects (validation/reference/diagnosis_18.*.json)
and, where the book saved none, against fresh R-generated references
(rgen_t3_ch18_*.json), under the VALIDATION_REPORT.md protocol:

    bias / mean_estimate / mean_estimand   ±0.02 · sd(Y)
    sd_estimate, rmse                      ±10 % relative
    power ±0.05          coverage ±0.03

Monte-Carlo-noise floor for mean-scale checks: both the reference and this
run are sims = 2000 Monte-Carlo estimates, so for high-variance estimators
(2SLS at 10 % compliance, Horvitz–Thompson) the protocol band can be smaller
than the noise of the comparison itself. For mean-scale diagnosands the band
is therefore max(protocol band, 3·√2·se_ref), where se_ref is the
reference's bootstrap SE — a 3σ band on the difference of two equally noisy
estimates (~2·10⁻⁵ false-failure rate per check). Bands are never widened
beyond this documented rule. One saved-reference cell (diagnosis_18.10_placebo
at compliance_rate = 0.3) was itself shown to be a +1.7-SE MC outlier of the
book's sims = 2000 run and is checked against a fresh sims = 10000 R rerun
instead — see validate_18_10_sweeps' docstring for the full investigation.

Helper fixtures: the private estimators in ch18.py (_lm_lin_est weighted and
unweighted, _iv_robust_est, _twfe_est, blocked/clustered difference in
means) are additionally checked digit-for-digit (±1e−6) against estimatr on
the fixed datasets embedded in rgen_t3_ch18_fixtures.json.

Book-index note: diagnosis_18.8 sweeps declaration_18.7; diagnosis_18.9
diagnoses declaration_18.8; diagnosis_18.10_* sweep declarations 18.9b/c;
diagnosis_18.11 covers declarations 18.10 and 18.11.

Usage:  .venv/bin/python validation/validate_t3_ch18.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch18 import (
    _iv_robust_est,
    _lm_lin_est,
    _twfe_est,
    declaration_18_2,
    declaration_18_3,
    declaration_18_4,
    declaration_18_5,
    declaration_18_6,
    declaration_18_7,
    declaration_18_8,
    declaration_18_9_encouragement,
    declaration_18_9_placebo,
    declaration_18_10,
    declaration_18_11,
    declaration_18_12,
    declaration_18_13,
)
from declarepy.estimators import difference_in_means as _dim_fit

HERE = Path(__file__).parent
REF = HERE / "reference"

results: list[dict[str, object]] = []

MEAN_KEYS = ("mean_estimand", "mean_estimate", "bias")
STANDARD_KEYS: list[tuple[str, float, str]] = [
    ("mean_estimand", 0.02, "mean"),
    ("mean_estimate", 0.02, "mean"),
    ("bias", 0.02, "mean"),
    ("sd_estimate", 0.10, "rel"),
    ("rmse", 0.10, "rel"),
    ("power", 0.05, "abs"),
    ("coverage", 0.03, "abs"),
]


def load_ref(name: str) -> list[dict[str, object]]:
    with open(REF / f"{name}.json") as f:
        obj = json.load(f)
    rows: list[dict[str, object]] = obj["diagnosands"] if isinstance(obj, dict) else obj
    return rows


def check(
    element: str,
    diagnosand: str,
    ours: object,
    ref: object,
    tol: float,
    kind: str = "abs",
) -> None:
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        return
    ours_f = float(ours) if ours is not None else float("nan")
    ref_f = float(ref)  # type: ignore[arg-type]
    delta = ours_f - ref_f
    if kind == "rel":
        ok = abs(delta) <= abs(tol * ref_f)
        band = f"±{tol:.0%} rel"
    else:
        ok = abs(delta) <= tol
        band = f"±{tol:.4g}"
    results.append(
        {
            "element": element,
            "diagnosand": diagnosand,
            "python": round(ours_f, 5),
            "reference": round(ref_f, 5),
            "delta": round(delta, 5),
            "band": band,
            "pass": bool(ok),
        }
    )


def check_row(
    element: str,
    mine: "pd.Series[object]",
    ref_row: dict[str, object],
    sd_y: float,
    keys: Optional[list[tuple[str, float, str]]] = None,
) -> None:
    """Standard 7-diagnosand comparison with the documented MC-noise floor."""
    for key, tol, kind in keys or STANDARD_KEYS:
        if key not in mine.index and ref_row.get(key) is None:
            continue
        if kind == "mean":
            se_ref = ref_row.get(f"se({key})") or 0.0
            band = max(tol * sd_y, 3.0 * np.sqrt(2.0) * float(se_ref))  # type: ignore[arg-type]
            check(element, key, mine.get(key), ref_row.get(key), band, "abs")
        else:
            check(element, key, mine.get(key), ref_row.get(key), tol, kind)


def sd_outcome(design: dp.Design, outcome: str = "Y", seed: int = 1) -> float:
    df = dp.draw_data(design, rng=seed)
    return float(df[outcome].std(ddof=1))


def match_ref(
    rows: list[dict[str, object]], **key_values: object
) -> dict[str, object]:
    for row in rows:
        ok = True
        for k, v in key_values.items():
            rv = row.get(k)
            if isinstance(v, float) or isinstance(rv, float):
                ok = ok and rv is not None and np.isclose(float(rv), float(v))  # type: ignore[arg-type]
            else:
                ok = ok and rv == v
            if not ok:
                break
        if ok:
            return row
    raise KeyError(f"no reference row matching {key_values}")


def frame_from_columns(obj: dict[str, list[object]]) -> pd.DataFrame:
    return pd.DataFrame({k: v for k, v in obj.items()})


# --------------------------------------------------------------------------
# helper fixtures (exact estimatr parity)
# --------------------------------------------------------------------------


def validate_fixtures() -> None:
    with open(REF / "rgen_t3_ch18_fixtures.json") as f:
        fx = json.load(f)
    data = frame_from_columns(fx["data"])
    twfe = frame_from_columns(fx["twfe_data"])
    blocked = frame_from_columns(fx["blocked_data"])
    clustered = frame_from_columns(fx["clustered_data"])
    exp = fx["expected"]

    def run_est(step: object, df: pd.DataFrame) -> dict[str, float]:
        res = step.fn(df)  # type: ignore[attr-defined]
        return {
            "estimate": float(res.estimate),
            "std_error": float(res.std_error),
            "df": float(res.df),
            "p_value": float(res.p_value),
            "conf_low": float(res.conf_low),
            "conf_high": float(res.conf_high),
        }

    cases: list[tuple[str, dict[str, float]]] = [
        ("lm_lin", run_est(_lm_lin_est(covariates=("X",)), data)),
        (
            "lm_lin_weighted",
            run_est(_lm_lin_est(covariates=("X",), weights="w"), data),
        ),
        ("iv_robust", run_est(_iv_robust_est(y="Y2", d="D", z="Zi"), data)),
        (
            "twfe_cr2",
            run_est(
                _twfe_est(
                    y="Y", z="Z", fixed_effects=("unit", "period"), clusters="unit"
                ),
                twfe,
            ),
        ),
    ]
    res_b = _dim_fit(blocked, y="Y", z="Z", blocks="b")
    cases.append(
        (
            "dim_blocked",
            {
                "estimate": res_b.estimate, "std_error": res_b.std_error,
                "df": res_b.df, "p_value": res_b.p_value,
                "conf_low": res_b.conf_low, "conf_high": res_b.conf_high,
            },
        )
    )
    cl = clustered.copy()
    cl["Shigh"] = (cl["S"] == "high").astype(int)
    res_c = _dim_fit(cl, y="Y", z="Shigh", clusters="g")
    cases.append(
        (
            "dim_clustered",
            {
                "estimate": res_c.estimate, "std_error": res_c.std_error,
                "df": res_c.df, "p_value": res_c.p_value,
                "conf_low": res_c.conf_low, "conf_high": res_c.conf_high,
            },
        )
    )
    for name, mine in cases:
        want = exp[name]
        for field in ("estimate", "std_error", "df", "p_value", "conf_low", "conf_high"):
            check(f"helper {name}", field, mine[field], want[field][0], 1e-6, "abs")


# --------------------------------------------------------------------------
# per-declaration validation
# --------------------------------------------------------------------------


def validate_18_2(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.2")
    n_grid = list(range(100, 1101, 200))
    r_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    grid = dp.redesign(declaration_18_2, N=n_grid, r_sq=r_grid)
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    for n in n_grid:
        for r in r_grid:
            mine = diag[
                (diag["N"] == n) & np.isclose(diag["r_sq"].astype(float), r)
            ].iloc[0]
            sd_y = sd_outcome(declaration_18_2(N=n, r_sq=r))
            row = match_ref(ref, N=n, r_sq=r)
            check_row(f"18.2 N={n} r_sq={r}", mine, row, sd_y)


def validate_18_3(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.3")
    slope_grid = [-1.0, -0.5, 0.0, 0.5, 1.0]
    prob_grid = [round(p, 1) for p in np.arange(0.1, 0.91, 0.1)]
    grid = dp.redesign(declaration_18_3, control_slope=slope_grid, prob=prob_grid)
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    for cs in slope_grid:
        sd_y = sd_outcome(declaration_18_3(prob=0.5, control_slope=cs))
        for p in prob_grid:
            sub = diag[
                np.isclose(diag["control_slope"].astype(float), cs)
                & np.isclose(diag["prob"].astype(float), p)
            ]
            for est in ("DIM", "OLS", "Lin"):
                mine = sub[sub["estimator"] == est].iloc[0]
                row = match_ref(ref, control_slope=cs, prob=p, estimator=est)
                check_row(f"18.3 cs={cs} p={p} {est}", mine, row, sd_y)


def validate_18_4(sims: int, seed: int) -> None:
    ref = load_ref("rgen_t3_ch18_18.4")[0]
    design = declaration_18_4()
    sd_y = sd_outcome(design)
    diag = dp.diagnose(design, sims=sims, seed=seed).diagnosands.iloc[0]
    keys: list[tuple[str, float, str]] = [
        ("mean_estimate", 0.02, "mean"),
        ("sd_estimate", 0.10, "rel"),
        ("power", 0.05, "abs"),
    ]
    check_row("18.4 Lin ipw", diag, ref, sd_y, keys=keys)


def validate_18_5(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.5")
    icc_grid = [0.1, 0.5, 0.9]
    grid = dp.redesign(declaration_18_5, ICC=icc_grid)
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    for icc in icc_grid:
        mine = diag[np.isclose(diag["ICC"].astype(float), icc)].iloc[0]
        sd_y = sd_outcome(declaration_18_5(ICC=icc))
        check_row(f"18.5 ICC={icc}", mine, match_ref(ref, ICC=icc), sd_y)


def validate_18_6(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.6")
    n_grid = list(range(20, 981, 96))
    grid = dp.redesign(declaration_18_6, n_x1=n_grid)
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    # The three inquiries are constants of the bundled fixed population.
    run = dp.run_design(declaration_18_6(), rng=seed)
    sd_y = sd_outcome(declaration_18_6())
    for n in n_grid:
        for inquiry in ("CATE_X0", "CATE_X1"):
            row = match_ref(ref, n_x1=n, inquiry=inquiry)
            check(
                f"18.6 n_x1={n} {inquiry}", "mean_estimand",
                run.estimands[inquiry], row.get("mean_estimand"), 0.02 * sd_y,
            )
        mine = diag[diag["n_x1"] == n].iloc[0]
        row = match_ref(ref, n_x1=n, inquiry="diff_in_CATEs", estimator="estimator")
        check_row(f"18.6 n_x1={n} diff_in_CATEs", mine, row, sd_y)


def validate_18_7(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.7")
    n_grid = list(range(500, 3001, 500))
    grid = dp.redesign(declaration_18_7, N=n_grid)
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    pairs = sorted({(str(r["estimator"]), str(r["inquiry"])) for r in ref})
    for n in n_grid:
        sd_y = sd_outcome(declaration_18_7(N=n))
        sub = diag[diag["N"] == n]
        for est, inquiry in pairs:
            mine = sub[(sub["estimator"] == est) & (sub["inquiry"] == inquiry)].iloc[0]
            row = match_ref(ref, N=n, estimator=est, inquiry=inquiry)
            check_row(f"18.7 N={n} est={est} {inquiry}", mine, row, sd_y)


def validate_18_8_sweep(sims: int, seed: int) -> None:
    """diagnosis_18.8: declaration_18.7 swept over CATE_Z1_Z2_0."""
    ref = load_ref("diagnosis_18.8")
    cate_grid = [round(c, 2) for c in np.arange(0.0, 0.51, 0.05)]
    grid = dp.redesign(
        declaration_18_7, CATE_Z1_Z2_0=cate_grid, CATE_Z2_Z1_0=0.2, interaction=0.0
    )
    diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
    pairs = sorted({(str(r["estimator"]), str(r["inquiry"])) for r in ref})
    sd_y = sd_outcome(
        declaration_18_7(CATE_Z1_Z2_0=0.25, CATE_Z2_Z1_0=0.2, interaction=0.0)
    )
    for c in cate_grid:
        sub = diag[np.isclose(diag["CATE_Z1_Z2_0"].astype(float), c)]
        for est, inquiry in pairs:
            mine = sub[(sub["estimator"] == est) & (sub["inquiry"] == inquiry)].iloc[0]
            row = match_ref(ref, CATE_Z1_Z2_0=c, estimator=est, inquiry=inquiry)
            check_row(f"18.8 c1={c} est={est} {inquiry}", mine, row, sd_y)


def validate_18_9(sims: int, seed: int) -> None:
    """diagnosis_18.9: declaration_18.8 (compliance types + 3 estimators)."""
    ref = load_ref("diagnosis_18.9")
    design = declaration_18_8()
    sd_y = sd_outcome(design)
    diag = dp.diagnose(design, sims=sims, seed=seed).diagnosands
    for row in ref:
        est, inquiry = str(row["estimator"]), str(row["inquiry"])
        mine = diag[(diag["estimator"] == est) & (diag["inquiry"] == inquiry)].iloc[0]
        check_row(f"18.9 {est} {inquiry}", mine, row, sd_y)


def validate_18_10_sweeps(sims: int, seed: int) -> None:
    """diagnosis_18.10_*: declarations 18.9b/c over compliance_rate.

    Documented reference substitution (NOT a band widening): the saved
    diagnosis_18.10_placebo cell at compliance_rate = 0.3 is an MC-noise
    outlier of the book's own sims = 2000 run — its mean_estimate 0.51006
    (bootstrap SE 0.00582) sits +1.7 of its own SEs above the exact truth
    0.5 (the CACE is 0.5 by construction and complier-subset OLS under
    complete RA is unbiased). Investigation (2026-07-18): at sims = 10000
    declarepy gives 0.49743 (seed 464) / 0.50285 (seed 2026) and a fresh R
    rerun of the same cell gives 0.50434 — Python vs fresh-R delta 0.0069
    (1.9 combined MC-SEs), so the original 3.1-sigma failure was two-sided
    Monte-Carlo noise, not a translation bug. This one cell is therefore
    checked against the fresh sims = 10000 R reference
    (rgen_t3_ch18_18.10_placebo_c03.json, generated by section 5 of
    r_scripts/t3_ch18_reference.R); the protocol bands are unchanged.
    """
    c_grid = [round(c, 1) for c in np.arange(0.1, 0.91, 0.1)]
    for suffix, factory in (
        ("encouragment", declaration_18_9_encouragement),
        ("placebo", declaration_18_9_placebo),
    ):
        ref = load_ref(f"diagnosis_18.10_{suffix}")
        grid = dp.redesign(factory, compliance_rate=c_grid)
        diag = dp.diagnose_all(grid, sims=sims, seed=seed).diagnosands
        for c in c_grid:
            sd_y = sd_outcome(factory(compliance_rate=c))
            mine = diag[np.isclose(diag["compliance_rate"].astype(float), c)].iloc[0]
            element = f"18.10 {suffix} c={c}"
            if suffix == "placebo" and np.isclose(c, 0.3):
                # See docstring: saved sims=2000 cell is a +1.7-SE MC
                # outlier; use the fresh sims=10000 R reference instead.
                row = match_ref(
                    load_ref("rgen_t3_ch18_18.10_placebo_c03"),
                    compliance_rate=c,
                )
                element += " [ref: fresh R sims=10000 noise-check]"
            else:
                row = match_ref(ref, compliance_rate=c)
            check_row(element, mine, row, sd_y)


def validate_18_11(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.11")
    e_grid = [round(e, 2) for e in np.arange(0.0, 0.751, 0.05)]
    sw = dp.diagnose_all(
        dp.redesign(declaration_18_10, effect_size=e_grid), sims=sims, seed=seed
    ).diagnosands
    s100 = dp.diagnose_all(
        dp.redesign(declaration_18_11, n_units=100, effect_size=e_grid),
        sims=sims, seed=seed,
    ).diagnosands
    s200 = dp.diagnose_all(
        dp.redesign(declaration_18_11, n_units=200, effect_size=e_grid),
        sims=sims, seed=seed,
    ).diagnosands
    sd_sw = sd_outcome(declaration_18_10())
    sd_100 = sd_outcome(declaration_18_11(n_units=100))
    sd_200 = sd_outcome(declaration_18_11(n_units=200))
    for e in e_grid:
        mine = sw[np.isclose(sw["effect_size"].astype(float), e)].iloc[0]
        row = match_ref(ref, estimator="TWFE", n_units=100, effect_size=e)
        check_row(f"18.11 TWFE e={e}", mine, row, sd_sw)
        for tab, n_units, sd_y in ((s100, 100, sd_100), (s200, 200, sd_200)):
            sub = tab[
                (tab["n_units"] == n_units)
                & np.isclose(tab["effect_size"].astype(float), e)
            ].iloc[0]
            row = match_ref(ref, estimator="DIM", n_units=n_units, effect_size=e)
            check_row(f"18.11 DIM n={n_units} e={e}", sub, row, sd_y)


def validate_18_12(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.12")
    design = declaration_18_12()
    sd_y = sd_outcome(design)
    diag = dp.diagnose(design, sims=sims, seed=seed).diagnosands
    for row in ref:
        est = str(row["estimator"])
        mine = diag[diag["estimator"] == est].iloc[0]
        check_row(f"18.12 {est}", mine, row, sd_y)


def validate_18_13(sims: int, seed: int) -> None:
    ref = load_ref("diagnosis_18.13")
    design = declaration_18_13()
    sd_y = sd_outcome(design)
    diag = dp.diagnose(design, sims=sims, seed=seed).diagnosands
    for row in ref:
        est, inquiry = str(row["estimator"]), str(row["inquiry"])
        mine = diag[(diag["estimator"] == est) & (diag["inquiry"] == inquiry)].iloc[0]
        check_row(f"18.13 {est} {inquiry}", mine, row, sd_y)


# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    ap.add_argument(
        "--only", type=str, default=None,
        help="comma-separated subset, e.g. '18.2,18.13,fixtures'",
    )
    args = ap.parse_args()
    sims, seed = args.sims, args.seed

    stages: list[tuple[str, Callable[[], None]]] = [
        ("fixtures", validate_fixtures),
        ("18.2", lambda: validate_18_2(sims, seed)),
        ("18.3", lambda: validate_18_3(sims, seed)),
        ("18.4", lambda: validate_18_4(sims, seed)),
        ("18.5", lambda: validate_18_5(sims, seed)),
        ("18.6", lambda: validate_18_6(sims, seed)),
        ("18.7", lambda: validate_18_7(sims, seed)),
        ("18.8", lambda: validate_18_8_sweep(sims, seed)),
        ("18.9", lambda: validate_18_9(sims, seed)),
        ("18.10", lambda: validate_18_10_sweeps(sims, seed)),
        ("18.11", lambda: validate_18_11(sims, seed)),
        ("18.12", lambda: validate_18_12(sims, seed)),
        ("18.13", lambda: validate_18_13(sims, seed)),
    ]
    wanted = set(args.only.split(",")) if args.only else None
    for name, fn in stages:
        if wanted is not None and name not in wanted:
            continue
        t0 = time.time()
        fn()
        n_fail = sum(
            1
            for r in results
            if not r["pass"] and str(r["element"]).startswith(name.replace("fixtures", "helper"))
        )
        print(f"[{name}] done in {time.time() - t0:.1f}s ({n_fail} failures so far in stage)")

    table = pd.DataFrame(results)
    out = HERE / "t3_results_ch18.csv"
    if wanted is not None and out.exists():
        # Partial (--only) rerun: splice this run's stage rows into the
        # existing full CSV in place of the old rows for those stages.
        # Stages are deterministic (explicit sims/seed per diagnose call),
        # so the merged file equals a full rerun under the current harness.
        prefixes = [w.replace("fixtures", "helper") for w in sorted(wanted)]

        def stage_of(element: object) -> Optional[str]:
            for p in prefixes:
                if str(element).startswith(p):
                    return p
            return None

        old = pd.read_csv(out)
        merged: list[dict[str, object]] = []
        spliced: set[str] = set()
        for rec in old.to_dict("records"):
            p = stage_of(rec["element"])
            if p is None:
                merged.append(rec)
            elif p not in spliced:
                merged.extend(
                    r for r in results if stage_of(r["element"]) == p
                )
                spliced.add(p)
        for p in prefixes:  # stages absent from the old CSV go at the end
            if p not in spliced:
                merged.extend(
                    r for r in results if stage_of(r["element"]) == p
                )
        table = pd.DataFrame(merged)
    pd.set_option("display.width", 200)
    fails = table[~table["pass"].astype(bool)]
    if len(fails):
        print("\nFAILING CHECKS:")
        print(fails.to_string(index=False))
    n_fail = int((~table["pass"].astype(bool)).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={sims}, seed={seed})")
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
