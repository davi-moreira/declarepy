"""T3 acceptance: validate declarations 4.1, 5.1 and 7.1.

Compares declarepy translations against R reference diagnoses under the
VALIDATION_REPORT.md protocol:

    bias / mean_estimate / mean_estimand  ±0.02 · sd(Y)
    sd_estimate, rmse                     ±10% relative
    power ±0.05        coverage ±0.03

References
----------
* ``rgen_t3_ch04_05_07_{4.1,5.1,7.1}.json`` — fresh 2000-sim R diagnoses of
  the UNMODIFIED book declarations (validation/r_scripts/
  t3_ch04_05_07_reference.R). These are the primary references, checked at
  strict protocol bands. 5.1 and 7.1 have no saved diagnosis object in the
  book archive; 7.1 (inquiries, no estimator) uses estimand diagnosands
  (mean_estimand, sd_estimand per inquiry).
* ``diagnosis_4.1.json`` — the book's saved diagnosis, which was built with
  only sims = 100 (see diagnoses/diagnosis_4.1.R). Its own Monte-Carlo
  error exceeds the protocol bands (bootstrap se(mean_estimate) = 0.031 >
  the ±0.02·sd(Y) = ±0.021 band), so strict-band comparison against it is
  statistically miscalibrated. It is checked SECONDARILY with the
  reference's reported bootstrap se added to the band (±(tol + 2·se_ref) —
  documented, not silent widening; the strict rgen check above is the
  binding one).

Usage:  .venv/bin/python validation/validate_t3_ch04_05_07.py [--sims 2000] [--seed 464]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch04 import declaration_4_1
from declarepy.library.ch05 import declaration_5_1
from declarepy.library.ch07 import _diagnose_estimands, declaration_7_1

HERE = Path(__file__).parent
REF = HERE / "reference"

results: list[dict[str, object]] = []


def load_ref(name: str) -> list[dict]:
    """Load a reference JSON: either a bare row array or {'diagnosands': [...]}."""
    with open(REF / f"{name}.json") as f:
        obj = json.load(f)
    rows = obj["diagnosands"] if isinstance(obj, dict) else obj
    return list(rows)


def check(
    element: str,
    diagnosand: str,
    ours: float,
    ref: Optional[float],
    tol: float,
    kind: str = "abs",
    se_ref: float = 0.0,
) -> None:
    """One tolerance check; ``se_ref > 0`` adds the reference's own MC error
    (2·se) to the band — used only for the saved 100-sim diagnosis_4.1."""
    if ref is None or (isinstance(ref, float) and np.isnan(ref)):
        return
    delta = ours - ref
    if kind == "rel":
        limit = abs(tol * ref) + 2.0 * se_ref
        band = f"±{tol:.0%} rel" + (f"+2·{se_ref:.4g}" if se_ref else "")
    else:
        limit = tol + 2.0 * se_ref
        band = f"±{tol:.4g}" + (f"+2·{se_ref:.4g}" if se_ref else "")
    results.append(
        {
            "element": element,
            "diagnosand": diagnosand,
            "python": round(float(ours), 5),
            "reference": round(float(ref), 5),
            "delta": round(float(delta), 5),
            "band": band,
            "pass": abs(delta) <= limit,
        }
    )


def sd_outcome(design: dp.Design, outcome: str = "Y", draws: int = 30) -> float:
    """sd of the outcome pooled over several draws (stable for small-N designs)."""
    cols = [dp.draw_data(design, rng=s)[outcome].to_numpy() for s in range(1, draws + 1)]
    return float(np.std(np.concatenate(cols), ddof=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    args = ap.parse_args()
    sims, seed = args.sims, args.seed

    protocol = [
        ("mean_estimand", "sdY", "abs"),
        ("mean_estimate", "sdY", "abs"),
        ("bias", "sdY", "abs"),
        ("sd_estimate", 0.10, "rel"),
        ("rmse", 0.10, "rel"),
        ("power", 0.05, "abs"),
        ("coverage", 0.03, "abs"),
    ]

    # ---- declaration_4.1: DiM on the PATE after sampling ---------------
    d4 = declaration_4_1()
    sdY4 = sd_outcome(d4)
    diag4 = dp.diagnose(d4, sims=sims, seed=seed).diagnosands.iloc[0]
    # Primary: fresh 2000-sim R reference, strict protocol bands.
    ref4 = load_ref("rgen_t3_ch04_05_07_4.1")[0]
    for key, tol, kind in protocol:
        t = 0.02 * sdY4 if tol == "sdY" else float(tol)
        check("declaration_4.1 [vs rgen sims=2000]", key, diag4[key], ref4.get(key), t, kind)
    # Secondary: the book's saved diagnosis (sims = 100) — its bootstrap
    # se(diagnosand) is added to the band (2·se), since at 100 sims the
    # reference's own MC error exceeds the protocol band.
    ref4s = load_ref("diagnosis_4.1")[0]
    for key, tol, kind in protocol:
        t = 0.02 * sdY4 if tol == "sdY" else float(tol)
        check(
            "declaration_4.1 [vs saved sims=100]",
            key,
            diag4[key],
            ref4s.get(key),
            t,
            kind,
            se_ref=float(ref4s.get(f"se({key})", 0.0)),
        )

    # ---- declaration_5.1: subgroup mean via lm_robust(Y ~ 1, subset) ---
    d5 = declaration_5_1()
    sdY5 = sd_outcome(d5)
    diag5 = dp.diagnose(d5, sims=sims, seed=seed).diagnosands.iloc[0]
    ref5 = load_ref("rgen_t3_ch04_05_07_5.1")[0]
    for key, tol, kind in protocol:
        t = 0.02 * sdY5 if tol == "sdY" else float(tol)
        check("declaration_5.1 [vs rgen sims=2000]", key, diag5[key], ref5.get(key), t, kind)

    # ---- declaration_7.1: estimand-only design (no estimator) ----------
    # sd(Y) is exactly 1 by construction (Y = 1 + U, U ~ N(0,1)); estimating
    # it from a drawn population of N=20 would itself be noisy.
    d7 = declaration_7_1()
    sdY7 = 1.0
    diag7 = _diagnose_estimands(d7, sims=sims, seed=seed).set_index("inquiry")
    for row in load_ref("rgen_t3_ch04_05_07_7.1"):
        inq = str(row["inquiry"])
        mine = diag7.loc[inq]
        el = f"declaration_7.1 {inq} [vs rgen sims=2000]"
        if inq == "superpopulation_mean":
            # Deterministic inquiry: exactly 1 with zero spread on both sides.
            check(el, "mean_estimand", mine["mean_estimand"], row["mean_estimand"], 1e-9)
            check(el, "sd_estimand", mine["sd_estimand"], row["sd_estimand"], 1e-9)
        else:
            check(el, "mean_estimand", mine["mean_estimand"], row["mean_estimand"], 0.02 * sdY7)
            check(el, "sd_estimand", mine["sd_estimand"], row["sd_estimand"], 0.10, "rel")

    # ---- report --------------------------------------------------------
    table = pd.DataFrame(results)
    pd.set_option("display.width", 160)
    print(table.to_string(index=False))
    n_fail = int((~table["pass"]).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={sims}, seed={seed})")
    out = HERE / "t3_results_ch04_05_07.csv"
    table.to_csv(out, index=False)
    print(f"written {out}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
