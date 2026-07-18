"""T4: validate the ch21 standalone designs vs diagnosis_21a/21b.rds.

Usage: .venv/bin/python validation/validate_t4_ch21.py [--sims 2000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import declarepy as dp
from declarepy.library.ch21 import declaration_21a, declaration_21b

HERE = Path(__file__).parent
results: list[dict[str, object]] = []


def check(element: str, name: str, ours: float, ref: object, tol: float, kind: str = "abs") -> None:
    if ref is None:
        return
    delta = float(ours) - float(ref)
    ok = abs(delta) <= (abs(tol * float(ref)) if kind == "rel" else tol)
    results.append(
        {"element": element, "diagnosand": name, "python": round(float(ours), 5),
         "reference": round(float(ref), 5), "delta": round(delta, 5),
         "band": f"±{tol:.0%} rel" if kind == "rel" else f"±{tol:.4g}", "pass": ok}
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=464)
    args = ap.parse_args()

    with open(HERE / "reference" / "diagnosis_21a.json") as f:
        ref_a = json.load(f)["diagnosands"]
    probs = [round(x, 6) for x in np.linspace(1 / 12, 11 / 12, 11)]
    grid = dp.redesign(declaration_21a, prob=probs)
    diag = dp.diagnose_all(grid, sims=args.sims, seed=args.seed).diagnosands
    for row in ref_a:
        p = float(row["prob"])
        mine = diag[np.isclose(diag["prob"].astype(float), p)].iloc[0]
        check(f"declaration_21a prob={p:.3f}", "mean_estimate", mine["mean_estimate"], row["mean_estimate"], 0.005)
        check(f"declaration_21a prob={p:.3f}", "sd_estimate", mine["sd_estimate"], row["sd_estimate"], 0.10, "rel")
        check(f"declaration_21a prob={p:.3f}", "power", mine["power"], row["power"], 0.05)

    with open(HERE / "reference" / "diagnosis_21b.json") as f:
        ref_b = json.load(f)["diagnosands"][0]
    diag_b = dp.diagnose(declaration_21b(), sims=args.sims, seed=args.seed).diagnosands.iloc[0]
    sd_y = float(dp.draw_data(declaration_21b(), rng=1)["Y"].std(ddof=1))
    for key, tol, kind in [
        ("mean_estimand", 0.02 * sd_y, "abs"), ("mean_estimate", 0.02 * sd_y, "abs"),
        ("bias", 0.02 * sd_y, "abs"), ("sd_estimate", 0.10, "rel"), ("rmse", 0.10, "rel"),
        ("power", 0.05, "abs"), ("coverage", 0.03, "abs"),
    ]:
        check("declaration_21b", key, diag_b[key], ref_b.get(key), tol, kind)

    table = pd.DataFrame(results)
    print(table.to_string(index=False))
    n_fail = int((~table["pass"].astype(bool)).sum())
    print(f"\n{len(table)} checks, {n_fail} failures (sims={args.sims})")
    table.to_csv(HERE / "t4_results_ch21.csv", index=False)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
