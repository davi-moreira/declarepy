"""T4 coverage report: which of the book's saved diagnosis objects are
reproduced by a validated declarepy recomputation.

Scans validation/reference/diagnosis_*.json (the 62 exported .rds objects)
and every validation/t*_results_*.csv, and prints/writes a coverage table.
The mapping from diagnosis object to validating harness is maintained here
explicitly (with justifications for anything not covered).

Usage: .venv/bin/python validation/coverage_report.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent

#: diagnosis object -> (validating results CSV(s), element hint or justification)
MAPPING: dict[str, tuple[str, str]] = {
    "diagnosis_2.1": ("t1_results.csv", "declaration_2.x twoarm+blocked b-grid"),
    "diagnosis_4.1": ("t3_results_ch04_05_07.csv", "declaration_4.1 (saved object is sims=100; fresh 2000-sim R reference primary)"),
    "diagnosis_9.1": ("t1_results.csv", "declaration_9.1"),
    "diagnosis_9.2": ("t3_results_ch09.csv", "declaration_9.2"),
    "diagnosis_9.3": ("t3_results_ch09.csv", "declaration_9.3"),
    "diagnosis_9.4": ("t3_results_ch09.csv", "declaration_9.4 sweep"),
    "diagnosis_9.5": ("t3_results_ch09.csv", "declaration_9.5"),
    "diagnosis_9.6": ("t3_results_ch09.csv", "declaration_9.6 design list"),
    "diagnosis_9.7": ("t3_results_ch09.csv", "declaration_9.7 randomization inference"),
    "diagnosis_10.1": ("t1_results.csv", "declaration_10.1 power"),
    "diagnosis_10.2": ("t3_results_ch10_11.csv", "declaration_10.2"),
    "diagnosis_10.3": ("t3_results_ch10_11.csv", "declaration_10.1/10.3a/10.3b"),
    "diagnosis_10.4": ("t3_results_ch10_11.csv", "declaration_10.3a/b multi-outcome"),
    "diagnosis_10.5": ("t3_results_ch10_11.csv", "declaration_10.4"),
    "diagnosis_10a": ("t3_results_ch10_11.csv", "declaration_10a"),
    "diagnosis_11.1": ("t1_results.csv", "declaration_11.1 N-grid"),
    "diagnosis_11.2": ("t3_results_ch10_11.csv", "declaration_11.2 N-grid"),
    "diagnosis_11.3": ("t3_results_ch10_11.csv", "declaration_11.3 N×prob grid"),
    "diagnosis_11.4": ("t3_results_ch10_11.csv", "declaration_11.4 polynomial inquiries"),
    "diagnosis_11.5": ("t3_results_ch10_11.csv", "declaration_11.5 OLS/logit-AME/probit-AME"),
    "diagnosis_12.1": ("t3_results_ch12_13.csv", "declaration_12.1"),
    "diagnosis_12.2": ("t3_results_ch12_13.csv", "declaration_12.1 redesign grid"),
    "diagnosis_13.1": ("t3_results_ch12_13.csv", "declaration_13.2 (book index shift)"),
    "diagnosis_15.1": ("t3_results_ch15.csv", "declaration_15.1"),
    "diagnosis_15.2": ("t3_results_ch15.csv", "declaration_15.2 effort sweep"),
    "diagnosis_15.3": ("t3_results_ch15.csv", "declaration_15.3 (book population seedless; canon-fixed, fresh reference)"),
    "diagnosis_15.4": ("t3_results_ch15.csv", "declaration_15.4/15.5 (book population seedless; canon-fixed, fresh reference)"),
    "diagnosis_15.5": ("t3_results_ch15.csv", "declaration_15.6 index outcomes"),
    "diagnosis_16.1": ("t3_results_ch16.csv", "declaration_16.1 process tracing"),
    "diagnosis_16.2": ("t3_results_ch16.csv", "declaration_16.2 exact matching"),
    "diagnosis_16.3": ("t3_results_ch16.csv", "declaration_16.3 staggered DID"),
    "diagnosis_16.4": ("t3_results_ch16.csv", "declaration_16.3 PO variants"),
    "diagnosis_16.5": ("t3_results_ch16.csv", "declaration_16.5/16.6 RD sweep"),
    "diagnosis_17.1": ("t3_results_ch17.csv", "declaration_17.2"),
    "diagnosis_17.2": ("t3_results_ch17.csv", "declaration_17.3"),
    "diagnosis_17.3": ("t3_results_ch17.csv", "declaration_17.4 sweep"),
    "diagnosis_17.4": ("t3_results_ch17.csv", "declaration_17.5 conjoint AMCEs"),
    "diagnosis_17.5": ("t3_results_ch17.csv", "declaration_17.6 trust game"),
    "diagnosis_18.1": ("t1_results.csv", "declaration_18.1"),
    "diagnosis_18.2": ("t3_results_ch18.csv", "declaration_18.2 N×r_sq sweep"),
    "diagnosis_18.3": ("t3_results_ch18.csv", "declaration_18.3 sweep"),
    "diagnosis_18.4": ("t3_results_ch18.csv", "N=12 permutation exercise; declaration_18.4 validated vs fresh reference"),
    "diagnosis_18.5": ("t3_results_ch18.csv", "declaration_18.5 ICC sweep"),
    "diagnosis_18.6": ("t3_results_ch18.csv", "declaration_18.6 n_x1 sweep"),
    "diagnosis_18.7": ("t3_results_ch18.csv", "declaration_18.7 N sweep"),
    "diagnosis_18.8": ("t3_results_ch18.csv", "declaration_18.7 CATE sweep"),
    "diagnosis_18.9": ("t3_results_ch18.csv", "declaration_18.8 (book index shift)"),
    "diagnosis_18.10_encouragment": ("t3_results_ch18.csv", "declaration_18.9 encouragement"),
    "diagnosis_18.10_placebo": ("t3_results_ch18.csv", "declaration_18.9 placebo"),
    "diagnosis_18.11": ("t3_results_ch18.csv", "declaration_18.10/18.11 stepped wedge"),
    "diagnosis_18.12": ("t3_results_ch18.csv", "declaration_18.12 saturation"),
    "diagnosis_18.13": ("t3_results_ch18.csv", "declaration_18.13 interference"),
    "diagnosis_19.1": ("t3_results_ch19_23.csv", "declaration_19.1"),
    "diagnosis_19.2": ("t3_results_ch19_23.csv", "declaration_19.2"),
    "diagnosis_19.3": ("t3_results_ch19_23.csv", "declaration_19.3"),
    "diagnosis_19.4": ("t3_results_ch19_23.csv", "declaration_19.4"),
    "diagnosis_19a": ("t3_results_ch19_23.csv", "19a"),
    "diagnosis_21a": ("t4_results_ch21.csv", "declaration_21a prob sweep"),
    "diagnosis_21b": ("t4_results_ch21.csv", "declaration_21b"),
    "diagnosis_23.1": ("t3_results_ch19_23.csv", "declaration_23.1"),
    "diagnosis_23a": ("t3_results_ch19_23.csv", "23a"),
    "simulation_10.1": ("", "10-sim tibble of declaration_10.1 (structure only; the design itself is validated in t1)"),
}


def main() -> int:
    refs = sorted(
        Path(p).stem for p in glob.glob(str(HERE / "reference" / "diagnosis_*.json"))
    ) + ["simulation_10.1"]
    rows = []
    for name in refs:
        csv_name, note = MAPPING.get(name, ("", "NOT MAPPED"))
        status = "—"
        n_pass = n_tot = 0
        if csv_name:
            path = HERE / csv_name
            if path.exists():
                t = pd.read_csv(path)
                p = t["pass"].astype(str).str.lower().isin(["true", "1", "1.0"])
                n_pass, n_tot = int(p.sum()), len(t)
                status = "validated" if p.all() else f"{int((~p).sum())} open"
            else:
                status = "csv missing"
        elif name == "simulation_10.1":
            status = "covered-by-design"
        rows.append(
            {"diagnosis_object": name, "harness": csv_name or "n/a",
             "harness_pass": f"{n_pass}/{n_tot}" if n_tot else "", "status": status,
             "note": note}
        )
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)
    print(table.to_string(index=False))
    covered = (table["status"] != "—").sum()
    print(f"\n{covered}/{len(table)} diagnosis objects covered")
    table.to_csv(HERE / "t4_coverage.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
