# VALIDATION_REPORT — translation validation protocol + current status

## Protocol (applies to every translated design element)

Because R and NumPy RNG streams are incompatible (`SEMANTIC_DIFFERENCES.md`
§1), "validated" means **statistical agreement, not digit equality**:

1. **Structural check** — the Python translation declares the same M/I/D/A
   steps as the R source (same n, assignment scheme, estimand, estimator).
2. **Diagnosand tolerance check** — with ≥1,000 Monte-Carlo replicates,
   translated diagnosands match the book/reference values within:
   bias ±0.02·sd(Y), power ±0.05, coverage ±0.03, RMSE ±10% relative.
3. **Known-truth check** — where the estimand is analytically computable
   (e.g., fixed simulated ATE), the translation recovers it as reps → large.
4. **Real-data check** — analyses of the shipped datasets (e.g., foos_etal
   difference-in-means; lapop_brazil summaries) match values computed
   independently in the validation harness (and, where available, published
   replication outputs) to 3 decimals — real data has no RNG excuse.
5. Every validated element gets a row below, with the seed, reps, and deltas.

## Status ledger

| Element | Source | Python home | Method | Status |
|---|---|---|---|---|
| world/PO simulation pattern | declaration-style model (ch. 6–7 concept) | nb04 inline | known-truth check (§3) | ✅ validated — nb04 self-checks passed in the Phase F execute-all audit (2026-07-18; TRUE_ATE = 2.0 exact) |
| complete random assignment | randomizr `complete_ra` | nb07/nb11 inline | structural + balance check | ✅ validated — nb07 self-checks passed (Phase F audit, 2026-07-18) |
| difference-in-means + SE | estimatr `difference_in_means` | nb09/nb13 inline | real-data check on foos_etal (§4) | ✅ validated — nb13 self-checks passed (Phase F audit; foos_etal DiM +3.4pp independently recomputed) |
| OLS + HC2 | estimatr `lm_robust` | nb09 inline | cross-check vs classical SEs + statsmodels docs | ✅ validated — nb09 self-checks passed (Phase F audit; HC2 vs classical cross-check) |
| Monte-Carlo diagnosis (bias/power/coverage) | `diagnose_design` (ch. 10) | nb10/nb11 inline | tolerance vs analytic power (§2–3) | ✅ validated — nb10/nb11 self-checks passed (Phase F audit; power monotonicity + |bias| < 0.15 asserted) |
| redesign grid | `redesign` (ch. 11) | nb11 inline | monotonicity checks (power ↑ with n) | ✅ validated — nb11 self-checks passed (Phase F audit) |
| power simulation (PS1 concept) | rdss-problem-set-1 | nb10 inline | tolerance vs `statsmodels.stats.power` | ✅ validated — nb10 self-checks passed (Phase F audit) |
| DAG drawing | make_dag_df.R | nb04 inline | structural (edge list) only | ✅ validated — structural check in nb04 (Phase F audit; visual aid, no numbers) |
| full declaration objects (declaration_2.1 … 23.1d) | replication archive | — (Tranche 3) | full protocol | ⬜ not started (parallel project) |
| design-based SE variants (blocked/clustered) | estimatr | — (Tranche 2) | PS answer keys | ⬜ not started |
| diagnosis objects (.rds) recomputation | replication archive | — (Tranche 4) | tolerance | ⬜ not started |

"validated in nbNN self-checks" means the notebook carries executable
assertion cells (the course's self-check convention) implementing the method
listed; they run on every notebook execution, including the Phase F
execute-all audit. Course-inline scope only — the parity package (Tranches
1–5) re-validates everything under this same protocol when built.

## Standing tolerances (restated for the parity project)

- RNG: never compare streams; compare distributions/diagnosands (§2).
- Real-data: exact to 3 decimals or investigate (§4).
- Any tolerance failure is recorded here with its investigation before the
  element may be marked validated.
