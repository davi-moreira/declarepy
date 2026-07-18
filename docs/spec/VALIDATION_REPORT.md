# VALIDATION_REPORT — translation validation protocol + current status

This is the **declarepy repo's** evolving copy (seeded from the HONR 46400
course planning docs). The course-inline ledger lives in the course repo;
rows here validate the **package**.

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
   independently (here: by estimatr in R) to 3 decimals — real data has no
   RNG excuse.
5. Every validated element gets a row below, with the seed, reps, and deltas.
   Any tolerance failure is recorded with its investigation before the
   element may be marked validated.

## Status ledger — Tranche 1 (core engine), validated 2026-07-18

Reference outputs: the book's saved `diagnosis_objects/*.rds`, exported to
`validation/reference/*.json`; fresh estimatr/prop.test references generated
by `validation/r_scripts/estimatr_reference.R` (R 4.6.1, estimatr from CRAN
2026-07). Harness: `validation/validate_t1.py`; per-check deltas in
`validation/t1_results.csv`. Tests: `pytest` (66 tests) pins every number.

### Course-notebook exact parity (seed 464 — method: bit-for-bit)

| Element | Course source | Check | Result |
|---|---|---|---|
| `complete_ra` stream compatibility | nb07/nb09/nb11 inline | same generator state ⇒ identical assignment vector | ✅ identical (tests/test_ra.py) |
| DiM estimate on nb09's world | nb09 hard-coded self-check | est == 2.0372 at 4dp; se == 0.8650 | ✅ exact |
| OLS(HC2) ≡ DiM on Z | nb09 | equality at 4dp | ✅ exact |
| covariate adjustment | nb09 | balance 50.03/50.27; adj est 1.849; SE shrinks | ✅ exact |
| course.diagnose canonical | nb11 (reps=1000, seed=464) | bias −0.022, power 0.152, coverage 0.947 | ✅ exact |
| sick designs X/Y/Z | nb11 | bias>10 & coverage 0.0; power 0.093; coverage 0.685 | ✅ exact |
| equal-cost fixes | nb11 | powers 0.121 / 0.189 / 0.152; fix₁ wins | ✅ exact |
| course.run_design spreads | nb10 | sd 1.9532 (n=100), 1.0205 (n=400), ratio 0.5224 | ✅ exact |
| course.power_at grid | nb10 | 0.153 / 0.284 / 0.516 / 0.157 | ✅ exact |
| foos_etal DiM | nb13 | est +0.034074; interval excludes 0 | ✅ exact (se: Welch vs binomial agree to 4dp) |
| cliningsmith DiM | nb13 | +0.474834, se 0.162672 | ✅ exact to 6dp |
| la_voter_file facts | nb07 | mean age 48.8; NPP ≈ 37.0 | ✅ exact |

### Estimator fidelity vs estimatr (real-data check §4 — method: R cross-run)

| Element | Data | Agreement | Result |
|---|---|---|---|
| `difference_in_means` (Welch + Satterthwaite df) | foos_etal, cliningsmith | ≤1e-6 abs on est/SE/p/CI/df | ✅ machine precision |
| `difference_in_means(blocks=)` (Σw·DiM_b, df=N−2B) | fixed blocked fixture | ≤1e-8 | ✅ machine precision |
| `lm_robust` HC2 / HC1 / HC3 / classical, t(N−k) | cliningsmith, lapop_brazil | ≤1e-8 | ✅ machine precision |
| `lm_robust` intercept-only, n=3 (HC2≡classical, df=2) | fixed tiny sample | ≤1e-6 | ✅ machine precision |
| `prop_test` (Yates χ², Wilson-cc CI) | x=45/n=100; x=3/n=10 | ≤1e-6 | ✅ matches R prop.test |

### Declaration tolerance checks (§1–§3 — sims=2000, seed=464)

97 checks, **97 pass** (`validation/t1_results.csv` holds every delta):

| Element | Reference | Checks | Worst delta vs band | Result |
|---|---|---|---|---|
| declaration_18.1 | diagnosis_18.1.rds | 7 diagnosands | bias Δ+0.0095 vs ±0.0174; power Δ−0.001 vs ±0.05 | ✅ validated |
| declaration_10.1 | diagnosis_10.1.rds | power | Δ+0.005 vs ±0.05 | ✅ validated |
| declaration_9.1 | diagnosis_9.1.rds | 7 diagnosands (lm_robust n=3 intercept) | power Δ−0.006; coverage Δ+0.0015 | ✅ validated |
| declaration_11.1 (N=100..1000) | diagnosis_11.1.rds | 30 (mean/sd/power × 10 N) | power Δ≤0.02; mean_estimate Δ≤0.002 | ✅ validated |
| declaration_2.1 (twoarm, b=0..3) | diagnosis_2.1.rds | 26 (success/failure × 13 b) | ≤0.025 vs ±0.05 | ✅ validated |
| declaration_2.2 (blocked, b=0..3) | diagnosis_2.1.rds | 26 | ≤0.045 vs ±0.05 | ✅ validated |

Note (investigated, not a failure): at sims=500 one blocked-arm check
(b=1.25 success) exceeded the band by 0.001 — pure Monte-Carlo noise
(≈2.6·MC-SE); at the protocol's sims=2000 it passes with margin. Recorded
per protocol rule 5.

### Known-truth checks (§3)

| Element | Check | Result |
|---|---|---|
| declaration_18.1 at sims=3000, alternate seed 7 | bias → 0 (±0.012), coverage → 0.95 (±0.02) | ✅ (tests/test_engine.py) |
| redesign power monotonicity (declaration_11.1) | power strictly rises N=100→400→900 | ✅ |

## Remaining tranches

| Element | Source | Python home | Method | Status |
|---|---|---|---|---|
| PS answer keys 1–4 reproduction; CR2/clustered SEs | exercises/ + estimatr | Tranche 2 | tolerance + real-data | ⬜ next |
| full declaration library (ch.9→23) | replication archive | Tranche 3 | full protocol | ⬜ |
| remaining diagnosis objects (.rds) recomputation | replication archive | Tranche 4 | tolerance | ⬜ (5 of 62 done in T1) |
| priority figures | figure scripts | Tranche 4 | visual/structural | ⬜ |

## Standing tolerances (restated)

- RNG: never compare streams; compare distributions/diagnosands (§2).
- Real-data: exact to 3 decimals or investigate (§4) — T1 achieved ≤1e-6.
- Any tolerance failure is recorded here with its investigation before the
  element may be marked validated.
