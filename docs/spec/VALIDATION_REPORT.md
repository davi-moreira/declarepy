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

## Status ledger — Tranche 2 (estimator fidelity), validated 2026-07-18

References: fresh R runs of the problem-set answer keys' own code
(`validation/r_scripts/ps_reference.R` → `rgen_ps1_4.json`, sims per key),
estimatr clustered fixtures (`cluster_reference.R` → `rgen_cluster.json`),
glm/margins on foos (`logit_reference.R` → `rgen_logit.json`). Harness:
`validation/validate_t2.py` → `validation/t2_results.csv`.

### Cluster-robust fidelity vs estimatr (real-data-style §4 — fixed fixtures)

| Element | Agreement | Result |
|---|---|---|
| `lm_robust(clusters=)` CR2 (Pustejovsky–Tipton) + Bell–McCaffrey Satterthwaite df | ≤5e-13 on est/SE/t/p/CI/df | ✅ machine precision |
| `lm_robust(clusters=)` CR0 and stata/CR1 (G−1 df) | ≤5e-13 | ✅ machine precision |
| CR2 with within-cluster treatment variation | ≤4e-13 | ✅ machine precision |
| clustered `difference_in_means` (delegates to CR2) | ≤5e-13 | ✅ machine precision |
| `glm_logit` MLE coefficients / Wald z (R glm binomial) | ≤2e-8 | ✅ |
| `glm_logit` profile-likelihood CIs (R confint.glm) | ≤1e-7 on foos | ✅ |
| `logit_ame` (margins-package AME, delta-method SE) | est ≤1e-11, SE ≤2e-8 | ✅ |
| `lm_robust` `.attrs` r_squared (broom::glance) | ≤1e-13 | ✅ |

### Problem-set answer keys 1–4 (§2 — sims=2000 py vs the keys' R runs)

**55 checks, 55 pass** (`validation/t2_results.csv`):

| Element | Checks | Key decisions reproduced | Result |
|---|---|---|---|
| PS1 two-arm power analysis | 17 | power ≈ 0.075; MDE = 0.6; N = 400 at ees 0.3 | ✅ validated |
| PS2 covariate adjustment (lapop resample) | 17 | estimand ≈ analytic 0.35·P(trust<7); R² ≈ 0.249; powers 0.737/0.583; same 80 %-power grid Ns | ✅ validated |
| PS3 interaction power | 7 | power 0.076 at N=100; 80 % crossing at N≈3000 (knife-edge documented) | ✅ validated |
| PS4 logit vs OLS vs AME | 13 | OLS & AME unbiased, logit coef biased (log-odds scale band ±0.10 documented) | ✅ validated |

Investigation log (protocol rule 5): the initial PS4 run failed logit-row
coverage by +0.0355 — systematic, not MC noise: the reference coverage uses
R's **profile-likelihood** CIs (broom::tidy conf.int=TRUE) while declarepy
used Wald. Fixed by implementing profile CIs in `glm_logit` (verified vs
`confint.glm` to 1e-7); coverage delta fell to +0.021, within ±0.03. The
PS3 "smallest N with 80 % power" decision is a knife-edge (reference power
at N=3000 is 0.802); the harness accepts 3000/5000 as the same substantive
answer, documented inline.

## Status ledger — Tranche 3 (declaration library), validated 2026-07-18

The full replication-archive declaration set (chs. 2–23, ~66 designs) is
translated in `src/declarepy/library/ch*.py`, one parameterized factory per
declaration with a provenance docstring. Validation: per-chapter harnesses
(`validation/validate_t3_*.py`), every check recorded in
`validation/t3_results_*.csv`; references are the book's saved diagnosis
objects, with fresh 2000-sim R references generated
(`validation/r_scripts/t3_*_reference.R`) where the archive has none or
where the saved object is itself too noisy/seedless (each case documented).

**6,828 tranche-3 checks, all passing** (sims=2000, seed=464 unless noted):

| Chapter group | Checks | Highlights / documented notes |
|---|---|---|
| ch04+05+07 | 27/27 | saved diagnosis_4.1 is sims=100 (its own MC error exceeds protocol bands) — fresh 2000-sim R reference is primary, documented; 7.1 is estimand-only (private estimand-diagnosis helper) |
| ch09 | 168/168 | lh_robust linear-hypothesis test (≤1e-12); rstanarm log-gaussian posterior by deterministic quadrature (rstanarm not installable locally — semantics reconstructed, documented); weighted HC2 + weighted-clustered CR2/BM transcribed from estimatr internals (1e-12); randomization inference on foos (observed estimate ≤1e-8) |
| ch10+11 | 2234/2234 | random-ATE designs (per-condition uniform draws); 11.4's fifty polynomial inquiries; 11.5 OLS/logit-AME/probit-AME (probit fixture vs margins); initial failures traced to a torn mid-edit harness state — full re-run on final code is clean |
| ch12+13 | 93/93 | three-arm clustered survey experiment (cluster_rs/strata_rs/cluster_ra prob_each incl. fractional remainders); cost diagnosand exact; book index shift (diagnosis_13.1 ↔ declaration_13.2) documented |
| ch15 | 223/223 | glmer Laplace GLMM (≈2e-5 vs lme4); post-stratification (7e-15); princomp fix_sign scores (9e-16, incl. the book's Y_2-typo kept faithfully); two book-seedless populations canon-fixed at seed 464 with fresh references (documented) |
| ch16 | 391/391 | process-tracing posteriors vs rdss (1e-9); MatchIt exact matching (3e-15); TWFE (3e-14) + DIDmultiplegt 0.1.0 exact; iv_robust (1e-14); full rdrobust 4.0.0 sharp-RD port (8.5e-13); 69 group-conditional/heavy-tailed cells use documented max(protocol, 3√2·se_ref) bands |
| ch17 | 379/379 | the book's own label-case bug in 17.1 preserved and reproduced; conjoint AMCEs (cjoint cluster-CR1) validated vs the saved diagnosis; trust-game behavioral functions exact (±1e-9) |
| ch18 | 2804/2804 | lm_lin (fixture-exact), iv_robust, absorbed-FE CR2, stepped wedge, saturation, Aronow–Samii interference (re-implemented from published sources; queen adjacency validated end-to-end); 18.10-placebo c=0.3 investigated across seeds/sims=10000 + fresh R — a two-sided MC coincidence; that cell now validated against a fresh sims=10000 R reference under tighter bands |
| ch19+23 | 509/509 | metafor rma REML port (FE exact to 1e-14; REML to its own convergence threshold, documented ±1e-5); **real bug found & fixed**: causal-forest emulation leaf-size miscalibration (systematic 10σ targeting shift) recalibrated against diagnosis_19.1; 19a re-adjudicated NA-aware vs a fresh 500-sim R reference with a documented binomial-SE floor |

## Status ledger — Tranche 4 (diagnosis objects + figures), validated 2026-07-18

* **Diagnosis-object coverage: 62/62** of the book's saved `.rds` objects
  have a validated declarepy recomputation (or a documented structural
  cover) — see `validation/coverage_report.py` → `validation/t4_coverage.csv`
  for the object-by-object mapping. The ch21 standalone designs (no
  declaration files; declared inside their diagnosis scripts) validate
  40/40 (`validation/validate_t4_ch21.py`).
* **Figures: matplotlib** (tranche decision, per SEMANTIC_DIFFERENCES §9 —
  message over aesthetics; plotnine is not a dependency).
  `declarepy.viz` implements the book's figure idioms (parameter sweeps,
  power curves, sampling distributions, CI caterpillars, grid heatmaps);
  `examples/book_figures.py` reproduces the six course-priority figure
  families (chs. 2/9/10/11/18) with structural checks asserting the plotted
  arrays are the validated diagnosand tables. Smoke/structural tests in
  `tests/test_viz.py`.

## Remaining tranches

| Element | Source | Python home | Method | Status |
|---|---|---|---|---|

## Standing tolerances (restated)

- RNG: never compare streams; compare distributions/diagnosands (§2).
- Real-data: exact to 3 decimals or investigate (§4) — T1 achieved ≤1e-6.
- Any tolerance failure is recorded here with its investigation before the
  element may be marked validated.
