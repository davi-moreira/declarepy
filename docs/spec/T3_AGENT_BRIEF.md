# T3 AGENT BRIEF — translating a chapter of the declaration library

You are translating R declarations from the replication archive of *Research
Design in the Social Sciences* (Blair, Coppock & Humphreys 2023) into
`declarepy`, and validating each against the book's saved diagnosis objects.
Read this whole brief before writing code.

## Paths

- **This repo (work here):** `/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy`
- **Python:** `.venv/bin/python` (package installed editable; `import declarepy as dp`)
- **R sources (READ-ONLY — never modify anything under the course repo):**
  `/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book/replication-materials/code/{declarations,diagnoses}/`
- **Reference diagnosands:** `validation/reference/diagnosis_*.json` (exported
  from the book's `.rds`; fields per estimator/param row). If a declaration
  has no usable reference, generate one with R (below).
- **R:** `/usr/local/bin/Rscript`; packages DeclareDesign/rdss/estimatr/
  randomizr/fabricatr/margins are installed in `Sys.getenv("R_LIBS_USER")` —
  every R script must start with
  `ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))`.

## Hard rules

1. NEVER modify the course repo (the `2026F_...HONR464` tree) — read-only.
2. NEVER run `git commit`/`git push` — the orchestrator commits.
3. Touch ONLY your assigned files: your chapter module(s) under
   `src/declarepy/library/`, your `validation/validate_t3_chXX.py`, your
   `validation/t3_results_chXX.csv`, optional `validation/r_scripts/
   t3_chXX_reference.R` + `validation/reference/rgen_t3_chXX*.json`, and (if
   truly needed) new dataset CSVs under `src/declarepy/data/`. Do NOT edit
   `library/__init__.py`, `data/README.md`, shared modules
   (`steps.py`/`estimators.py`/`diagnose.py`), or other chapters' files.
4. New procedures/estimators a declaration needs (e.g. `cluster_ra`,
   `strata_rs`, `lm_lin`, `iv_robust`, fabricatr-style hierarchical
   builders) → implement as PRIVATE helpers inside your chapter module
   (prefix `_`), validated against R, and list them in your report for later
   promotion to the shared modules.
5. Datasets: if a declaration needs an rdss dataset we don't ship, export it
   from the installed rdss package to `src/declarepy/data/<name>.csv`
   (`write.csv(rdss::<name>, ..., row.names=FALSE)` — rdss is MIT; list it in
   your report so attribution docs get updated). Do not export anything from
   the course repo's private archive.

## declarepy API cheat sheet

```python
import declarepy as dp
import numpy as np, pandas as pd

design = dp.Design(
    dp.Model(n=100, build=lambda n, rng: {"U": rng.normal(size=n)}),
    # declare_model(N) alone -> dp.Model(n=N)  (ID-only frame)
    # declare_model(data=..., handler=resample_data) -> dp.Model.resample(df, n=N)
    # a second declare_model step -> dp.Model(transform=lambda df, rng: {...})
    dp.potential_outcomes(lambda df, z, rng: 0.2*float(z) + df["U"].to_numpy()),
    #   -> adds Y0/Y1 (template="{outcome}{condition}"); fn is evaluated ONCE
    #      PER CONDITION, so randomness inside fn is redrawn per condition —
    #      exactly like DeclareDesign's potential_outcomes.
    dp.Inquiry("ATE", lambda df: float((df["Y1"]-df["Y0"]).mean())),  # position matters
    dp.Sampling.complete(n=150),            # declare_sampling(S=complete_rs(N,n))
    dp.Assignment.complete(prob=0.5),       # complete_ra; .block(blocks="col"), .simple(prob)
    dp.Assignment(lambda df, rng: my_custom_ra(...), name="Z"),   # custom
    dp.reveal_outcomes(),                   # Y = np.where(Z==1, Y1, Y0); k-condition supported
    dp.Measurement(lambda df, rng: {"Ynoisy": ...}),
    dp.Estimator.lm_robust("Y ~ Z", inquiry="ATE"),               # HC2, t(N-k)
    dp.Estimator.lm_robust("Y ~ Z + X", term="Z", label="adj"),
    dp.Estimator.lm_robust("Y ~ Z", clusters="cl"),               # CR2 + BM df
    dp.Estimator.difference_in_means(blocks="b", inquiry="ATE"),
    dp.Estimator.logit("Y ~ Z + X", term="Z"),                    # profile CIs
    dp.Estimator.logit_ame("Y ~ Z + X", term="Z"),
    dp.Estimator(lambda df: my_tidy_row(df), label="custom", inquiry=None),
    #   custom fn returns EstimatorResult / tidy DataFrame / dict with at
    #   least estimate; std_error/p_value/conf_low/conf_high drive diagnosands
)
run  = dp.run_design(design, rng=464)     # .data .estimands .estimates
df   = dp.draw_data(design, rng=464)
diag = dp.diagnose(design, sims=2000, seed=464).diagnosands
grid = dp.redesign(factory, N=[100, 200]) # factory(**params) -> Design
tab  = dp.diagnose_all(grid, sims=2000, seed=464).diagnosands   # + param cols
dgs  = dp.Diagnosands(success=lambda d: float(((d["estimate"]>0.3) & (d["p_value"]<0.05)).mean()))
# raw procedures: dp.complete_ra/block_ra/simple_ra/complete_rs/simple_rs/block_rs(rng=...)
# raw estimators: dp.difference_in_means / dp.lm_robust / dp.glm_logit / dp.logit_ame / dp.prop_test
```

Default diagnosands (DeclareDesign's): mean_estimand, mean_estimate, bias,
sd_estimate, rmse, power (p ≤ 0.05), coverage — grouped by
(estimator, inquiry, outcome, term).

## Translation conventions (from SEMANTIC_DIFFERENCES.md — honor all)

- `declare_estimator(Y ~ Z, ...)` with **no `.method`** = `lm_robust` (HC2).
  DD's default `term` = the first non-intercept coefficient (declarepy's
  `term=None` does this).
- R `sample(0:80, N, replace=TRUE)` → `rng.integers(0, 81, n)`;
  `sample(c(0,1), N, TRUE)` → `rng.integers(0, 2, n)`; `pmin` → `np.minimum`;
  `rbinom(N, 1, p)` → `rng.binomial(1, p, n)`; `if_else` → `np.where`.
- `rnorm(N, mean, sd)` → `rng.normal(mean, sd, n)` (R argument order is
  (n, mean, sd); NumPy's is (loc, scale, size) — do not transpose!).
- fabricatr `fabricate(add_level(...))` hierarchies → build explicitly with
  pandas (`np.repeat` for cluster-level draws), documented in a comment.
- Estimand/PO naming: course convention `Y0`/`Y1` via the default template.
- Every factory takes the parameters the book's diagnosis script sweeps
  (`redesign(...)` arguments) as Python keyword args with the R defaults.
- Never seed global numpy; all randomness flows through the `rng` argument.

## Validation protocol (docs/spec/VALIDATION_REPORT.md — apply per declaration)

1. Structural check: same M/I/D/A steps as the R source.
2. Diagnosand tolerance vs the reference JSON, sims=2000, seed=464:
   bias ±0.02·sd(Y), power ±0.05, coverage ±0.03, RMSE (and sd_estimate)
   ±10 % relative, mean_estimate/mean_estimand ±0.02·sd(Y). For diagnosands
   on other scales (log-odds etc.) choose and DOCUMENT a defensible band.
3. If no reference exists for a declaration: write
   `validation/r_scripts/t3_chXX_reference.R` that sources the book's
   declaration + runs `diagnose_design(..., sims=2000)` and
   `jsonlite::write_json(as.data.frame(get_diagnosands(d)), ...)` →
   `validation/reference/rgen_t3_chXX_<decl>.json`; validate against that.
4. Record EVERY check as a row in your `validation/t3_results_chXX.csv`
   (columns: element,diagnosand,python,reference,delta,band,pass) — follow
   `validation/validate_t1.py`'s structure.
5. A tolerance failure must be investigated (more sims, structural bug,
   semantic difference) and either fixed or documented in your report with
   your investigation. Do not silently widen bands.

## Workflow expected of you

1. Read your chapter's R declaration files AND the matching
   `diagnoses/diagnosis_*.R` scripts (they tell you sims, redesign sweeps,
   custom diagnosands, and which designs the .rds actually contains).
   Inspect the matching `validation/reference/diagnosis_*.json` to see the
   reference table's shape FIRST — it defines what you must reproduce.
2. Write the chapter module: one `declaration_X_Y()` factory per
   declaration, module docstring with provenance, per-factory docstrings.
3. Write + run your `validation/validate_t3_chXX.py` (support `--sims`,
   default 2000, seed 464). Iterate until checks pass or deviations are
   understood and documented.
4. Run `.venv/bin/python -m pytest -q` and `.venv/bin/python -m mypy` —
   your additions must not break either (mypy checks `src/declarepy`; typed
   defs required).
5. Return (as your final message) a JSON report:
   `{"module_files": [...], "declarations": [{"name": "...", "status":
   "validated" | "validated-with-note" | "blocked", "checks": "12/12",
   "worst_delta": "...", "notes": "..."}], "new_helpers": [...],
   "datasets_added": [...], "r_scripts_added": [...]}`.
