# SEMANTIC_DIFFERENCES — where R and Python quietly disagree

Traps for the translation (course-inline helpers now, parity package later).
Each entry: the difference, the risk, the course's chosen convention.

## 1. Random number generation is NOT portable

R's `set.seed(464)` + `rnorm()` and NumPy's `default_rng(464).normal()` produce
**different streams** — identical seeds never reproduce the book's exact
numbers. **Convention:** the course validates translations against the book's
*qualitative* results and published summary statistics (direction, magnitude,
diagnosand patterns), never digit-for-digit equality; `VALIDATION_REPORT.md`
records tolerance bands. Parity package: same policy, documented prominently.

## 2. 1-based vs 0-based indexing

R slices are 1-based and inclusive; pandas `.iloc` is 0-based, end-exclusive.
Risk: off-by-one in "first N rows" convenience-sample demos. **Convention:**
always `.head(n)` / `.iloc[:n]` with a printed shape check.

## 3. `NA` semantics

R propagates `NA` through `mean()` unless `na.rm=TRUE`; pandas **silently
skips** NaN in `.mean()` (`skipna=True` default). The SAME code can hide
missingness in Python that R would surface. **Convention:** every course cell
that aggregates real data prints `df[col].isna().sum()` first (this is also
pedagogy: who's missing?).

## 4. Factors vs strings/categoricals

R models auto-encode factors; statsmodels formulas encode `C(x)` or infer from
dtype — reference categories can differ from R's. **Convention:** explicit
`C(var, Treatment(reference=...))` whenever a categorical enters a model.

## 5. Formula interfaces differ in defaults

`lm_robust` defaults to **HC2** robust SEs; statsmodels `.fit()` defaults to
classical SEs. Naive translation silently changes every uncertainty statement.
**Convention:** course code always passes `cov_type="HC2"` and says why.

## 6. `sample()` semantics

R's `sample(x, n)` vs pandas `.sample(n, random_state=)` vs `rng.choice(...,
replace=False)` — replacement defaults and reproducibility mechanics differ.
**Convention:** seeded `rng`/`random_state=464` on every draw; replacement
always explicit.

## 7. Data-frame mutation styles

dplyr pipelines are copy-on-modify; pandas chained assignment can raise
`SettingWithCopyWarning` or silently no-op. **Convention:** `.assign()` or
explicit `df = df.copy()` before mutation in all course helpers.

## 8. Logical coercion in aggregation

Both `mean(Y > 0)` (R) and `(Y > 0).mean()` (pandas) work, but pandas boolean
columns with NaN become `object` dtype and break `.mean()` in surprising ways.
**Convention:** cast with `.astype(float)` after any comparison on real data.

## 9. ggplot2 grammar vs matplotlib

Figure scripts (~90) are grammar-of-graphics; matplotlib is imperative.
**Convention:** course re-implements the *message* of each figure, not its
aesthetics; parity project may adopt plotnine for closer translation (roadmap
decision, not course scope — plotnine is NOT a course dependency).

## 10. Environments/quosures have no Python analog

DeclareDesign's tidy-eval design steps (bare-name capture) cannot be translated
literally. The parity package will take explicit callables/column names — an
API departure, documented as intentional in `TRANSLATION_ROADMAP.md`.
