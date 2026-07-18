# declarepy for DeclareDesign users

A translation table from the R DeclareDesign ecosystem to `declarepy`. The
philosophy differs in one deliberate way (see
[`spec/SEMANTIC_DIFFERENCES.md`](spec/SEMANTIC_DIFFERENCES.md) §10): there is
**no tidy-eval** — every step takes explicit callables and column names, and
potential outcomes are ordinary columns (`Y0`, `Y1`) you can print.

## The pipeline

| R (DeclareDesign) | Python (declarepy) |
|---|---|
| `declare_model(N = 100, U = rnorm(N))` | `dp.Model(n=100, build=lambda n, rng: {"U": rng.normal(size=n)})` |
| `declare_model(N = 100)` | `dp.Model(n=100)` (ID-only frame) |
| `declare_model(data = df, handler = resample_data)` | `dp.Model.resample(df, n=N)` |
| a second `declare_model(...)` step | `dp.Model(transform=lambda df, rng: {...})` |
| `potential_outcomes(Y ~ 0.2 * Z + U)` | `dp.potential_outcomes(lambda df, z, rng: 0.2*float(z) + df["U"].to_numpy())` |
| `declare_inquiry(ATE = mean(Y_Z_1 - Y_Z_0))` | `dp.Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean()))` |
| `declare_sampling(S = complete_rs(N, n = 150))` | `dp.Sampling.complete(n=150)` |
| `declare_assignment(Z = complete_ra(N, prob = 0.5))` | `dp.Assignment.complete(prob=0.5)` |
| `declare_assignment(Z = block_ra(blocks = b))` | `dp.Assignment.block(blocks="b")` |
| `declare_measurement(Y = reveal_outcomes(Y ~ Z))` | `dp.reveal_outcomes()` |
| `declare_measurement(M = latent + rnorm(N, 0, s))` | `dp.Measurement(lambda df, rng: {"M": df["latent"] + rng.normal(0, s, len(df))})` |
| `declare_estimator(Y ~ Z, inquiry = "ATE")` | `dp.Estimator.lm_robust("Y ~ Z", inquiry="ATE")` (DD's default method is lm_robust) |
| `declare_estimator(Y ~ Z, .method = difference_in_means)` | `dp.Estimator.difference_in_means(inquiry="ATE")` |
| `declare_test(handler = ...)` | `dp.Estimator(fn, label=..., inquiry=None)` |
| `design_1 + design_2` / step composition with `+` | identical: `dp.Design(...) + step`, `step + step` |

## Run, diagnose, redesign

| R | Python |
|---|---|
| `draw_data(design)` | `dp.draw_data(design, rng=464)` |
| `run_design(design)` | `dp.run_design(design, rng=464)` → `.data`, `.estimands`, `.estimates` |
| `diagnose_design(design, sims = 1000)` | `dp.diagnose(design, sims=1000, seed=464)` → `.diagnosands`, `.simulations` |
| `declare_diagnosands(power = mean(p.value <= 0.05))` | `dp.Diagnosands(power=lambda d: float((d["p_value"] <= 0.05).mean()))` |
| `redesign(design, N = c(100, 200))` | `dp.redesign(design_factory, N=[100, 200])` — designs are **factory functions** of their parameters |
| `diagnose_designs(designs, sims = 1000)` | `dp.diagnose_all(grid, sims=1000, seed=464)` |
| bootstrap `se(diagnosand)` columns | `dp.diagnose(..., bootstrap_sims=500)` |

The biggest structural difference: where DeclareDesign's `redesign()` edits
variables captured in step environments, declarepy asks you to write the
design as a plain function of its parameters — `def my_design(N=100, effect=0.2) ->
dp.Design: ...` — and sweeps that. Explicit, and refactor-safe.

## Estimators (estimatr → declarepy)

| R | Python | Fidelity |
|---|---|---|
| `difference_in_means(Y ~ Z, data)` | `dp.difference_in_means(df)` | Welch SE + Satterthwaite df; ≤1e-8 vs estimatr |
| `difference_in_means(..., blocks = b)` | `dp.difference_in_means(df, blocks="b")` | block-size-weighted, df = N − 2B |
| `difference_in_means(..., clusters = cl)` | `dp.difference_in_means(df, clusters="cl")` | CR2 + Bell–McCaffrey df |
| `lm_robust(Y ~ Z + X, data)` | `dp.lm_robust("Y ~ Z + X", df)` | HC2 default, t(N − k); ≤1e-8 |
| `lm_robust(..., se_type = "stata")` | `dp.lm_robust(..., se_type="stata")` | HC1 |
| `lm_robust(..., clusters = cl)` | `dp.lm_robust(..., clusters="cl")` | CR2 (Pustejovsky–Tipton) + Satterthwaite df; ≤5e-13 |
| `glm(..., family = binomial)` + `broom::tidy(conf.int = TRUE)` | `dp.glm_logit(...)` | profile-likelihood CIs, ≤1e-7 vs `confint.glm` |
| `margins::margins(fit)` | `dp.logit_ame(...)` | AME + delta-method SE |
| `prop.test(x, n, p)` | `dp.prop_test(x, n, p)` | Yates χ² + Wilson-cc CI, matches R |

## Randomization procedures (randomizr → declarepy.ra)

`complete_ra`, `block_ra`, `simple_ra`, `complete_rs`, `simple_rs`,
`block_rs` — same `m`/`prob` semantics (floor/ceiling rule for fractional
expectations). All take `rng=` (an int seed or a `numpy.random.Generator`);
none touch global random state.

## What does NOT carry over

- **Seeds.** R and NumPy RNG streams are incompatible. The same design with
  `set.seed(464)` in R and `seed=464` here gives *statistically equivalent*,
  never digit-identical, simulations. Validation is tolerance-based
  ([`spec/VALIDATION_REPORT.md`](spec/VALIDATION_REPORT.md)).
- **Bare-name capture.** Write `lambda df: df["Y1"] - df["Y0"]`, not `Y_Z_1 -
  Y_Z_0`.
- **`fabricate()` hierarchies.** Build multilevel data explicitly with
  `np.repeat` in a `Model` build callable.
- Missing values never silently skip: estimators raise on NA (drop or impute
  explicitly first).
