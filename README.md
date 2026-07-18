# declarepy

**A transparent, tested Python translation of the DeclareDesign / rdss
declare–diagnose–redesign research-design engine.**

`declarepy` re-implements, in plain scientific Python (numpy / pandas / scipy /
statsmodels), the MIDA framework — **M**odel, **I**nquiry, **D**ata strategy,
**A**nswer strategy — from:

> Blair, Graeme, Alexander Coppock, and Macartan Humphreys. 2023.
> *Research Design in the Social Sciences: Declaration, Diagnosis, and
> Redesign.* Princeton University Press. Free online at
> [book.declaredesign.org](https://book.declaredesign.org).

and from the R packages of the [DeclareDesign](https://declaredesign.org)
ecosystem (`DeclareDesign`, `randomizr`, `fabricatr`, `estimatr`, `rdss`).
It is an independent translation, **not affiliated with or endorsed by the
DeclareDesign authors**, built as the parity companion to Purdue's
HONR 46400 *Evidence-Driven Research* course.

## Design philosophy

1. **Transparent core.** Design steps take **explicit callables and column
   names** — no tidy-eval / quosure emulation. Potential outcomes are explicit
   `Y0` / `Y1` columns; revealing outcomes is a visible `np.where` switch.
   The fundamental problem of causal inference stays visible in the code.
2. **Validated, not assumed.** R and NumPy random-number streams are not
   portable, so "parity" means **statistical agreement within tolerance**
   (bias ±0.02·sd(Y), power ±0.05, coverage ±0.03, RMSE ±10% relative;
   real-data results exact to 3 decimals), never digit equality. Every
   translated element passes the validation protocol in
   [`docs/spec/VALIDATION_REPORT.md`](docs/spec/VALIDATION_REPORT.md) against
   the book's published reference outputs before it is called "at parity".
3. **Reproducible by construction.** Every stochastic step is driven by an
   explicit `numpy.random.Generator`; nothing seeds global state.

## Quick example

```python
import declarepy as dp

design = dp.Design(
    dp.Model(n=100,
             draw=lambda n, rng: {"U": (U := rng.normal(size=n)),
                                  "Y0": U, "Y1": 0.2 + U}),
    dp.Inquiry("ATE", lambda df: (df["Y1"] - df["Y0"]).mean()),
    dp.Assignment(lambda df, rng: dp.complete_ra(len(df), rng=rng)),
    dp.Measurement.reveal_outcomes(),          # Y = np.where(Z == 1, Y1, Y0)
    dp.Estimator.difference_in_means(inquiry="ATE"),
)

diagnosis = dp.diagnose(design, sims=1000, seed=464)
print(diagnosis.diagnosands)      # bias, sd_estimate, rmse, power, coverage ...

sweep = dp.redesign(design, n=[100, 200, 400])
print(dp.diagnose_all(sweep, sims=1000, seed=464).diagnosands)
```

## What's here

| Layer | Contents |
|---|---|
| `declarepy` (core) | `Model`, `Inquiry`, `Sampling`, `Assignment`, `Measurement`, `Estimator`, `Diagnosands`, `Design`, `diagnose()`, `redesign()`, `run_design()`, `draw_data()` |
| `declarepy.ra` | `complete_ra`, `block_ra`, `simple_rs`, `complete_rs` (randomizr-style procedures) |
| `declarepy.estimators` | `difference_in_means` (incl. blocked/clustered), `lm_robust` (HC0–HC3, CR2) — estimatr-style, **HC2 default** |
| `declarepy.data` | Five MIT-licensed datasets from `rdss` (with attribution) |
| `docs/spec/` | The translation spec: inventory, API mapping, semantic differences, validation protocol, roadmap, parity matrix |
| `validation/` | Reference outputs (from the book's replication archive + fresh R runs) and the comparison harness |

## Semantic differences from R (deliberate)

See [`docs/spec/SEMANTIC_DIFFERENCES.md`](docs/spec/SEMANTIC_DIFFERENCES.md)
for the full list. Headlines:

- **RNG streams differ** — same seed never reproduces R digits; validation is
  tolerance-based against published outputs.
- **No tidy-eval** — steps take explicit callables/column names by design.
- **HC2 robust SEs are the default** in `lm_robust` and estimator steps,
  matching `estimatr` (statsmodels alone defaults to classical SEs).
- Missing values are surfaced, not silently skipped: aggregation helpers
  report NA counts.

## Status

Working through the tranches of
[`docs/spec/TRANSLATION_ROADMAP.md`](docs/spec/TRANSLATION_ROADMAP.md);
per-element status lives in
[`docs/spec/PARITY_MATRIX.csv`](docs/spec/PARITY_MATRIX.csv) and validated
results in [`docs/spec/VALIDATION_REPORT.md`](docs/spec/VALIDATION_REPORT.md).

This package is **not yet on PyPI**; per the project's guardrails the
distribution name will be coordinated with the DeclareDesign authors before
any release.

## License

MIT — see [`LICENSE`](LICENSE), which carries the full attribution notice to
Blair, Coppock & Humphreys and notes upstream `randomizr` / `estimatr`
licenses (their behavior is re-implemented from published descriptions; no
GPL source code is copied).
