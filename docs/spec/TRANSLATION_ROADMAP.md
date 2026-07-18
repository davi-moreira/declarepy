# TRANSLATION_ROADMAP — rdss/DeclareDesign → Python, as a parallel project

> **Status (2026-07-18): BUILT.** Tranches 0-4 are complete and validated
> in this repository (see VALIDATION_REPORT.md for the ledgers: T1 97/97,
> T2 55/55, T3 6,828/6,828, T4 62/62 diagnosis objects + figures). T5
> (polish) is complete except the public release itself, which is
> deliberately withheld: **the PyPI/distribution name must be coordinated
> with the DeclareDesign authors first** (professor's action item). The
> course still does NOT depend on this package (locked decision #4):
> notebooks ship self-contained inline helpers this term; swaps happen
> between terms.

## Guiding principles

1. **Course first:** priorities P1/P2 mirror the notebook calendar
   (`PARITY_MATRIX.csv` `course_date` column); parity work never changes a
   shipped notebook mid-semester — swaps happen between terms.
2. **Transparent core:** the package's design steps take explicit callables and
   column names (no tidy-eval emulation — see `SEMANTIC_DIFFERENCES.md` §10).
3. **Validated, not assumed:** every translated declaration/diagnosis is run
   against the book's published outputs within tolerance
   (`VALIDATION_REPORT.md` protocol) before it is called "at parity".
4. **MIT throughout**, with attribution to Blair, Coppock & Humphreys.

## Tranches

### Tranche 0 — already done (course-inline, Fall 2026 build)

The 16 course-load-bearing patterns exist as narrated inline helpers inside
nb04–nb13 (see `API_MAPPING.md`). They are the de-facto reference
implementations for Tranche 1.

### Tranche 1 — core engine as a package (`declarepy` working name)

Extract the inline helpers into `src/` as a tested mini-package:
`Model`, `Inquiry`, `Sampling`, `Assignment`, `Measurement`, `Estimator`,
`Diagnosands`, `diagnose()`, `redesign()` + `complete_ra/simple_rs/complete_rs`.
Acceptance: reproduces the course notebooks' outputs exactly (same seeds), and
declaration_2.1 / 9.1 / 10.1 / 11.1 / 18.1 within tolerance of book outputs.
Effort: ~2–3 weeks part-time. **Prereq for everything below.**

### Tranche 2 — estimator fidelity

Design-based SEs matching estimatr (HC1/HC2/CR2, blocked + clustered DiM),
via statsmodels/linearmodels. Acceptance: problem-set answer keys 1–4
reproduce within tolerance. Unlocks faithful PS translations for future course
offerings.

### Tranche 3 — declaration library, course chapters first

Translate the replication declarations in course-priority order:
ch. 9 (7 files) → ch. 10–11 (10) → ch. 18 (13) → ch. 15–16 (13) → ch. 17 (7) →
ch. 19 + 23 (8). Each file: translation + validation entry + provenance line.

### Tranche 4 — diagnosis objects + figures

Recompute the book's saved diagnosis objects (`.rds`) in Python; re-implement
priority figures (matplotlib or plotnine — decide at tranche start).

### Tranche 5 — polish & release

Docs, PyPI packaging, CI matrix, a "for DeclareDesign users" migration guide;
coordinate with the DeclareDesign authors before any public release naming.

## Non-goals / exclusions (each justified)

- **Tidy-eval / quosure emulation** — deliberate API departure
  (SEMANTIC_DIFFERENCES §10): steps take explicit callables/column names.
- **The `dataverse` download client** — the needed datasets ship as CSVs
  with attribution (data/README.md).
- **`DesignLibrary` breadth** — post-parity goal; the book's own designs
  are covered by `declarepy.library`.
- **Blocked + clustered `difference_in_means` combined** — no book design
  requires it; raises NotImplementedError with a pointer here.
- **MCMC-backed `stan_glm`** — rstanarm is not installable locally;
  declaration 9.3's posterior is reproduced by deterministic quadrature
  (documented in ch09), which is exact for the 2-parameter model.
- **Pixel-level figure parity** — figures reproduce the *message*
  (SEMANTIC_DIFFERENCES §9) with structural checks; matplotlib, not
  plotnine (T4 decision).
- **Digit-level RNG parity with R** — impossible across RNG streams
  (SEMANTIC_DIFFERENCES §1); the tolerance protocol is the contract.

## Risks

| Risk | Mitigation |
|---|---|
| RNG non-portability makes "parity" ambiguous | tolerance-based validation protocol (VALIDATION_REPORT.md), stated up front |
| Scope creep into the course build | this directory is the only place parity work lives; notebooks never wait on it |
| Upstream API drift (DeclareDesign updates) | pin to rdss 1.0.14 / book 2023 edition as the reference surface |
