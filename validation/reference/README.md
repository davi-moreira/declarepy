# Reference outputs (ground truth for validation)

`*.json` files here are **derived numeric summaries** (diagnosand tables) of
the saved diagnosis objects shipped with the replication materials of
*Research Design in the Social Sciences: Declaration, Diagnosis, and Redesign*
(Blair, Coppock & Humphreys, 2023, Princeton University Press —
book.declaredesign.org). They were exported with
`validation/extract_rds_reference.R` (reads each `diagnosis_*.rds` and writes
its `diagnosands_df` as JSON; Monte-Carlo simulation draws are not included).

These are the published reference values every `declarepy` translation is
validated against (see `docs/spec/VALIDATION_REPORT.md` for the tolerance
protocol). Attribution: Blair, Coppock & Humphreys (2023); DeclareDesign
ecosystem (declaredesign.org).

Files named `rgen_*.json` (if present) are freshly generated reference runs
produced by the scripts in `validation/r_scripts/` using the CRAN packages
`DeclareDesign` / `rdss` / `estimatr` / `randomizr`.
