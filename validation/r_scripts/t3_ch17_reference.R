# T3 ch17 reference generation.
#
# declaration_17.1 has no saved diagnosis object in the replication archive
# (diagnosis_17.1.rds diagnoses declaration_17.2), so this script sources the
# book's declaration_17.1.R and diagnoses it with the default diagnosands at
# sims = 2000, writing validation/reference/rgen_t3_ch17_17_1.json.
#
# Note: the book's declaration_17.1.R draws `type` with lowercase labels
# ("Never-responder", "Always-responder") but tests them against the
# capitalized "Never-Responder"/"Always-Responder" inside if_else, so both
# comparisons are always FALSE: Y_Z_white = 1 and Y_Z_latino = 0 for every
# unit, and the difference in means is exactly 1 while the estimand stays
# ~0.05. The reference (and the declarepy translation) preserve this
# behavior faithfully rather than "fixing" the case mismatch.
#
# Usage: /usr/local/bin/Rscript validation/r_scripts/t3_ch17_reference.R

ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(DeclareDesign)
  library(dplyr)
  library(jsonlite)
})

repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
book <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book",
  "replication-materials/code"
)

set.seed(464)

# ---- declaration_17.1: fresh reference diagnosis --------------------------
source(file.path(book, "declarations", "declaration_17.1.R"))
diag_17_1 <- diagnose_design(declaration_17.1, sims = 2000, bootstrap_sims = FALSE)
out <- list(
  kind = "diagnosis",
  source = "declaration_17.1.R, diagnose_design(sims = 2000), default diagnosands",
  diagnosands = get_diagnosands(diag_17_1)
)
path <- file.path(repo, "validation", "reference", "rgen_t3_ch17_17_1.json")
write_json(out, path, digits = 12, auto_unbox = TRUE, dataframe = "rows")
cat("written", path, "\n")

# ---- randomizr num_arms pair-assignment sanity check ----------------------
# declaration_17.6 uses complete_ra(N = 400, num_arms = 200) to form pairs;
# confirm every arm gets exactly 2 units (declarepy's _complete_ra_conditions
# reproduces this exactly-N/k allocation).
z <- randomizr::complete_ra(N = 400, num_arms = 200)
cat("complete_ra(400, num_arms=200) arm sizes:", paste(range(table(z)), collapse = ".."), "\n")

# ---- declaration_17.6_a behavioral-function spot checks -------------------
# Deterministic values of the trust-game functions, for exact (1e-12)
# validation of declarepy's private _invested/_returned/_average_invested/
# _average_returned helpers.
source(file.path(book, "declarations", "declaration_17.6_a.R"))
a_grid <- c(0.05, 0.2, 1 / 3, 0.5, 0.65, 0.8, 0.95)
helpers <- list(
  a_grid = a_grid,
  average_invested = sapply(a_grid, average_invested),
  average_returned = sapply(a_grid, average_returned),
  invested_a1 = c(0.3, 0.7, 0.9),
  invested_a2 = c(0.6, 0.2, 0.5),
  invested_values = invested(c(0.3, 0.7, 0.9), c(0.6, 0.2, 0.5)),
  returned_x1 = c(0.5, 0.9, 1.0),
  returned_a2 = c(0.4, 1 / 3, 0.25),
  returned_values = returned(c(0.5, 0.9, 1.0), c(0.4, 1 / 3, 0.25))
)
hpath <- file.path(repo, "validation", "reference", "rgen_t3_ch17_helpers.json")
write_json(helpers, hpath, digits = 12)
cat("written", hpath, "\n")
