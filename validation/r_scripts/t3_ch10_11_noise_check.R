# t3_ch10_11_noise_check.R — quantify the saved references' own MC noise.
#
# Failure-investigation pass for the T3 ch10_11 group: reruns two of the
# book's diagnoses at sims = 2000 with a DIFFERENT seed and saves the fresh
# diagnosand tables. Comparing fresh-R-vs-saved-reference deltas against
# python-vs-saved-reference deltas shows how much of any remaining discrepancy is the
# reference's own Monte-Carlo dispersion (the saved objects were produced at
# sims = 2000 with bootstrap_sims = FALSE for 11.3/11.4, so they carry no
# uncertainty information of their own).
#
#   1. diagnosis_11.4 (full 50-inquiry x 6-estimator grid, 2000 sims)
#        -> reference/rgen_t3_ch10_11_noise_11.4.json
#   2. diagnosis_11.3 subset: prob = 0.1 (the noisiest column: ~N/10 treated
#      units), N in the cells that failed in the stale low-sims CSV,
#      2000 sims -> reference/rgen_t3_ch10_11_noise_11.3.json
#   3. a fixed-data fixture for declaration_11.4's estimator handler:
#      lm(Y ~ poly(X, k)) predictions on x_range for k = 1..6 from a
#      deterministic dataset -> reference/rgen_t3_ch10_11_poly_fixture.json
#      (proves the numpy polynomial predictions are the same numbers).
#
# Run from the declarepy repo root:
#   /usr/local/bin/Rscript validation/r_scripts/t3_ch10_11_noise_check.R

ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))

suppressPackageStartupMessages({
  # tidyverse meta-package is not installed; attach the pieces the book's
  # declaration files actually use.
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(purrr)
  library(stringr)
  library(DeclareDesign)
})

code_dir <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book",
  "replication-materials/code/declarations"
)
out_dir <- file.path("validation", "reference")
sims <- 2000

# ---- 1. fresh diagnosis_11.4 --------------------------------------------
set.seed(20260718)  # deliberately != the book's environment
source(file.path(code_dir, "declaration_11.4.R"))
fresh_11.4 <- diagnose_design(declaration_11.4, sims = sims,
                              bootstrap_sims = FALSE)
jsonlite::write_json(
  list(kind = "fresh-rerun seed=20260718 sims=2000",
       diagnosands = as.data.frame(get_diagnosands(fresh_11.4))),
  file.path(out_dir, "rgen_t3_ch10_11_noise_11.4.json"),
  digits = 10, na = "null"
)
message("fresh 11.4 written")

# ---- 2. fresh diagnosis_11.3 subset (prob = 0.1 failing cells) ----------
N <- 100
prob <- 0.5  # must exist in the sourcing env for redesign() to rebind it
             # (the book's master run inherited it from earlier scripts)
source(file.path(code_dir, "declaration_11.3.R"))
diagnosands_11.3 <-
  declare_diagnosands(cost = unique(N * 2 + prob * N * 20),
                      rmse = sqrt(mean((estimate - estimand) ^ 2)))
set.seed(20260719)
subset_designs <- redesign(
  declaration_11.3,
  N = c(140, 150, 220, 290, 360, 380, 490, 790, 950),
  prob = 0.1
)
fresh_11.3 <- diagnose_designs(subset_designs,
                               diagnosands = diagnosands_11.3,
                               sims = sims, bootstrap_sims = FALSE)
jsonlite::write_json(
  list(kind = "fresh-rerun seed=20260718 sims=2000 prob=0.1 subset",
       diagnosands = as.data.frame(get_diagnosands(fresh_11.3))),
  file.path(out_dir, "rgen_t3_ch10_11_noise_11.3.json"),
  digits = 10, na = "null"
)
message("fresh 11.3 subset written")

# ---- 3. deterministic poly-prediction fixture ---------------------------
# Same functional pieces as declaration_11.4's estimator handler, on fixed
# data, so declarepy's numpy Polynomial.fit predictions can be compared
# number-for-number (they share the polynomial column space with poly()).
dip <- function(x) (x <= 1) * x + (x > 1) * (x - 2) ^ 2 + 0.2
x_range <- seq(from = 0, to = 3, length.out = 50)
set.seed(464)
fix <- tibble(X = runif(100, 0, 3), Y = dip(X) + rnorm(100, 0, 0.5))
preds <- map(1:6, ~ predict(lm(Y ~ poly(X, .), data = fix),
                            newdata = tibble(X = x_range))) |>
  set_names(nm = str_c("A", 1:6)) |>
  map(unname) |>
  as_tibble() |>
  mutate(X = x_range)
jsonlite::write_json(
  list(kind = "poly fixture: lm(Y~poly(X,k)) predictions, fixed data",
       data = fix, predictions = preds),
  file.path(out_dir, "rgen_t3_ch10_11_poly_fixture.json"),
  digits = 12, na = "null"
)
message("poly fixture written")
