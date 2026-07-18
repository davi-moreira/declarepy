# T3 ch16 reference generation.
#
# 1. declaration_16.4 (IV / LATE) has no saved diagnosis object in the book's
#    replication archive, so we diagnose the book's own declaration here
#    (sims = 2000, bootstrap_sims = 2000) -> rgen_t3_ch16_16.4.json.
# 2. The process-tracing estimates of declaration_16.1 are deterministic
#    given the observed single-case data, so we also export the exact
#    posterior for every strategy x possible (X, M, W, Y) combination
#    -> rgen_t3_ch16_pt_posteriors.json (an exact, seed-free reference).
#
# Usage: Rscript validation/r_scripts/t3_ch16_reference.R
ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(DeclareDesign); library(rdss); library(CausalQueries)
  library(dplyr); library(jsonlite)
})

repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
decl <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book",
  "replication-materials/code/declarations")

## ---------------- declaration_16.4 (IV) ----------------
set.seed(464)
source(file.path(decl, "declaration_16.4.R"))
diag_16_4 <- diagnose_design(declaration_16.4, sims = 2000, bootstrap_sims = 2000)
write_json(
  list(kind = "diagnosis", sims = 2000,
       diagnosands = as.data.frame(get_diagnosands(diag_16_4))),
  file.path(repo, "validation/reference/rgen_t3_ch16_16.4.json"),
  digits = 10, na = "null")
cat("16.4 done\n")

## ---------------- declaration_16.1 exact process-tracing posteriors ------
source(file.path(decl, "declaration_16.1a.R"))  # causal_model + strategies

grid <- expand.grid(X = 0:1, M = 0:1, W = 0:1, Y = 0:1)
rows <- list()
for (i in seq_len(nrow(grid))) {
  dat <- grid[i, , drop = FALSE]
  res <- tryCatch(
    process_tracing_estimator(causal_model, query = "Y[X=1] - Y[X=0]",
                              data = dat, strategies = strategies),
    error = function(e) NULL)
  if (!is.null(res)) {
    res$X <- dat$X; res$M <- dat$M; res$W <- dat$W; res$Y <- dat$Y
    rows[[length(rows) + 1]] <- res
  }
}
write_json(
  list(kind = "pt_posteriors", posteriors = bind_rows(rows)),
  file.path(repo, "validation/reference/rgen_t3_ch16_pt_posteriors.json"),
  digits = 12, na = "null")
cat("pt posteriors done\n")
