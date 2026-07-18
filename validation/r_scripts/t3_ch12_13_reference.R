# T3 ch12/ch13 reference generation.
#
# declaration_13.1 has no saved diagnosis object in the replication archive
# (diagnosis_13.1.rds diagnoses declaration_13.2), so this script sources the
# book's declaration_13.1.R and diagnoses it with the default diagnosands at
# sims = 2000, writing validation/reference/rgen_t3_ch12_13_13_1.json.
#
# It also prints an empirical check that randomizr's complete_ra(N = 500,
# prob_each = c(.25, .375, .375)) splits the remainder unit between the two
# 0.375 arms (counts 125/187/188 or 125/188/187), the behavior declarepy's
# private _complete_ra_each reproduces by randomized-systematic allocation.
#
# Usage: /usr/local/bin/Rscript validation/r_scripts/t3_ch12_13_reference.R

ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(DeclareDesign)
  library(jsonlite)
})

repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
book <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book",
  "replication-materials/code"
)

set.seed(464)

# ---- declaration_13.1: fresh reference diagnosis --------------------------
source(file.path(book, "declarations", "declaration_13.1.R"))
diag_13_1 <- diagnose_design(declaration_13.1, sims = 2000, bootstrap_sims = FALSE)
out <- list(
  kind = "diagnosis",
  source = "declaration_13.1.R, diagnose_design(sims = 2000), default diagnosands",
  diagnosands = get_diagnosands(diag_13_1)
)
path <- file.path(repo, "validation", "reference", "rgen_t3_ch12_13_13_1.json")
write_json(out, path, digits = 12, auto_unbox = TRUE, dataframe = "rows")
cat("written", path, "\n")

# ---- randomizr remainder-allocation sanity check --------------------------
counts <- t(replicate(2000, {
  z <- randomizr::complete_ra(
    N = 500,
    conditions = c("neutral", "personal", "social"),
    prob_each = c(0.250, 0.375, 0.375)
  )
  as.integer(table(z)[c("neutral", "personal", "social")])
}))
cat("complete_ra(500, prob_each=c(.25,.375,.375)) count patterns:\n")
print(table(apply(counts, 1, paste, collapse = "/")))
