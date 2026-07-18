ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
# Fresh 2000-sim reference diagnoses for declarations 4.1, 5.1 and 7.1.
#
# Why: the book's saved diagnosis_4.1.rds was built with sims = 100, whose
# Monte-Carlo error exceeds the validation protocol's tolerance bands
# (se(mean_estimate) = 0.031 > the ±0.02·sd(Y) band), and 5.1 / 7.1 have no
# saved diagnosis object at all.  So we re-diagnose the *unmodified* book
# declarations at sims = 2000 (bootstrap_sims = 100 for se columns) and use
# these as the primary references.
#
# declaration_7.1 has inquiries but no estimator, so the default diagnosands
# are almost all NA; we ask for estimand-focused diagnosands instead
# (mean_estimand, sd_estimand per inquiry).
suppressMessages({
  library(DeclareDesign)
  library(jsonlite)
})

decl <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book",
  "replication-materials/code/declarations"
)
ref <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "declarepy/validation/reference"
)

source(file.path(decl, "declaration_4.1.R"))
source(file.path(decl, "declaration_5.1.R"))
source(file.path(decl, "declaration_7.1.R"))

dump <- function(diagnosis, name) {
  df <- as.data.frame(get_diagnosands(diagnosis))
  write_json(df, file.path(ref, name), digits = 12)
  cat("written", name, "\n")
}

set.seed(464)
dump(diagnose_design(declaration_4.1, sims = 2000), "rgen_t3_ch04_05_07_4.1.json")

set.seed(464)
dump(diagnose_design(declaration_5.1, sims = 2000), "rgen_t3_ch04_05_07_5.1.json")

# Estimand-only design: summarize the inquiries themselves.
estimand_diagnosands <- declare_diagnosands(
  mean_estimand = mean(estimand),
  sd_estimand = sd(estimand)
)
set.seed(464)
dump(
  diagnose_design(declaration_7.1, sims = 2000, diagnosands = estimand_diagnosands),
  "rgen_t3_ch04_05_07_7.1.json"
)
