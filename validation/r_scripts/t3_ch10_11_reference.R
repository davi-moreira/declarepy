# Fresh R references for T3 ch10-11 where the book saved no diagnosis object:
#   1. declaration_10.4 (diagnosis_10.4.rds diagnoses declaration_10.2, not 10.4)
#      -> rgen_t3_ch10_11_decl_10.4.json  (diagnose_design, sims = 2000)
#   2. a fixed-data margins fixture for the logit/probit AME estimators of
#      declaration_11.5 -> rgen_t3_ch10_11_probit_fixture.json
ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(DeclareDesign); library(broom); library(jsonlite); library(margins)
})

repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
book <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book/replication-materials"

## ---------------- declaration_10.4: full default-diagnosand rows ----------
set.seed(464)
source(file.path(book, "code/declarations/declaration_10.4.R"))  # read-only
diag_10.4 <- diagnose_design(get("declaration_10.4"), sims = 2000, bootstrap_sims = FALSE)
write_json(
  list(
    kind = "diagnosis", n_sims = 2000,
    diagnosands = as.data.frame(get_diagnosands(diag_10.4))
  ),
  file.path(repo, "validation/reference/rgen_t3_ch10_11_decl_10.4.json"),
  digits = 12, na = "null"
)
cat("declaration_10.4 reference written\n")

## ---------------- margins fixture (declaration_11.5's tidy_margins) -------
set.seed(464)
N <- 100
Z <- randomizr::complete_ra(N, prob = 0.5)
Y <- rbinom(N, 1, prob = 0.2 * Z + 0.6)
dat <- data.frame(Y = Y, Z = Z)
tidy_margins <- function(x) tidy(margins(x, data = x$data), conf.int = TRUE)
lfit <- glm(Y ~ Z, family = binomial("logit"), data = dat)
pfit <- glm(Y ~ Z, family = binomial("probit"), data = dat)
write_json(
  list(
    data = dat,
    logit = tidy_margins(lfit),
    probit = tidy_margins(pfit)
  ),
  file.path(repo, "validation/reference/rgen_t3_ch10_11_probit_fixture.json"),
  digits = 12, na = "null"
)
cat("margins fixture written\n")
