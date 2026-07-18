# Reference values for the ch09 private helpers (T3, group ch09).
#
# Generates rgen_t3_ch09_*.json consumed by validation/validate_t3_ch09.py:
#   * rgen_t3_ch09_lh_robust.json  — estimatr::lh_robust linear-hypothesis rows
#       (declaration_9.2's declare_test). The installed estimatr 1.0.6 errors on
#       single-coefficient fits (`var(lm_robust_fit$df > 0)` is NA when df has
#       length 1), so the intercept-only case reproduces lh_robust's internals
#       verbatim — car::linearHypothesis value + HC2 vcov + df of the first
#       coefficient — bypassing only that version-specific guard.
#   * rgen_t3_ch09_wls_hc2.json    — weighted lm_robust, HC2 (declaration_9.6).
#   * rgen_t3_ch09_wls_cr2.json    — weighted + clustered lm_robust, CR2 with
#       Bell–McCaffrey df (declaration_9.7's answer strategy).
#   * rgen_t3_ch09_foos_observed.json — the observed Foos et al. estimate the
#       book's diagnosis_9.7.R compares the randomization distribution against.
#
# Fixture data are embedded in each JSON so Python reads identical inputs.
ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({library(estimatr); library(jsonlite)})
requireNamespace("car")

repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
ref_dir <- file.path(repo, "validation", "reference")

tidy_rows <- function(td) {
  cols <- c("term", "estimate", "std.error", "statistic", "p.value",
            "conf.low", "conf.high", "df", "outcome")
  lapply(seq_len(nrow(td)), function(i) as.list(td[i, cols]))
}

# ---- 1. lh_robust ----------------------------------------------------------
# (a) two-coefficient model: lh_robust runs end-to-end in estimatr 1.0.6.
d_lh2 <- data.frame(y = c(1.2, 3.4, 2.2, 5.1, 4.4, 6.0),
                    z = c(0, 0, 0, 1, 1, 1))
f2 <- lh_robust(y ~ z, data = d_lh2, linear_hypothesis = "z = 0.5")

# (b) intercept-only model (declaration_9.2's exact form): reproduce
# lh_robust()'s computation (see header note on the 1.0.6 guard bug).
d_lh1 <- data.frame(age = c(10, 44, 71))
fit1 <- lm_robust(age ~ 1, data = d_lh1)
lht1 <- car::linearHypothesis(fit1, hypothesis.matrix = "(Intercept) = 20",
                              level = 1 - fit1$alpha)
est1 <- drop(attr(lht1, "value"))
se1 <- sqrt(diag(attr(lht1, "vcov")))
df1 <- fit1$df[1]
stat1 <- est1 / se1
p1 <- 2 * pt(abs(stat1), df1, lower.tail = FALSE)
ci1 <- est1 + se1 %o% qt(c(fit1$alpha / 2, 1 - fit1$alpha / 2), df1)
row1 <- list(term = "(Intercept) = 20", estimate = unname(est1),
             std.error = unname(se1), statistic = unname(stat1),
             p.value = unname(p1), conf.low = unname(ci1[, 1]),
             conf.high = unname(ci1[, 2]), df = unname(df1), outcome = "age")

write_json(
  list(two_coef = list(data = as.list(d_lh2),
                       linear_hypothesis = "z = 0.5",
                       tidy = tidy_rows(tidy(f2))),
       intercept_only = list(data = as.list(d_lh1),
                             linear_hypothesis = "(Intercept) = 20",
                             row = row1)),
  file.path(ref_dir, "rgen_t3_ch09_lh_robust.json"),
  digits = 12, auto_unbox = TRUE
)

# ---- 2. weighted lm_robust, HC2 (declaration_9.6's Weighted estimator) -----
set.seed(11)
n <- 20
zw <- rep(c(0, 1), each = 10)
d_hc2 <- data.frame(z = zw,
                    y = rnorm(n, 1 + 0.5 * zw),
                    w = runif(n, 0.5, 3))
f_hc2 <- lm_robust(y ~ z, weights = w, data = d_hc2)
write_json(
  list(data = as.list(d_hc2), tidy = tidy_rows(tidy(f_hc2))),
  file.path(ref_dir, "rgen_t3_ch09_wls_hc2.json"),
  digits = 12, auto_unbox = TRUE
)

# ---- 3. weighted + clustered lm_robust, CR2 + Bell–McCaffrey df ------------
set.seed(21)
sizes <- c(4, 5, 6, 5, 4, 6)
cl <- rep(1:6, times = sizes)
zc <- c(0, 1, 0, 1, 0, 1)[cl]
d_cr2 <- data.frame(cl = cl, z = zc,
                    y = rnorm(30, 0.4 * zc + rep(rnorm(6), times = sizes)),
                    w = runif(30, 0.5, 3))
f_cr2 <- lm_robust(y ~ z, weights = w, clusters = cl, data = d_cr2)
write_json(
  list(data = as.list(d_cr2), tidy = tidy_rows(tidy(f_cr2))),
  file.path(ref_dir, "rgen_t3_ch09_wls_cr2.json"),
  digits = 12, auto_unbox = TRUE
)

# ---- 4. Foos et al. observed estimate (diagnosis_9.7.R's benchmark) --------
foos <- read.csv(file.path(repo, "src/declarepy/data/foos_etal.csv"))
obs <- lm_robust(marked_register_2014 ~ treat + ward,
                 weights = weights, clusters = street, data = foos)
write_json(
  list(tidy = tidy_rows(tidy(obs))),
  file.path(ref_dir, "rgen_t3_ch09_foos_observed.json"),
  digits = 12, auto_unbox = TRUE
)

cat("written rgen_t3_ch09_{lh_robust,wls_hc2,wls_cr2,foos_observed}.json\n")
