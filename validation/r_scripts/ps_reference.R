# Reference outputs for problem sets 1-4 (T2 acceptance), run with the keys' seed.
ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
if (!requireNamespace("margins", quietly = TRUE))
  install.packages("margins", repos = "https://cloud.r-project.org", lib = .libPaths()[1])
suppressMessages({library(DeclareDesign); library(rdss); library(dplyr); library(broom); library(jsonlite); library(margins)})
repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
out <- list()
tidy_diag <- function(dg) as.data.frame(get_diagnosands(dg))

## ---------------- PS1 ----------------
set.seed(343)
expected_effect_size <- 0.1
ps1_design <-
  declare_model(N = 100, U = rnorm(N),
                potential_outcomes(Y ~ expected_effect_size * Z + U)) +
  declare_inquiry(ATE = mean(Y_Z_1 - Y_Z_0)) +
  declare_assignment(Z = simple_ra(N = N, prob = 0.5)) +
  declare_measurement(Y = reveal_outcomes(Y ~ Z)) +
  declare_estimator(Y ~ Z, .method = difference_in_means, inquiry = "ATE")
out$ps1_power_base <- tidy_diag(diagnose_design(ps1_design, sims = 1000, bootstrap_sims = FALSE))
designs <- redesign(ps1_design, expected_effect_size = seq(0.1, 1, 0.1))
out$ps1_mde <- tidy_diag(diagnose_designs(designs, sims = 1000, bootstrap_sims = FALSE))
expected_effect_size <- 0.3
N <- 100
ps1_bonus <-
  declare_model(N = N, U = rnorm(N),
                potential_outcomes(Y ~ expected_effect_size * Z + U)) +
  declare_inquiry(ATE = mean(Y_Z_1 - Y_Z_0)) +
  declare_assignment(Z = simple_ra(N = N, prob = 0.5)) +
  declare_measurement(Y = reveal_outcomes(Y ~ Z)) +
  declare_estimator(Y ~ Z, .method = difference_in_means, inquiry = "ATE")
out$ps1_bonus <- tidy_diag(diagnose_designs(redesign(ps1_bonus, N = seq(100, 500, 100)), sims = 1000, bootstrap_sims = FALSE))
cat("PS1 done\n")

## ---------------- PS2 ----------------
set.seed(343)
N <- 1000
ps2_design <-
  declare_model(N = N, data = lapop_brazil, handler = resample_data) +
  declare_model(potential_outcomes(Y ~ pmin(7, trust_police + rbinom(N, 1, prob = 0.35) * Z))) +
  declare_inquiry(ATE = mean(Y_Z_1 - Y_Z_0))
est <- draw_estimand(ps2_design)
dat <- draw_data(ps2_design)
out$ps2_estimand <- est$estimand
out$ps2_sd_yz0 <- sd(dat$Y_Z_0)
ps2_design <- ps2_design +
  declare_assignment(Z = simple_ra(N, prob = 0.5)) +
  declare_measurement(Y = reveal_outcomes(Y ~ Z))
dat <- draw_data(ps2_design)
fit <- estimatr::lm_robust(Y ~ govt_pride + trust_military + trust_supreme_court + support_political_system, data = dat)
out$ps2_r_squared <- glance(fit)$r.squared
ps2_design <- ps2_design +
  declare_estimator(Y ~ Z, .method = estimatr::lm_robust, label = "unadjusted") +
  declare_estimator(Y ~ Z + govt_pride + trust_military + trust_supreme_court + support_political_system,
                    .method = estimatr::lm_robust, label = "adjusted")
out$ps2_power_1000 <- tidy_diag(diagnose_design(ps2_design, sims = 1000, bootstrap_sims = FALSE))
out$ps2_power_grid <- tidy_diag(diagnose_designs(redesign(ps2_design, N = c(1000, 1500, 2000, 2500, 3000)),
                                                 sims = 1000, bootstrap_sims = FALSE))
cat("PS2 done\n")

## ---------------- PS3 ----------------
set.seed(343)
N <- 100
ps3_design <-
  declare_model(N = N, X = rbinom(N, 1, 0.5),
                potential_outcomes(Y ~ rbinom(N, 1, prob = 0.3 + 0.1 * X + 0.1 * Z + 0.1 * X * Z))) +
  declare_assignment(Z = complete_ra(N, prob = 0.5)) +
  declare_measurement(Y = reveal_outcomes(Y ~ Z)) +
  declare_estimator(Y ~ X + Z + X:Z, .method = estimatr::lm_robust, term = "X:Z")
out$ps3_power_100 <- tidy_diag(diagnose_design(ps3_design, sims = 500, bootstrap_sims = FALSE))
out$ps3_power_grid <- tidy_diag(diagnose_designs(redesign(ps3_design, N = c(100, 500, 1000, 3000, 5000)),
                                                 sims = 500, bootstrap_sims = FALSE))
cat("PS3 done\n")

## ---------------- PS4 ----------------
set.seed(343)
tidy_margins <- function(x) tidy(margins(x, data = x$data), conf.int = TRUE)
ps4_design <-
  declare_model(N = 100, U = runif(N), X = rnorm(N),
                potential_outcomes(Y ~ if_else(0.1 * Z + U > 0.8, 1, 0))) +
  declare_inquiry(ATE = mean(Y_Z_1 - Y_Z_0)) +
  declare_assignment(Z = complete_ra(N, prob = 0.5)) +
  declare_measurement(Y = reveal_outcomes(Y ~ Z)) +
  declare_estimator(Y ~ Z + X, .method = lm, inquiry = "ATE", label = "ols") +
  declare_estimator(Y ~ Z + X, .method = glm, family = binomial("logit"), inquiry = "ATE", label = "logit") +
  declare_estimator(Y ~ Z + X, .method = glm, family = binomial("logit"), .summary = tidy_margins,
                    inquiry = "ATE", term = "Z", label = "logit-marginal-effect")
out$ps4_diagnosis <- tidy_diag(diagnose_design(ps4_design, sims = 1000, bootstrap_sims = FALSE))
cat("PS4 done\n")

write_json(out, file.path(repo, "validation/reference/rgen_ps1_4.json"), digits = 10, na = "null")
cat("written rgen_ps1_4.json\n")
