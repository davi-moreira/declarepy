# Fresh, higher-sims R reference for declaration 19.2's alpha × n sweep.
#
# The book's diagnosis_19.2/19a were saved with only 100 simulations, and
# several cells are NA-sensitive: bbmle's profile confint returns NA on a
# side whose profile deviance never reaches the chi-square cutoff inside
# the box constraints — and can hard-error outright ("Hessian is
# ill-behaved or missing") on boundary fits, which crashes a plain
# DeclareDesign simulate_designs run at higher sims (that is why this
# script loops manually with tryCatch instead of using diagnose_design).
# DeclareDesign's default diagnosands do NOT na.rm, so one NA sim turns
# power/coverage into NA — the book's saved values are conditional on a
# no-NA 100-sim draw. To adjudicate declarepy deltas beyond that noise
# floor, this script simulates each design at sims = 500 and records
# NA-AWARE diagnosands per design × term: mean/sd/rmse of the estimates
# (na.rm), na.rm'd power and coverage, and the NA/error rates themselves.
#
# The data-generating process is the declaration's own (N = 200,
# type ~ Bern(alpha), Z = complete_ra, y ~ Beta(pi k, (1-pi) k)); the fit
# is the declaration's mle2 call verbatim.
#
# Output: validation/reference/rgen_t3_ch19_23_19a_sims500.json
# Usage:  /usr/local/bin/Rscript validation/r_scripts/t3_ch19_23_19a_fresh.R

ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(bbmle)
  library(randomizr)
  library(dplyr)
  library(jsonlite)
})

offer <- function(n, d) sum(sapply(2:n[1], function(t) ((-1)^t) * (d^{t - 1})))

sims <- 500
N <- 200
delta <- 0.8
kappa <- 2

run_cell <- function(alpha, n_rounds) {
  rows <- vector("list", sims)
  for (s in seq_len(sims)) {
    type <- rbinom(N, 1, alpha)
    Z <- complete_ra(N)
    off <- offer(n_rounds, delta)
    pi <- type * .75 + (1 - type) * (Z * off + (1 - Z) * (1 - off))
    y <- rbeta(N, pi * kappa, (1 - pi) * kappa)
    likelihood <- function(k, d, a) {
      m <- Z * offer(n_rounds, d) + (1 - Z) * (1 - offer(n_rounds, d))
      R <- a * dbeta(y, k * .75, k * .25) +
        (1 - a) * dbeta(y, k * m, k * (1 - m))
      -sum(log(R))
    }
    fit <- tryCatch(
      suppressWarnings(mle2(likelihood,
           start = list(k = 2, d = 0.50, a = 0.50),
           lower = list(k = 0.10, d = 0.01, a = 0.01),
           upper = list(k = 100, d = 0.99, a = 0.99),
           method = "L-BFGS-B")),
      error = function(e) NULL)
    if (is.null(fit)) {
      rows[[s]] <- data.frame(term = c("k", "d", "a"), estimate = NA_real_,
                              std.error = NA_real_, p.value = NA_real_,
                              conf.low = NA_real_, conf.high = NA_real_,
                              fit_error = TRUE)
      next
    }
    summ <- tryCatch(coef(summary(fit)), error = function(e) NULL)
    ci <- tryCatch(
      suppressWarnings(suppressMessages(as.data.frame(confint(fit)))),
      error = function(e) data.frame(lo = rep(NA_real_, 3), hi = rep(NA_real_, 3)))
    if (is.null(summ)) {
      est <- coef(fit); se <- rep(NA_real_, 3); p <- rep(NA_real_, 3)
    } else {
      est <- summ[, 1]; se <- summ[, 2]; p <- summ[, 4]
    }
    rows[[s]] <- data.frame(term = c("k", "d", "a"), estimate = unname(est),
                            std.error = unname(se), p.value = unname(p),
                            conf.low = ci[[1]], conf.high = ci[[2]],
                            fit_error = FALSE)
  }
  bind_rows(rows) |>
    mutate(truth = case_when(term == "k" ~ kappa, term == "d" ~ delta,
                             term == "a" ~ alpha)) |>
    group_by(term) |>
    summarize(
      alpha = alpha, n = n_rounds, n_sims = n(),
      fit_error_rate = mean(fit_error),
      mean_estimate = mean(estimate, na.rm = TRUE),
      sd_estimate = sd(estimate, na.rm = TRUE),
      bias = mean(estimate - truth, na.rm = TRUE),
      rmse = sqrt(mean((estimate - truth)^2, na.rm = TRUE)),
      power_narm = mean(p.value <= 0.05, na.rm = TRUE),
      coverage_narm = mean(conf.low <= truth & truth <= conf.high, na.rm = TRUE),
      na_p_rate = mean(is.na(p.value)),
      na_ci_rate = mean(is.na(conf.low) | is.na(conf.high)),
      .groups = "drop"
    )
}

set.seed(464)
grid <- expand.grid(alpha = c(0.25, 0.5, 0.75), n = c(2, 8))
out <- bind_rows(Map(function(a, n) run_cell(a, n), grid$alpha, grid$n))

write_json(
  list(kind = "narm_diagnosands", sims = sims, data = out),
  path = "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy/validation/reference/rgen_t3_ch19_23_19a_sims500.json",
  digits = 10, auto_unbox = TRUE
)
cat("written rgen_t3_ch19_23_19a_sims500.json\n")
