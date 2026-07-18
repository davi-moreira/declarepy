# Fixture generator for T3 group ch19_23 (declarations 19.1-19.4, 23.1).
#
# The book ships saved diagnosis objects for every design in this group
# (diagnosis_19.1/19.2/19.3/19.4/19a/23.1), so no fresh *diagnosis*
# references are needed. What IS generated here are exact-match *estimator
# fixtures* on fixed, exported data, used by validate_t3_ch19_23.py to
# verify that the private Python ports of the R estimation machinery are
# numerically faithful before any Monte-Carlo comparison:
#
#   * bbmle::mle2 on declaration 19.2's bargaining likelihood (n = 2 and
#     n = 8): coefficients, Hessian-based SEs, z p-values, and *profile*
#     confidence intervals (DeclareDesign's tidy_try uses coef(summary(.))
#     + confint(.), which for mle2 is profile-likelihood).
#   * metafor::rma (REML and FE) on 19.3-style data (k = 200) and
#     19.4-style data (k = 5): mu, se(mu), z, p, CI, tau2, se(tau2), and
#     the Q-profile confint for tau2 (rdss::rma_mu_tau's tau_sq row).
#   * rdss::best_predictor's binned-R^2 estimand on a fixed 19.1-style
#     draw (cut(x, 20) ANOVA R^2 per covariate, which.max).
#
# Output: validation/reference/rgen_t3_ch19_23_fixtures.json
#
# Usage: /usr/local/bin/Rscript validation/r_scripts/t3_ch19_23_reference.R

ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(bbmle)
  library(metafor)
  library(estimatr)
  library(dplyr)
  library(jsonlite)
})

out <- list()

# ---------------------------------------------------------------- 19.2 mle2
offer <- function(n, d) sum(sapply(2:n[1], function(t) ((-1)^t) * (d^{t - 1})))

mle_fixture <- function(seed, n_rounds, delta = 0.8, kappa = 2, alpha = 0.5,
                        N = 200) {
  set.seed(seed)
  type <- rbinom(N, 1, alpha)
  Z <- sample(rep(c(0L, 1L), N / 2))                    # complete_ra(N)
  pi <- type * .75 +
    (1 - type) * (Z * offer(n_rounds, delta) + (1 - Z) * (1 - offer(n_rounds, delta)))
  y <- rbeta(N, pi * kappa, (1 - pi) * kappa)
  likelihood <- function(k, d, a) {
    m <- Z * offer(n_rounds, d) + (1 - Z) * (1 - offer(n_rounds, d))
    R <- a * dbeta(y, k * .75, k * .25) +
      (1 - a) * dbeta(y, k * m, k * (1 - m))
    -sum(log(R))
  }
  fit <- mle2(likelihood,
              start = list(k = 2, d = 0.50, a = 0.50),
              lower = list(k = 0.10, d = 0.01, a = 0.01),
              upper = list(k = 100, d = 0.99, a = 0.99),
              method = "L-BFGS-B")
  summ <- coef(summary(fit))
  ci <- suppressMessages(suppressWarnings(as.data.frame(confint(fit))))
  list(
    n_rounds = n_rounds, offer = offer(n_rounds, delta),
    y = y, Z = Z,
    term = rownames(summ),
    estimate = unname(summ[, 1]), std_error = unname(summ[, 2]),
    statistic = unname(summ[, 3]), p_value = unname(summ[, 4]),
    conf_low = ci[[1]], conf_high = ci[[2]],
    nll = as.numeric(-logLik(fit))
  )
}

out$mle_n2 <- mle_fixture(seed = 4641, n_rounds = 2)
out$mle_n8 <- mle_fixture(seed = 4642, n_rounds = 8)

# ---------------------------------------------------------------- 19.3 rma
rma_fixture <- function(seed, k, mu = 0.2, tau = 1) {
  set.seed(seed)
  sei <- pmax(0.1, abs(rnorm(k, mean = 0.8, sd = 0.5)))
  theta <- rnorm(k, mean = mu, sd = tau)
  yi <- rnorm(k, mean = theta, sd = sei)
  fx <- function(fit, method) {
    o <- list(
      method = method,
      b = as.numeric(fit$beta), se = fit$se, zval = fit$zval, pval = fit$pval,
      ci_lb = fit$ci.lb, ci_ub = fit$ci.ub,
      tau2 = fit$tau2, se_tau2 = fit$se.tau2
    )
    if (method == "REML") {
      tci <- confint(fit)$random
      o$tau2_ci_lb <- tci["tau^2", "ci.lb"]
      o$tau2_ci_ub <- tci["tau^2", "ci.ub"]
    }
    o
  }
  list(
    yi = yi, sei = sei,
    reml = fx(rma(yi = yi, sei = sei, method = "REML"), "REML"),
    fe = fx(rma(yi = yi, sei = sei, method = "FE"), "FE")
  )
}

out$rma_k200 <- rma_fixture(seed = 4643, k = 200, tau = 1)
out$rma_k200_tau0 <- rma_fixture(seed = 4645, k = 200, tau = 0)
out$rma_k5 <- rma_fixture(seed = 4644, k = 5, tau = 0.1)

# ------------------------------------------------------- 19.1 best_predictor
set.seed(4646)
N <- 1000
X <- matrix(rnorm(10 * N), N)
U <- rnorm(N)
f_Y <- function(z, X.1, X.2, X.3, X.4, u)
  z * X.1 + z * X.2^2 + z * exp(X.3) + z * X.3 * X.4 + u
tau <- f_Y(1, X[, 1], X[, 2], X[, 3], X[, 4], U) -
  f_Y(0, X[, 1], X[, 2], X[, 3], X[, 4], U)
dat <- as.data.frame(X)
names(dat) <- paste0("X.", 1:10)
dat$tau <- tau
r2 <- sapply(paste0("X.", 1:10), function(j)
  lm_robust(tau ~ cut(dat[[j]], 20), data = dat)$r.squared)
out$best_predictor <- list(
  X = as.data.frame(X), tau = tau,
  r_squared = unname(r2), estimand = which.max(r2)
)

write_json(
  out,
  path = "validation/reference/rgen_t3_ch19_23_fixtures.json",
  digits = 12, auto_unbox = TRUE
)
cat("written validation/reference/rgen_t3_ch19_23_fixtures.json\n")
