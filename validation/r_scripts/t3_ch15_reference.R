# T3 ch15 reference generation — observational descriptive designs.
#
# Exports the chapter's fixed populations to src/declarepy/data/ and generates
# fresh R reference diagnosands where the book's saved diagnosis objects are
# not reproducible:
#
#   * declaration_15.1 / 15.2: the book's own `set.seed(343)` fixes portola, so
#     the shipped diagnosis_15.1.json / diagnosis_15.2.json ARE reproducible —
#     we only export the population CSV (and assert the book's estimand
#     4.2919047619 to prove the export equals the book's population).
#   * declaration_15.3: the book draws two_nigerian_states with NO seed, so
#     the shipped diagnosis_15.3.json's population is unrecoverable. We fix
#     set.seed(464), export the population, and diagnose the same sweep
#     (cluster_prob 0.1..0.9, sims = 2000) -> rgen_t3_ch15_15_3.json.
#   * declaration_15.4/15.5: same problem (state_shock drawn with no seed).
#     Fixed with set.seed(464), states frame exported, declaration_15.5
#     diagnosed (3 estimators, sims = 2000) -> rgen_t3_ch15_15_4.json.
#     rdss::post_stratification_helper needs `marginaleffects` (not installed);
#     we inline the same computation with predict(). Note: the helper passes
#     re.form = NA, but the book's published diagnosis (partial-pooling
#     sd_estimate 0.14, between full 0.017 and no pooling 0.198) can only
#     arise from predictions that INCLUDE the state random intercepts —
#     verified empirically (RE-excluded predictions give sd 0.018). We
#     therefore include REs (predict.merMod default re.form = NULL).
#   * fit-level check objects (rgen_t3_ch15_fitchecks.json): princomp first
#     scores on fixed r>0 / r<0 datasets (declaration_15.6's Y_first_factor)
#     and a fixed-dataset glmer + post-stratification fit, so the Python
#     private helpers can be checked at the fit level, not only through
#     Monte-Carlo diagnosands.
#
# Usage: /usr/local/bin/Rscript validation/r_scripts/t3_ch15_reference.R

ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(DeclareDesign)
  library(fabricatr)
  library(randomizr)
  library(estimatr)
  library(lme4)
  library(dplyr)
  library(jsonlite)
})

repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
data_dir <- file.path(repo, "src/declarepy/data")
ref_dir <- file.path(repo, "validation/reference")
sims <- 2000

write_diag <- function(dg, path, extra = list()) {
  out <- c(list(kind = "diagnosis",
                diagnosands = as.data.frame(get_diagnosands(dg))),
           extra)
  write_json(out, path, digits = 12, na = "null", auto_unbox = TRUE)
  cat("wrote", path, "\n")
}

## ---- populations -----------------------------------------------------------

# portola (book's own seed; identical to the population under diagnosis_15.1/2)
set.seed(343) # as in code/declarations/declaration_15.1.R
portola <- fabricate(N = 2100, Y_star = rnorm(N))
write.csv(portola, file.path(data_dir, "ch15_portola.csv"), row.names = FALSE)
stopifnot(abs(mean(as.numeric(cut(portola$Y_star, 7))) - 4.2919047619) < 1e-9)
cat("portola exported; book estimand reproduced (4.2919047619)\n")

# two_nigerian_states (book has no seed; canon fixed here)
set.seed(464)
ICC <- 0.4
two_nigerian_states <-
  fabricate(
    state = add_level(N = 2,
                      state_name = c("taraba", "kwara"),
                      state_mean = c(-0.2, 0.2)),
    locality = add_level(
      N = 500,
      locality_shock = rnorm(N, state_mean, sqrt(ICC))
    ),
    individual = add_level(
      N = 100,
      individual_shock = rnorm(N, sd = sqrt(1 - ICC)),
      Y_star = locality_shock + individual_shock
    )
  )
write.csv(two_nigerian_states[, c("state", "locality", "Y_star")],
          file.path(data_dir, "ch15_two_nigerian_states.csv"), row.names = FALSE)

# states (book has no seed for state_shock; canon fixed here)
set.seed(464)
sx <- as.data.frame(state.x77)
states <- data.frame(
  state = rownames(state.x77),
  prop_of_US = sx$Population / sum(sx$Population),
  prob_HS = sx$`HS Grad` / 100
)
states$state_n <- round(states$prop_of_US * 1998.6) # exactly 2,000 total
states$state_shock <- rnorm(50, sd = 0.5)
states$state_mean <- states$prob_HS * pnorm(0.2 + states$state_shock) +
  (1 - states$prob_HS) * pnorm(states$state_shock)
write.csv(states, file.path(data_dir, "ch15_states.csv"), row.names = FALSE)
stopifnot(sum(states$state_n) == 2000)
cat("populations exported\n")

# Read populations back so the reference diagnoses run on the CSV canon
# (bit-identical to what declarepy loads), not the in-memory doubles.
two_nigerian_states <- read.csv(file.path(data_dir, "ch15_two_nigerian_states.csv"))
states <- read.csv(file.path(data_dir, "ch15_states.csv"))

## ---- fit-level check objects ----------------------------------------------

fitchecks <- list()

# princomp first component, r > 0 and r < 0 (declaration_15.6's typo formula
# `~ Y_1 + Y_2 + Y_2`: R terms() dedupes, so the PCA is on (Y_1, Y_2) only).
set.seed(11)
n <- 40
Y_1 <- rnorm(n); Y_2 <- 0.5 * Y_1 + rnorm(n)
pc <- princomp(~ Y_1 + Y_2 + Y_2, cor = TRUE)
fitchecks$princomp_pos <- list(Y_1 = Y_1, Y_2 = Y_2, scores = as.vector(pc$scores[, 1]))
set.seed(12)
Y_1 <- rnorm(n); Y_2 <- -0.5 * Y_1 + rnorm(n)
pc <- princomp(~ Y_1 + Y_2 + Y_2, cor = TRUE)
fitchecks$princomp_neg <- list(Y_1 = Y_1, Y_2 = Y_2, scores = as.vector(pc$scores[, 1]))

# one fixed draw of declaration_15.4's data + all three fitted estimators
expanded <- states[rep(1:50, states$state_n), ]
N <- nrow(expanded)
set.seed(99)
HS <- rbinom(N, 1, expanded$prob_HS)
individual_shock <- rnorm(N, sd = 0.5)
policy_support <- rbinom(N, 1, pnorm(0.2 * HS + individual_shock + expanded$state_shock))
dat <- data.frame(
  policy_support = policy_support, HS = HS, state = expanded$state,
  PS_weight = ifelse(HS == 0, 1 - expanded$prob_HS, expanded$prob_HS)
)

ps_estimates <- function(pred, data) {
  pw <- tapply(pred * data$PS_weight, data$state, sum)
  w <- tapply(data$PS_weight, data$state, sum)
  est <- pw / w
  est[order(names(est))]
}

fit_pp <- glmer(policy_support ~ HS + (1 | state), data = dat,
                family = binomial(link = "logit"))
fit_np <- lm_robust(policy_support ~ HS + state, data = dat)
fit_fp <- lm_robust(policy_support ~ HS, data = dat)
fitchecks$glmm <- list(
  policy_support = dat$policy_support, HS = dat$HS, state = dat$state,
  PS_weight = dat$PS_weight,
  fixef = as.vector(fixef(fit_pp)), theta = as.vector(getME(fit_pp, "theta")),
  states_sorted = sort(unique(as.character(dat$state))),
  est_partial = as.vector(ps_estimates(predict(fit_pp, newdata = dat, type = "response"), dat)),
  est_no_pooling = as.vector(ps_estimates(predict(fit_np, newdata = dat), dat)),
  est_full_pooling = as.vector(ps_estimates(predict(fit_fp, newdata = dat), dat))
)
write_json(fitchecks, file.path(ref_dir, "rgen_t3_ch15_fitchecks.json"),
           digits = 15, na = "null", auto_unbox = TRUE)
cat("wrote fitchecks\n")

## ---- declaration_15.3 reference sweep --------------------------------------

budget_function <- function(cluster_prob) {
  budget <- 20000
  cluster_cost <- 20
  individual_cost <- 2
  n_clusters <- 1000
  n_individuals_per_cluster <- 100
  total_cluster_cost <- cluster_prob * n_clusters * cluster_cost
  remaining_funds <- budget - total_cluster_cost
  sampleable_individuals <- cluster_prob * n_clusters * n_individuals_per_cluster
  individual_prob <- (remaining_funds / individual_cost) / sampleable_individuals
  pmin(individual_prob, 1)
}

cluster_prob <- 0.5
declaration_15.3 <-
  declare_model(data = two_nigerian_states) +
  declare_measurement(Y = as.numeric(cut(Y_star, 7))) +
  declare_inquiry(Y_bar = mean(Y)) +
  declare_sampling(
    S_cluster = strata_and_cluster_rs(
      strata = state, clusters = locality, prob = cluster_prob),
    filter = S_cluster == 1) +
  declare_sampling(
    S_individual = strata_rs(strata = locality, prob = budget_function(cluster_prob)),
    filter = S_individual == 1) +
  declare_estimator(Y ~ 1, clusters = locality, se_type = "stata", inquiry = "Y_bar")

set.seed(464)
t0 <- Sys.time()
diagnosis_15.3 <-
  declaration_15.3 |>
  redesign(cluster_prob = seq(0.1, 0.9, 0.1)) |>
  diagnose_designs(sims = sims, bootstrap_sims = FALSE)
cat("15.3 sweep took", round(as.numeric(Sys.time() - t0, units = "mins"), 1), "min\n")
write_diag(diagnosis_15.3, file.path(ref_dir, "rgen_t3_ch15_15_3.json"),
           extra = list(n_sims = sims, seed_note = "population set.seed(464); diagnosis set.seed(464)"))

## ---- declaration_15.4/15.5 reference ---------------------------------------

# Inline replacement for rdss::post_stratification_helper (see header note:
# marginaleffects is unavailable; REs are included, matching the published
# diagnosis_15.4 values).
ps_helper <- function(model_fit, data) {
  pred <- if (inherits(model_fit, "merMod")) {
    predict(model_fit, newdata = data, type = "response")
  } else {
    predict(model_fit, newdata = data)
  }
  data.frame(pred = as.vector(pred), state = data$state, w = data$PS_weight) |>
    group_by(state) |>
    summarize(estimate = weighted.mean(pred, w), .groups = "drop")
}

declaration_15.4 <-
  declare_model(
    data = states[rep(1:50, states$state_n), ],
    HS = rbinom(n = N, size = 1, prob = prob_HS),
    PS_weight =
      case_when(HS == 0 ~ (1 - prob_HS),
                HS == 1 ~ prob_HS),
    individual_shock = rnorm(n = N, sd = 0.5),
    policy_support =
      rbinom(N, 1, prob = pnorm(0.2 * HS + individual_shock + state_shock))
  ) +
  declare_inquiry(
    handler = function(data) {
      states |> transmute(
        state,
        inquiry = "mean_policy_support",
        estimand = state_mean
      )
    }
  ) +
  declare_estimator(handler = label_estimator(function(data) {
    model_fit <- glmer(
      formula = policy_support ~ HS + (1 | state),
      data = data,
      family = binomial(link = "logit")
    )
    ps_helper(model_fit, data)
  }),
  label = "Partial pooling",
  inquiry = "mean_policy_support")

declaration_15.5 <-
  declaration_15.4 +
  declare_estimator(
    handler = label_estimator(function(data) {
      model_fit <- lm_robust(
        formula = policy_support ~ HS + state,
        data = data
      )
      ps_helper(model_fit, data)
    }),
    label = "No pooling",
    inquiry = "mean_policy_support") +
  declare_estimator(
    handler = label_estimator(function(data) {
      model_fit <- lm_robust(
        formula = policy_support ~ HS,
        data = data
      )
      ps_helper(model_fit, data)
    }),
    label = "Full pooling",
    inquiry = "mean_policy_support")

set.seed(464)
t0 <- Sys.time()
diagnosis_15.4 <- diagnose_design(declaration_15.5, sims = sims, bootstrap_sims = FALSE)
cat("15.4 diagnosis took", round(as.numeric(Sys.time() - t0, units = "mins"), 1), "min\n")
write_diag(diagnosis_15.4, file.path(ref_dir, "rgen_t3_ch15_15_4.json"),
           extra = list(n_sims = sims, seed_note = "states set.seed(464); diagnosis set.seed(464)"))

cat("done\n")
