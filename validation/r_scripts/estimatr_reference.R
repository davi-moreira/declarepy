# Reference values from estimatr/prop.test for declarepy's estimator tests.
ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({library(estimatr); library(jsonlite)})
repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
out <- list()

tidy1 <- function(fit) {
  td <- tidy(fit)
  lapply(split(td, td$term), function(r) as.list(r[1, c("estimate","std.error","statistic","p.value","conf.low","conf.high","df")]))
}

foos <- read.csv(file.path(repo, "src/declarepy/data/foos_etal.csv"))
out$foos_dim <- tidy1(difference_in_means(marked_register_2014 ~ treat, data = foos))
hajj <- read.csv(file.path(repo, "src/declarepy/data/cliningsmith_etal.csv"))
out$hajj_dim <- tidy1(difference_in_means(views ~ success, data = hajj))
out$hajj_lm_hc2 <- tidy1(lm_robust(views ~ success, data = hajj))
lapop <- read.csv(file.path(repo, "src/declarepy/data/lapop_brazil.csv"))
out$lapop_lm_hc2 <- tidy1(lm_robust(trust_police ~ govt_responsive + ideology, data = lapop))
out$lapop_lm_classical <- tidy1(lm_robust(trust_police ~ govt_responsive + ideology, data = lapop, se_type = "classical"))
out$lapop_lm_hc1 <- tidy1(lm_robust(trust_police ~ govt_responsive + ideology, data = lapop, se_type = "stata"))
out$lapop_lm_hc3 <- tidy1(lm_robust(trust_police ~ govt_responsive + ideology, data = lapop, se_type = "HC3"))

blk <- read.csv(file.path(repo, "validation/reference/blocked_fixture.csv"))
out$blocked_dim <- tidy1(difference_in_means(Y ~ Z, blocks = block, data = blk))

pt <- prop.test(45, 100, p = 0.5, correct = TRUE)
out$prop_test_45_100 <- list(estimate = unname(pt$estimate), statistic = unname(pt$statistic),
                             p.value = pt$p.value, conf.low = pt$conf.int[1], conf.high = pt$conf.int[2])
pt2 <- prop.test(3, 10, p = 0.5, correct = TRUE)
out$prop_test_3_10 <- list(estimate = unname(pt2$estimate), statistic = unname(pt2$statistic),
                           p.value = pt2$p.value, conf.low = pt2$conf.int[1], conf.high = pt2$conf.int[2])
# intercept-only lm_robust on a tiny fixed sample (declaration_9.1's shape)
tiny <- data.frame(age = c(12, 47, 71))
out$tiny_lm_intercept <- tidy1(lm_robust(age ~ 1, data = tiny))

write_json(out, file.path(repo, "validation/reference/rgen_estimatr.json"), digits = 12, auto_unbox = TRUE)
cat("written rgen_estimatr.json\n")
