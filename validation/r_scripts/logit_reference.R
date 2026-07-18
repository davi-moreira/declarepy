ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({library(margins); library(broom); library(jsonlite); library(estimatr)})
repo <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy"
foos <- read.csv(file.path(repo, "src/declarepy/data/foos_etal.csv"))
out <- list()
fit <- glm(marked_register_2014 ~ treat, data = foos, family = binomial("logit"))
td <- tidy(fit)
out$foos_logit <- lapply(split(td, td$term), function(r) as.list(r[1, c("estimate","std.error","statistic","p.value")]))
mg <- tidy(margins(fit, data = fit$data), conf.int = TRUE)
out$foos_logit_ame <- lapply(split(mg, mg$term), function(r) as.list(r[1, c("estimate","std.error","statistic","p.value","conf.low","conf.high")]))
out$foos_lm_r2 <- glance(lm_robust(marked_register_2014 ~ treat, data = foos))$r.squared
write_json(out, file.path(repo, "validation/reference/rgen_logit.json"), digits = 12, auto_unbox = TRUE)
cat("written rgen_logit.json\n")
