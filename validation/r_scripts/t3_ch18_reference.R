# T3 ch18 reference generation (run from anywhere; absolute paths below).
#
# Produces:
#   1. validation/reference/rgen_t3_ch18_fixtures.json
#      Exact estimatr outputs (estimate/SE/df/p/CI) on small fixed datasets for
#      every private helper ch18.py implements: lm_lin (plain + weighted),
#      iv_robust (2SLS, HC2), TWFE lm_robust with fixed effects + CR2 clusters,
#      blocked difference_in_means (5/15 per block), clustered
#      difference_in_means on a string condition. The fixture DATA are embedded
#      in the JSON so the Python validator refits on identical numbers.
#   2. validation/reference/rgen_t3_ch18_18.4.json
#      Fresh sims = 2000 diagnosis of the *unmodified* book declaration_18.4:
#      the book's saved diagnosis_18.4.rds is the N = 12 permutation exercise
#      of the chapter text, NOT a diagnosis of declaration_18.4, so a fresh
#      reference is required by the validation protocol.
#   3. src/declarepy/data/fixed_pop_18_6.csv
#      The fixed population declaration_18.6 conditions on, generated exactly
#      as in the book's declaration file (set.seed(343) + fabricate) — the
#      estimands are constants of this exact draw, so parity requires it.
#   4. src/declarepy/data/fairfax.csv + fairfax_adjacency.csv
#      Attribute columns of rdss::fairfax (MIT license) and a queen-contiguity
#      0/1 adjacency matrix computed from shared polygon vertices (sf/spdep are
#      not installed here; shared-vertex contiguity is poly2nb(queen = TRUE)'s
#      criterion of "one or more shared boundary points" applied to vertices).
#   5. validation/reference/rgen_t3_ch18_18.10_placebo_c03.json
#      Fresh sims = 10000 R rerun of the single diagnosis_18.10_placebo cell
#      at compliance_rate = 0.3 (declaration_18.9c). The book's saved
#      sims = 2000 cell has mean_estimate = 0.51006 (bootstrap SE 0.00582),
#      +1.7 of its own SEs above the true value 0.5 (CACE is exactly 0.5 by
#      construction and the complier-subset OLS is unbiased under complete
#      RA); the declarepy sims = 2000 seed-464 draw landed low (0.48445),
#      producing a spurious 3.1-combined-sigma tolerance failure. At
#      sims = 10000 both sides converge (R 0.50434, Python 0.49743 seed 464 /
#      0.50285 seed 2026; delta 1.9 combined MC-SEs) — reference-side MC
#      noise, not a translation bug. validate_t3_ch18.py checks this one
#      cell against the fresh high-precision reference instead.

ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({
  library(DeclareDesign)
  library(estimatr)
  library(jsonlite)
})

decl <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book",
  "replication-materials/code/declarations"
)
repo <- file.path(
  "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research",
  "declarepy"
)
ref <- file.path(repo, "validation", "reference")
datadir <- file.path(repo, "src", "declarepy", "data")

# ---------------------------------------------------------------- fixtures --
tidy_row <- function(fit, term) {
  t <- tidy(fit)
  r <- t[t$term == term, ]
  list(term = term, estimate = r$estimate, std_error = r$std.error,
       df = r$df, p_value = r$p.value, conf_low = r$conf.low,
       conf_high = r$conf.high)
}

set.seed(99)
N <- 40
fx <- data.frame(
  X = rnorm(N), U = rnorm(N),
  Z = rep(c(0, 1), each = N / 2),
  w = runif(N, 0.5, 2)
)
fx$Y <- 0.3 * fx$Z + 0.5 * fx$X + fx$U
set.seed(100)
fx$Zi <- rep(c(0, 1), 20)
fx$D <- ifelse(fx$Zi == 1 & fx$X > -0.5, 1, 0)
fx$Y2 <- 0.4 * fx$D + fx$U

set.seed(7)
n_units <- 10; n_per <- 3
tw <- expand.grid(unit = 1:n_units, period = 1:n_per)
wave <- sample(rep(1:3, length.out = n_units))
tw$wave <- wave[tw$unit]
tw$Z <- as.integer(tw$period >= tw$wave)
tw$Y <- rnorm(n_units)[tw$unit] + rnorm(n_per)[tw$period] + 0.4 * tw$Z +
  rnorm(nrow(tw))
tw <- tw[tw$period < 3, ]

set.seed(101)
bl <- data.frame(b = rep(1:3, each = 20))
bl$Z <- unlist(lapply(1:3, function(i) sample(rep(c(1, 0), c(5, 15)))))
bl$Y <- rnorm(60) + 0.3 * bl$Z

set.seed(102)
cd <- data.frame(g = rep(1:10, each = 4))
cd$S <- rep(sample(c("low", "high"), 10, TRUE), each = 4)
cd$Y <- rnorm(40) + 0.4 * (cd$S == "high")

fixtures <- list(
  data = fx,
  twfe_data = tw,
  blocked_data = bl,
  clustered_data = cd,
  expected = list(
    lm_lin = tidy_row(lm_lin(Y ~ Z, covariates = ~X, data = fx), "Z"),
    lm_lin_weighted = tidy_row(
      lm_lin(Y ~ Z, covariates = ~X, weights = w, data = fx), "Z"),
    iv_robust = tidy_row(iv_robust(Y2 ~ D | Zi, data = fx), "D"),
    twfe_cr2 = tidy_row(
      lm_robust(Y ~ Z, fixed_effects = ~ unit + period, clusters = unit,
                data = tw), "Z"),
    dim_blocked = tidy_row(
      difference_in_means(Y ~ Z, blocks = b, data = bl), "Z"),
    dim_clustered = tidy_row(
      difference_in_means(Y ~ S, condition1 = "low", condition2 = "high",
                          clusters = g, data = cd), "Shigh")
  )
)
write_json(fixtures, file.path(ref, "rgen_t3_ch18_fixtures.json"),
           digits = 12, dataframe = "columns")
cat("written rgen_t3_ch18_fixtures.json\n")

# ------------------------------------------------- declaration_18.4 fresh ---
source(file.path(decl, "declaration_18.4.R"))
set.seed(464)
d184 <- diagnose_design(declaration_18.4, sims = 2000)
write_json(as.data.frame(get_diagnosands(d184)),
           file.path(ref, "rgen_t3_ch18_18.4.json"), digits = 12)
cat("written rgen_t3_ch18_18.4.json\n")

# ----------------------- 18.10 placebo c=0.3 noise check (header item 5) ----
suppressMessages(library(dplyr))  # case_when/if_else in declaration_18.9a
source(file.path(decl, "declaration_18.9a.R"))  # MI (compliance_rate = 0.2)
source(file.path(decl, "declaration_18.9c.R"))  # declaration_18.9_placebo
set.seed(464)
d1810p <- redesign(declaration_18.9_placebo, compliance_rate = 0.3) |>
  diagnose_designs(sims = 10000, bootstrap_sims = 500)
write_json(
  list(kind = "noise-check", n_sims = 10000, bootstrap_sims = 500,
       note = paste("fresh R rerun of the diagnosis_18.10_placebo cell",
                    "compliance_rate=0.3; the saved sims=2000 reference cell",
                    "is a +1.7-SE MC outlier (see header item 5)"),
       diagnosands = as.data.frame(get_diagnosands(d1810p))),
  file.path(ref, "rgen_t3_ch18_18.10_placebo_c03.json"),
  digits = 12, auto_unbox = TRUE
)
cat("written rgen_t3_ch18_18.10_placebo_c03.json\n")

# --------------------------------------------- fixed population for 18.6 ----
# Verbatim from declaration_18.6.R (book replication materials).
set.seed(343)
fixed_pop <-
  fabricate(N = 10000,
            X = rbinom(N, 1, 0.2),
            potential_outcomes(
              Y ~ rbinom(N, 1,
                         prob = 0.7 + 0.1 * Z - 0.4 * X - 0.2 * Z * X))
  )
write.csv(fixed_pop, file.path(datadir, "fixed_pop_18_6.csv"),
          row.names = FALSE)
cat("written fixed_pop_18_6.csv\n")

# ------------------------------------------------------- fairfax exports ----
suppressMessages(library(rdss))
data(fairfax, package = "rdss")
geom <- fairfax$geometry
# sf is not installed, so the sfc list-column cannot go through
# as.data.frame(); extract the attribute columns by name instead.
attr_names <- setdiff(names(fairfax), "geometry")
attrs <- as.data.frame(lapply(attr_names, function(nm) fairfax[[nm]]))
names(attrs) <- attr_names
write.csv(attrs, file.path(datadir, "fairfax.csv"), row.names = FALSE)
cat("written fairfax.csv  (", nrow(attrs), "rows )\n")

# Vertex extraction from the sfc list-column without sf: recursively collect
# every coordinate matrix of each (MULTI)POLYGON.
collect_coords <- function(g) {
  if (is.matrix(g)) return(list(g))
  if (is.list(g)) return(do.call(c, lapply(g, collect_coords)))
  list()
}
n <- length(geom)
vertex_keys <- vector("list", n)
for (i in seq_len(n)) {
  mats <- collect_coords(unclass(geom[[i]]))
  xy <- do.call(rbind, mats)
  vertex_keys[[i]] <- unique(paste(round(xy[, 1], 4), round(xy[, 2], 4)))
}
adj <- matrix(0L, n, n)
vertex_map <- new.env(hash = TRUE)
for (i in seq_len(n)) {
  for (key in vertex_keys[[i]]) {
    ids <- vertex_map[[key]]
    if (is.null(ids)) assign(key, i, envir = vertex_map)
    else assign(key, c(ids, i), envir = vertex_map)
  }
}
for (key in ls(vertex_map)) {
  ids <- unique(vertex_map[[key]])
  if (length(ids) > 1) {
    for (a in ids) for (b in ids) if (a != b) adj[a, b] <- 1L
  }
}
cat("queen adjacency: n =", n, " edges =", sum(adj) / 2,
    " isolates =", sum(rowSums(adj) == 0), "\n")
write.table(adj, file.path(datadir, "fairfax_adjacency.csv"),
            sep = ",", row.names = FALSE, col.names = FALSE)
cat("written fairfax_adjacency.csv\n")
