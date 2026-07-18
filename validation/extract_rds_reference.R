# Export the book's saved diagnosis objects (.rds) to JSON reference files.
# Reads the course repo's private archive IN PLACE (never copied); only the
# derived diagnosand summaries are written into declarepy/validation/reference/.
suppressMessages({library(jsonlite)})
ul <- Sys.getenv("R_LIBS_USER"); if (nzchar(ul)) .libPaths(c(ul, .libPaths()))
suppressMessages({library(DeclareDesign)})

src <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/2026F_evidence_driven_research_purdue_HONR464/_adm/_references/book/replication-materials/diagnosis_objects"
out <- "/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy/validation/reference"
dir.create(out, showWarnings = FALSE, recursive = TRUE)

files <- list.files(src, pattern = "\\.rds$", full.names = TRUE)
for (f in files) {
  nm <- sub("\\.rds$", "", basename(f))
  obj <- tryCatch(readRDS(f), error = function(e) NULL)
  if (is.null(obj)) { cat("SKIP (read error):", nm, "\n"); next }
  res <- NULL
  if (inherits(obj, "diagnosis")) {
    res <- list(kind = "diagnosis",
                diagnosands = as.data.frame(obj$diagnosands_df),
                n_sims = if (!is.null(obj$simulations_df)) nrow(obj$simulations_df) else NA)
  } else if (is.data.frame(obj)) {
    res <- list(kind = "data.frame", data = as.data.frame(obj))
  } else if (is.list(obj)) {
    dd <- tryCatch(obj$diagnosands_df, error = function(e) NULL)
    if (!is.null(dd)) {
      res <- list(kind = "diagnosis-like", diagnosands = as.data.frame(dd))
    } else {
      res <- list(kind = paste(class(obj), collapse = ","),
                  summary = tryCatch(capture.output(str(obj, max.level = 1)), error = function(e) "unreadable"))
    }
  }
  write_json(res, file.path(out, paste0(nm, ".json")), digits = 10, na = "null", auto_unbox = TRUE)
  cat("OK:", nm, "->", res$kind, "\n")
}
