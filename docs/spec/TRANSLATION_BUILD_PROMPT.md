# DECLAREPY BUILD PROMPT — the parallel R→Python parity project (read, then execute)

> This is the **parallel project**, NOT the course. It builds `declarepy` — a real, tested
> Python translation of the `rdss` / DeclareDesign instructional ecosystem — in its **OWN
> separate repository**. The Fall 2026 course repo is **read-only reference** and must
> never be modified. The `translation/` docs in the course repo are the spec; the course
> notebooks are the reference implementations. Reading this file = executing it. Don't ask
> questions unless genuinely blocked; make documented assumptions.

Course repo (read-only reference), referred to below as **$COURSE**:
`/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/2026F_evidence_driven_research_purdue_HONR464`

## STEP 0 — READ FIRST (the spec is already written, in $COURSE/translation/)
- `$COURSE/translation/R_TO_PYTHON_INVENTORY.md` — the 16-function load-bearing surface + what each R source contains, by chapter.
- `$COURSE/translation/API_MAPPING.md` — **the canonical Python reference implementations** (the inline helpers the course ships); `declarepy` extracts and generalizes these.
- `$COURSE/translation/SEMANTIC_DIFFERENCES.md` — the 10 R↔Python traps (RNG non-portability, NA semantics, HC2 default, tidy-eval, …); honor every stated **Convention**.
- `$COURSE/translation/VALIDATION_REPORT.md` — the tolerance-based validation protocol (5 checks) + the status ledger.
- `$COURSE/translation/TRANSLATION_ROADMAP.md` — the 6 tranches (T0 done → T5 release), acceptance gates, non-goals, risks.
- `$COURSE/translation/PARITY_MATRIX.csv` — the per-function status table (columns: r_function, r_package, python_target, status, course_notebook, course_date, priority, notes).
- `$COURSE/course_config.yaml` + `$COURSE/CLAUDE.md` — conventions, seed `464`, MIT + attribution to Blair, Coppock & Humphreys.

Reference material (read in place, NEVER modify or copy the private archive):
- R source: `$COURSE/_adm/_references/book/replication-materials/code/{declarations,diagnoses,figures,utilities}/*.R`; saved diagnosis objects `.../diagnosis_objects/*.rds`; problem sets `.../exercises/*.Rmd`.
- Reference Python impls: `$COURSE/notebooks/instructor/nb0{4,7,9}_*.ipynb`, `nb1{0,1,3}_*.ipynb`.
- Small MIT datasets: `$COURSE/notebooks/data/*.csv` — copy the ones you need into the new repo (attribution required); do NOT touch `_adm/`.

## STEP 0.5 — BOOTSTRAP THE SEPARATE REPO (do this first)
1. Create the project OUTSIDE the course repo. Suggested sibling path (adjust if you prefer):
   `/Users/dcordeir/Dropbox/academic/cursos/cursos-davi/evidence_based_research/declarepy`
   (so `$COURSE` is reachable as `../2026F_evidence_driven_research_purdue_HONR464`).
2. `git init` + `git branch -M main`. Create a **PUBLIC** GitHub repo:
   `gh repo create davi-moreira/declarepy --public --source=. --remote=origin` (push later).
   *(Per the professor's decision the repo is public as `davi-moreira/declarepy`. Because it
   is public from the first commit, keep it release-quality from day one: a real README, MIT
   LICENSE + attribution, and NO private material — it only READS the course repo's `_adm/`
   archive, never copies it. Still coordinate the PyPI/distribution package name with the
   DeclareDesign authors before publishing a release — T5.)*
3. Copy the six `$COURSE/translation/*` spec docs (+ this prompt) into the new repo's
   `docs/spec/` as its planning home, and add a top-level `README.md` and `LICENSE` (MIT,
   attribution to Blair, Coppock & Humphreys; note `randomizr`/`estimatr` upstream licenses).
4. Create `pyproject.toml` (PEP 621): package `declarepy` under `src/declarepy/`, with
   `tests/`, `examples/`, `docs/`, typed public interfaces where practical, docstrings,
   `pytest` + `mypy` config, reproducible seeds, MIT license, authors/attribution.
5. Create the repo's own `.venv` and install numpy/pandas/scipy/statsmodels/scikit-learn/
   networkx + dev tools (`pytest`, `mypy`, `build`; `plotnine` only if chosen at T4).

## MISSION
Build `declarepy` — a transparent, tested Python translation of the MIDA
declare → diagnose → redesign engine + estimators, assignment/sampling procedures, the
declaration library, and the diagnosis objects the book uses — validated against the
book's published/reference outputs within tolerance.

## GUARDRAILS (do not violate)
1. **Isolation = a separate repo.** ALL work happens in the new `declarepy` repo. NEVER
   modify anything under `$COURSE` (notebooks, planning, schedule.qmd, docs, the private
   `_adm/` archive) — it is read-only reference. Never run `quarto render` in the course repo.
2. **Transparent core.** Design steps take **explicit callables and column names** — no
   tidy-eval / quosure emulation (SEMANTIC_DIFFERENCES §10). Explicit `Y0`/`Y1`
   potential-outcome columns and printed shape/NA checks carry into the API.
3. **Validated, not assumed.** RNG streams are NOT portable (§1) — "parity" means
   **statistical agreement within tolerance**, never digit equality. Every element runs
   the VALIDATION_REPORT protocol before it is marked validated.
4. **MIT + attribution** throughout; carry a LICENSE/citation notice to Blair, Coppock &
   Humphreys and note upstream `randomizr`/`estimatr` licenses.
5. **Public repo, release-quality hygiene.** The repo is public as `davi-moreira/declarepy`
   (professor's decision) — never commit the private `_adm/` archive or any course-repo
   content; ship only MIT-licensed datasets with attribution. Coordinate the PyPI/
   distribution package name with the DeclareDesign authors before a release (T5).

## ENVIRONMENT
- **R is available locally** (`/usr/local/bin/Rscript`). Use it to generate **reference
  outputs** for validation: install `rdss` + `DeclareDesign` + `estimatr` + `randomizr`
  from CRAN, run the book's declaration/diagnosis `.R` files (from `$COURSE/_adm/...`),
  and emit structured JSON/CSV the Python harness compares against. Read the book's saved
  `diagnosis_objects/*.rds` via `Rscript -e '... jsonlite::write_json(...)'` as ground
  truth. `rpy2` is allowed as an optional dev/validation aid — never a runtime dependency.

## EXECUTE — tranche by tranche (autonomous; stop only on an unresolvable validation gate)
Maintain the repo's own copy of `PARITY_MATRIX.csv` + `VALIDATION_REPORT.md` (seed the
originals from $COURSE, then evolve them here). Update after each element; commit per element/tranche.

**T1 — core engine (`declarepy`), the prereq for everything.** Extract the API_MAPPING
helpers into composable step objects: `Model` (+ `potential_outcomes`), `Inquiry`,
`Sampling`, `Assignment`, `Measurement` (+ `reveal_outcomes`), `Estimator`, `Diagnosands`,
`diagnose()`, `redesign()`, `run_design()`/`draw_data()`, and procedures
`complete_ra`/`block_ra`/`simple_rs`/`complete_rs`. **Acceptance:** reproduces the course
notebooks' outputs exactly (same seed 464) AND recovers declaration_2.1 / 9.1 / 10.1 /
11.1 / 18.1 diagnosands within tolerance (bias ±0.02·sd(Y), power ±0.05, coverage ±0.03,
RMSE ±10%).

**T2 — estimator fidelity.** Design-based SEs matching `estimatr` (HC1/HC2/CR2, blocked &
clustered difference-in-means) via statsmodels/linearmodels. **Acceptance:** problem-set
answer keys 1–4 reproduce within tolerance; HC2 default (SEMANTIC_DIFFERENCES §5).

**T3 — declaration library (course chapters first).** Translate replication declarations
in course-priority order: ch.9 (7) → ch.10–11 (10) → ch.18 (13) → ch.15–16 (13) → ch.17
(7) → ch.19+23 (8). Each: translation + validation row + provenance line.

**T4 — diagnosis objects + figures.** Recompute the book's `diagnosis_objects/*.rds` in
Python within tolerance; re-implement priority figures (matplotlib vs plotnine — decide at
tranche start; plotnine is a parity-project choice only).

**T5 — polish & release.** `docs/`, examples, CI matrix, a "for DeclareDesign users"
migration guide, packaging. Coordinate naming with the DeclareDesign authors before any
public release.

**Non-goals:** tidy-eval/quosure emulation; the `dataverse` download client (ship CSVs);
`DesignLibrary` breadth before T5.

## VALIDATION PROTOCOL (from VALIDATION_REPORT.md — apply to every element)
1. Structural check (same M/I/D/A steps as the R source).
2. Diagnosand tolerance check (≥1,000 reps; bands above).
3. Known-truth check (analytic estimand recovered as reps→large).
4. Real-data check (shipped datasets, e.g. foos_etal DiM, lapop_brazil summaries — exact
   to 3 decimals; real data has no RNG excuse).
5. Record every validated element in the ledger with seed, reps, deltas. Log any tolerance
   failure with its investigation before marking the element validated.

## WORKING DISCIPLINE
- All commits/pushes go to the **`declarepy` repo** (`main`). Never commit to or push the
  course repo. Stage files **by name**; never `git add .`.
- Commit messages `<type>: <subject>` (feat|fix|test|docs|build) + trailing
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `pytest` + `mypy` green before each commit; keep the tree clean.
- Optionally track progress on a private issue (its own `declarepy-tasks` companion, or a
  section of the course tracker).

## DEFINITION OF DONE (do NOT claim "complete" until all hold — brief §16)
- Every in-scope exported function AND dataset has a `PARITY_MATRIX.csv` entry.
- T1 acceptance is proven (the course notebooks *could* run on `declarepy`), though the
  notebooks stay on their inline helpers this term.
- Tests pass; numerical tolerances are justified and recorded.
- Semantic differences documented; every exclusion justified in the roadmap.
- `VALIDATION_REPORT.md` shows a validated row for every translated element.
Report status honestly per tranche; label "conceptual parity" vs "exact parity" distinctly.
When all tranches are green, summarize and stop.
