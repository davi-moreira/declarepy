# R_TO_PYTHON_INVENTORY — the rdss/DeclareDesign surface, inventoried

> **Framing (locked decision #4):** full Python parity for `rdss`/DeclareDesign
> is a **separate parallel project**, NOT a course deliverable. The HONR 46400
> notebooks implement RDSS *concepts* directly (numpy/pandas/scipy/statsmodels
> + small inline declare→diagnose→redesign helpers). This directory is the
> parallel project's planning home: what exists in R, what the course actually
> needs, and the roadmap for real parity later.

## Source packages (verified locally)

| Package | Version | License | Role |
|---|---|---|---|
| `rdss` | 1.0.14 (CRAN, 2025-01-09) | MIT | book companion: datasets + helpers; imports dplyr, ggplot2, estimatr, randomizr, broom, purrr, tidyr, dataverse |
| `DeclareDesign` | (suggested by rdss) | MIT | the declare/diagnose/redesign engine |
| `randomizr` | (imported) | GPL-3-ish (verify at parity time) | random assignment/sampling procedures |
| `fabricatr` | (ecosystem) | — | data fabrication for models |
| `estimatr` | (imported) | — | `lm_robust`, `difference_in_means` |
| `DesignLibrary` | (ecosystem) | — | pre-built designs |

## The course-load-bearing surface (16 functions)

Verified against the local replication code (`_adm/_references/book/
replication-materials/code/`), which uses these in the declarations/diagnoses
the course translates:

| # | R function | Package | What it does |
|---|---|---|---|
| 1 | `declare_model()` | DeclareDesign | define units + potential outcomes (M of MIDA) |
| 2 | `potential_outcomes()` | DeclareDesign | PO shorthand inside the model |
| 3 | `declare_inquiry()` | DeclareDesign | define the estimand (I) |
| 4 | `declare_sampling()` | DeclareDesign | sampling step (D) |
| 5 | `declare_assignment()` | DeclareDesign | assignment step (D) |
| 6 | `declare_measurement()` | DeclareDesign | measurement step (D) |
| 7 | `reveal_outcomes()` | DeclareDesign | switch POs by realized assignment |
| 8 | `declare_estimator()` | DeclareDesign | answer strategy (A) |
| 9 | `declare_diagnosands()` | DeclareDesign | bias/power/coverage/etc. definitions |
| 10 | `diagnose_design()` | DeclareDesign | Monte-Carlo diagnosis |
| 11 | `redesign()` | DeclareDesign | parameter-sweep redesign |
| 12 | `run_design()` / `draw_data()` | DeclareDesign | simulate one run / one dataset |
| 13 | `complete_ra()` / `block_ra()` | randomizr | complete/blocked random assignment |
| 14 | `simple_rs()` / `complete_rs()` | randomizr | random sampling procedures |
| 15 | `difference_in_means()` | estimatr | DiM with design-appropriate SEs |
| 16 | `lm_robust()` | estimatr | OLS with robust (HC2) SEs |

Plus one book utility: `make_dag_df()` (`…/code/utilities/make_dag_df.R`) —
DAG layout helper for plotting.

## What the replication archive contains (translation targets, by chapter)

~70 declaration files (`declaration_2.1` … `declaration_23.1d`), ~60 diagnosis
files, ~90 figure scripts. Chapter clusters: 9.x answer strategies (7 files),
10.x–11.x diagnosis/redesign (10), 15.x–16.x observational designs (13), 17.x
experimental descriptive (7), 18.x experimental causal (13 — largest), 19.x
complex designs (4), 23.x integration (4).

## What the COURSE actually uses (the priority set)

| Course use | R source | Course notebook |
|---|---|---|
| inquiry declaration pattern | declaration_7.1 | nb04 |
| sampling declaration | declaration_2.1, ch.8 patterns | nb07 |
| answer-strategy declarations | declaration_9.1 (+9.2–9.7 concepts) | nb09 |
| power simulation | problem-set-1, ch.10 | nb10 |
| declare→diagnose loop | declaration_10.1 + diagnosis_10.1 | nb11 |
| redesign comparison | declaration_11.1 + diagnosis_11.x | nb11 |
| two-arm trial | declaration_18.1 | nb13 |
| DAG helper | make_dag_df.R | nb04 |

Everything else in the archive is **out of course scope** and belongs to the
parity roadmap (`TRANSLATION_ROADMAP.md`).
