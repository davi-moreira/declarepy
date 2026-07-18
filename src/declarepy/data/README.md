# Bundled datasets — provenance and attribution

All five CSVs originate from the **`rdss` R package v1.0.14** (Blair, Coppock &
Humphreys, **MIT License**), the companion package to *Research Design in the
Social Sciences: Declaration, Diagnosis, and Redesign* (Princeton University
Press, 2023; free online at book.declaredesign.org). Redistribution with
attribution is permitted by the MIT license. Files are byte-identical to the
book's replication archive.

> **Attribution line:** *Dataset from the `rdss` package (Blair, Coppock &
> Humphreys, MIT License), companion to* Research Design in the Social
> Sciences *(2023).*

| File | Shape | What it is |
|---|---|---|
| `lapop_brazil.csv` | 10,000 × 10 | AmericasBarometer (LAPOP) Brazil survey items — a 10,000-row resample **with replacement** of the original, suitable for planning/teaching, NOT for substantive research claims (per the package documentation) |
| `la_voter_file.csv` | 1,000 × 4 | Los Angeles voter-file extract (party, age, census tract, 2012 turnout) |
| `foos_etal.csv` | 8,375 × 5 | Foos et al. UK get-out-the-vote field experiment replication (treatment, 2014 turnout, ward/street, weights) |
| `cliningsmith_etal.csv` | 958 × 8 | Clingingsmith, Khwaja & Kremer Hajj-lottery study replication (lottery success, views toward other groups) |
| `bonilla_tillery.csv` | 849 × 10 | Bonilla & Tillery survey experiment replication (treatment `Z`, BLM support, linked fate, demographics) |
| `fairfax.csv` | 238 × k | Attribute columns of `rdss::fairfax` (Fairfax County precincts; the spatial geometry is not shipped) |

## Derived / seeded simulation exports (not rdss data)

These support specific declaration-library translations; none is an rdss
dataset. Each is fully regenerable from the noted source.

| File | What it is |
|---|---|
| `fairfax_adjacency.csv` | 238×238 queen-contiguity 0/1 matrix of the fairfax precincts, computed by the shared-vertex rule (675 edges, 0 isolates) in `validation/r_scripts/t3_ch18_reference.R`; validated end-to-end by declaration 18.13's diagnosand parity |
| `fixed_pop_18_6.csv` | The fixed population declaration 18.6 conditions on — regenerated verbatim by the book's own `set.seed(343)` fabricate code from the public replication materials |
| `ch15_portola.csv` | `fabricate(N = 2100, Y_star = rnorm(N))` under the book's own `set.seed(343)` — the population behind the book's saved diagnosis 15.1/15.2 |
| `ch15_states.csv` | Declaration 15.4's state frame, derived from base-R `state.x77` (public-domain 1977 US census figures) plus a state-shock draw canon-fixed with `set.seed(464)` (the book draws it seedless) |
| `ch15_two_nigerian_states.csv` | Declaration 15.3's two-state hierarchy (2 × 500 localities × 100 individuals, ICC 0.4), canon-fixed with `set.seed(464)` because the book draws it seedless |

Load any of them via `declarepy.data.load(<name>)`.
