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

Load them via `declarepy.data.load(<name>)`.
