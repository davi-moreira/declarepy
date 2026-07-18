"""Chapter 4 declarations: the full MIDA walk-through design.

Provenance: replication-materials ``code/declarations/declaration_4.1.R``
(Blair, Coppock & Humphreys 2023). Reference outputs: ``diagnosis_4.1.rds``
(sims = 100, set.seed(42)) — NOTE the book's saved diagnosis was run with
only 100 simulations, so its own Monte-Carlo error (bootstrap
se(mean_estimate) ≈ 0.031) exceeds the validation protocol's ±0.02·sd(Y)
band. The primary validation reference is therefore a fresh 2000-sim R run
of the unmodified book declaration
(``validation/reference/rgen_t3_ch04_05_07_4.1.json``); the saved 100-sim
diagnosis is checked secondarily with the reference's Monte-Carlo error
added to the band (documented in ``validation/validate_t3_ch04_05_07.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..steps import (
    Assignment,
    Design,
    Estimator,
    Inquiry,
    Model,
    Sampling,
    potential_outcomes,
    reveal_outcomes,
)

__all__ = ["declaration_4_1"]


def declaration_4_1(N: int = 100, n: int = 50, effect: float = 0.25) -> Design:
    """declaration_4.1: sample n of N, complete assignment, DiM on the PATE.

    R source: ``declare_model(N = 100, U = rnorm(N), potential_outcomes(
    Y ~ 0.25 * Z + U))`` + PATE inquiry + ``complete_rs(N, n = 50)`` +
    ``complete_ra(N, prob = 0.5)`` + ``reveal_outcomes`` +
    ``difference_in_means``. The potential outcomes are deterministic given
    ``U``, so the PATE is exactly ``effect`` in every draw; the inquiry
    sits before sampling (a population estimand).
    """

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = effect * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=N, build=lambda n_, rng: {"U": rng.normal(size=n_)}, label="model"),
        potential_outcomes(po),
        Inquiry("PATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Sampling.complete(n=n),
        Assignment.complete(prob=0.5),
        reveal_outcomes(),
        Estimator.difference_in_means(inquiry="PATE"),
    )
