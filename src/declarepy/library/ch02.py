"""Chapter 2 declarations: the running two-arm example, plain and blocked.

Provenance: replication-materials ``code/declarations/declaration_2.1.R``
and ``declaration_2.2.R`` (Blair, Coppock & Humphreys 2023). Reference
outputs: ``diagnosis_2.1.rds`` (success/failure diagnosands over a
``b`` grid for both designs, sims = 2000).
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

__all__ = ["declaration_2_1", "declaration_2_2"]


def _model_and_inquiry(b: float) -> Design:
    """N=1000 world with a random per-draw treatment effect ~ U(0, 0.5).

    R source: ``potential_outcomes(Y ~ b * history + runif(1, 0, 0.5) * Z +
    rnorm(N))``. As in DeclareDesign, the formula is evaluated once per
    condition, so the uniform effect and the normal noise are drawn
    independently for Z=0 and Z=1 within each simulation.
    """

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        n = len(df)
        effect = rng.uniform(0.0, 0.5)
        result: np.ndarray = (
            b * df["history"].to_numpy() + effect * float(z) + rng.normal(size=n)  # type: ignore[arg-type]
        )
        return result

    return Design(
        Model(
            n=1000,
            build=lambda n, rng: {"history": rng.integers(0, 2, n)},
            label="model",
        ),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
    )


def declaration_2_1(b: float = 0.0) -> Design:
    """declaration_2.1: sample 150 of 1000, complete assignment, DiM."""
    return _model_and_inquiry(b) + Design(
        Sampling.complete(n=150),
        Assignment.complete(),
        reveal_outcomes(),
        Estimator.difference_in_means(inquiry="ATE"),
    )


def declaration_2_2(b: float = 0.0) -> Design:
    """declaration_2.2: sample 100, block on history, blocked DiM."""
    return _model_and_inquiry(b) + Design(
        Sampling.complete(n=100),
        Assignment.block(blocks="history"),
        reveal_outcomes(),
        Estimator.difference_in_means(blocks="history", inquiry="ATE"),
    )
