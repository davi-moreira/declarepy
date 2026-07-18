"""Chapter 18 declarations: experimental causal designs.

Provenance: replication-materials ``code/declarations/declaration_18.1.R``
(Blair, Coppock & Humphreys 2023). Reference outputs:
``diagnosis_18.1.rds`` (sims = 2000): mean_estimand 0.2, bias ≈ −0.005,
sd_estimate ≈ 0.202, rmse ≈ 0.202, power ≈ 0.167, coverage ≈ 0.948.
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
    potential_outcomes,
    reveal_outcomes,
)

__all__ = ["declaration_18_1"]


def declaration_18_1(N: int = 100, effect: float = 0.2) -> Design:
    """declaration_18.1: the basic two-arm randomized trial (prob=0.5)."""

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = effect * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=N, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(prob=0.5),
        reveal_outcomes(),
        Estimator.difference_in_means(inquiry="ATE"),
    )
