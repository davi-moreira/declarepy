"""Chapter 10 declarations: diagnosis.

Provenance: replication-materials ``code/declarations/declaration_10.1.R``
(Blair, Coppock & Humphreys 2023). Reference outputs:
``diagnosis_10.1.rds`` (power = 0.1605 from a 2000-rep hand-rolled loop
with classical OLS p-values) and, for the full diagnosand row, the
structurally identical ``diagnosis_18.1.rds``.
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

__all__ = ["declaration_10_1"]


def declaration_10_1(N: int = 100, effect: float = 0.2) -> Design:
    """declaration_10.1: the canonical two-arm ATE design (N=100, ATE=0.2)."""

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = effect * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=N, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(),
        reveal_outcomes(),
        Estimator.difference_in_means(inquiry="ATE"),
    )
