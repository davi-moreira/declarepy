"""Chapter 11 declarations: redesign.

Provenance: replication-materials ``code/declarations/declaration_11.1.R``
(Blair, Coppock & Humphreys 2023). Reference outputs:
``diagnosis_11.1.rds`` — the design redesigned over N = 100..1000 (step
100), sims = 2000 per N.
"""

from __future__ import annotations

import pandas as pd

from ..estimators import prop_test
from ..steps import Design, Estimator, Measurement, Model

__all__ = ["declaration_11_1"]


def declaration_11_1(N: int = 100) -> Design:
    """declaration_11.1: is a 55% coin fair? prop.test against p = 0.5.

    R source uses ``declare_test`` with ``prop.test(x = table(data$Y),
    p = 0.5)``: R's ``table`` puts the count of ``Y == 0`` first, so the
    tested proportion is the share of zeros (mean estimate ≈ 0.45).
    """

    def test(df: pd.DataFrame) -> pd.DataFrame:
        x = int((df["Y"] == 0).sum())
        return prop_test(x, len(df), p=0.5, correct=True).to_frame()

    return Design(
        Model(n=N, label="model"),
        Measurement(
            lambda df, rng: {"Y": rng.binomial(1, 0.55, len(df))},
            label="measurement",
        ),
        Estimator(test, label="test"),
    )
