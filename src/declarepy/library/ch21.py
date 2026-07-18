"""Chapter 21 designs: reporting and assignment-probability sweeps.

Provenance: replication-materials ``code/diagnoses/diagnosis_21a.R`` and
``diagnosis_21b.R`` (Blair, Coppock & Humphreys 2023) — these designs are
declared inline inside the diagnosis scripts (no separate declaration files).
Reference outputs: ``diagnosis_21a.rds`` (prob sweep 1/12 … 11/12, sims=2000)
and ``diagnosis_21b.rds``.
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

__all__ = ["declaration_21a", "declaration_21b"]


def declaration_21a(prob: float = 0.5) -> Design:
    """declaration_21a: two-arm trial swept over assignment probability.

    N=100, U ~ N(0, 0.2), ATE 0.15, no declared inquiry (power/sd only).
    """

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = 0.15 * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=100, build=lambda n, rng: {"U": rng.normal(0, 0.2, n)}, label="model"),
        potential_outcomes(po),
        Assignment.complete(prob=prob),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z"),
    )


def declaration_21b() -> Design:
    """declaration_21b: the basic N=100 / ATE 0.1 trial with SATE inquiry."""

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = 0.1 * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=100, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("SATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", inquiry="SATE"),
    )
