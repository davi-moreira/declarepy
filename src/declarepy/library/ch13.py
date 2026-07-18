"""Chapter 13 declarations: the two-arm trial, plain and with a covariate.

Provenance: replication-materials ``code/declarations/declaration_13.1.R``
and ``declaration_13.2.R`` (Blair, Coppock & Humphreys 2023). Reference
outputs: ``diagnosis_13.1.rds`` diagnoses **declaration_13.2** (sims =
2000, default diagnosands; DIM power ≈ 0.29, OLS power ≈ 0.87);
declaration_13.1 has no saved diagnosis object, so it is validated against
a freshly generated R reference
(``validation/reference/rgen_t3_ch12_13_13_1.json``, sims = 2000).
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

__all__ = ["declaration_13_1", "declaration_13_2"]


def declaration_13_1(N: int = 100) -> Design:
    """declaration_13.1: N=100 two-arm trial, complete assignment, lm_robust OLS."""

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = 0.2 * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=N, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        # complete_ra(N): randomizr's default half-of-N assignment.
        Assignment.complete(),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", term="Z", inquiry="ATE", label="OLS"),
    )


def declaration_13_2(N: int = 1000) -> Design:
    """declaration_13.2: sampled two-arm trial with a covariate-adjusted arm.

    ``X = U + rnorm(N, sd = 0.5)`` is a pre-treatment covariate correlated
    with the disturbance; ``DIM`` is ``declare_estimator(Y ~ Z)`` (no
    ``.method``, i.e. DeclareDesign's default lm_robust/HC2 — numerically
    the Welch difference in means) and ``OLS`` adjusts for ``X``. The ATE
    inquiry sits before sampling, so it is the population estimand.
    """

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        U = rng.normal(size=n)
        return {"U": U, "X": U + rng.normal(0.0, 0.5, n)}

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = 0.2 * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=N, build=build, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Sampling.simple(prob=0.2),
        Assignment.complete(),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", inquiry="ATE", label="DIM"),
        Estimator.lm_robust("Y ~ Z + X", term="Z", inquiry="ATE", label="OLS"),
    )
