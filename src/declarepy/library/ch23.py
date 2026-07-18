"""Chapter 23 declarations: to control or not to control (reanalysis).

Provenance: replication-materials ``code/declarations/declaration_23.1a.R``
… ``declaration_23.1d.R`` (Blair, Coppock & Humphreys 2023). Reference
output: ``diagnosis_23.1.rds`` (sims = 2000): three N = 100 models crossed
with two estimators — ``A`` (``lm_robust(Y ~ Z)``) and ``A_prime``
(``lm_robust(Y ~ Z + X)``):

* ``design_1`` (23.1a): X is *not* a confounder, measured pretreatment —
  both estimators unbiased, A_prime slightly more precise.
* ``design_2`` (23.1b): X confounds (Z ~ Bernoulli(plogis(0.5 + X))) —
  A biased (≈ +0.21), A_prime unbiased.
* ``design_3`` (23.1d): X is a *posttreatment* consequence of Y and Z —
  A unbiased, A_prime badly biased (≈ −0.11) with coverage ≈ 0.21.

The true ATE is 0.1 throughout; the inquiry is ATE = mean(Y1 − Y0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..steps import (
    Design,
    Estimator,
    Inquiry,
    Measurement,
    Model,
    potential_outcomes,
    reveal_outcomes,
)

__all__ = ["declaration_23_1"]


def _expit(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _po_with_x(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
    """potential_outcomes(Y ~ 0.1 * Z + 0.25 * X + U) — models 1 and 2."""
    result: np.ndarray = (
        0.1 * float(z) + 0.25 * df["X"].to_numpy() + df["U"].to_numpy()  # type: ignore[arg-type]
    )
    return result


def _po_without_x(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
    """potential_outcomes(Y ~ 0.1 * Z + U) — model 3."""
    result: np.ndarray = 0.1 * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
    return result


def _inquiry_and_answers() -> Design:
    """The shared I + A + A_prime tail (declarations 23.1c and 23.1d's I)."""
    return Design(
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Estimator.lm_robust("Y ~ Z", inquiry="ATE", label="A"),
        Estimator.lm_robust("Y ~ Z + X", inquiry="ATE", label="A_prime"),
    )


def _model_1(N: int = 100) -> Design:
    """23.1a: X is not a confounder and is measured pretreatment."""

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        return {
            "U": rng.normal(size=n),
            "X": rng.normal(size=n),
            "Z": rng.binomial(1, _expit(0.5), n),
        }

    return Design(
        Model(n=N, build=build, label="model"),
        potential_outcomes(_po_with_x),
        reveal_outcomes(),
    )


def _model_2(N: int = 100) -> Design:
    """23.1b: X is a confounder (it drives assignment) measured pretreatment."""

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        u = rng.normal(size=n)
        x = rng.normal(size=n)
        p = 1.0 / (1.0 + np.exp(-(0.5 + x)))
        return {"U": u, "X": x, "Z": rng.binomial(1, p)}

    return Design(
        Model(n=N, build=build, label="model"),
        potential_outcomes(_po_with_x),
        reveal_outcomes(),
    )


def _model_3(N: int = 100) -> Design:
    """23.1d: X is not a confounder but is measured *posttreatment*."""

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        return {
            "U": rng.normal(size=n),
            "Z": rng.binomial(1, _expit(0.5), n),
        }

    return Design(
        Model(n=N, build=build, label="model"),
        potential_outcomes(_po_without_x),
        reveal_outcomes(),
        Measurement(
            lambda df, rng: {
                "X": 0.1 * df["Z"].to_numpy() + 5.0 * df["Y"].to_numpy() + rng.normal(size=len(df))
            },
            label="posttreatment_X",
        ),
    )


def declaration_23_1(N: int = 100) -> dict[str, Design]:
    """declaration_23.1: the three-design controlling-for-X comparison.

    Returns ``{"design_1": …, "design_2": …, "design_3": …}`` mirroring the
    R source's ``list(design_1, design_2, design_3)``; pass the dict to
    :func:`declarepy.diagnose_all` to reproduce ``diagnosis_23.1``.
    """
    return {
        "design_1": _model_1(N) + _inquiry_and_answers(),
        "design_2": _model_2(N) + _inquiry_and_answers(),
        "design_3": _model_3(N) + _inquiry_and_answers(),
    }
