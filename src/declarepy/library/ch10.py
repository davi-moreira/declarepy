"""Chapter 10 declarations: diagnosis.

Provenance: replication-materials ``code/declarations/declaration_10.1.R``,
``declaration_10.2.R``, ``declaration_10.3.R`` (which defines 10.3a and
10.3b), ``declaration_10.4.R`` and ``declaration_10a.R`` (Blair, Coppock &
Humphreys 2023). Reference outputs:

* ``diagnosis_10.1.rds`` — power of 10.1 (0.1605 from a 2000-rep
  hand-rolled loop with classical OLS p-values); the full default row is
  cross-checked against the structurally identical ``diagnosis_18.1.rds``.
* ``diagnosis_10.2.rds`` / ``diagnosis_10.3.rds`` — custom diagnosands
  (power; bias/true_se/power/coverage) of declaration 10.1.
* ``diagnosis_10.4.rds`` — default diagnosands of declaration 10.2.
* ``diagnosis_10.5.rds`` — declarations 10.3a and 10.3b side by side.
* ``diagnosis_10a.rds`` — ``var_estimate`` (pop. variance of estimates) and
  ``mean_var_hat`` (mean squared SE) for declaration 10a over the
  heteroskedasticity × prob_treated sweep.
* declaration 10.4 has no saved diagnosis object; it is validated against a
  fresh R run (``validation/reference/rgen_t3_ch10_11_decl_10.4.json``).
"""

from __future__ import annotations

from typing import Callable

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

__all__ = [
    "declaration_10_1",
    "declaration_10_2",
    "declaration_10_3a",
    "declaration_10_3b",
    "declaration_10_4",
    "declaration_10a",
]

#: DeclareDesign's potential-outcome column naming, used where the R source
#: has several outcomes (Y1/Y2) or declares Y_Z_0/Y_Z_1 columns directly.
_DD_TEMPLATE = "{outcome}_Z_{condition}"


def declaration_10_1(N: int = 100, effect: float = 0.2) -> Design:
    """declaration_10.1: the canonical two-arm ATE design (N=100, ATE=0.2)."""

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = effect * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    # declare_estimator(Y ~ Z, inquiry = "ATE") with no .method uses
    # DeclareDesign's default, lm_robust (HC2) — numerically the Welch SE
    # for a two-group regression, with df = N − 2.
    return Design(
        Model(n=N, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", inquiry="ATE"),
    )


def declaration_10_2(N: int = 200) -> Design:
    """declaration_10.2: a two-arm trial whose true ATE is a U(0, 0.5) draw.

    R source: ``potential_outcomes(Y ~ runif(n = 1, min = 0, max = 0.5) * Z
    + U)`` — one random ATE per simulation. As in DeclareDesign, the PO
    formula is evaluated once per condition, so the uniform draw enters only
    the Z = 1 branch (the Z = 0 branch multiplies its own draw by zero).
    """

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        effect = rng.uniform(0.0, 0.5)
        result: np.ndarray = effect * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=N, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(prob=0.5),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", inquiry="ATE"),
    )


def _effect_po(
    effect: float,
) -> Callable[[pd.DataFrame, object, np.random.Generator], np.ndarray]:
    """PO function ``effect · Z + U`` (both conditions share the same U)."""

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = effect * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return po


def _declaration_10_3(effect_y1: float, effect_y2: float) -> Design:
    """Shared body of declarations 10.3a/10.3b: two outcomes, two inquiries.

    R source: M1/M2 declare ``potential_outcomes(Y1 ~ ...)`` and
    ``potential_outcomes(Y2 ~ ...)`` on the same U, then IDA adds ATE1/ATE2
    inquiries, complete assignment, both reveals, and one lm_robust
    (DeclareDesign's default method) per outcome.
    """
    return Design(
        Model(n=200, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(_effect_po(effect_y1), outcome="Y1", template=_DD_TEMPLATE),
        potential_outcomes(_effect_po(effect_y2), outcome="Y2", template=_DD_TEMPLATE),
        Inquiry(
            ATE1=lambda df: float((df["Y1_Z_1"] - df["Y1_Z_0"]).mean()),
            ATE2=lambda df: float((df["Y2_Z_1"] - df["Y2_Z_0"]).mean()),
        ),
        Assignment.complete(),
        reveal_outcomes(outcome="Y1", template=_DD_TEMPLATE),
        reveal_outcomes(outcome="Y2", template=_DD_TEMPLATE),
        Estimator.lm_robust("Y1 ~ Z", inquiry="ATE1", label="DIM1"),
        Estimator.lm_robust("Y2 ~ Z", inquiry="ATE2", label="DIM2"),
    )


def declaration_10_3a() -> Design:
    """declaration_10.3a: Y1 carries the 0.2 effect, Y2 carries none."""
    return _declaration_10_3(effect_y1=0.2, effect_y2=0.0)


def declaration_10_3b() -> Design:
    """declaration_10.3b: Y2 carries the 0.2 effect, Y1 carries none."""
    return _declaration_10_3(effect_y1=0.0, effect_y2=0.2)


def declaration_10_4(effect_size: float = 0.1) -> Design:
    """declaration_10.4: covariate adjustment — unadjusted vs adjusted OLS.

    N = 100 with a prognostic covariate X; ``Y ~ effect_size · Z + X + U``.
    Both estimators report the Z coefficient (DeclareDesign's default term
    is the first non-intercept coefficient), with HC2 robust SEs.
    """

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = (
            effect_size * float(z) + df["X"].to_numpy() + df["U"].to_numpy()  # type: ignore[arg-type]
        )
        return result

    return Design(
        Model(
            n=100,
            build=lambda n, rng: {"U": rng.normal(size=n), "X": rng.normal(size=n)},
            label="model",
        ),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", inquiry="ATE", label="unadjusted"),
        Estimator.lm_robust("Y ~ Z + X", inquiry="ATE", label="adjusted"),
    )


def declaration_10a(
    heteroskedasticity: float = 0.0, prob_treated: float = 0.5
) -> Design:
    """declaration_10a: classical vs HC2 SEs under heteroskedasticity.

    ``Y_Z_0 ~ N(0, 1 − h)`` and ``Y_Z_1 ~ N(1, 1 + h)`` are declared
    directly as model columns (no potential_outcomes call in the R source);
    assignment is ``complete_ra(N, prob = prob_treated)`` (randomizr's
    floor/ceiling rule when N·prob is not an integer). The book sweeps
    ``heteroskedasticity ∈ {−0.4, 0, 0.4}`` × ``prob_treated ∈
    seq(0.1, 0.9, length.out = 7)`` and compares ``var_estimate`` with
    ``mean_var_hat`` per estimator. The R estimators declare no inquiry;
    DeclareDesign pairs them with the design's only inquiry (ATE), which we
    make explicit here.
    """
    h = heteroskedasticity
    return Design(
        Model(
            n=100,
            build=lambda n, rng: {
                "Y_Z_0": rng.normal(0.0, 1.0 - h, n),
                "Y_Z_1": rng.normal(1.0, 1.0 + h, n),
            },
            label="model",
        ),
        Inquiry("ATE", lambda df: float((df["Y_Z_1"] - df["Y_Z_0"]).mean())),
        Assignment.complete(prob=prob_treated),
        reveal_outcomes(template=_DD_TEMPLATE),
        Estimator.lm_robust(
            "Y ~ Z",
            se_type="classical",
            inquiry="ATE",
            label="Classical standard error",
        ),
        Estimator.lm_robust(
            "Y ~ Z", se_type="HC2", inquiry="ATE", label="HC2 robust standard error"
        ),
    )
