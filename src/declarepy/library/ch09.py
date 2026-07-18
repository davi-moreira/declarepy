"""Chapter 9 declarations: answer strategies.

Provenance: replication-materials ``code/declarations/declaration_9.1.R``
(Blair, Coppock & Humphreys 2023). Reference outputs:
``diagnosis_9.1.rds`` (sims = 2000, set.seed(42) in R — RNG streams are not
portable, so validation is tolerance-based).
"""

from __future__ import annotations

from ..steps import Design, Estimator, Inquiry, Model, Sampling

__all__ = ["declaration_9_1"]


def declaration_9_1() -> Design:
    """declaration_9.1: mean age of 100 people, estimated from a sample of 3.

    ``lm_robust(age ~ 1)`` on a complete random sample of n=3 — the book's
    deliberately tiny-sample illustration of an answer strategy.
    """
    return Design(
        Model(
            n=100,
            build=lambda n, rng: {"age": rng.integers(0, 81, n)},
            label="model",
        ),
        Inquiry("mean_age", lambda df: float(df["age"].mean())),
        Sampling.complete(n=3),
        Estimator.lm_robust("age ~ 1", inquiry="mean_age"),
    )
