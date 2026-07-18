"""Course-inline-compatible helpers (HONR 46400 exact-parity surface).

These functions are numerically **identical** to the inline helpers the
HONR 46400 course notebooks ship (nb10/nb11): same call signatures, same RNG
discipline (a fresh ``default_rng(seed)`` per call, one continuous stream),
same fixed z = 1.96 detection rule. Given the course seed 464 they reproduce
the notebooks' printed numbers bit-for-bit — that is this module's contract,
verified in ``tests/test_course_parity.py``.

The package-level API (:func:`declarepy.diagnose` on :class:`Design`
objects) is the DeclareDesign-style interface; this module is the simple
three-diagnosand teaching interface.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import pandas as pd

__all__ = ["diagnose", "run_design", "power_at"]

SEED = 464

DesignFn = Callable[[np.random.Generator], Tuple[float, float, float]]


def diagnose(design_fn: DesignFn, reps: int = 1000, seed: int = SEED) -> pd.Series:
    """Run a (est, se, truth) design function ``reps`` times; summarize.

    The course's diagnosand trio: bias = mean(est − truth); power =
    share of runs with |est/se| > 1.96; coverage = share of runs whose
    est ± 1.96·se interval contains the truth.
    """
    rng = np.random.default_rng(seed)
    rows = [design_fn(rng) for _ in range(reps)]
    d = pd.DataFrame(rows, columns=["est", "se", "truth"])
    return pd.Series(
        {
            "bias": (d.est - d.truth).mean(),
            "power": (np.abs(d.est / d.se) > 1.96).mean(),
            "coverage": (
                (d.est - 1.96 * d.se <= d.truth) & (d.truth <= d.est + 1.96 * d.se)
            ).mean(),
        }
    )


def run_design(
    n: int,
    effect: float = 2.0,
    noise: float = 2.0,
    reps: int = 1000,
    seed: int = SEED,
) -> pd.DataFrame:
    """The course's vectorized two-arm design engine (nb10, verbatim math).

    Simulates ``reps`` studies of size ``n`` in one shot: Y0 = N(50,10) +
    N(0,noise), Y1 = Y0 + effect, complete assignment of exactly n//2 per
    run via argsort of uniforms, Welch SE with ddof=1. Returns a DataFrame
    with columns ``est`` and ``se``.
    """
    rng = np.random.default_rng(seed)
    Y0 = rng.normal(50, 10, (reps, n)) + rng.normal(0, noise, (reps, n))
    Y1 = Y0 + effect
    m = n // 2
    treated = np.argsort(rng.random((reps, n)), axis=1) < m
    Y = np.where(treated, Y1, Y0)
    mean1 = (Y * treated).sum(1) / m
    mean0 = (Y * ~treated).sum(1) / (n - m)
    var1 = ((Y - mean1[:, None]) ** 2 * treated).sum(1) / (m - 1)
    var0 = ((Y - mean0[:, None]) ** 2 * ~treated).sum(1) / (n - m - 1)
    est = mean1 - mean0
    se = np.sqrt(var1 / m + var0 / (n - m))
    return pd.DataFrame({"est": est, "se": se})


def power_at(
    n: int,
    effect: float,
    noise: float = 2.0,
    reps: int = 1000,
    seed: int = SEED,
) -> float:
    """Share of simulated studies that detect the effect (|est/se| > 1.96)."""
    d = run_design(n=n, effect=effect, noise=noise, reps=reps, seed=seed)
    return float((np.abs(d.est / d.se) > 1.96).mean())
