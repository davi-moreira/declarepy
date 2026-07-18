"""Chapter 11 declarations: redesign.

Provenance: replication-materials ``code/declarations/declaration_11.1.R``
… ``declaration_11.5.R`` (Blair, Coppock & Humphreys 2023). Reference
outputs:

* ``diagnosis_11.1.rds`` — 11.1 redesigned over N = 100..1000 (step 100).
* ``diagnosis_11.2.rds`` — 11.2 (declaration 10.2's random-ATE trial with a
  free N) over N ∈ {100, 500, 1000}.
* ``diagnosis_11.3.rds`` — 11.3 over N = seq(100, 1000, 10) ×
  prob ∈ {0.1, 0.3, 0.5} with custom ``cost``/``rmse`` diagnosands.
* ``diagnosis_11.4.rds`` — 11.4's fifty conditional-mean inquiries answered
  by polynomial regressions of degree 1–6.
* ``diagnosis_11.5.rds`` — 11.5's OLS vs logit-AME vs probit-AME
  comparison (the AMEs via R's ``margins``; the probit helper here is also
  fixture-checked against ``margins`` in
  ``validation/reference/rgen_t3_ch10_11_probit_fixture.json``).

All sims = 2000 in the book's diagnosis scripts.
"""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..estimators import prop_test
from ..steps import (
    Assignment,
    Design,
    Estimator,
    Inquiry,
    Measurement,
    Model,
    potential_outcomes,
    reveal_outcomes,
)
from .ch10 import declaration_10_2

__all__ = [
    "declaration_11_1",
    "declaration_11_2",
    "declaration_11_3",
    "declaration_11_4",
    "declaration_11_5",
]


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


def declaration_11_2(N: int = 100) -> Design:
    """declaration_11.2: the random-ATE trial of declaration 10.2, free N.

    The R source is line-for-line declaration 10.2 with ``N`` lifted to a
    variable (default 100) so the book can ``redesign(N = c(100, 500,
    1000))``; we delegate to :func:`declaration_10_2` accordingly.
    """
    return declaration_10_2(N=N)


def declaration_11_3(N: int = 100, prob: float = 0.5) -> Design:
    """declaration_11.3: fixed ATE 0.2, free N and treatment share.

    The R source leaves ``prob`` unbound (it exists only through
    ``redesign(N = seq(100, 1000, 10), prob = seq(0.1, 0.5, 0.2))``); the
    Python default 0.5 is declarepy's, chosen for a runnable factory. The
    book's diagnosis adds ``cost = unique(N·2 + prob·N·20)`` and rmse.
    """

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = 0.2 * float(z) + df["U"].to_numpy()  # type: ignore[arg-type]
        return result

    return Design(
        Model(n=N, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(prob=prob),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", inquiry="ATE"),
    )


# --------------------------------------------------------------------------
# declaration_11.4: fifty conditional-mean inquiries, six polynomial fits
# --------------------------------------------------------------------------

#: The book's evaluation grid: seq(from = 0, to = 3, length.out = 50).
_X_RANGE: np.ndarray = np.linspace(0.0, 3.0, 50)

#: Polynomial degrees the estimator handler sweeps (R: polynomial_degrees).
_POLY_DEGREES: tuple[int, ...] = tuple(range(1, 7))


def _dip(x: np.ndarray) -> np.ndarray:
    """The book's kinked truth: (x ≤ 1)·x + (x > 1)·(x − 2)² + 0.2."""
    arr = np.asarray(x, dtype=float)
    result: np.ndarray = (arr <= 1) * arr + (arr > 1) * (arr - 2.0) ** 2 + 0.2
    return result


def _dip_scalar(v: float) -> float:
    """Scalar version of :func:`_dip` (for the deterministic estimands)."""
    return (v if v <= 1 else (v - 2.0) ** 2) + 0.2


def _x_grid() -> list[float]:
    """_X_RANGE as a plain list of floats (typed iteration helper)."""
    return cast("list[float]", _X_RANGE.tolist())


def _r_chr(x: float) -> str:
    """Format a double the way R's ``as.character`` does (15 sig. digits).

    Reproduces the inquiry labels ``str_c("X_", X)`` builds in the R source
    (e.g. ``X_0.0612244897959184``) so declarepy's tables match the saved
    diagnosis object's row names.
    """
    return format(float(x), ".15g")


def _const(v: float) -> Callable[[pd.DataFrame], float]:
    """A data-independent estimand (R: declare_inquiry(..., data = NULL))."""

    def fn(df: pd.DataFrame) -> float:
        return v

    return fn


class _TidyRowsEstimator(Estimator):
    """Estimator whose handler emits rows carrying their own labels.

    DeclareDesign's ``declare_estimator(handler = ...)`` idiom returns a
    tidy frame with per-row ``estimator``/``inquiry`` columns (here: A1–A6 ×
    fifty inquiries). The base :class:`Estimator` stamps the step's single
    label over those columns; this subclass keeps them. Private to ch11
    pending promotion to the shared steps module.
    """

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        res = self.fn(df)
        if not isinstance(res, pd.DataFrame):
            raise TypeError("_TidyRowsEstimator fn must return a DataFrame")
        rows = res.copy()
        if "estimator" not in rows.columns:
            rows["estimator"] = self.label
        if "inquiry" not in rows.columns:
            rows["inquiry"] = self.inquiry
        return rows


def _poly_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """R handler: lm(Y ~ poly(X, k)) for k = 1..6, predicted on _X_RANGE.

    Predictions from R's orthogonal-polynomial basis are identical to any
    least-squares polynomial fit of the same degree (same column space);
    numpy's scaled-domain Polynomial.fit is the numerically stable analog.
    """
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    labels = [f"X_{_r_chr(v)}" for v in _x_grid()]
    frames: list[pd.DataFrame] = []
    for k in _POLY_DEGREES:
        poly = np.polynomial.polynomial.Polynomial.fit(x, y, k)
        frames.append(
            pd.DataFrame(
                {
                    "estimator": f"A{k}",
                    "inquiry": labels,
                    "estimate": np.asarray(poly(_X_RANGE), dtype=float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def declaration_11_4() -> Design:
    """declaration_11.4: which polynomial degree recovers E[Y | X]?

    Fifty deterministic inquiries (the true regression function ``dip``
    evaluated on a grid — ``declare_inquiry(..., data = NULL, handler =
    tibble)``), answered by predictions from polynomial regressions of
    degree 1–6. The estimator rows carry no p-values or CIs, so power and
    coverage are undefined (NA in the book's diagnosis).
    """
    inquiries: dict[str, Any] = {
        f"X_{_r_chr(v)}": _const(_dip_scalar(v)) for v in _x_grid()
    }
    return Design(
        Model(
            n=100,
            build=lambda n, rng: {"X": rng.uniform(0.0, 3.0, n)},
            label="model",
        ),
        Inquiry(**inquiries),
        Measurement(
            lambda df, rng: {
                "Y": _dip(df["X"].to_numpy()) + rng.normal(0.0, 0.5, len(df))
            },
            label="measurement",
        ),
        _TidyRowsEstimator(_poly_predictions, label="polynomials"),
    )


# --------------------------------------------------------------------------
# declaration_11.5: OLS vs logit-AME vs probit-AME
# --------------------------------------------------------------------------


def _probit_ame(formula: str, data: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Average marginal effects of a probit fit (R ``margins`` on a probit glm).

    Mirrors :func:`declarepy.estimators.logit_ame` — statsmodels
    ``get_margeff(at="overall", method="dydx")`` with delta-method SEs and
    Wald z inference, numerically the same estimator as
    ``margins::margins`` on ``glm(family = binomial("probit"))``. Private to
    ch11 pending promotion to the shared estimators module.
    """
    import statsmodels.formula.api as smf

    fit = smf.probit(formula, data=data).fit(disp=False)
    mfx = fit.get_margeff(at="overall", method="dydx")
    params = np.asarray(mfx.margeff)
    bse = np.asarray(mfx.margeff_se)
    names = [t for t in fit.params.index if t != "Intercept"]
    zstats = params / bse
    pvals = 2 * stats.norm.sf(np.abs(zstats))
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    return pd.DataFrame(
        {
            "term": names,
            "estimate": params,
            "std_error": bse,
            "statistic": zstats,
            "p_value": pvals,
            "conf_low": params - zcrit * bse,
            "conf_high": params + zcrit * bse,
            "df": np.nan,
            "outcome": formula.split("~")[0].strip(),
        }
    )


def declaration_11_5() -> Design:
    """declaration_11.5: binary outcome — OLS, logit AME, probit AME.

    ``Y ~ rbinom(N, 1, 0.2·Z + 0.6)``: both potential outcomes are fresh
    Bernoulli draws (evaluated per condition, like DeclareDesign), so the
    per-simulation estimand mean(Y1 − Y0) is itself random around 0.2. The
    logit and probit estimators report ``margins``-style average marginal
    effects (``.summary = tidy_margins`` in the R source); OLS is
    DeclareDesign's default lm_robust (HC2).
    """

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = rng.binomial(1, 0.2 * float(z) + 0.6, len(df))  # type: ignore[arg-type]
        return result

    def probit(df: pd.DataFrame) -> pd.DataFrame:
        tidy = _probit_ame("Y ~ Z", df)
        return tidy[tidy["term"] == "Z"].copy()

    return Design(
        Model(n=100, build=lambda n, rng: {"U": rng.normal(size=n)}, label="model"),
        potential_outcomes(po),
        Inquiry("ATE", lambda df: float((df["Y1"] - df["Y0"]).mean())),
        Assignment.complete(prob=0.5),
        reveal_outcomes(),
        Estimator.lm_robust("Y ~ Z", term="Z", inquiry="ATE", label="OLS"),
        Estimator.logit_ame("Y ~ Z", term="Z", inquiry="ATE", label="logit"),
        Estimator(probit, inquiry="ATE", label="probit"),
    )
