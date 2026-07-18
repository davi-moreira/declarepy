"""Answer strategies: estimatr-style estimators in plain scientific Python.

Re-implements the *behavior* of ``estimatr``'s ``difference_in_means`` and
``lm_robust`` (Blair, Cooper, Coppock, Humphreys & Sonnet,
declaredesign.org/r/estimatr) from their published methodological notes —
no upstream source code is copied.

Conventions (see docs/spec/SEMANTIC_DIFFERENCES.md):

* **HC2 is the default** robust standard error for ``lm_robust``, matching
  estimatr (statsmodels alone defaults to classical SEs — §5).
* Inference is **t-based** with estimatr's degrees of freedom (Welch–
  Satterthwaite for the unblocked difference in means; ``N - k`` for
  ``lm_robust`` with HC standard errors).
* ``difference_in_means`` results unpack as ``est, se = ...`` — the exact
  two numbers the HONR 46400 course notebooks compute inline — while also
  exposing p-values, confidence intervals and degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["EstimatorResult", "difference_in_means", "lm_robust", "prop_test"]


@dataclass
class EstimatorResult:
    """One estimate row: the estimand-answering numbers plus their uncertainty.

    Unpacks as ``est, se = result`` for course-notebook compatibility; the
    full inference (statistic, p-value, CI, df) is available as attributes
    and via :meth:`to_frame`.
    """

    estimate: float
    std_error: float
    statistic: float = float("nan")
    p_value: float = float("nan")
    conf_low: float = float("nan")
    conf_high: float = float("nan")
    df: float = float("nan")
    term: str = ""
    outcome: str = ""
    estimator: str = ""
    extra: dict[str, float] = field(default_factory=dict)

    def __iter__(self) -> Iterator[float]:
        # Two-tuple unpacking: `est, se = difference_in_means(df)`.
        yield self.estimate
        yield self.std_error

    def to_frame(self) -> pd.DataFrame:
        """One-row tidy DataFrame (estimatr::tidy-style columns)."""
        return pd.DataFrame(
            [
                {
                    "estimator": self.estimator,
                    "term": self.term,
                    "estimate": self.estimate,
                    "std_error": self.std_error,
                    "statistic": self.statistic,
                    "p_value": self.p_value,
                    "conf_low": self.conf_low,
                    "conf_high": self.conf_high,
                    "df": self.df,
                    "outcome": self.outcome,
                }
            ]
        )


def _welch(
    y1: np.ndarray, y0: np.ndarray, alpha: float
) -> tuple[float, float, float, float, float, float, float]:
    """Welch difference in means: est, se, t, p, ci_low, ci_high, df."""
    n1, n0 = len(y1), len(y0)
    if n1 < 2 or n0 < 2:
        raise ValueError("each arm needs at least 2 units for a variance")
    m1, m0 = float(np.mean(y1)), float(np.mean(y0))
    v1 = float(np.var(y1, ddof=1))
    v0 = float(np.var(y0, ddof=1))
    est = m1 - m0
    se = float(np.sqrt(v1 / n1 + v0 / n0))
    # Welch–Satterthwaite degrees of freedom (estimatr's unblocked default).
    num = (v1 / n1 + v0 / n0) ** 2
    den = (v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1)
    dof = num / den if den > 0 else float(n1 + n0 - 2)
    tstat = est / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(tstat), dof)) if se > 0 else float("nan")
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return est, se, tstat, p, est - crit * se, est + crit * se, dof


def difference_in_means(
    data: pd.DataFrame,
    y: str = "Y",
    z: str = "Z",
    blocks: Optional[str] = None,
    alpha: float = 0.05,
    condition1: object = 0,
    condition2: object = 1,
) -> EstimatorResult:
    """Difference in means with design-appropriate standard errors.

    Unblocked: Welch SE ``sqrt(s1²/n1 + s0²/n0)`` (sample variances,
    ``ddof=1``) with Welch–Satterthwaite degrees of freedom — the estimate
    and SE are numerically identical to the course notebooks' inline helper.

    Blocked (``blocks=`` a column name): the block-size-weighted average of
    within-block differences, ``Σ (N_b/N)·DiM_b``, with variance
    ``Σ (N_b/N)²·(s²1b/n1b + s²0b/n0b)`` and ``N - 2B`` degrees of freedom
    (estimatr's blocked design without clusters).

    Missing values in ``y`` or ``z`` raise — surface missingness, never skip
    it silently (SEMANTIC_DIFFERENCES §3).
    """
    cols = [y, z] + ([blocks] if blocks else [])
    sub = data[cols]
    n_na = int(sub.isna().sum().sum())
    if n_na:
        raise ValueError(
            f"{n_na} missing values in {cols}; drop or impute explicitly "
            "before estimating (declarepy never skips NAs silently)"
        )
    if blocks is None:
        y1 = sub.loc[sub[z] == condition2, y].to_numpy(dtype=float)
        y0 = sub.loc[sub[z] == condition1, y].to_numpy(dtype=float)
        est, se, tstat, p, lo, hi, dof = _welch(y1, y0, alpha)
        return EstimatorResult(
            estimate=est, std_error=se, statistic=tstat, p_value=p,
            conf_low=lo, conf_high=hi, df=dof, term=z, outcome=y,
            estimator="difference_in_means",
        )

    N = len(sub)
    est = 0.0
    var = 0.0
    n_blocks = 0
    for _, grp in sub.groupby(blocks, sort=False):
        n_blocks += 1
        g1 = grp.loc[grp[z] == condition2, y].to_numpy(dtype=float)
        g0 = grp.loc[grp[z] == condition1, y].to_numpy(dtype=float)
        if len(g1) < 2 or len(g0) < 2:
            raise ValueError("each block needs ≥2 treated and ≥2 control units")
        w = len(grp) / N
        est += w * (float(np.mean(g1)) - float(np.mean(g0)))
        var += w**2 * (
            float(np.var(g1, ddof=1)) / len(g1) + float(np.var(g0, ddof=1)) / len(g0)
        )
    se = float(np.sqrt(var))
    dof = float(N - 2 * n_blocks)
    tstat = est / se if se > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(tstat), dof))
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return EstimatorResult(
        estimate=est, std_error=se, statistic=tstat, p_value=p,
        conf_low=est - crit * se, conf_high=est + crit * se, df=dof,
        term=z, outcome=y, estimator="difference_in_means",
    )


_HC_TYPES = {"HC0", "HC1", "HC2", "HC3", "classical", "stata"}


def lm_robust(
    formula: str,
    data: pd.DataFrame,
    se_type: str = "HC2",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """OLS with robust standard errors, estimatr-style (HC2 default).

    Fits ``formula`` by OLS via statsmodels, then applies estimatr's
    inference: heteroskedasticity-robust (or classical) standard errors with
    **t** statistics on ``N - k`` degrees of freedom and t-based confidence
    intervals. Returns a tidy DataFrame with one row per term
    (``term, estimate, std_error, statistic, p_value, conf_low, conf_high,
    df, outcome``).

    ``se_type``: ``"HC2"`` (default, estimatr's default), ``"HC0"``,
    ``"HC1"``/``"stata"``, ``"HC3"``, or ``"classical"``. Clustered (CR2)
    errors are a Tranche-2 target (see docs/spec/TRANSLATION_ROADMAP.md).
    """
    if se_type not in _HC_TYPES:
        raise NotImplementedError(
            f"se_type={se_type!r} not implemented; available: {sorted(_HC_TYPES)} "
            "(CR2/clustered SEs are a roadmap item)"
        )
    import statsmodels.formula.api as smf

    model = smf.ols(formula, data=data)
    if model.exog.shape[0] < len(data):
        n_dropped = len(data) - model.exog.shape[0]
        raise ValueError(
            f"{n_dropped} rows with missing values in the model frame; drop or "
            "impute explicitly before estimating (declarepy never skips NAs silently)"
        )
    if se_type == "classical":
        fit = model.fit()
        bse = np.asarray(fit.bse)
    else:
        cov = "HC1" if se_type == "stata" else se_type
        fit = model.fit(cov_type=cov)
        bse = np.asarray(fit.bse)
    params = np.asarray(fit.params)
    names = list(fit.params.index)
    n = int(fit.nobs)
    k = len(params)
    dof = float(n - k)
    tstats = params / bse
    pvals = 2 * stats.t.sf(np.abs(tstats), dof)
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    outcome = formula.split("~")[0].strip()
    return pd.DataFrame(
        {
            "term": names,
            "estimate": params,
            "std_error": bse,
            "statistic": tstats,
            "p_value": pvals,
            "conf_low": params - crit * bse,
            "conf_high": params + crit * bse,
            "df": dof,
            "outcome": outcome,
        }
    )


def prop_test(
    x: int,
    n: int,
    p: float = 0.5,
    correct: bool = True,
    alpha: float = 0.05,
) -> EstimatorResult:
    """One-sample proportion test, matching R's ``prop.test``.

    Chi-square test of ``x`` successes in ``n`` trials against probability
    ``p``, with Yates continuity correction by default, and a Wilson score
    confidence interval (continuity-corrected when ``correct=True``).
    """
    if not 0 <= x <= n:
        raise ValueError("need 0 <= x <= n")
    phat = x / n
    yates = min(0.5, abs(x - n * p)) if correct else 0.0
    stat = (abs(x - n * p) - yates) ** 2 / (n * p * (1 - p))
    pval = float(stats.chi2.sf(stat, 1))
    z = float(stats.norm.ppf(1 - alpha / 2))
    # Wilson score interval, continuity-corrected as in R's prop.test.
    cc = (0.5 / n) if correct else 0.0
    p_lo = max(phat - cc, 0.0)
    p_hi = min(phat + cc, 1.0)
    z2 = z * z
    lo = (p_lo + z2 / (2 * n) - z * np.sqrt(p_lo * (1 - p_lo) / n + z2 / (4 * n * n))) / (1 + z2 / n)
    hi = (p_hi + z2 / (2 * n) + z * np.sqrt(p_hi * (1 - p_hi) / n + z2 / (4 * n * n))) / (1 + z2 / n)
    lo = float(max(lo, 0.0)) if x != 0 else 0.0
    hi = float(min(hi, 1.0)) if x != n else 1.0
    return EstimatorResult(
        estimate=float(phat), std_error=float(np.sqrt(phat * (1 - phat) / n)),
        statistic=float(stat), p_value=pval, conf_low=lo, conf_high=hi,
        df=1.0, term="p", estimator="prop_test",
    )
