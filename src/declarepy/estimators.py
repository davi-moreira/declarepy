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

__all__ = [
    "EstimatorResult",
    "difference_in_means",
    "lm_robust",
    "prop_test",
    "glm_logit",
    "logit_ame",
]


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
    clusters: Optional[str] = None,
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

    Clustered (``clusters=`` a column name, no blocks): estimatr's clustered
    difference in means — OLS of y on z with CR2 cluster-robust variance and
    Bell–McCaffrey/Satterthwaite degrees of freedom.

    Missing values in ``y`` or ``z`` raise — surface missingness, never skip
    it silently (SEMANTIC_DIFFERENCES §3).
    """
    cols = [y, z] + ([blocks] if blocks else []) + ([clusters] if clusters else [])
    sub = data[cols]
    n_na = int(sub.isna().sum().sum())
    if n_na:
        raise ValueError(
            f"{n_na} missing values in {cols}; drop or impute explicitly "
            "before estimating (declarepy never skips NAs silently)"
        )
    if clusters is not None:
        if blocks is not None:
            raise NotImplementedError(
                "blocked + clustered difference_in_means is a roadmap item"
            )
        tidy = lm_robust(f"{y} ~ {z}", data, se_type="CR2", clusters=clusters, alpha=alpha)
        row = tidy[tidy["term"] == z].iloc[0]
        return EstimatorResult(
            estimate=float(row["estimate"]), std_error=float(row["std_error"]),
            statistic=float(row["statistic"]), p_value=float(row["p_value"]),
            conf_low=float(row["conf_low"]), conf_high=float(row["conf_high"]),
            df=float(row["df"]), term=z, outcome=y, estimator="difference_in_means",
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
_CR_TYPES = {"CR0", "CR2", "stata"}


def _sym_inv_sqrt(mat: np.ndarray) -> np.ndarray:
    """Symmetric pseudo-inverse square root (for CR2's (I − H_gg)^{−1/2})."""
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    inv_sqrt = np.where(vals > 1e-12, 1.0 / np.sqrt(np.clip(vals, 1e-12, None)), 0.0)
    result: np.ndarray = vecs @ np.diag(inv_sqrt) @ vecs.T
    return result


def _cluster_robust(
    X: np.ndarray,
    resid: np.ndarray,
    cluster_ids: np.ndarray,
    se_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster-robust covariance and per-coefficient df, estimatr-style.

    CR0: plain sandwich, df = G − 1 (with Stata's small-sample factor for
    ``"stata"``/CR1). CR2: Pustejovsky–Tipton bias-reduced linearization with
    Bell–McCaffrey/Satterthwaite per-coefficient degrees of freedom.
    """
    n, k = X.shape
    M = np.linalg.inv(X.T @ X)
    labels, inverse = np.unique(cluster_ids, return_inverse=True)
    G = len(labels)
    idx_by_g = [np.flatnonzero(inverse == g) for g in range(G)]

    if se_type in ("CR0", "stata"):
        meat = np.zeros((k, k))
        for idx in idx_by_g:
            s = X[idx].T @ resid[idx]
            meat += np.outer(s, s)
        V = M @ meat @ M
        if se_type == "stata":
            V *= (G / (G - 1)) * ((n - 1) / (n - k))
        dof = np.full(k, float(G - 1))
        return V, dof

    # CR2
    H_parts = [X[idx] @ M for idx in idx_by_g]  # X_g (X'X)^{-1}
    A_by_g = []
    meat = np.zeros((k, k))
    for idx, XgM in zip(idx_by_g, H_parts):
        Xg = X[idx]
        Hgg = XgM @ Xg.T
        A = _sym_inv_sqrt(np.eye(len(idx)) - Hgg)
        A_by_g.append(A)
        s = Xg.T @ (A @ resid[idx])
        meat += np.outer(s, s)
    V = M @ meat @ M

    # Bell–McCaffrey Satterthwaite df, per coefficient.
    H = X @ M @ X.T
    ImH = np.eye(n) - H
    dof = np.empty(k)
    for j in range(k):
        cols = np.empty((n, G))
        for g, (idx, A) in enumerate(zip(idx_by_g, A_by_g)):
            p_g = A @ (X[idx] @ M[:, j])
            cols[:, g] = ImH[:, idx] @ p_g
        omega = cols.T @ cols
        lam = np.linalg.eigvalsh(omega)
        dof[j] = lam.sum() ** 2 / (lam**2).sum()
    return V, dof


def lm_robust(
    formula: str,
    data: pd.DataFrame,
    se_type: Optional[str] = None,
    clusters: Optional[str] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """OLS with robust standard errors, estimatr-style (HC2/CR2 defaults).

    Fits ``formula`` by OLS via statsmodels, then applies estimatr's
    inference: robust (or classical) standard errors with **t** statistics —
    ``N − k`` degrees of freedom for HC types, Bell–McCaffrey/Satterthwaite
    per-coefficient df for CR2, ``G − 1`` for CR0/stata — and t-based
    confidence intervals. Returns a tidy DataFrame with one row per term
    (``term, estimate, std_error, statistic, p_value, conf_low, conf_high,
    df, outcome``) carrying ``r_squared``/``adj_r_squared``/``nobs`` in
    ``.attrs``.

    ``se_type``: without clusters — ``"HC2"`` (default, estimatr's default),
    ``"HC0"``, ``"HC1"``/``"stata"``, ``"HC3"``, ``"classical"``; with
    ``clusters=`` a column name — ``"CR2"`` (default), ``"CR0"``, or
    ``"stata"`` (CR1 with G−1 df).
    """
    import statsmodels.formula.api as smf

    if clusters is None:
        se_type = se_type or "HC2"
        if se_type not in _HC_TYPES:
            raise NotImplementedError(
                f"se_type={se_type!r} not implemented without clusters; "
                f"available: {sorted(_HC_TYPES)}"
            )
    else:
        se_type = se_type or "CR2"
        if se_type not in _CR_TYPES:
            raise NotImplementedError(
                f"se_type={se_type!r} not implemented with clusters; "
                f"available: {sorted(_CR_TYPES)}"
            )
        if data[clusters].isna().any():
            raise ValueError(f"missing values in cluster column {clusters!r}")

    model = smf.ols(formula, data=data)
    if model.exog.shape[0] < len(data):
        n_dropped = len(data) - model.exog.shape[0]
        raise ValueError(
            f"{n_dropped} rows with missing values in the model frame; drop or "
            "impute explicitly before estimating (declarepy never skips NAs silently)"
        )
    n = model.exog.shape[0]
    k = model.exog.shape[1]

    if clusters is None:
        if se_type == "classical":
            fit = model.fit()
        else:
            fit = model.fit(cov_type="HC1" if se_type == "stata" else se_type)
        params = np.asarray(fit.params)
        names = list(fit.params.index)
        bse = np.asarray(fit.bse)
        dof_arr = np.full(k, float(n - k))
        rsq, arsq = float(fit.rsquared), float(fit.rsquared_adj)
    else:
        fit = model.fit()
        params = np.asarray(fit.params)
        names = list(fit.params.index)
        V, dof_arr = _cluster_robust(
            np.asarray(model.exog),
            np.asarray(fit.resid),
            data[clusters].to_numpy(),
            se_type,
        )
        bse = np.sqrt(np.diag(V))
        rsq, arsq = float(fit.rsquared), float(fit.rsquared_adj)

    tstats = params / bse
    pvals = 2 * stats.t.sf(np.abs(tstats), dof_arr)
    crit = stats.t.ppf(1 - alpha / 2, dof_arr)
    outcome = formula.split("~")[0].strip()
    tidy = pd.DataFrame(
        {
            "term": names,
            "estimate": params,
            "std_error": bse,
            "statistic": tstats,
            "p_value": pvals,
            "conf_low": params - crit * bse,
            "conf_high": params + crit * bse,
            "df": dof_arr,
            "outcome": outcome,
        }
    )
    tidy.attrs.update({"r_squared": rsq, "adj_r_squared": arsq, "nobs": n})
    return tidy


def _profile_ci_logit(
    endog: np.ndarray,
    exog: np.ndarray,
    j: int,
    beta_hat: np.ndarray,
    se_j: float,
    llmax: float,
    alpha: float,
) -> tuple[float, float]:
    """Profile-likelihood CI for one logit coefficient (R confint.glm-style).

    Solves 2·(llmax − profile-ll(c)) = χ²₁(1−α) on each side of the MLE by
    Brent root-finding, profiling out the other coefficients via an offset
    refit. Falls back to ±∞ on a side where the deviance never reaches the
    cutoff within ±10 SE (quasi-separation).
    """
    import statsmodels.api as sm
    from scipy import optimize

    cutoff = float(stats.chi2.ppf(1 - alpha, 1))
    others = [c for c in range(exog.shape[1]) if c != j]
    X_rest = exog[:, others]
    x_j = exog[:, j]

    def dev(c: float) -> float:
        offset = c * x_j
        if X_rest.shape[1]:
            fit_c = sm.Logit(endog, X_rest, offset=offset).fit(
                disp=False, start_params=beta_hat[others], method="lbfgs", maxiter=200
            )
            ll = float(fit_c.llf)
        else:
            p = 1 / (1 + np.exp(-offset))
            ll = float(np.sum(endog * np.log(p) + (1 - endog) * np.log1p(-p)))
        return 2 * (llmax - ll) - cutoff

    b = float(beta_hat[j])
    bounds: list[float] = []
    for sign in (-1.0, 1.0):
        hi = None
        for k in (2.0, 4.0, 6.0, 10.0):
            cand = b + sign * k * se_j
            try:
                if dev(cand) > 0:
                    hi = cand
                    break
            except Exception:
                break
        if hi is None:
            bounds.append(float("-inf") if sign < 0 else float("inf"))
            continue
        try:
            bounds.append(float(optimize.brentq(dev, b, hi, xtol=1e-6 * max(1.0, abs(b)))))
        except Exception:
            bounds.append(float("-inf") if sign < 0 else float("inf"))
    return bounds[0], bounds[1]


def glm_logit(
    formula: str,
    data: pd.DataFrame,
    alpha: float = 0.05,
    ci: str = "profile",
) -> pd.DataFrame:
    """Logistic regression, R ``glm(family = binomial)``-style tidy output.

    Maximum-likelihood logit via statsmodels with Wald z tests (matching R's
    ``summary.glm``). Confidence intervals are **profile-likelihood** by
    default (matching ``broom::tidy(conf.int = TRUE)`` / ``confint.glm``,
    which is what DeclareDesign's coverage diagnosand sees); pass
    ``ci="wald"`` for normal-approximation intervals.
    """
    if ci not in ("profile", "wald"):
        raise ValueError("ci must be 'profile' or 'wald'")
    import statsmodels.formula.api as smf

    model = smf.logit(formula, data=data)
    if model.exog.shape[0] < len(data):
        raise ValueError(
            "rows with missing values in the model frame; drop or impute "
            "explicitly before estimating"
        )
    fit = model.fit(disp=False)
    params = np.asarray(fit.params)
    bse = np.asarray(fit.bse)
    zstats = params / bse
    pvals = 2 * stats.norm.sf(np.abs(zstats))
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    if ci == "wald":
        lo = params - zcrit * bse
        hi = params + zcrit * bse
    else:
        endog = np.asarray(model.endog, dtype=float)
        exog = np.asarray(model.exog, dtype=float)
        llmax = float(fit.llf)
        lo_list, hi_list = [], []
        for jj in range(len(params)):
            l, h = _profile_ci_logit(endog, exog, jj, params, float(bse[jj]), llmax, alpha)
            lo_list.append(l)
            hi_list.append(h)
        lo, hi = np.asarray(lo_list), np.asarray(hi_list)
    tidy = pd.DataFrame(
        {
            "term": list(fit.params.index),
            "estimate": params,
            "std_error": bse,
            "statistic": zstats,
            "p_value": pvals,
            "conf_low": lo,
            "conf_high": hi,
            "df": np.nan,
            "outcome": formula.split("~")[0].strip(),
        }
    )
    tidy.attrs.update({"nobs": int(fit.nobs)})
    return tidy


def logit_ame(
    formula: str,
    data: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Average marginal effects of a logit fit (the ``margins`` package's AME).

    Fits the logit by MLE, then reports the sample-average derivative of the
    predicted probability with respect to each regressor, with delta-method
    standard errors and Wald z inference — statsmodels'
    ``get_margeff(at="overall")``, numerically the same estimator as R's
    ``margins::margins``.
    """
    import statsmodels.formula.api as smf

    fit = smf.logit(formula, data=data).fit(disp=False)
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
