"""Chapter 16 declarations: observational designs for causal identification.

Provenance: replication-materials ``code/declarations/declaration_16.1a.R`` /
``declaration_16.1b.R`` (process tracing with a CausalQueries binary causal
model), ``declaration_16.2.R`` (exact matching on a binary covariate),
``declaration_16.3.R`` (staggered-adoption difference-in-differences with a
two-way fixed-effects and a de Chaisemartin–D'Haultfœuille estimator),
``declaration_16.4.R`` (instrumental variables / LATE), ``declaration_16.5.R``
and ``declaration_16.6.R`` (regression discontinuity with an rdrobust-style
MSE-optimal local estimator and a bandwidth-swept local-linear estimator)
from Blair, Coppock & Humphreys 2023. Reference outputs:
``diagnosis_16.1.rds`` … ``diagnosis_16.5.rds`` (sims = 2000 each;
``diagnosis_16.4.rds`` diagnoses the 16.3 design under three
treatment-effect-trend variants, all reachable here via
``declaration_16_3(treatment_trend=...)``; ``diagnosis_16.5.rds`` sweeps
``declaration_16_6`` over ``bandwidth`` 0.05 … 0.50).

Instead of depending on CausalQueries / MatchIt / DIDmultiplegt / estimatr /
rdrobust, this module re-implements the *behavior* each declaration needs as
private helpers (enumerated-causal-type posteriors, exact-matching weights,
the DID_M point estimator, 2SLS with HC2 errors, and the bias-corrected
local-polynomial RD estimator with MSE-optimal bandwidths following
Calonico, Cattaneo & Titiunik 2014). Every helper is validated numerically
against its R origin in ``validation/validate_t3_ch16.py``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .. import estimators as _est
from ..steps import (
    Assignment,
    Design,
    Estimator,
    Inquiry,
    Measurement,
    Model,
    reveal_outcomes,
)

__all__ = [
    "declaration_16_1",
    "declaration_16_2",
    "declaration_16_3",
    "declaration_16_4",
    "declaration_16_5",
    "declaration_16_6",
]


# =====================================================================
# declaration_16.1 — process tracing on a binary causal model
# =====================================================================
#
# The R source builds the CausalQueries model  X -> M -> Y <- W -> M  and
# restricts M's and Y's nodal types with
#   set_restrictions("(M[X=1] < M[X=0]) | (M[X=1, W=1] == M[X=0, W=1])")
#   set_restrictions("(Y[M=1] < Y[M=0]) | (Y[M=1, W=1] == Y[M=0, W=1])")
# leaving exactly three admissible nodal types for M (as a function of X, W)
# and for Y (as a function of M, W), verified against the package's
# ``parameters_df`` / ``interpret_type``:
#   "0001": node = 1  iff  parent = 1 and W = 1        (AND gate)
#   "0011": node = parent                              (pure parent effect)
#   "1011": node = 0  iff  parent = 0 and W = 1        (parent OR not-W)
# With CausalQueries' default flat parameters, every full causal type
# (theta_W, theta_X, theta_M, theta_Y) has probability
# (1/2)(1/2)(1/3)(1/3) = 1/36, so posteriors are exact counts over the 36
# enumerated types. The process-tracing estimate for a strategy (the set of
# observed nodes) is E[query | observed data] under those flat parameters —
# deterministic given the data, matching rdss::process_tracing_estimator.

_PT_TYPE_LABELS: Tuple[str, str, str] = ("0001", "0011", "1011")

#: nodal-type value as a function of (non-W parent a, W w).
_PT_TYPE_FNS: Dict[str, Dict[Tuple[int, int], int]] = {
    "0001": {(a, w): int(a == 1 and w == 1) for a in (0, 1) for w in (0, 1)},
    "0011": {(a, w): int(a) for a in (0, 1) for w in (0, 1)},
    "1011": {(a, w): int(not (a == 0 and w == 1)) for a in (0, 1) for w in (0, 1)},
}

_PT_STRATEGIES: Dict[str, Tuple[str, ...]] = {
    "X-Y": ("X", "Y"),
    "X-Y-M": ("X", "Y", "M"),
    "X-Y-W": ("X", "Y", "W"),
    "X-Y-W-M": ("X", "Y", "W", "M"),
}


def _pt_realize(theta_w: int, theta_x: int, m_type: str, y_type: str) -> Dict[str, int]:
    """Observed data and case-level causal effect implied by a causal type."""
    f_m = _PT_TYPE_FNS[m_type]
    f_y = _PT_TYPE_FNS[y_type]
    w, x = theta_w, theta_x
    m = f_m[(x, w)]
    y = f_y[(m, w)]
    effect = f_y[(f_m[(1, w)], w)] - f_y[(f_m[(0, w)], w)]
    return {"X": x, "M": m, "W": w, "Y": y, "effect": effect}


#: all 36 equally-likely causal types with their observables and effects.
_PT_TYPES: Tuple[Dict[str, int], ...] = tuple(
    _pt_realize(tw, tx, mt, yt)
    | {"theta_W": tw, "theta_X": tx, "theta_M": mi, "theta_Y": yi}
    for tw in (0, 1)
    for tx in (0, 1)
    for mi, mt in enumerate(_PT_TYPE_LABELS)
    for yi, yt in enumerate(_PT_TYPE_LABELS)
)


def _pt_posterior(observed: Mapping[str, int], nodes: Tuple[str, ...]) -> float:
    """E[Y(X=1) − Y(X=0) | observed values of ``nodes``], flat type prior."""
    total = 0
    hits = 0
    for t in _PT_TYPES:
        if all(t[n] == observed[n] for n in nodes):
            total += 1
            hits += t["effect"]
    if total == 0:
        return float("nan")
    return hits / total


def declaration_16_1() -> Design:
    """declaration_16.1: single-case process tracing (CoE query).

    One causal type is drawn per run; the inquiry is that case's causal
    effect of X on Y; the estimator reports, for each of the four clue
    strategies, the posterior mean effect given the strategy's observed
    nodes. Estimate rows carry an ``XY`` column (e.g. ``"X0Y0"``) matching
    the reference diagnosis' ``make_groups = vars(XY)``.
    """

    def draw_type(n: int, rng: np.random.Generator) -> Dict[str, object]:
        return {
            "theta_W": [int(rng.integers(0, 2))],
            "theta_X": [int(rng.integers(0, 2))],
            "theta_M": [int(rng.integers(0, 3))],
            "theta_Y": [int(rng.integers(0, 3))],
        }

    def coe(df: pd.DataFrame) -> float:
        row = df.iloc[0]
        t = _pt_realize(
            int(row["theta_W"]),
            int(row["theta_X"]),
            _PT_TYPE_LABELS[int(row["theta_M"])],
            _PT_TYPE_LABELS[int(row["theta_Y"])],
        )
        return float(t["effect"])

    def make_data(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        row = df.iloc[0]
        t = _pt_realize(
            int(row["theta_W"]),
            int(row["theta_X"]),
            _PT_TYPE_LABELS[int(row["theta_M"])],
            _PT_TYPE_LABELS[int(row["theta_Y"])],
        )
        return pd.DataFrame(
            {"W": [t["W"]], "X": [t["X"]], "M": [t["M"]], "Y": [t["Y"]]}
        )

    def process_tracing(df: pd.DataFrame) -> pd.DataFrame:
        obs = {k: int(df.iloc[0][k]) for k in ("X", "M", "W", "Y")}
        rows = [
            {
                "term": label,
                "estimate": _pt_posterior(obs, nodes),
                "XY": f"X{obs['X']}Y{obs['Y']}",
            }
            for label, nodes in _PT_STRATEGIES.items()
        ]
        return pd.DataFrame(rows)

    return Design(
        Model(n=1, build=draw_type, label="draw_causal_type"),
        Inquiry("CoE", coe),
        Measurement(make_data, label="make_data"),
        Estimator(process_tracing, inquiry="CoE", label="estimator"),
    )


# =====================================================================
# declaration_16.2 — exact matching on a binary covariate
# =====================================================================


def _exact_match(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """MatchIt-style exact matching on ``X`` (ATT weights).

    Strata are the unique values of ``X``; strata lacking treated or control
    units are dropped. Treated units get weight 1; control units get weight
    proportional to the stratum ratio n1s/n0s, rescaled so control weights
    average 1 (``MatchIt::match.data``'s scaling).
    """
    keep_parts = []
    for _, grp in df.groupby("X", sort=True):
        n1 = int((grp["D"] == 1).sum())
        n0 = int((grp["D"] == 0).sum())
        if n1 == 0 or n0 == 0:
            continue
        out = grp.copy()
        out["weights"] = np.where(out["D"] == 1, 1.0, n1 / n0)
        keep_parts.append(out)
    matched = pd.concat(keep_parts, ignore_index=True)
    ctrl = matched["D"] == 0
    raw_sum = float(matched.loc[ctrl, "weights"].sum())
    matched.loc[ctrl, "weights"] *= float(ctrl.sum()) / raw_sum
    return matched


def _difference_in_means_weighted(
    df: pd.DataFrame,
    y: str = "Y",
    z: str = "D",
    weights: str = "weights",
    alpha: float = 0.05,
) -> _est.EstimatorResult:
    """estimatr's weighted difference in means: WLS with HC2, df = N − 2."""
    w = df[weights].to_numpy(dtype=float)
    yy = df[y].to_numpy(dtype=float)
    zz = df[z].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(df)), zz])
    sw = np.sqrt(w)
    Xt = X * sw[:, None]
    yt = yy * sw
    XtX_inv = np.linalg.inv(Xt.T @ Xt)
    beta = XtX_inv @ Xt.T @ yt
    resid = yt - Xt @ beta
    hii = np.sum((Xt @ XtX_inv) * Xt, axis=1)
    meat = (Xt * (resid**2 / (1.0 - hii))[:, None]).T @ Xt
    V = XtX_inv @ meat @ XtX_inv
    est = float(beta[1])
    se = float(np.sqrt(V[1, 1]))
    dof = float(len(df) - 2)
    tstat = est / se
    p = float(2 * stats.t.sf(abs(tstat), dof))
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return _est.EstimatorResult(
        estimate=est, std_error=se, statistic=tstat, p_value=p,
        conf_low=est - crit * se, conf_high=est + crit * se, df=dof,
        term=z, outcome=y, estimator="difference_in_means",
    )


def declaration_16_2(N: int = 100) -> Design:
    """declaration_16.2: exact matching, matched vs raw difference-in-means."""

    def build(n: int, rng: np.random.Generator) -> Dict[str, object]:
        U = rng.normal(size=n)
        X = rng.binomial(1, 0.5, n)
        D = rng.binomial(1, 0.25 + 0.5 * X, n)
        Y_D_0 = 0.2 * X + U
        return {"U": U, "X": X, "D": D, "Y_D_0": Y_D_0, "Y_D_1": Y_D_0 + 0.5}

    return Design(
        Model(n=N, build=build, label="model"),
        Inquiry("ATE", lambda df: float((df["Y_D_1"] - df["Y_D_0"]).mean())),
        Measurement(_exact_match, label="exact_matching"),
        reveal_outcomes(outcome="Y", assignment="D", template="{outcome}_D_{condition}"),
        Estimator(
            lambda df: _difference_in_means_weighted(df),
            inquiry="ATE",
            label="Matched difference-in-means",
        ),
        Estimator.difference_in_means(
            y="Y", z="D", inquiry="ATE", label="Raw difference-in-means"
        ),
    )


# =====================================================================
# declaration_16.3 — staggered-adoption difference-in-differences
# =====================================================================


def _lm_robust_twfe(
    df: pd.DataFrame,
    y: str,
    d: str,
    fe1: str,
    fe2: str,
    alpha: float = 0.05,
) -> _est.EstimatorResult:
    """``lm_robust(Y ~ D, fixed_effects = ~fe1 + fe2)`` — HC2, df = N − k.

    estimatr's fixed-effects HC2 numbers equal the full-dummy regression
    (verified numerically), so the fit uses explicit dummies.
    """
    yy = df[y].to_numpy(dtype=float)
    dd = df[d].to_numpy(dtype=float)
    parts = [np.ones((len(df), 1)), dd[:, None]]
    for fe in (fe1, fe2):
        codes, _ = pd.factorize(df[fe], sort=True)
        n_lev = codes.max() + 1
        dummies = np.zeros((len(df), n_lev - 1))
        mask = codes > 0
        dummies[np.arange(len(df))[mask], codes[mask] - 1] = 1.0
        parts.append(dummies)
    X = np.hstack(parts)
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ yy
    resid = yy - X @ beta
    hii = np.sum((X @ XtX_inv) * X, axis=1)
    meat = (X * (resid**2 / (1.0 - hii))[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv
    est = float(beta[1])
    se = float(np.sqrt(V[1, 1]))
    dof = float(n - k)
    tstat = est / se
    p = float(2 * stats.t.sf(abs(tstat), dof))
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return _est.EstimatorResult(
        estimate=est, std_error=se, statistic=tstat, p_value=p,
        conf_low=est - crit * se, conf_high=est + crit * se, df=dof,
        term=d, outcome=y, estimator="lm_robust",
    )


def _did_multiplegt(
    df: pd.DataFrame,
    y: str = "Y",
    group: str = "units",
    time: str = "periods",
    d: str = "D",
) -> float:
    """DIDmultiplegt 0.1.0's ``did_multiplegt(...)$effect`` (the DID_M
    estimator of de Chaisemartin & D'Haultfœuille 2020) for a balanced
    binary-treatment panel.

    For each consecutive period pair, joiners (D: 0→1) are compared with
    not-yet-treated stayers (0→0) and leavers (1→0) with still-treated
    stayers (1→1); the effect is the switcher-count-weighted average.
    Matches the R package exactly on this chapter's staggered-adoption data
    (where only joiners occur).
    """
    wide_y = df.pivot(index=group, columns=time, values=y).sort_index(axis=1)
    wide_d = df.pivot(index=group, columns=time, values=d).sort_index(axis=1)
    Y = wide_y.to_numpy(dtype=float)
    D = wide_d.to_numpy(dtype=float)
    num = 0.0
    den = 0.0
    for t in range(1, Y.shape[1]):
        dy = Y[:, t] - Y[:, t - 1]
        prev, now = D[:, t - 1], D[:, t]
        joiners = (prev == 0) & (now == 1)
        stay0 = (prev == 0) & (now == 0)
        leavers = (prev == 1) & (now == 0)
        stay1 = (prev == 1) & (now == 1)
        if joiners.any() and stay0.any():
            n_j = int(joiners.sum())
            num += n_j * float(dy[joiners].mean() - dy[stay0].mean())
            den += n_j
        if leavers.any() and stay1.any():
            n_l = int(leavers.sum())
            num += n_l * float(dy[stay1].mean() - dy[leavers].mean())
            den += n_l
    return num / den if den > 0 else float("nan")


def declaration_16_3(
    treatment_trend: float = -1.0,
    N_units: int = 20,
    N_time_periods: int = 20,
) -> Design:
    """declaration_16.3: staggered adoption, TWFE vs de Chaisemartin DID_M.

    The unit-period effect is ``0.2 + treatment_trend * (D_time - t)``:
    ``treatment_trend=-1`` is the book's declaration_16.3 (later-treated
    periods have larger effects); ``0`` and ``+1`` are diagnosis_16.4's
    ``PO_homogenous`` and ``PO_later_lower`` variants.
    """

    def build(n: int, rng: np.random.Generator) -> pd.DataFrame:
        # fabricate(units = add_level(...), periods = add_level(..., nest
        # = FALSE), unit_period = cross_levels(...)) — built explicitly:
        # unit-level draws are repeated over the 20 x 20 cross (period-major
        # order, matching the R frame).
        U_unit = rng.normal(size=N_units)
        D_unit = (U_unit > np.median(U_unit)).astype(int)
        D_time = rng.integers(1, N_time_periods + 1, N_units)
        U_time = rng.normal(size=N_time_periods)
        U = rng.normal(size=N_units * N_time_periods)
        units = np.tile(np.arange(1, N_units + 1), N_time_periods)
        periods = np.repeat(np.arange(1, N_time_periods + 1), N_units)
        uu, tt = units - 1, periods - 1
        effect = 0.2 + treatment_trend * (D_time[uu] - periods)
        Y_D_0 = U + U_unit[uu] + U_time[tt]
        D = ((D_unit[uu] == 1) & (periods >= D_time[uu])).astype(int)
        # lag_by_group(D, groups = units, order_by = periods): D is
        # deterministic in (unit, period), so the lag is D at period t-1.
        D_prev = ((D_unit[uu] == 1) & (periods - 1 >= D_time[uu])).astype(float)
        D_lag = np.where(periods == 1, np.nan, D_prev)
        return pd.DataFrame(
            {
                "units": units, "periods": periods, "U_unit": U_unit[uu],
                "D_unit": D_unit[uu], "D_time": D_time[uu],
                "U_time": U_time[tt], "U": U,
                "Y_D_0": Y_D_0, "Y_D_1": Y_D_0 + effect,
                "D": D, "D_lag": D_lag,
            }
        )

    def att(df: pd.DataFrame) -> float:
        sub = df[df["D"] == 1]
        return float((sub["Y_D_1"] - sub["Y_D_0"]).mean())

    def att_switchers(df: pd.DataFrame) -> float:
        sub = df[(df["D"] == 1) & (df["D_lag"] == 0) & df["D_lag"].notna()]
        return float((sub["Y_D_1"] - sub["Y_D_0"]).mean())

    steps = [
        Model(n=N_units * N_time_periods, build=lambda n, rng: build(n, rng), label="model"),
        Inquiry("ATT", att),
        Inquiry("ATT_switchers", att_switchers),
        reveal_outcomes(outcome="Y", assignment="D", template="{outcome}_D_{condition}"),
    ]
    for inquiry in ("ATT", "ATT_switchers"):
        steps.append(
            Estimator(
                lambda df: _lm_robust_twfe(df, "Y", "D", "units", "periods"),
                inquiry=inquiry,
                label="twoway-fe",
            )
        )
        steps.append(
            Estimator(
                lambda df: {"estimate": _did_multiplegt(df)},
                inquiry=inquiry,
                label="chaisemartin",
            )
        )
    return Design(*steps)


# =====================================================================
# declaration_16.4 — instrumental variables (LATE)
# =====================================================================


def _iv_robust(
    df: pd.DataFrame,
    y: str = "Y",
    d: str = "D",
    z: str = "Z",
    alpha: float = 0.05,
) -> _est.EstimatorResult:
    """estimatr's ``iv_robust(Y ~ D | Z)`` — 2SLS with HC2 SEs, t(N − 2).

    Just-identified single-endogenous-regressor 2SLS: HC2 leverage comes
    from the projected (second-stage) design matrix, residuals from the
    structural equation at the actual D.
    """
    yy = df[y].to_numpy(dtype=float)
    dd = df[d].to_numpy(dtype=float)
    zz = df[z].to_numpy(dtype=float)
    n = len(df)
    Zmat = np.column_stack([np.ones(n), zz])
    dhat = Zmat @ np.linalg.lstsq(Zmat, dd, rcond=None)[0]
    Xhat = np.column_stack([np.ones(n), dhat])
    XtX_inv = np.linalg.inv(Xhat.T @ Xhat)
    beta = XtX_inv @ Xhat.T @ yy
    resid = yy - np.column_stack([np.ones(n), dd]) @ beta
    hii = np.sum((Xhat @ XtX_inv) * Xhat, axis=1)
    meat = (Xhat * (resid**2 / (1.0 - hii))[:, None]).T @ Xhat
    V = XtX_inv @ meat @ XtX_inv
    est = float(beta[1])
    se = float(np.sqrt(V[1, 1]))
    dof = float(n - 2)
    tstat = est / se
    p = float(2 * stats.t.sf(abs(tstat), dof))
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    return _est.EstimatorResult(
        estimate=est, std_error=se, statistic=tstat, p_value=p,
        conf_low=est - crit * se, conf_high=est + crit * se, df=dof,
        term=d, outcome=y, estimator="iv_robust",
    )


def declaration_16_4(N: int = 100) -> Design:
    """declaration_16.4: binary IV with one-sided noncompliance structure."""

    def build(n: int, rng: np.random.Generator) -> Dict[str, object]:
        U = rng.normal(size=n)
        D_Z_0 = (0 + U > 0).astype(int)
        D_Z_1 = (1 + U > 0).astype(int)
        Y_D_0 = 0.25 + U
        return {
            "U": U, "D_Z_0": D_Z_0, "D_Z_1": D_Z_1,
            "Y_D_0": Y_D_0, "Y_D_1": Y_D_0 + 0.1,
            "complier": (D_Z_1 == 1) & (D_Z_0 == 0),
        }

    def late(df: pd.DataFrame) -> float:
        sub = df[df["complier"]]
        return float((sub["Y_D_1"] - sub["Y_D_0"]).mean())

    return Design(
        Model(n=N, build=build, label="model"),
        Inquiry("LATE", late),
        Assignment.complete(prob=0.5),
        reveal_outcomes(outcome="D", assignment="Z", template="{outcome}_Z_{condition}"),
        reveal_outcomes(outcome="Y", assignment="D", template="{outcome}_D_{condition}"),
        Estimator(lambda df: _iv_robust(df), inquiry="LATE", label="estimator"),
    )


# =====================================================================
# declaration_16.5 / 16.6 — regression discontinuity
# =====================================================================
#
# The "optimal" estimator re-implements rdrobust's default sharp-RD path
# (p=1, q=2, triangular kernel, bwselect="mserd", vce="nn" with 3 matches,
# no covariates/clusters; Calonico, Cattaneo & Titiunik 2014, Econometrica;
# Calonico, Cattaneo, Farrell & Titiunik 2017). The reported row is the
# "Bias-Corrected" one: bias-corrected point estimate paired with the
# conventional standard error and normal-theory inference, exactly as
# rdrobust's tidy method reports it. Validated to ~1e-13 against
# rdrobust 4.0.0 on fixed datasets (see validation/validate_t3_ch16.py).


def _quantile_type2(x: np.ndarray, p: float) -> float:
    """R ``quantile(type = 2)``: inverse ECDF, averaging at discontinuities."""
    xs = np.sort(x)
    n = len(xs)
    j = int(np.floor(n * p))
    g = n * p - j
    if g > 0:
        return float(xs[min(j, n - 1)])
    return float((xs[max(j - 1, 0)] + xs[min(j, n - 1)]) / 2.0)


def _rd_kweight(x: np.ndarray, c: float, h: float) -> np.ndarray:
    """Triangular kernel weights ((1 − |u|)+ / h), u = (x − c)/h."""
    u = (x - c) / h
    w: np.ndarray = (1.0 - np.abs(u)) * (np.abs(u) <= 1) / h
    return np.maximum(w, 0.0)


def _rd_nn_res(x: np.ndarray, y: np.ndarray, matches: int = 3) -> np.ndarray:
    """Nearest-neighbor residuals sqrt(J/(J+1))·(y_i − mean of J neighbors).

    ``x`` must be sorted ascending without exact duplicates (a continuous
    running variable); neighbors come from the same side's sample.
    """
    n = len(y)
    res = np.empty(n)
    J_target = min(matches, n - 1)
    for pos in range(n):
        lpos = 0
        rpos = 0
        while lpos + rpos < J_target:
            if pos - lpos - 1 < 0:
                rpos += 1
            elif pos + rpos + 1 > n - 1:
                lpos += 1
            elif (x[pos] - x[pos - lpos - 1]) > (x[pos + rpos + 1] - x[pos]):
                rpos += 1
            elif (x[pos] - x[pos - lpos - 1]) < (x[pos + rpos + 1] - x[pos]):
                lpos += 1
            else:
                rpos += 1
                lpos += 1
        idx = np.arange(pos - lpos, min(pos + rpos, n - 1) + 1)
        ysum = y[idx].sum() - y[pos]
        J = len(idx) - 1
        res[pos] = np.sqrt(J / (J + 1.0)) * (y[pos] - ysum / J)
    return res


def _rd_bw_stage(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    o: int,
    nu: int,
    o_B: int,
    h_V: float,
    h_B: float,
    scale: float,
    cache: Dict[Tuple[int, int], Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    """One side's (V, B, R, rate) for one bandwidth-selection stage."""
    key = (o, nu)
    if key in cache:
        V_V, BConst = cache[key]
    else:
        w = _rd_kweight(x, c, h_V)
        ind = w > 0
        eY, eX, eW = y[ind], x[ind], w[ind]
        R_V = np.vander(eX - c, N=o + 1, increasing=True)
        invG = np.linalg.inv((R_V * eW[:, None]).T @ R_V)
        res = _rd_nn_res(eX, eY)
        RW = R_V * eW[:, None]
        aux = (RW * res[:, None] ** 2).T @ RW
        V_V = float((invG @ aux @ invG)[nu, nu])
        v = RW.T @ (((eX - c) / h_V) ** (o + 1))
        Hp = h_V ** np.arange(o + 1)
        BConst = float((Hp * (invG @ v))[nu])
        cache[key] = (V_V, BConst)

    w = _rd_kweight(x, c, h_B)
    ind = w > 0
    eY, eX, eW = y[ind], x[ind], w[ind]
    R_B = np.vander(eX - c, N=o_B + 1, increasing=True)
    invG_B = np.linalg.inv((R_B * eW[:, None]).T @ R_B)
    beta_B = invG_B @ (R_B * eW[:, None]).T @ eY

    BWreg = 0.0
    if scale > 0:
        res_B = _rd_nn_res(eX, eY)
        RWB = R_B * eW[:, None]
        aux_B = (RWB * res_B[:, None] ** 2).T @ RWB
        V_B = float((invG_B @ aux_B @ invG_B)[o + 1, o + 1])
        BWreg = 3.0 * BConst**2 * V_B

    B = float(np.sqrt(2.0 * (o + 1 - nu)) * BConst * beta_B[o + 1])
    V = float((2.0 * nu + 1.0) * h_V ** (2 * nu + 1) * V_V)
    R = float(scale * (2.0 * (o + 1 - nu)) * BWreg)
    rate = 1.0 / (2.0 * o + 3.0)
    return V, B, R, rate


def _rd_mserd_bw(y: np.ndarray, x: np.ndarray, c: float = 0.0) -> Tuple[float, float]:
    """The common MSE-optimal (h, b) bandwidths for p=1, q=2 ("mserd")."""
    p, q = 1, 2
    ind_l = x < c
    X_l, Y_l = x[ind_l], y[ind_l]
    X_r, Y_r = x[~ind_l], y[~ind_l]
    N = len(x)
    x_iq = _quantile_type2(x, 0.75) - _quantile_type2(x, 0.25)
    BWp = min(float(np.std(x, ddof=1)), x_iq / 1.349)
    c_bw = 2.576 * BWp * N ** (-1 / 5)  # C_c for the triangular kernel
    range_l = abs(c - float(x.min()))
    range_r = abs(c - float(x.max()))
    bw_max = max(range_l, range_r)
    c_bw = min(c_bw, bw_max)

    cache_l: Dict[Tuple[int, int], Tuple[float, float]] = {}
    cache_r: Dict[Tuple[int, int], Tuple[float, float]] = {}
    V_dl, B_dl, _, rate_d = _rd_bw_stage(
        Y_l, X_l, c, q + 1, q + 1, q + 2, c_bw, range_l, 0.0, cache_l
    )
    V_dr, B_dr, _, _ = _rd_bw_stage(
        Y_r, X_r, c, q + 1, q + 1, q + 2, c_bw, range_r, 0.0, cache_r
    )
    d_bw = min(((V_dl + V_dr) / (B_dr - B_dl) ** 2) ** rate_d, bw_max)
    V_bl, B_bl, R_bl, rate_b = _rd_bw_stage(
        Y_l, X_l, c, q, p + 1, q + 1, c_bw, d_bw, 1.0, cache_l
    )
    V_br, B_br, R_br, _ = _rd_bw_stage(
        Y_r, X_r, c, q, p + 1, q + 1, c_bw, d_bw, 1.0, cache_r
    )
    b_bw = min(
        ((V_bl + V_br) / ((B_br - B_bl) ** 2 + (R_br + R_bl))) ** rate_b, bw_max
    )
    V_hl, B_hl, R_hl, rate_h = _rd_bw_stage(
        Y_l, X_l, c, p, 0, q, c_bw, b_bw, 1.0, cache_l
    )
    V_hr, B_hr, R_hr, _ = _rd_bw_stage(
        Y_r, X_r, c, p, 0, q, c_bw, b_bw, 1.0, cache_r
    )
    h_bw = min(
        ((V_hl + V_hr) / ((B_hr - B_hl) ** 2 + (R_hr + R_hl))) ** rate_h, bw_max
    )
    return float(h_bw), float(b_bw)


def _rd_side(
    y: np.ndarray, x: np.ndarray, c: float, h: float, b: float, p: int, q: int
) -> Tuple[float, float, float, float]:
    """One side's (tau_cl, tau_bc, V_cl, V_rb) intercept contributions."""
    w_h = _rd_kweight(x, c, h)
    w_b = _rd_kweight(x, c, b)
    ind = w_b > 0 if h <= b else w_h > 0
    eY, eX = y[ind], x[ind]
    W_h, W_b = w_h[ind], w_b[ind]
    u = (eX - c) / h
    R_q = np.vander(eX - c, N=q + 1, increasing=True)
    R_p = R_q[:, : p + 1]
    L = (R_p * W_h[:, None]).T @ (u ** (p + 1))
    invG_q = np.linalg.inv((R_q * W_b[:, None]).T @ R_q)
    invG_p = np.linalg.inv((R_p * W_h[:, None]).T @ R_p)
    e_p1 = np.zeros(q + 1)
    e_p1[p + 1] = 1.0
    t_vec = R_q @ (invG_q @ e_p1)
    Q_q = R_p * W_h[:, None] - h ** (p + 1) * (W_b * t_vec)[:, None] * L[None, :]
    beta_p = invG_p @ (R_p * W_h[:, None]).T @ eY
    beta_bc = invG_p @ Q_q.T @ eY
    res_h = _rd_nn_res(eX, eY)
    RW = R_p * W_h[:, None]
    V_cl = float((invG_p @ ((RW * res_h[:, None] ** 2).T @ RW) @ invG_p)[0, 0])
    V_rb = float((invG_p @ ((Q_q * res_h[:, None] ** 2).T @ Q_q) @ invG_p)[0, 0])
    return float(beta_p[0]), float(beta_bc[0]), V_cl, V_rb


def _rdrobust(
    df: pd.DataFrame,
    y: str = "Y",
    x: str = "X",
    c: float = 0.0,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Sharp-RD ``rdrobust(y, x, c)`` with defaults, tidied like R.

    Returns the three tidy rows (Conventional / Bias-Corrected / Robust)
    with normal-theory inference: the Bias-Corrected row pairs the
    bias-corrected point estimate with the *conventional* standard error.
    """
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    order = np.argsort(xv, kind="stable")
    xv, yv = xv[order], yv[order]
    h, b = _rd_mserd_bw(yv, xv, c)
    ind_l = xv < c
    tau_l, tbc_l, Vcl_l, Vrb_l = _rd_side(yv[ind_l], xv[ind_l], c, h, b, 1, 2)
    tau_r, tbc_r, Vcl_r, Vrb_r = _rd_side(yv[~ind_l], xv[~ind_l], c, h, b, 1, 2)
    tau_cl = tau_r - tau_l
    tau_bc = tbc_r - tbc_l
    se_cl = float(np.sqrt(Vcl_l + Vcl_r))
    se_rb = float(np.sqrt(Vrb_l + Vrb_r))
    zcrit = float(stats.norm.ppf(1 - alpha / 2))
    rows = []
    for term, est, se in [
        ("Conventional", tau_cl, se_cl),
        ("Bias-Corrected", tau_bc, se_cl),
        ("Robust", tau_bc, se_rb),
    ]:
        zstat = est / se
        rows.append(
            {
                "term": term, "estimate": est, "std_error": se,
                "statistic": zstat,
                "p_value": float(2 * stats.norm.sf(abs(zstat))),
                "conf_low": est - zcrit * se, "conf_high": est + zcrit * se,
                "df": np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs.update({"h": h, "b": b})
    return out


_RD_CUTOFF = 0.5


def _rd_control(x: np.ndarray) -> np.ndarray:
    """control(X): poly(X, 4) %*% c(.7, -.8, .5, 1)."""
    result: np.ndarray = 0.7 * x - 0.8 * x**2 + 0.5 * x**3 + 1.0 * x**4
    return result


def _rd_treatment(x: np.ndarray) -> np.ndarray:
    """treatment(X): poly(X, 4) %*% c(0, -1.5, .5, .8) + .15."""
    result: np.ndarray = -1.5 * x**2 + 0.5 * x**3 + 0.8 * x**4 + 0.15
    return result


def declaration_16_5(N: int = 500) -> Design:
    """declaration_16.5: sharp RD, MSE-optimal bias-corrected estimator.

    The inquiry is the book's ``LATE = treatment(0.5) - control(0.5)``
    (evaluated at the raw cutoff value, per the source declaration) while
    potential outcomes evaluate the polynomials on the *centered* running
    variable — reproduced verbatim, mismatch and all.
    """

    def build(n: int, rng: np.random.Generator) -> Dict[str, object]:
        U = rng.normal(0.0, 0.1, n)
        X = rng.uniform(0.0, 1.0, n) + U - _RD_CUTOFF
        return {
            "U": U, "X": X, "D": (X > 0).astype(int),
            "Y_D_0": _rd_control(X) + U, "Y_D_1": _rd_treatment(X) + U,
        }

    late = float(
        _rd_treatment(np.array([_RD_CUTOFF]))[0]
        - _rd_control(np.array([_RD_CUTOFF]))[0]
    )

    def optimal(df: pd.DataFrame) -> pd.DataFrame:
        tidy = _rdrobust(df, y="Y", x="X", c=0.0)
        return tidy[tidy["term"] == "Bias-Corrected"].copy()

    return Design(
        Model(n=N, build=build, label="model"),
        Inquiry("LATE", lambda df: late),
        reveal_outcomes(outcome="Y", assignment="D", template="{outcome}_D_{condition}"),
        Estimator(optimal, inquiry="LATE", label="optimal"),
    )


def declaration_16_6(bandwidth: float = 0.5, N: int = 500) -> Design:
    """declaration_16.6: 16.5 plus a local-linear estimator within ±bandwidth.

    ``lm_robust(Y ~ X * D, subset = |X| < bandwidth)`` with DeclareDesign's
    default term — the first non-intercept coefficient, i.e. the slope
    ``X``, exactly as the book's reference diagnosis records it.
    """

    def linear(df: pd.DataFrame) -> pd.DataFrame:
        sub = df[(df["X"] > -bandwidth) & (df["X"] < bandwidth)]
        tidy = _est.lm_robust("Y ~ X * D", sub)
        return tidy[tidy["term"] == "X"].copy()

    return declaration_16_5(N=N) + Design(
        Estimator(linear, inquiry="LATE", label="linear"),
    )
