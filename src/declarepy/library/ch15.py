"""Chapter 15 declarations: observational descriptive designs.

Provenance: replication-materials ``code/declarations/declaration_15.1.R`` …
``declaration_15.6.R`` (Blair, Coppock & Humphreys 2023) — sample means under
coarsened measurement (15.1), nonresponse bias vs. survey effort (15.2),
budget-constrained stratified cluster sampling (15.3), multilevel regression
and post-stratification (15.4/15.5), and index construction (15.6).

Fixed populations (see ``validation/r_scripts/t3_ch15_reference.R``, which
exports them):

* ``ch15_portola.csv`` — ``fabricate(N = 2100, Y_star = rnorm(N))`` under the
  book's own ``set.seed(343)``; identical to the population behind the book's
  saved ``diagnosis_15.1``/``diagnosis_15.2`` (their estimand 4.2919047619 is
  reproduced exactly).
* ``ch15_two_nigerian_states.csv`` — the 15.3 hierarchy (2 states × 500
  localities × 100 individuals, ICC 0.4). The book draws it with **no seed**,
  so its saved diagnosis population is unrecoverable; the shipped canon is
  fixed with ``set.seed(464)`` and the reference diagnosands regenerated
  (``rgen_t3_ch15_15_3.json``).
* ``ch15_states.csv`` — the 15.4 state frame built from R's ``state.x77``
  (public-domain 1977 US census figures) plus a ``state_shock`` draw; again
  seedless in the book, so fixed with ``set.seed(464)`` and referenced by
  ``rgen_t3_ch15_15_4.json``.

Reference outputs: ``diagnosis_15.1.json`` (custom bias/rmse diagnosands),
``diagnosis_15.2.json`` (effort sweep 0..5), ``rgen_t3_ch15_15_3.json``
(cluster_prob sweep 0.1..0.9), ``rgen_t3_ch15_15_4.json`` (declaration_15.5's
three estimators), ``diagnosis_15.5.json`` (declaration_15.6, grouped by
outcome), all at sims = 2000.

Semantic notes (documented deviations, mirrored in the validation script):

* 15.2's ``lm_robust(Y ~ 1)`` runs on data with item nonresponse; R drops
  NAs silently, declarepy raises — the translation drops them **explicitly**
  (SEMANTIC_DIFFERENCES §3).
* 15.4/15.5's ``rdss::post_stratification_helper`` predictions include the
  fitted state random intercepts (the behavior behind the book's published
  diagnosis; see the header of ``t3_ch15_reference.R``).
* 15.6's ``princomp(~ Y_1 + Y_2 + Y_2, cor = TRUE)`` keeps the book's typo:
  R's ``terms()`` dedupes the repeated ``Y_2``, so the "first factor" is the
  leading component of (Y_1, Y_2) only — Y_3 never enters.
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .. import estimators as _est
from .. import ra as _ra
from ..steps import Design, Estimator, Inquiry, Measurement, Model, Sampling

__all__ = [
    "declaration_15_1",
    "declaration_15_2",
    "declaration_15_3",
    "declaration_15_4",
    "declaration_15_5",
    "declaration_15_6",
]


# --------------------------------------------------------------------------
# fixed populations
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _fixed_population(name: str) -> pd.DataFrame:
    ref = resources.files("declarepy").joinpath("data", f"{name}.csv")
    with resources.as_file(ref) as path:
        return pd.read_csv(path)


def _load(name: str) -> pd.DataFrame:
    """A fresh copy of a bundled fixed population (cached read)."""
    return _fixed_population(name).copy()


# --------------------------------------------------------------------------
# private helpers (candidates for promotion to shared modules)
# --------------------------------------------------------------------------


def _r_cut(x: np.ndarray, breaks: int) -> np.ndarray:
    """R's ``as.numeric(cut(x, breaks = n))``: equal-width bin codes 1..n.

    The range of ``x`` is split into ``n`` intervals of equal length whose
    outer limits are pushed out by 0.1% of the range; intervals are
    right-closed ``(lo, hi]`` (R's defaults).
    """
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo
    brks = np.linspace(lo, hi, breaks + 1)
    brks[0] = lo - span / 1000.0
    brks[-1] = hi + span / 1000.0
    codes: np.ndarray = np.searchsorted(brks, x, side="left").astype(float)
    return codes


def _strata_and_cluster_rs(
    df: pd.DataFrame,
    strata: str,
    clusters: str,
    prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """randomizr's ``strata_and_cluster_rs``: whole-cluster complete random
    sampling within each stratum, returned as a row-level 0/1 inclusion."""
    codes, _ = pd.factorize(df[clusters])
    first_idx = np.unique(codes, return_index=True)[1]
    strata_by_cluster = df[strata].to_numpy()[first_idx]
    incl_cluster = _ra.block_rs(strata_by_cluster, prob=prob, rng=rng)
    result: np.ndarray = incl_cluster[codes]
    return result


def _strata_rs(labels: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """randomizr's ``strata_rs``: complete random sampling within each stratum.

    Equivalent to :func:`declarepy.ra.block_rs` but grouped via one stable
    argsort instead of a per-block scan — the per-block scan is O(N·B), too
    slow for 15.3's up to 900 strata × 90k rows per simulation.
    """
    codes, _ = pd.factorize(labels)
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    out = np.zeros(len(labels), dtype=int)
    start = 0
    for count in counts:
        idx = order[start : start + count]
        out[idx] = _ra.complete_rs(int(count), prob=prob, rng=rng)
        start += int(count)
    return out


def _budget_function(cluster_prob: float) -> float:
    """15.3b's budget: individual-stage sampling share given cluster share."""
    budget = 20000.0
    cluster_cost = 20.0
    individual_cost = 2.0
    n_clusters = 1000.0
    n_individuals_per_cluster = 100.0
    total_cluster_cost = cluster_prob * n_clusters * cluster_cost
    remaining_funds = budget - total_cluster_cost
    sampleable_individuals = cluster_prob * n_clusters * n_individuals_per_cluster
    individual_prob = (remaining_funds / individual_cost) / sampleable_individuals
    return float(min(individual_prob, 1.0))


def _scale(v: np.ndarray) -> np.ndarray:
    """R's ``scale()``: center and divide by the sd with divisor n − 1."""
    result: np.ndarray = (v - v.mean()) / v.std(ddof=1)
    return result


def _princomp_first_score(y1: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """R's ``princomp(~ Y_1 + Y_2 + Y_2, cor = TRUE)$scores[, 1]``.

    R's ``terms()`` dedupes the repeated ``Y_2`` (the book's typo silently
    drops ``Y_3``), so this is the first principal component of the two
    standardized variables. ``princomp`` standardizes with divisor-N sds; for
    a 2×2 correlation matrix the leading eigenvector is analytic, and
    ``princomp``'s default ``fix_sign = TRUE`` rescales every loading column
    so its FIRST element is positive — hence PC1 = (1, sign(r))/√2 (verified
    against R in ``rgen_t3_ch15_fitchecks.json``, both correlation signs).
    """
    z1 = (y1 - y1.mean()) / y1.std(ddof=0)
    z2 = (y2 - y2.mean()) / y2.std(ddof=0)
    r = float(np.corrcoef(y1, y2)[0, 1])
    sign = 1.0 if r >= 0 else -1.0
    result: np.ndarray = (z1 + sign * z2) / np.sqrt(2.0)
    return result


def _expit(z: np.ndarray) -> np.ndarray:
    result: np.ndarray = 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))
    return result


def _logit_irls(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Plain logistic regression y ~ 1 + x by Newton scoring (start values)."""
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(25):
        mu = _expit(X @ beta)
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def _glmer_logit_intercept(
    y: np.ndarray, x: np.ndarray, group: np.ndarray, n_groups: int
) -> tuple[float, float, float, np.ndarray]:
    """``lme4::glmer(y ~ x + (1 | group), family = binomial)``, Laplace (nAGQ=1).

    For each candidate (β₀, β₁, θ) the spherical random effects u are profiled
    out at their joint penalized mode (per-group Newton — the modes separate
    because each u_j enters only its own group), and the Laplace deviance

        −2·loglik(y | η̂) + ‖û‖² + Σ_j log(1 + θ²·W_j),   W_j = Σ_{i∈j} μ̂ᵢ(1−μ̂ᵢ)

    is minimized over (β₀, β₁, θ) by Nelder–Mead (θ enters as |θ|; the
    deviance is symmetric in its sign). Returns (β₀, β₁, θ, û). Fit-level
    agreement with glmer is checked in ``rgen_t3_ch15_fitchecks.json``.
    """

    def mode_u(b0: float, b1: float, th: float) -> np.ndarray:
        off = b0 + b1 * x
        u = np.zeros(n_groups)
        for _ in range(60):
            mu = _expit(off + th * u[group])
            grad = -th * np.bincount(group, weights=y - mu, minlength=n_groups) + u
            hess = th * th * np.bincount(group, weights=mu * (1.0 - mu), minlength=n_groups) + 1.0
            step = np.clip(grad / hess, -5.0, 5.0)
            u -= step
            if float(np.max(np.abs(step))) < 1e-11:
                break
        return u

    def deviance(params: np.ndarray) -> float:
        b0, b1, th = float(params[0]), float(params[1]), abs(float(params[2]))
        u = mode_u(b0, b1, th)
        mu = np.clip(_expit(b0 + b1 * x + th * u[group]), 1e-12, 1.0 - 1e-12)
        ll = float(y @ np.log(mu) + (1.0 - y) @ np.log1p(-mu))
        w_j = np.bincount(group, weights=mu * (1.0 - mu), minlength=n_groups)
        return -2.0 * ll + float(u @ u) + float(np.sum(np.log1p(th * th * w_j)))

    b0_start, b1_start = _logit_irls(y, x)
    res = optimize.minimize(
        deviance,
        x0=np.array([b0_start, b1_start, 1.0]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 3000, "maxfev": 6000},
    )
    b0, b1, th = float(res.x[0]), float(res.x[1]), abs(float(res.x[2]))
    return b0, b1, th, mode_u(b0, b1, th)


def _post_stratify(pred: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """``rdss::post_stratification_helper``'s aggregation: per-state weighted
    mean of row-level predictions with ``PS_weight`` weights."""
    w = df["PS_weight"].to_numpy(dtype=float)
    tmp = pd.DataFrame({"state": df["state"].to_numpy(), "pw": pred * w, "w": w})
    agg = tmp.groupby("state", sort=True).sum()
    return pd.DataFrame(
        {"state": agg.index.to_numpy(), "estimate": (agg["pw"] / agg["w"]).to_numpy()}
    )


class _PerStateEstimator(Estimator):
    """Estimator whose fn returns per-state rows ``(state, estimate)``.

    DeclareDesign aligns 15.4's 50-row estimates with the 50-row inquiry by
    merging on the shared ``state`` column; declarepy's diagnosis aligns by
    inquiry *name*, so each row is tagged ``mean_policy_support[<state>]`` to
    match the per-state inquiries declared by :func:`declaration_15_4`.
    Diagnosands pooled across states (the book's single-row-per-estimator
    table) are recomputed from the simulations frame in the validation script.
    """

    def __init__(self, fn: Callable[[pd.DataFrame], pd.DataFrame], label: str) -> None:
        super().__init__(fn, inquiry=None, label=label)
        self._per_state_fn = fn

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = self._per_state_fn(df)
        return pd.DataFrame(
            {
                "estimator": self.label,
                "estimate": rows["estimate"].to_numpy(dtype=float),
                "inquiry": [
                    f"mean_policy_support[{s}]" for s in rows["state"].astype(str)
                ],
            }
        )


def _fit_partial_pooling(df: pd.DataFrame) -> pd.DataFrame:
    """glmer logit with state random intercepts → RE-inclusive predictions."""
    y = df["policy_support"].to_numpy(dtype=float)
    x = df["HS"].to_numpy(dtype=float)
    codes, uniques = pd.factorize(df["state"])
    b0, b1, th, u = _glmer_logit_intercept(y, x, codes, len(uniques))
    pred = _expit(b0 + b1 * x + th * u[codes])
    return _post_stratify(pred, df)


def _fit_no_pooling(df: pd.DataFrame) -> pd.DataFrame:
    """OLS ``policy_support ~ HS + state`` fitted values (exact, via FWL)."""
    y = df["policy_support"].to_numpy(dtype=float)
    x = df["HS"].to_numpy(dtype=float)
    codes, _ = pd.factorize(df["state"])
    counts = np.bincount(codes).astype(float)
    ybar = np.bincount(codes, weights=y) / counts
    xbar = np.bincount(codes, weights=x) / counts
    xt = x - xbar[codes]
    b = float(xt @ (y - ybar[codes])) / float(xt @ xt)
    fitted = ybar[codes] + b * xt
    return _post_stratify(fitted, df)


def _fit_full_pooling(df: pd.DataFrame) -> pd.DataFrame:
    """OLS ``policy_support ~ HS`` fitted values."""
    y = df["policy_support"].to_numpy(dtype=float)
    x = df["HS"].to_numpy(dtype=float)
    xc = x - x.mean()
    b = float(xc @ (y - y.mean())) / float(xc @ xc)
    fitted = y.mean() + b * xc
    return _post_stratify(fitted, df)


def _const(v: float) -> Callable[[pd.DataFrame], float]:
    def f(df: pd.DataFrame) -> float:
        return v

    return f


# --------------------------------------------------------------------------
# declarations
# --------------------------------------------------------------------------


def _cut_measurement() -> Measurement:
    return Measurement(
        lambda df, rng: {"Y": _r_cut(df["Y_star"].to_numpy(), 7)},
        label="Y = cut(Y_star, 7)",
    )


def declaration_15_1() -> Design:
    """declaration_15.1: mean 7-point attitude in Portola from a sample of 100.

    Fixed N=2100 population (the book's ``set.seed(343)``), latent ``Y_star``
    coarsened to a 1..7 scale, complete random sample of 100,
    ``lm_robust(Y ~ 1)`` for the population mean.
    """
    return Design(
        Model(data=_load("ch15_portola"), label="portola"),
        _cut_measurement(),
        Inquiry("Y_bar", lambda df: float(df["Y"].mean())),
        Sampling.complete(n=100),
        Estimator.lm_robust("Y ~ 1", inquiry="Y_bar"),
    )


def declaration_15_2(effort: float = 0.0) -> Design:
    """declaration_15.2: nonresponse follows the latent attitude minus effort.

    Response ``R ~ Bernoulli(pnorm(Y_star + effort))`` among the 100 sampled;
    ``Y`` is observed only for responders. The mean estimator drops the
    missing ``Y`` explicitly (R's lm_robust drops NAs silently; declarepy
    surfaces missingness — SEMANTIC_DIFFERENCES §3). The book sweeps
    ``effort`` 0..5 by 0.5.
    """

    def nonresponse(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        n = len(df)
        r = rng.binomial(1, stats.norm.cdf(df["Y_star"].to_numpy() + effort), n)
        return {"R": r, "Y": np.where(r == 1, df["Y"].to_numpy(), np.nan)}

    def mean_of_responders(df: pd.DataFrame) -> pd.DataFrame:
        return _est.lm_robust("Y ~ 1", df.loc[df["Y"].notna()])

    return Design(
        Model(data=_load("ch15_portola"), label="portola"),
        _cut_measurement(),
        Inquiry("Y_bar", lambda df: float(df["Y"].mean())),
        Sampling.complete(n=100),
        Measurement(nonresponse, label="R, Y | nonresponse"),
        Estimator(mean_of_responders, inquiry="Y_bar", label="estimator"),
        Estimator.lm_robust("R ~ 1", label="Response Rate"),
    )


def declaration_15_3(cluster_prob: float = 0.5) -> Design:
    """declaration_15.3: two-stage budget-constrained cluster sample in Nigeria.

    Fixed hierarchy (2 states × 500 localities × 100 individuals, ICC 0.4;
    canon seed 464 — see module docstring). Stage 1 samples whole localities
    within states with ``cluster_prob``; stage 2 samples individuals within
    sampled localities at the budget-implied rate; ``lm_robust(Y ~ 1)`` with
    Stata (CR1) locality-clustered SEs. The book sweeps ``cluster_prob``
    0.1..0.9.
    """
    p_individual = _budget_function(cluster_prob)
    return Design(
        Model(data=_load("ch15_two_nigerian_states"), label="two_nigerian_states"),
        _cut_measurement(),
        Inquiry("Y_bar", lambda df: float(df["Y"].mean())),
        Sampling(
            lambda df, rng: _strata_and_cluster_rs(
                df, "state", "locality", cluster_prob, rng
            ),
            label=f"strata_and_cluster_rs(prob={cluster_prob})",
        ),
        Sampling(
            lambda df, rng: _strata_rs(df["locality"].to_numpy(), p_individual, rng),
            label=f"strata_rs(prob={p_individual:.4f})",
        ),
        Estimator.lm_robust(
            "Y ~ 1", clusters="locality", se_type="stata", inquiry="Y_bar"
        ),
    )


def _states_design_base() -> Design:
    """Shared model + per-state inquiry steps of declarations 15.4/15.5."""
    states = _load("ch15_states")
    idx = np.repeat(np.arange(len(states)), states["state_n"].to_numpy())
    expanded = states.iloc[idx].reset_index(drop=True)

    def draw_individuals(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        n = len(df)
        p_hs = df["prob_HS"].to_numpy(dtype=float)
        hs = rng.binomial(1, p_hs, n)
        ps_weight = np.where(hs == 0, 1.0 - p_hs, p_hs)
        shock = rng.normal(0.0, 0.5, n)
        prob = stats.norm.cdf(0.2 * hs + shock + df["state_shock"].to_numpy(dtype=float))
        return {
            "HS": hs,
            "PS_weight": ps_weight,
            "individual_shock": shock,
            "policy_support": rng.binomial(1, prob, n),
        }

    # The book's inquiry handler returns 50 rows (state, estimand=state_mean),
    # constant across simulations because the states frame is fixed; here each
    # state is a named inquiry ``mean_policy_support[<state>]`` (see
    # _PerStateEstimator for how estimates align).
    named: dict[str, Any] = {
        f"mean_policy_support[{s}]": _const(float(m))
        for s, m in zip(states["state"], states["state_mean"])
    }
    return Design(
        Model(data=expanded, label="states[rep(1:50, state_n), ]"),
        Model(transform=draw_individuals, label="HS, PS_weight, policy_support"),
        Inquiry(**named),
    )


def declaration_15_4() -> Design:
    """declaration_15.4: MRP — partial pooling via a state-intercept GLMM.

    2,000 respondents allocated to 50 states by population share; per-state
    mean policy support post-stratified on high-school completion. The
    partial-pooling estimator is ``glmer(policy_support ~ HS + (1 | state))``
    with RE-inclusive predictions (see module docstring), post-stratified
    with ``PS_weight``.
    """
    return _states_design_base() + Design(
        _PerStateEstimator(_fit_partial_pooling, label="Partial pooling"),
    )


def declaration_15_5() -> Design:
    """declaration_15.5: 15.4 plus no-pooling and full-pooling comparators.

    No pooling: ``lm_robust(policy_support ~ HS + state)`` fitted values;
    full pooling: ``lm_robust(policy_support ~ HS)`` fitted values; both
    post-stratified per state with ``PS_weight``.
    """
    return declaration_15_4() + Design(
        _PerStateEstimator(_fit_no_pooling, label="No pooling"),
        _PerStateEstimator(_fit_full_pooling, label="Full pooling"),
    )


_OUTCOMES_15_6 = ("Y_avg", "Y_avg_adjusted", "Y_avg_rescaled", "Y_first_factor")


def declaration_15_6() -> Design:
    """declaration_15.6: four ways to build an index from three noisy measures.

    N=500, X alternating 0/1, ``Y_star = 1 + X + 2·N(0,1)``; three measures of
    ``Y_star`` with different loadings and noise; indices: equal-weighted mean
    of z-scores, control-group-standardized mean, rescaled mean, and the
    (typo-faithful) princomp first factor of (Y_1, Y_2). Estimator: mean of
    each index in the X=1 group (``lm_robust(cbind(...) ~ 1, subset = X == 1)``),
    against the standardized-Y_star inquiry.
    """

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        x = np.tile(np.array([0, 1]), n // 2)
        return {"X": x, "Y_star": 1.0 + x + 2.0 * rng.normal(size=n)}

    def measure(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        n = len(df)
        y_star = df["Y_star"].to_numpy()
        x0 = df["X"].to_numpy() == 0
        y_1 = 3.0 + 0.1 * y_star + rng.normal(0.0, 5.0, n)
        y_2 = 2.0 + 1.0 * y_star + rng.normal(0.0, 2.0, n)
        y_3 = 1.0 + 0.5 * y_star + rng.normal(0.0, 1.0, n)

        def adjusted(v: np.ndarray) -> np.ndarray:
            result: np.ndarray = (v - v[x0].mean()) / v[x0].std(ddof=1)
            return result

        return {
            "Y_1": y_1,
            "Y_2": y_2,
            "Y_3": y_3,
            "Y_avg": (_scale(y_1) + _scale(y_2) + _scale(y_3)) / 3.0,
            "Y_avg_adjusted": (adjusted(y_1) + adjusted(y_2) + adjusted(y_3)) / 3.0,
            "Y_avg_rescaled": _scale(_scale(y_1) + _scale(y_2) + _scale(y_3)),
            # princomp(~ Y_1 + Y_2 + Y_2, cor = TRUE): terms() dedupes, so Y_3
            # is absent from the "first factor" — the book's typo, kept.
            "Y_first_factor": _princomp_first_score(y_1, y_2),
        }

    def treated_means(df: pd.DataFrame) -> pd.DataFrame:
        sub = df.loc[df["X"] == 1]
        return pd.concat(
            [_est.lm_robust(f"{col} ~ 1", sub) for col in _OUTCOMES_15_6],
            ignore_index=True,
        )

    return Design(
        Model(n=500, build=build, label="X, Y_star"),
        Inquiry(
            "Y_bar_X1",
            lambda df: float(
                _scale(df["Y_star"].to_numpy())[df["X"].to_numpy() == 1].mean()
            ),
        ),
        Measurement(measure, label="indices"),
        Estimator(treated_means, inquiry="Y_bar_X1", label="Average"),
    )
