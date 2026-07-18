"""Chapter 19 declarations: complex designs.

Provenance: replication-materials ``code/declarations/declaration_19.1.R``
… ``declaration_19.4.R`` (Blair, Coppock & Humphreys 2023). Reference
outputs: ``diagnosis_19.1.rds`` (discovery diagnosands, sims = 2000),
``diagnosis_19.2.rds`` + ``diagnosis_19a.rds`` (structural-model rows,
sims = 100, alpha × n sweep), ``diagnosis_19.3.rds`` (meta-analysis over
tau ∈ {0, 1}, sims = 2000), ``diagnosis_19.4.rds`` (multi-site coordination,
sims = 2000).

The R sources lean on specialty machinery that declarepy ports privately
in this module (validated against fixed-data R fixtures in
``validation/reference/rgen_t3_ch19_23_fixtures.json``):

* ``bbmle::mle2`` maximum likelihood (19.2): L-BFGS-B on the bargaining
  mixture likelihood with box constraints, Hessian-based standard errors
  and **profile-likelihood** confidence intervals — DeclareDesign's
  ``tidy_try`` tidies an mle2 fit via ``coef(summary(.))`` (z tests) plus
  ``confint(.)``, which profiles.
* ``metafor::rma`` / ``rdss::rma_helper`` + ``rma_mu_tau`` (19.3, 19.4):
  fixed-effect and REML random-effects meta-analysis with z inference for
  ``mu`` and the Q-profile confidence interval for ``tau_sq`` (REML only,
  as in ``confint.rma.uni``).
* ``rdss::best_predictor`` (19.1): the covariate with the highest ANOVA
  R² from regressing the unit treatment effect on ``cut(x, 20)`` bins.
* ``rdss::causal_forest_handler`` (19.1): **approximated** — grf's honest
  causal forest is emulated with an R-learner (orthogonalized
  pseudo-outcome) random forest in scikit-learn; see
  :func:`_causal_forest_columns` for the mapping of grf defaults. Parity
  is distributional (targeting quality / variable-importance ranking),
  never bit-for-bit.
* ``randomizr::block_ra(blocks, block_prob = ...)`` (19.4): per-block
  assignment probabilities (:func:`_block_ra_prob`).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .. import ra as _ra
from ..estimators import lm_robust as _lm_robust
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
    "declaration_19_1",
    "declaration_19_2",
    "declaration_19_3",
    "declaration_19_4",
]


# =====================================================================
# Shared plumbing: one fitted model reused by several estimator steps
# =====================================================================

def _memo_last(compute: Callable[[pd.DataFrame], Any]) -> Callable[[pd.DataFrame], Any]:
    """Single-slot memo so consecutive estimator steps share one fit.

    DeclareDesign's ``declare_estimator(term = c(...), inquiry = c(...))``
    fits once and emits one row per (term, inquiry) pair. declarepy maps
    each pair to its own :class:`Estimator` step, so without care the model
    would be refit per step. The steps of one run receive the *same*
    DataFrame object back-to-back, so caching on ``(id, shape, first data
    row)`` reuses the fit within a run and recomputes on the next run.
    """
    slot: dict[str, Any] = {}

    def key_of(df: pd.DataFrame) -> tuple[object, ...]:
        num = df.select_dtypes("number")
        first = tuple(float(x) for x in num.iloc[0]) if len(num) else ()
        return (id(df), df.shape, first)

    def wrapper(df: pd.DataFrame) -> Any:
        key = key_of(df)
        if slot.get("key") != key:
            slot["key"] = key
            slot["value"] = compute(df)
        return slot["value"]

    return wrapper


# =====================================================================
# declaration_19.1 — discovery with causal forests
# =====================================================================

_COVARIATE_NAMES = [f"X.{j}" for j in range(1, 11)]

#: scikit-learn knobs for the grf causal-forest emulation. grf defaults are
#: num.trees = 2000, min.node.size = 5, sample.fraction = 0.5, mtry =
#: min(ceil(sqrt(p) + 20), p) = 10 (all features here), honest splitting.
#: We use 200 trees (the importance ranking and quintile targeting are
#: stable well before that), bootstrap draws of half the sample for
#: sample.fraction, all features per split, and min_samples_leaf 5 (m̂) /
#: 7 (effect forest) — the extra leaf smoothing stands in for grf's honest
#: estimation halves. The effect-forest leaf size is the emulation's one
#: free calibration knob and the targeting diagnosands are extremely
#: sensitive to it (leaf 6 → predicted-weakest-quintile true effect
#: ≈ +0.012, leaf 7 → ≈ +0.05-0.07, leaf 10 → ≈ +0.18; grf's honest
#: forest gives +0.047). Calibrated at leaf 7 against diagnosis_19.1's
#: targeting diagnosands over 400 fresh draws (mean tau in the
#: predicted-weakest quintile +0.068 vs grf's +0.047, lm estimate on that
#: subset +0.079 vs +0.067 — inside the ±0.02·sd(Y) ≈ ±0.05 protocol
#: band; top variable-importance pick = X.3 in 99.0 % vs grf's 99.3 % of
#: sims); see validation/t3_results_ch19_23.csv.
_CF_N_TREES = 200
_CF_MIN_LEAF_M = 5
_CF_MIN_LEAF_TAU = 7
_CF_MAX_SAMPLES = 0.5


def _f_y(z: float, x1: np.ndarray, x2: np.ndarray, x3: np.ndarray,
         x4: np.ndarray, u: np.ndarray) -> np.ndarray:
    """R source: f_Y <- function(z, X.1, X.2, X.3, X.4, u) ..."""
    result: np.ndarray = z * x1 + z * x2**2 + z * np.exp(x3) + z * x3 * x4 + u
    return result


def _cut_r(x: np.ndarray, cuts: int) -> np.ndarray:
    """Bin indices matching R's ``cut(x, cuts)``.

    R's ``cut.default`` with scalar ``breaks`` spaces the breaks evenly on
    [min, max] and then moves only the two *outermost* breaks outward by
    0.1 % of the range (interior breaks stay on the [min, max] grid).
    """
    lo, hi = float(np.min(x)), float(np.max(x))
    dx = (hi - lo) / 1000.0
    breaks = np.linspace(lo, hi, cuts + 1)
    breaks[0] -= dx
    breaks[-1] += dx
    # Right-closed intervals (lo, hi]: index of the left break strictly below x.
    idx: np.ndarray = np.searchsorted(breaks, x, side="left") - 1
    return np.clip(idx, 0, cuts - 1)


def _binned_r_squared(x: np.ndarray, tau: np.ndarray, cuts: int = 20) -> float:
    """R² of ``lm(tau ~ cut(x, cuts))`` — the ANOVA (between-bin) R²."""
    bins = _cut_r(x, cuts)
    sst = float(np.sum((tau - tau.mean()) ** 2))
    ssr = 0.0
    for b in np.unique(bins):
        grp = tau[bins == b]
        ssr += float(np.sum((grp - grp.mean()) ** 2))
    return 1.0 - ssr / sst


def _best_predictor(df: pd.DataFrame, covariate_names: Sequence[str],
                    cuts: int = 20) -> float:
    """rdss::best_predictor — 1-based index of the top binned-R² covariate."""
    r2 = [
        _binned_r_squared(df[j].to_numpy(dtype=float), df["tau"].to_numpy(dtype=float), cuts)
        for j in covariate_names
    ]
    return float(np.argmax(r2) + 1)


def _causal_forest_columns(
    df: pd.DataFrame,
    rng: np.random.Generator,
    covariate_names: Sequence[str],
    share_train: float = 0.5,
) -> dict[str, object]:
    """rdss::causal_forest_handler, emulated with an R-learner forest.

    grf's causal forest orthogonalizes (Y − m̂(x), W − ê(x)) and grows an
    honest forest on the treatment-effect moment. Emulation: (1) train/test
    split by ``complete_rs(prob = share_train)`` (as in rdss); (2) center Y
    with out-of-bag random-forest predictions m̂ and W with its training
    mean (assignment is simple_ra(0.5)); (3) fit a random forest to the
    R-learner pseudo-outcome (Y − m̂)/(W − Ŵ) with weights (W − Ŵ)²; (4)
    ``pred`` = OOB predictions for train rows, standard predictions for
    test rows; ``var_imp`` = 1-based argmax of the effect forest's feature
    importances (grf's ``variable_importance |> which.max``).
    """
    from sklearn.ensemble import RandomForestRegressor

    X = df[list(covariate_names)].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    w = df["Z"].to_numpy(dtype=float)
    train = _ra.complete_rs(len(df), prob=share_train, rng=rng).astype(bool)

    seed_m, seed_tau = (int(s) for s in rng.integers(0, 2**31, size=2))
    m_forest = RandomForestRegressor(
        n_estimators=_CF_N_TREES, min_samples_leaf=_CF_MIN_LEAF_M,
        max_features=None, max_samples=_CF_MAX_SAMPLES,
        oob_score=True, bootstrap=True, random_state=seed_m, n_jobs=-1,
    )
    m_forest.fit(X[train], y[train])
    m_hat = np.asarray(m_forest.oob_prediction_, dtype=float)

    w_bar = float(w[train].mean())
    resid_w = w[train] - w_bar
    pseudo = (y[train] - m_hat) / resid_w
    tau_forest = RandomForestRegressor(
        n_estimators=_CF_N_TREES, min_samples_leaf=_CF_MIN_LEAF_TAU,
        max_features=None, max_samples=_CF_MAX_SAMPLES,
        oob_score=True, bootstrap=True, random_state=seed_tau, n_jobs=-1,
    )
    tau_forest.fit(X[train], pseudo, sample_weight=resid_w**2)

    pred = np.empty(len(df), dtype=float)
    pred[train] = np.asarray(tau_forest.oob_prediction_, dtype=float)
    pred[~train] = tau_forest.predict(X[~train])
    var_imp = int(np.argmax(tau_forest.feature_importances_) + 1)
    return {
        "pred": pred,
        "var_imp": float(var_imp),
        "train": train.astype(bool),
        "test": ~train,
    }


def _lm_z_on_subset(subset_col: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """lm_robust(Y ~ Z, subset = <subset_col>), reporting the Z row (HC2)."""

    def fit(df: pd.DataFrame) -> pd.DataFrame:
        sub = df.loc[df[subset_col].astype(bool)]
        tidy = _lm_robust("Y ~ Z", sub)
        return tidy[tidy["term"] == "Z"].copy()

    return fit


def declaration_19_1() -> Design:
    """declaration_19.1: discovery of heterogeneous effects via causal forest.

    N = 1000, ten independent standard-normal covariates, simple 50/50
    assignment; effects are driven by X.1, X.2², exp(X.3) and X.3·X.4.
    Inquiries: the best binned predictor of tau, the ATE, and the mean
    effect among the truly-worst / predicted-weakest 20 % (test-half and
    full-sample versions). Estimators: full-sample OLS, OLS on the
    predicted-weak subsets, and the forest's top variable-importance index.
    """

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        cols: dict[str, object] = {
            name: rng.normal(size=n) for name in _COVARIATE_NAMES
        }
        cols["U"] = rng.normal(size=n)
        cols["Z"] = _ra.simple_ra(n, prob=0.5, rng=rng)
        return cols

    def add_pos(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        args = (
            df["X.1"].to_numpy(), df["X.2"].to_numpy(),
            df["X.3"].to_numpy(), df["X.4"].to_numpy(), df["U"].to_numpy(),
        )
        y1 = _f_y(1.0, *args)
        y0 = _f_y(0.0, *args)
        return {"Y1": y1, "Y0": y0, "tau": y1 - y0}

    def add_subsets(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        pred = df["pred"].to_numpy(dtype=float)
        test = df["test"].to_numpy(dtype=bool)
        low_test = test & (pred < np.quantile(pred[test], 0.2))
        low_all = pred < np.quantile(pred, 0.2)
        return {"low_test": low_test, "low_all": low_all}

    def _tau_q(df: pd.DataFrame, mask: Optional[np.ndarray] = None) -> float:
        tau = df["tau"].to_numpy(dtype=float)
        if mask is not None:
            tau = tau[mask]
        return float(tau[tau <= np.quantile(tau, 0.2)].mean())

    return Design(
        Model(n=1000, build=build, label="model"),
        Model(transform=add_pos, label="potential_outcomes"),
        Inquiry("best_predictor", lambda df: _best_predictor(df, _COVARIATE_NAMES)),
        reveal_outcomes(),
        Measurement(
            lambda df, rng: _causal_forest_columns(df, rng, _COVARIATE_NAMES, 0.5),
            label="causal_forest",
        ),
        Measurement(add_subsets, label="fabricate_subsets"),
        Inquiry(
            ate=lambda df: float(df["tau"].mean()),
            worst_effects_all=lambda df: _tau_q(df),
            worst_effects_test=lambda df: _tau_q(df, df["test"].to_numpy(dtype=bool)),
            weak_effects_all=lambda df: float(df.loc[df["low_all"], "tau"].mean()),
            weak_effects_test=lambda df: float(df.loc[df["low_test"], "tau"].mean()),
        ),
        Estimator.lm_robust("Y ~ Z", inquiry="ate"),
        Estimator(_lm_z_on_subset("low_test"), inquiry="weak_effects_test", label="lm_weak_test"),
        Estimator(_lm_z_on_subset("low_test"), inquiry="worst_effects_test", label="lm_weak_test"),
        Estimator(_lm_z_on_subset("low_all"), inquiry="weak_effects_all", label="lm_weak_all"),
        Estimator(_lm_z_on_subset("low_all"), inquiry="worst_effects_all", label="lm_weak_all"),
        Estimator(
            lambda df: {"estimate": float(df["var_imp"].iloc[0])},
            inquiry="best_predictor",
            label="cf",
        ),
    )


# =====================================================================
# declaration_19.2 — structural estimation of a bargaining model
# =====================================================================

def _offer(n_rounds: int, d: float) -> float:
    """Equilibrium offer for a game of length n: Σ_{t=2..n} (−1)^t d^(t−1)."""
    return float(sum(((-1.0) ** t) * d ** (t - 1) for t in range(2, n_rounds + 1)))


def _beta_logpdf(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized log dbeta(y, a, b) (faster than stats.beta.logpdf)."""
    result: np.ndarray = (
        special.xlogy(a - 1.0, y) + special.xlog1py(b - 1.0, -y) - special.betaln(a, b)
    )
    return result


_MLE_START = np.array([2.0, 0.50, 0.50])
_MLE_LOWER = np.array([0.10, 0.01, 0.01])
_MLE_UPPER = np.array([100.0, 0.99, 0.99])
_CHI2_95_HALF = float(stats.chi2.ppf(0.95, 1)) / 2.0


def _bargaining_nll(y: np.ndarray, z: np.ndarray, n_rounds: int) -> Callable[[np.ndarray], float]:
    """The declaration's likelihood(): beta mixture of norm and equilibrium types."""

    def nll(theta: np.ndarray) -> float:
        k, d, a = float(theta[0]), float(theta[1]), float(theta[2])
        off = _offer(n_rounds, d)
        m = np.where(z == 1, off, 1.0 - off)
        lf_norm = _beta_logpdf(y, np.full_like(y, k * 0.75), np.full_like(y, k * 0.25))
        lf_eq = _beta_logpdf(y, k * m, k * (1.0 - m))
        val = -float(
            np.sum(np.logaddexp(np.log(a) + lf_norm, np.log1p(-a) + lf_eq))
        )
        return val if np.isfinite(val) else 1e10

    return nll


def _num_hessian(fn: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
    """Central-difference Hessian (bbmle uses numDeriv's Richardson variant)."""
    p = len(x)
    h = 1e-4 * np.maximum(np.abs(x), 1.0)
    H = np.empty((p, p))
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ei[i] = h[i]
            ej = np.zeros(p); ej[j] = h[j]
            H[i, j] = H[j, i] = (
                fn(x + ei + ej) - fn(x + ei - ej) - fn(x - ei + ej) + fn(x - ei - ej)
            ) / (4.0 * h[i] * h[j])
    return H


def _profile_ci(
    nll: Callable[[np.ndarray], float],
    mle: np.ndarray,
    nll_min: float,
    j: int,
    se_j: float,
) -> tuple[float, float]:
    """Profile-likelihood CI for parameter ``j`` (bbmle's confint.mle2).

    Root-finds the profile deviance 2·(NLLprof(c) − NLLmin) = χ²₁(0.95) on
    each side, re-optimizing the other parameters (warm-started, same box
    constraints). A side whose profile never crosses the cutoff before the
    box bound is reported NaN, as bbmle does.
    """
    others = [i for i in range(len(mle)) if i != j]
    bounds = [( _MLE_LOWER[i], _MLE_UPPER[i]) for i in others]
    warm = {"x": mle[others].copy()}

    def prof(c: float) -> float:
        def sub_nll(rest: np.ndarray) -> float:
            full = np.empty(len(mle))
            full[j] = c
            full[others] = rest
            return nll(full)

        res = optimize.minimize(
            sub_nll, warm["x"], method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-11},
        )
        if res.success or res.fun < 1e9:
            warm["x"] = np.asarray(res.x)
        return float(res.fun)

    step = 0.5 * (se_j if np.isfinite(se_j) and se_j > 0 else
                  0.05 * (_MLE_UPPER[j] - _MLE_LOWER[j]))
    target = nll_min + _CHI2_95_HALF
    out: list[float] = []
    for sign in (-1.0, 1.0):
        warm["x"] = mle[others].copy()
        lo_c, lo_v = float(mle[j]), nll_min
        bound = float(_MLE_LOWER[j] if sign < 0 else _MLE_UPPER[j])
        found = float("nan")
        for i in range(1, 200):
            c = float(mle[j] + sign * i * step)
            hit_bound = (c <= bound) if sign < 0 else (c >= bound)
            if hit_bound:
                c = bound
            v = prof(c)
            if v >= target:
                root = optimize.brentq(
                    lambda cc: prof(float(cc)) - target, lo_c, c, xtol=1e-6
                )
                found = float(root)
                break
            lo_c, lo_v = c, v
            if hit_bound:
                break  # profile never reached the cutoff inside the box
        out.append(found)
    return out[0], out[1]


def _fit_bargaining(df: pd.DataFrame, n_rounds: int) -> pd.DataFrame:
    """mle2-style fit: tidy rows for terms k, d, a (Wald z + profile CI)."""
    y = df["y"].to_numpy(dtype=float)
    z = df["Z"].to_numpy(dtype=float)
    nll = _bargaining_nll(y, z, n_rounds)
    res = optimize.minimize(
        nll, _MLE_START, method="L-BFGS-B",
        bounds=list(zip(_MLE_LOWER, _MLE_UPPER)),
        options={"maxiter": 500, "ftol": 1e-12},
    )
    mle = np.asarray(res.x, dtype=float)
    nll_min = float(res.fun)
    H = _num_hessian(nll, mle)
    try:
        vcov = np.linalg.inv(H)
        se = np.sqrt(np.diag(vcov))
        se = np.where(np.isfinite(se), se, np.nan)
    except np.linalg.LinAlgError:
        se = np.full(3, np.nan)
    zstat = mle / se
    pval = 2.0 * stats.norm.sf(np.abs(zstat))
    rows = []
    for jj, term in enumerate(["k", "d", "a"]):
        lo, hi = _profile_ci(nll, mle, nll_min, jj, float(se[jj]))
        rows.append(
            {
                "term": term,
                "estimate": float(mle[jj]),
                "std_error": float(se[jj]),
                "statistic": float(zstat[jj]),
                "p_value": float(pval[jj]),
                "conf_low": lo,
                "conf_high": hi,
            }
        )
    return pd.DataFrame(rows)


def declaration_19_2(
    n: int = 2,
    alpha: float = 0.5,
    delta: float = 0.8,
    kappa: float = 2.0,
    N: int = 200,
) -> Design:
    """declaration_19.2: structural estimation of a bargaining game.

    A share ``alpha`` of players are behavioral (norm) types receiving
    payoff 0.75; the rest earn the equilibrium offer for a game of ``n``
    rounds at discount ``delta``. Payoffs are measured with Beta noise
    governed by ``kappa``; (kappa, delta, alpha) are recovered by maximum
    likelihood with term ↔ inquiry mapping k → kappa, d → delta, a → alpha.
    """

    def build(n_units: int, rng: np.random.Generator) -> dict[str, object]:
        return {
            "type": rng.binomial(1, alpha, n_units),
            "n": np.full(n_units, n),
        }

    def measure(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        off = _offer(n, delta)
        typ = df["type"].to_numpy(dtype=float)
        z = df["Z"].to_numpy(dtype=float)
        pi = typ * 0.75 + (1.0 - typ) * (z * off + (1.0 - z) * (1.0 - off))
        y = rng.beta(pi * kappa, (1.0 - pi) * kappa)
        return {"pi": pi, "y": y}

    fit = _memo_last(lambda df: _fit_bargaining(df, n))

    def row_for(term: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
        def get(df: pd.DataFrame) -> pd.DataFrame:
            tidy = fit(df)
            return tidy[tidy["term"] == term].copy()

        return get

    return Design(
        Model(n=N, build=build, label="model"),
        Inquiry(
            kappa=lambda df: kappa,
            delta=lambda df: delta,
            alpha=lambda df: alpha,
        ),
        Assignment.complete(),
        Measurement(measure, label="payoffs"),
        Estimator(row_for("k"), inquiry="kappa", label="Structural model"),
        Estimator(row_for("d"), inquiry="delta", label="Structural model"),
        Estimator(row_for("a"), inquiry="alpha", label="Structural model"),
    )


# =====================================================================
# metafor::rma port (used by 19.3 and 19.4)
# =====================================================================

_Z_975 = float(stats.norm.ppf(0.975))


def _rma_uni(yi: np.ndarray, sei: np.ndarray, method: str = "REML") -> dict[str, float]:
    """metafor::rma(yi, sei, method) — FE or REML intercept-only model.

    Returns mu, se(mu), z, p, z-based CI, tau2, se(tau2) (expected REML
    information), and the Q-profile CI for tau2 (REML only; the FE model
    fixes tau2 = 0 with no interval, as rdss::rma_mu_tau reports it).
    """
    v = sei.astype(float) ** 2
    k = len(yi)

    if method == "FE":
        w = 1.0 / v
        mu = float(np.sum(w * yi) / np.sum(w))
        se_mu = float(np.sqrt(1.0 / np.sum(w)))
        tau2, se_tau2 = 0.0, float("nan")
        tau2_lb = tau2_ub = float("nan")
    elif method == "REML":

        def neg_restricted_ll(t2: float) -> float:
            wi = 1.0 / (v + t2)
            mu_t = float(np.sum(wi * yi) / np.sum(wi))
            return 0.5 * float(
                np.sum(np.log(v + t2))
                + np.log(np.sum(wi))
                + np.sum(wi * (yi - mu_t) ** 2)
            )

        upper = max(1e-6, 100.0 * float(np.var(yi, ddof=1)))
        sol = optimize.minimize_scalar(
            neg_restricted_ll, bounds=(0.0, upper), method="bounded",
            options={"xatol": 1e-12},
        )
        tau2 = float(sol.x)
        if neg_restricted_ll(0.0) <= float(sol.fun):
            tau2 = 0.0
        w = 1.0 / (v + tau2)
        mu = float(np.sum(w * yi) / np.sum(w))
        se_mu = float(np.sqrt(1.0 / np.sum(w)))
        # Expected REML information: 0.5·tr(P²), P = W − W11'W/Σw.
        sw, sw2, sw3 = float(np.sum(w)), float(np.sum(w**2)), float(np.sum(w**3))
        info = 0.5 * (sw2 - 2.0 * sw3 / sw + (sw2 / sw) ** 2)
        se_tau2 = float(np.sqrt(1.0 / info)) if info > 0 else float("nan")
        tau2_lb, tau2_ub = _tau2_q_profile_ci(yi, v)
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown rma method: {method!r}")

    zval = mu / se_mu
    return {
        "mu": mu,
        "se": se_mu,
        "zval": zval,
        "pval": float(2.0 * stats.norm.sf(abs(zval))),
        "ci_lb": mu - _Z_975 * se_mu,
        "ci_ub": mu + _Z_975 * se_mu,
        "tau2": tau2,
        "se_tau2": se_tau2,
        "tau2_ci_lb": tau2_lb,
        "tau2_ci_ub": tau2_ub,
    }


def _tau2_q_profile_ci(yi: np.ndarray, v: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Q-profile CI for tau² (metafor's confint.rma.uni default).

    The generalized Q statistic Q(t²) = Σ w(t²)·(y − μ̂(t²))² is decreasing
    in t²; bounds solve Q = χ²_{k−1}(0.975) (lower) and χ²_{k−1}(0.025)
    (upper), truncated at 0.
    """
    k = len(yi)
    alpha = 1.0 - level

    def q_gen(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = float(np.sum(w * yi) / np.sum(w))
        return float(np.sum(w * (yi - mu) ** 2))

    crit_lo = float(stats.chi2.ppf(1.0 - alpha / 2.0, k - 1))
    crit_hi = float(stats.chi2.ppf(alpha / 2.0, k - 1))

    def solve(crit: float) -> float:
        if q_gen(0.0) <= crit:
            return 0.0
        hi = max(1.0, float(np.var(yi, ddof=1)))
        for _ in range(200):
            if q_gen(hi) < crit:
                break
            hi *= 2.0
        else:  # pragma: no cover - Q always falls below crit eventually
            return float("nan")
        return float(optimize.brentq(lambda t: q_gen(t) - crit, 0.0, hi, xtol=1e-10))

    return solve(crit_lo), solve(crit_hi)


def _rma_mu_row(fit: dict[str, float]) -> dict[str, float | str]:
    """rdss::rma_mu_tau's mu row (broom::tidy of the rma fit, z inference)."""
    return {
        "term": "mu",
        "estimate": fit["mu"],
        "std_error": fit["se"],
        "statistic": fit["zval"],
        "p_value": fit["pval"],
        "conf_low": fit["ci_lb"],
        "conf_high": fit["ci_ub"],
    }


def _rma_tau_row(fit: dict[str, float]) -> dict[str, float | str]:
    """rdss::rma_mu_tau's tau_sq row (no test statistic; CI for REML only)."""
    return {
        "term": "tau_sq",
        "estimate": fit["tau2"],
        "std_error": fit["se_tau2"],
        "statistic": float("nan"),
        "p_value": float("nan"),
        "conf_low": fit["tau2_ci_lb"],
        "conf_high": fit["tau2_ci_ub"],
    }


# =====================================================================
# declaration_19.3 — meta-analysis (random- vs fixed-effects)
# =====================================================================

def declaration_19_3(tau: float = 0.0, mu: float = 0.2, N: int = 200) -> Design:
    """declaration_19.3: 200 studies, REML vs FE meta-analysis of mu and tau².

    Study standard errors are max(0.1, |N(0.8, 0.5)|); site effects theta ~
    N(mu, tau); observed estimates ~ N(theta, se). The book diagnoses the
    redesign over tau ∈ {0, 1}.
    """

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        std_error = np.maximum(0.1, np.abs(rng.normal(0.8, 0.5, n)))
        theta = rng.normal(mu, tau, n)
        estimate = rng.normal(theta, std_error)
        return {
            "site": np.arange(1, n + 1),
            "std.error": std_error,
            "theta": theta,
            "estimate": estimate,
        }

    def fit_for(method: str) -> Callable[[pd.DataFrame], dict[str, float]]:
        return _memo_last(
            lambda df: _rma_uni(
                df["estimate"].to_numpy(dtype=float),
                df["std.error"].to_numpy(dtype=float),
                method,
            )
        )

    reml = fit_for("REML")
    fe = fit_for("FE")

    return Design(
        Model(n=N, build=build, label="model"),
        Inquiry(mu=lambda df: mu, tau_sq=lambda df: tau**2),
        Estimator(lambda df: _rma_mu_row(reml(df)), inquiry="mu", label="random-effects"),
        Estimator(lambda df: _rma_tau_row(reml(df)), inquiry="tau_sq", label="random-effects"),
        Estimator(lambda df: _rma_mu_row(fe(df)), inquiry="mu", label="fixed-effects"),
        Estimator(lambda df: _rma_tau_row(fe(df)), inquiry="tau_sq", label="fixed-effects"),
    )


# =====================================================================
# declaration_19.4 — multi-site studies and coordination
# =====================================================================

_STUDY_SIZES = (500, 1000, 1500, 2000, 2500)
_STUDY_PROBS = (0.5, 0.5, 0.6, 0.7, 0.8)
_STUDY_INTERCEPTS = (1.0, 2.0, 3.0, 4.0, 5.0)
_STUDY_PRIORS = tuple(np.linspace(0.0, 0.3, 5))  # seq(0, 0.3, length = 5)


def _block_ra_prob(
    blocks: pd.Series, probs: Sequence[float], rng: np.random.Generator
) -> np.ndarray:
    """randomizr::block_ra(blocks, block_prob = probs).

    Complete random assignment within each block with that block's own
    probability; ``probs`` align with the sorted unique block labels
    (randomizr's convention — identical to first-appearance order here).
    """
    labels = np.asarray(pd.Series(blocks).to_numpy())
    z = np.zeros(len(labels), dtype=int)
    uniq = pd.unique(labels)
    if len(uniq) != len(probs):
        raise ValueError("need exactly one probability per block")
    for lab, p in zip(sorted(uniq), probs):
        idx = np.flatnonzero(labels == lab)
        z[idx] = _ra.complete_ra(len(idx), prob=float(p), rng=rng)
    return z


def _site_lm_step(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """The declare_step: per-site lm_robust(Y ~ Z_implemented), Z rows only."""
    rows: list[dict[str, object]] = []
    for site, grp in df.groupby("sites", sort=True):
        tidy = _lm_robust("Y ~ Z_implemented", grp)
        z_row = tidy[tidy["term"] == "Z_implemented"].iloc[0]
        rows.append(
            {
                "sites": site,
                "estimate": float(z_row["estimate"]),
                "std.error": float(z_row["std_error"]),
            }
        )
    return pd.DataFrame(rows)


def declaration_19_4(study_coordination: str = "high") -> Design:
    """declaration_19.4: five sites, implemented vs common treatments.

    Sites of sizes 500…2500 have intercepts 1…5 and site effects tau_1 ~
    N(seq(0, 0.3, len 5), 0.1); with low coordination each site implements
    max(tau_1, tau_2) with tau_2 ~ N(0.3, 0.2). Assignment uses per-site
    probabilities (0.5, 0.5, 0.6, 0.7, 0.8); site-level OLS estimates are
    meta-analyzed by REML, answering both ATE_implemented and ATE_common.
    """
    if study_coordination not in ("high", "low"):
        raise ValueError("study_coordination must be 'high' or 'low'")

    def build(n: int, rng: np.random.Generator) -> pd.DataFrame:
        sizes = np.asarray(_STUDY_SIZES)
        tau_1 = rng.normal(np.asarray(_STUDY_PRIORS), 0.1)
        tau_2 = rng.normal(0.3, 0.2, 5)
        tau = tau_1 if study_coordination == "high" else np.maximum(tau_1, tau_2)
        total = int(sizes.sum())
        site_ids = np.repeat(np.arange(1, 6), sizes)
        return pd.DataFrame(
            {
                "sites": site_ids,
                "size": np.repeat(sizes, sizes),
                "intercept": np.repeat(np.asarray(_STUDY_INTERCEPTS), sizes),
                "tau_1": np.repeat(tau_1, sizes),
                "tau": np.repeat(tau, sizes),
                "U": rng.normal(size=total),
            }
        )

    def add_pos(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        base = df["intercept"].to_numpy() + df["U"].to_numpy()
        return {
            # potential_outcomes(Y ~ intercept + tau * Z_implemented + U)
            "Y_implemented_0": base,
            "Y_implemented_1": base + df["tau"].to_numpy(),
            # potential_outcomes(Y ~ intercept + tau_1 * Z_common + U)
            "Y_common_0": base,
            "Y_common_1": base + df["tau_1"].to_numpy(),
        }

    def site_mean_diff(df: pd.DataFrame, y1: str, y0: str) -> float:
        per_site = (
            df.assign(_d=df[y1] - df[y0]).groupby("sites")["_d"].mean()
        )
        return float(per_site.mean())

    rma_fit = _memo_last(
        lambda df: _rma_uni(
            df["estimate"].to_numpy(dtype=float),
            df["std.error"].to_numpy(dtype=float),
            "REML",
        )
    )

    return Design(
        Model(n=int(sum(_STUDY_SIZES)), build=build, label="model"),
        Model(transform=add_pos, label="potential_outcomes"),
        Inquiry(
            ATE_implemented=lambda df: site_mean_diff(df, "Y_implemented_1", "Y_implemented_0"),
            ATE_common=lambda df: site_mean_diff(df, "Y_common_1", "Y_common_0"),
        ),
        Assignment(
            lambda df, rng: _block_ra_prob(df["sites"], _STUDY_PROBS, rng),
            name="Z_implemented",
            label="block_ra(block_prob)",
        ),
        reveal_outcomes(outcome="Y", assignment="Z_implemented",
                        template="Y_implemented_{condition}"),
        Measurement(_site_lm_step, label="site_lm_robust"),
        Estimator(lambda df: _rma_mu_row(rma_fit(df)), inquiry="ATE_implemented", label="estimator"),
        Estimator(lambda df: _rma_mu_row(rma_fit(df)), inquiry="ATE_common", label="estimator"),
    )
