"""Chapter 12 declaration: the three-arm clustered survey experiment.

Provenance: replication-materials ``code/declarations/declaration_12.1a.R``
– ``declaration_12.1d.R`` (Blair, Coppock & Humphreys 2023), assembled as
``declaration_12.1 = model + inquiry + data strategy + answer strategy`` by
``code/diagnoses/diagnosis_12.1.R``. Reference outputs:
``diagnosis_12.1.rds`` (sims = 2000; custom diagnosands bias / rmse /
power / cost at n_villages = 192, citizens_per_village = 48) and
``diagnosis_12.2.rds`` (the same design redesigned over
``n_villages ∈ {192, 500} × citizens_per_village ∈ {25, 50, 75, 100}``).

The world is 660 villages of 100 citizens (fabricatr ``add_level``
hierarchy, built here explicitly with ``np.repeat`` for the village-level
shock). Latent potential outcomes are ``pnorm(U_citizen + U_village +
0.10·(Z == "personal") + 0.15·(Z == "social")``; the data strategy cluster-
samples villages, stratum-samples citizens within sampled villages,
cluster-assigns villages to the three conditions with ``prob_each =
(0.250, 0.375, 0.375)``, and observes a Bernoulli draw of the latent
probability. The answer strategy is ``lm_robust(Y_observed ~ Z, clusters =
villages, se_type = "stata")`` with terms ``Zpersonal``/``Zsocial`` paired
positionally to inquiries ``ATE_personal``/``ATE_social``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .. import estimators as _est
from .. import ra as _ra
from ..steps import (
    Assignment,
    Design,
    Estimator,
    Inquiry,
    Measurement,
    Model,
    Sampling,
    potential_outcomes,
)

__all__ = ["declaration_12_1"]

ArrayLike = Union[Sequence[object], np.ndarray, pd.Series]

#: The fixed population: 660 villages of 100 citizens each.
_N_VILLAGES_POP = 660
_CITIZENS_PER_VILLAGE_POP = 100
_CONDITIONS = ("neutral", "personal", "social")
_PROB_EACH = (0.250, 0.375, 0.375)


# ---------------------------------------------------------------------------
# Private randomizr-style procedures (candidates for promotion to ra.py).
# Parity with randomizr is distributional (SEMANTIC_DIFFERENCES §1); each
# helper reproduces the documented sampling distribution, not R's stream.
# ---------------------------------------------------------------------------


def _cluster_rs(clusters: ArrayLike, n: int, rng: np.random.Generator) -> np.ndarray:
    """randomizr ``cluster_rs(clusters, n)``: sample exactly ``n`` whole clusters.

    Complete random sampling over the unique cluster labels (order of first
    appearance), mapped back to units: every unit of a sampled cluster gets
    inclusion 1.
    """
    codes, uniques = pd.factorize(np.asarray(pd.Series(clusters).to_numpy()))
    incl = _ra.complete_rs(len(uniques), n=n, rng=rng)
    result: np.ndarray = incl[codes]
    return result


def _strata_rs(strata: ArrayLike, n: int, rng: np.random.Generator) -> np.ndarray:
    """randomizr ``strata_rs(strata, n)``: sample exactly ``n`` units per stratum.

    Vectorized complete random sampling within every stratum: rank an i.i.d.
    uniform draw within each stratum and keep the ``n`` smallest — a
    uniformly random size-``n`` subset per stratum, identical in
    distribution to per-stratum ``complete_rs`` but O(N log N) overall
    (the per-block loop in :func:`declarepy.ra.block_rs` is quadratic in
    the number of strata and prohibitive at this design's 500+ strata).
    """
    codes, uniques = pd.factorize(np.asarray(pd.Series(strata).to_numpy()))
    N = len(codes)
    counts = np.bincount(codes, minlength=len(uniques))
    if (counts < n).any():
        raise ValueError(f"every stratum needs >= {n} units for strata_rs(n={n})")
    order = np.lexsort((rng.random(N), codes))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    ranks = np.arange(N) - starts[codes[order]]
    s = np.zeros(N, dtype=int)
    s[order] = (ranks < n).astype(int)
    return s


def _complete_ra_each(
    N: int, prob_each: Sequence[float], rng: np.random.Generator
) -> np.ndarray:
    """randomizr ``complete_ra(N, prob_each=...)``: condition indices 0..k−1.

    Assigns ``floor(N·prob_each)`` units to each condition, then allocates
    the remaining units by randomized-systematic sampling on the fractional
    parts, so each condition receives an extra unit with probability exactly
    equal to its fractional part (expected counts = ``N·prob_each``, the
    property randomizr documents). The counts vector is then randomly
    permuted across units.
    """
    p = np.asarray(prob_each, dtype=float)
    if p.min() < 0 or not np.isclose(p.sum(), 1.0):
        raise ValueError("prob_each must be nonnegative and sum to 1")
    exact = N * p
    m = np.floor(exact).astype(int)
    r = int(N - m.sum())
    if r > 0:
        frac = exact - np.floor(exact)  # sums to r
        cum = np.cumsum(frac)
        points = rng.random() + np.arange(r)
        extra = np.searchsorted(cum, points, side="right")
        np.add.at(m, extra, 1)
    z = np.repeat(np.arange(len(p)), m)
    return rng.permutation(z)


def _cluster_ra(
    clusters: ArrayLike,
    conditions: Sequence[object],
    prob_each: Sequence[float],
    rng: np.random.Generator,
) -> np.ndarray:
    """randomizr ``cluster_ra(clusters, conditions, prob_each)``.

    Complete random assignment of the unique clusters (order of first
    appearance) to ``conditions`` with ``prob_each``, mapped back so every
    unit carries its cluster's condition.
    """
    codes, uniques = pd.factorize(np.asarray(pd.Series(clusters).to_numpy()))
    zc = _complete_ra_each(len(uniques), prob_each, rng)
    conds = np.asarray(conditions, dtype=object)
    result: np.ndarray = conds[zc][codes]
    return result


class _LmRobustTermInquiries(Estimator):
    """``lm_robust`` reporting several terms, each answering its own inquiry.

    DeclareDesign's ``declare_estimator(..., term = c(t1, t2), inquiry =
    c(i1, i2))`` pairs terms with inquiries positionally; the shared
    :class:`~declarepy.steps.Estimator` carries a single ``inquiry``, so
    this private subclass keeps a per-row ``inquiry`` column produced by the
    fit. Patsy's treatment-coded names (``Z[T.personal]``) are renamed to
    R's (``Zpersonal``) so diagnosis rows align with the book's tables.
    """

    def __init__(
        self,
        formula: str,
        term_to_inquiry: Mapping[str, str],
        clusters: Optional[str] = None,
        se_type: Optional[str] = None,
        label: str = "estimator",
    ) -> None:
        self._formula = formula
        self._map = dict(term_to_inquiry)
        self._clusters = clusters
        self._se_type = se_type
        super().__init__(self._tidy, inquiry=None, label=label)

    def _tidy(self, df: pd.DataFrame) -> pd.DataFrame:
        tidy = _est.lm_robust(
            self._formula, df, se_type=self._se_type, clusters=self._clusters
        ).copy()
        tidy["term"] = tidy["term"].str.replace(
            r"^([A-Za-z_]\w*)\[T\.(.+)\]$", r"\1\2", regex=True
        )
        out = tidy[tidy["term"].isin(self._map)].copy()
        missing = set(self._map) - set(out["term"])
        if missing:
            raise KeyError(f"terms not in fit: {sorted(missing)}")
        out["inquiry"] = out["term"].map(self._map)
        return out

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = self._tidy(df)
        rows["estimator"] = self.label
        return rows


# ---------------------------------------------------------------------------
# The declaration.
# ---------------------------------------------------------------------------


def _build_population(n: int, rng: np.random.Generator) -> dict[str, object]:
    """fabricatr hierarchy: villages(660, U_village ~ N(0, 0.1²)) ⊃ citizens(100)."""
    v, c = _N_VILLAGES_POP, _CITIZENS_PER_VILLAGE_POP
    return {
        "villages": np.repeat(np.arange(1, v + 1), c),
        "U_village": np.repeat(rng.normal(0.0, 0.1, v), c),
        "U_citizen": rng.normal(size=v * c),
    }


def _po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
    """Y ~ pnorm(U_citizen + U_village + 0.10·(Z=='personal') + 0.15·(Z=='social'))."""
    latent = (
        df["U_citizen"].to_numpy()
        + df["U_village"].to_numpy()
        + 0.10 * float(z == "personal")
        + 0.15 * float(z == "social")
    )
    result: np.ndarray = stats.norm.cdf(latent)
    return result


def _reveal_and_binarize(
    df: pd.DataFrame, rng: np.random.Generator
) -> dict[str, object]:
    """Y_latent = reveal_outcomes(Y ~ Z); Y_observed = rbinom(N, 1, Y_latent)."""
    z = df["Z"].to_numpy()
    latent = np.select(
        [z == c for c in _CONDITIONS],
        [df[f"Y_Z_{c}"].to_numpy() for c in _CONDITIONS],
    )
    return {"Y_latent": latent, "Y_observed": rng.binomial(1, latent)}


def declaration_12_1(n_villages: int = 192, citizens_per_village: int = 48) -> Design:
    """declaration_12.1: cluster-sampled, cluster-assigned three-arm experiment.

    ``n_villages`` and ``citizens_per_village`` are the data-strategy knobs
    the book sweeps in ``diagnosis_12.2.R`` (``redesign(n_villages =
    c(192, 500), citizens_per_village = c(25, 50, 75, 100))``); defaults
    are ``declaration_12.1c.R``'s 192 and 48.
    """
    return Design(
        Model(
            n=_N_VILLAGES_POP * _CITIZENS_PER_VILLAGE_POP,
            build=_build_population,
            label="model_12.1",
        ),
        potential_outcomes(
            _po, conditions=_CONDITIONS, template="{outcome}_Z_{condition}"
        ),
        Inquiry(
            ATE_personal=lambda df: float(
                (df["Y_Z_personal"] - df["Y_Z_neutral"]).mean()
            ),
            ATE_social=lambda df: float((df["Y_Z_social"] - df["Y_Z_neutral"]).mean()),
        ),
        Sampling(
            lambda df, rng: _cluster_rs(df["villages"], n_villages, rng),
            label=f"cluster_rs(villages, n={n_villages})",
        ),
        Sampling(
            lambda df, rng: _strata_rs(df["villages"], citizens_per_village, rng),
            label=f"strata_rs(villages, n={citizens_per_village})",
        ),
        Assignment(
            lambda df, rng: _cluster_ra(df["villages"], _CONDITIONS, _PROB_EACH, rng),
            name="Z",
            label="cluster_ra(villages, prob_each=(0.250, 0.375, 0.375))",
        ),
        Measurement(_reveal_and_binarize, label="reveal + rbinom(Y_latent)"),
        _LmRobustTermInquiries(
            "Y_observed ~ Z",
            {"Zpersonal": "ATE_personal", "Zsocial": "ATE_social"},
            clusters="villages",
            se_type="stata",
            label="estimator",
        ),
    )
