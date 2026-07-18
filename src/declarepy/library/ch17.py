"""Chapter 17 declarations: experimental descriptive designs.

Provenance: replication-materials ``code/declarations/declaration_17.1.R``
through ``declaration_17.6_b.R`` (Blair, Coppock & Humphreys 2023).
Reference outputs:

* ``declaration_17.1`` — no saved diagnosis in the archive (the book's
  ``diagnosis_17.1.rds`` diagnoses declaration 17.2), so it is validated
  against the freshly R-generated
  ``validation/reference/rgen_t3_ch17_17_1.json`` (sims = 2000).
* ``declaration_17.2`` — ``diagnosis_17.1.rds`` (sims = 2000).
* ``declaration_17.3`` — ``diagnosis_17.2.rds`` (custom diagnosands: bias
  and mean CI width, sims = 2000).
* ``declaration_17.4`` — ``diagnosis_17.3.rds`` (redesign over
  ``proportion_hiding`` in {0, .1, .2, .3} x ``N`` in {500, ..., 2500}).
* ``declaration_17.5`` — ``diagnosis_17.4.rds`` (conjoint AMCEs, five
  attribute-level inquiries).
* ``declaration_17.6`` — ``diagnosis_17.5.rds`` (redesign over ``deceive``
  in {TRUE, FALSE}); the behavioral functions from ``declaration_17.6_a.R``
  are the private helpers ``_invested`` / ``_returned`` /
  ``_average_invested`` / ``_average_returned`` below.

Translation notes
-----------------
* declaration_17.1 (audit study): the book's R code draws ``type`` with
  lowercase labels ("Never-responder", "Always-responder") but compares
  against the capitalized "Never-Responder" / "Always-Responder" inside
  ``if_else``, so both tests are always FALSE — every unit has
  ``Y_Z_white = 1`` and ``Y_Z_latino = 0`` and the difference in means is
  exactly 1 while the estimand stays ~0.05. The translation preserves this
  faithfully (the R-generated reference is built from the same code).
  declaration_17.2 spells the types consistently and behaves as intended.
* declaration_17.5 (conjoint): rdss's ``conjoint_inquiries`` draws a FRESH
  conjoint assignment inside the inquiry for every attribute-level
  comparison, forces profile 2's attribute to the reference vs. the level,
  and takes the mean difference in profile-2 choice. Because declarepy
  inquiries are deterministic functions of the frame, that stochastic
  computation lives in a ``Model(transform=...)`` prep step (which receives
  the run's rng) that stores each estimand as a constant column the Inquiry
  step reads back.
* declaration_17.5's estimator is cjoint's ``amce`` as tidied by
  ``rdss::tidy.amce``: OLS of choice on all attribute-level dummies
  (baselines = first levels: Man / Left / North), respondent-clustered
  Stata-style CR1 sandwich SEs (``cluster_se_glm``'s
  ``(M/(M-1))((N-1)/(N-K))`` correction), and z-based p-values and CIs.
  Each AMCE row answers the same-named inquiry, so the private estimator
  subclass keeps a per-row ``inquiry`` column.
* declaration_17.6: ``rho`` is defined in the R source but never used by the
  design, so it is not a factory parameter. R's
  ``pivot_wider(id_cols = pair, names_from = role, values_from = c(ID, a))``
  step is a ``Model(transform=...)``. The deception measurement draws
  ``runif(N)`` unconditionally (as R does) before mixing by ``deceive``.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..estimators import lm_robust as _lm_robust
from ..ra import block_ra as _block_ra
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

__all__ = [
    "declaration_17_1",
    "declaration_17_2",
    "declaration_17_3",
    "declaration_17_4",
    "declaration_17_5",
    "declaration_17_6",
]


# ---------------------------------------------------------------------------
# Private helpers (candidates for promotion to shared modules)
# ---------------------------------------------------------------------------


def _complete_ra_conditions(
    n: int, conditions: Sequence[object], rng: np.random.Generator
) -> np.ndarray:
    """randomizr's ``complete_ra(N, conditions = ...)`` / ``num_arms = k``.

    Each of the k conditions receives exactly ``n // k`` units; the
    ``n % k`` remainder conditions are chosen at random (randomizr's rule)
    and get one extra; the labeled vector is then permuted. Returns an
    object array of condition labels in unit order.
    """
    k = len(conditions)
    counts = np.full(k, n // k, dtype=int)
    r = n % k
    if r:
        counts[rng.choice(k, size=r, replace=False)] += 1
    idx = rng.permutation(np.repeat(np.arange(k), counts))
    return np.asarray(conditions, dtype=object)[idx]


# ---- declaration_17.6_a: behavioral functions of the trust game -----------

#: seq(0, 1, .01) — the a_2 grid average_invested integrates over.
_A2_GRID: np.ndarray = np.linspace(0.0, 1.0, 101)
#: seq(0.01, 1, .01) — the x1 grid average_returned integrates over.
_X1_GRID: np.ndarray = np.linspace(0.01, 1.0, 100)


def _invested(a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Amount player 1 invests given norms (a_1, a_2) — 17.6_a's ``invested``.

    ``u_a`` is the utility of giving a_1, ``u_b`` of giving everything;
    player 1 gives a_1 when u_a > u_b, else 1. log(0) = -inf cases (a_2 at
    the grid endpoints) resolve exactly as in R.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        u_a = (1 - a1) * np.log(1 - a1) + a1 * np.log(2 * a1)
        u_b = (1 - a1) * np.log(2 * a2) + a1 * np.log(2 * (1 - a2))
    return np.where(u_a > u_b, a1, 1.0)


def _returned(x1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Share player 2 returns of an investment x1 — 17.6_a's ``returned``."""
    ind = (x1 > (1 - a2) / (1 + a2)).astype(float)
    result: np.ndarray = ((2 * a2 * x1 - (1 - a2) * (1 - x1)) / (2 * x1)) * ind
    return result


def _average_invested(a: np.ndarray) -> np.ndarray:
    """``average_invested`` for each a_1 in ``a``: mean over the a_2 grid."""
    return np.asarray(_invested(np.asarray(a)[:, None], _A2_GRID[None, :]).mean(axis=1))


def _average_returned(a2: np.ndarray) -> np.ndarray:
    """``average_returned`` for each a_2 in ``a2``: mean over the x1 grid."""
    return np.asarray(_returned(_X1_GRID[None, :], np.asarray(a2)[:, None]).mean(axis=1))


# ---- declaration_17.5: conjoint helpers (rdss + cjoint) -------------------

#: The book's levels_list — order fixes each attribute's reference level.
_CONJOINT_LEVELS: dict[str, tuple[str, ...]] = {
    "gender": ("Man", "Woman"),
    "party": ("Left", "Right"),
    "region": ("North", "South", "East", "West"),
}

#: Inquiry/term names in rdss::conjoint_inquiries' order.
_AMCE_NAMES: tuple[str, ...] = tuple(
    f"{attr}{lev}" for attr, levels in _CONJOINT_LEVELS.items() for lev in levels[1:]
)


def _utility_v(gender: np.ndarray, party: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Deterministic part of the book's ``conjoint_utility`` (add uij for U)."""
    result: np.ndarray = 0.25 * (
        (gender == "Woman") & np.isin(region, ("North", "East"))
    ) + 0.5 * ((party == "Right") & np.isin(region, ("North", "South")))
    return result


def _conjoint_inquiry_prep(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    """rdss::conjoint_inquiries as a stochastic prep step.

    For each (attribute, level) comparison: draw a fresh complete-RA
    assignment of ALL attributes over the full frame, force profile 2's
    attribute to the reference (A) vs. the level (B), recompute utilities
    with the realized ``uij``, and store
    ``mean(1{U1 <= U2^B} - 1{U1 <= U2^A})`` (profile-2 choice, ties to
    profile 2, exactly as ``conjoint_measurement``) as a constant column
    ``estimand_<attribute><level>`` for the Inquiry step to read.
    """
    n = len(df)
    uij = df["uij"].to_numpy()
    p1 = slice(0, None, 2)  # rows alternate profile 1, profile 2
    p2 = slice(1, None, 2)
    out: dict[str, object] = {}
    for attr, levels in _CONJOINT_LEVELS.items():
        reference = levels[0]
        for lev in levels[1:]:
            assign = {
                a: _complete_ra_conditions(n, lv, rng)
                for a, lv in _CONJOINT_LEVELS.items()
            }
            u1 = _utility_v(
                assign["gender"][p1], assign["party"][p1], assign["region"][p1]
            ) + uij[p1]
            second = {a: v[p2] for a, v in assign.items()}
            n2 = len(uij[p2])
            forced_a = dict(second)
            forced_a[attr] = np.full(n2, reference, dtype=object)
            forced_b = dict(second)
            forced_b[attr] = np.full(n2, lev, dtype=object)
            u2a = _utility_v(forced_a["gender"], forced_a["party"], forced_a["region"]) + uij[p2]
            u2b = _utility_v(forced_b["gender"], forced_b["party"], forced_b["region"]) + uij[p2]
            out[f"estimand_{attr}{lev}"] = float(
                np.mean((u1 <= u2b).astype(float) - (u1 <= u2a).astype(float))
            )
    return out


def _conjoint_measure(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    """rdss::conjoint_measurement: U from the utility fn, forced choice.

    Within each (subject, task) pair, profile 1's choice is ``1{U1 > U2}``
    and profile 2's is ``1{U1 <= U2}`` (ties go to profile 2, as in rdss).
    """
    u = _utility_v(
        df["gender"].to_numpy(), df["party"].to_numpy(), df["region"].to_numpy()
    ) + df["uij"].to_numpy()
    first_wins = (u[0::2] > u[1::2]).astype(float)
    choice = np.empty(len(df), dtype=float)
    choice[0::2] = first_wins
    choice[1::2] = 1.0 - first_wins
    return {"U": u, "choice": choice}


def _amce_fit(
    df: pd.DataFrame,
    levels_list: Mapping[str, Sequence[str]],
    respondent_id: str,
) -> pd.DataFrame:
    """cjoint::amce + rdss::tidy.amce for a uniform, no-interaction conjoint.

    OLS of ``choice`` on all attribute-level dummies (baseline = first
    level of each attribute), respondent-clustered Stata-style CR1 sandwich
    variance (cjoint's ``cluster_se_glm``: ``dfc = (M/(M-1))((N-1)/(N-K))``),
    z statistics, normal p-values, and ``estimate ± z_{.975}·se`` intervals
    exactly as ``rdss::tidy.amce`` reports them. Each attribute-level row
    carries ``inquiry`` = its term name (how the book's diagnosis joins
    AMCEs to conjoint_inquiries' estimands).
    """
    y = df["choice"].to_numpy(dtype=float)
    cols: list[np.ndarray] = [np.ones(len(df))]
    names: list[str] = ["(Intercept)"]
    for attr, levels in levels_list.items():
        vals = df[attr].to_numpy()
        for lev in levels[1:]:
            cols.append((vals == lev).astype(float))
            names.append(f"{attr}{lev}")
    x = np.column_stack(cols)
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    _, inverse = np.unique(df[respondent_id].to_numpy(), return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    scores = x * resid[:, None]
    sums = np.zeros((n_clusters, k))
    np.add.at(sums, inverse, scores)
    dfc = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    cov = dfc * xtx_inv @ (sums.T @ sums) @ xtx_inv
    se = np.sqrt(np.diag(cov))
    z = beta / se
    p = 2 * stats.norm.sf(np.abs(z))
    zcrit = float(stats.norm.ppf(0.975))
    tidy = pd.DataFrame(
        {
            "term": names,
            "estimate": beta,
            "std_error": se,
            "statistic": z,
            "p_value": p,
            "conf_low": beta - zcrit * se,
            "conf_high": beta + zcrit * se,
            "df": np.nan,
            "outcome": "choice",
        }
    )
    tidy = tidy[tidy["term"] != "(Intercept)"].reset_index(drop=True)
    tidy["inquiry"] = tidy["term"]
    return tidy


class _AmceEstimator(Estimator):
    """The conjoint AMCE step: one estimate row per attribute level.

    :class:`~declarepy.steps.Estimator`'s ``run`` stamps a single
    design-level inquiry on every row; the AMCE estimator instead answers
    one same-named inquiry per term, so this subclass keeps the per-row
    ``inquiry`` column produced by :func:`_amce_fit`.
    """

    def __init__(
        self,
        levels_list: Mapping[str, Sequence[str]],
        respondent_id: str = "subject",
        label: str = "estimator",
    ) -> None:
        self._levels = dict(levels_list)
        self._respondent = respondent_id
        super().__init__(
            lambda df: _amce_fit(df, self._levels, self._respondent),
            inquiry=None,
            label=label,
        )

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = _amce_fit(df, self._levels, self._respondent)
        rows["estimator"] = self.label
        return rows


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def declaration_17_1() -> Design:
    """declaration_17.1: the audit-experiment design (N = 500).

    Estimand: share of anti-Latino discriminators (~0.05). See the module
    docstring: the R source's case-mismatched ``if_else`` labels make
    ``Y_Z_white = 1`` and ``Y_Z_latino = 0`` for every unit, and the
    translation preserves that behavior faithfully.
    """

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        type_ = rng.choice(
            np.array(
                ["Always-responder", "Anti-Latino discriminator", "Never-responder"],
                dtype=object,
            ),
            size=n,
            p=[0.30, 0.05, 0.65],
        )
        return {
            "type": type_,
            # Case mismatch preserved from the R source: both always False.
            "Y_Z_white": np.where(type_ == "Never-Responder", 0, 1),
            "Y_Z_latino": np.where(type_ == "Always-Responder", 1, 0),
        }

    return Design(
        Model(n=500, build=build, label="model"),
        Inquiry(
            "anti_latino_discrimination",
            lambda df: float((df["type"] == "Anti-Latino discriminator").mean()),
        ),
        Assignment(
            lambda df, rng: _complete_ra_conditions(len(df), ("latino", "white"), rng),
            name="Z",
            label="complete_ra(conditions = latino/white)",
        ),
        reveal_outcomes(
            outcome="Y",
            assignment="Z",
            conditions=("latino", "white"),
            template="{outcome}_Z_{condition}",
        ),
        # declare_estimator(Y ~ Z) with no .method -> lm_robust (HC2);
        # patsy names the reported term "Z[T.white]" (R: "Zwhite").
        Estimator.lm_robust("Y ~ Z", inquiry="anti_latino_discrimination"),
    )


def declaration_17_2() -> Design:
    """declaration_17.2: the audit experiment nested in a causal design.

    N = 5000 callers; D randomizes an anti-discrimination treatment that
    converts half the discriminators; the audit outcome interacts caller
    name (Z) with D and the ``Zwhite:D`` interaction answers the ATE
    (~ -0.025) on the discriminator share.
    """

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        type_d_0 = rng.choice(
            np.array(
                ["Always-Responder", "Anti-Latino Discriminator", "Never-Responder"],
                dtype=object,
            ),
            size=n,
            p=[0.30, 0.05, 0.65],
        )
        type_tau_i = rng.binomial(1, 0.5, n)
        type_d_1 = np.where(
            (type_d_0 == "Anti-Latino Discriminator") & (type_tau_i == 1),
            "Always-Responder",
            type_d_0,
        )
        return {"type_D_0": type_d_0, "type_tau_i": type_tau_i, "type_D_1": type_d_1}

    def ate(df: pd.DataFrame) -> float:
        return float(
            (
                (df["type_D_1"] == "Anti-Latino Discriminator").astype(float)
                - (df["type_D_0"] == "Anti-Latino Discriminator").astype(float)
            ).mean()
        )

    return Design(
        # This part of the design is about causal inference.
        Model(n=5000, build=build, label="model"),
        Inquiry("ATE", ate),
        Assignment.complete(name="D"),
        reveal_outcomes(
            outcome="type", assignment="D", conditions=(0, 1),
            template="{outcome}_D_{condition}",
        ),
        # This part is about descriptive inference in each condition.
        Model(
            transform=lambda df, rng: {
                "Y_Z_white": np.where(df["type"] == "Never-Responder", 0, 1),
                "Y_Z_latino": np.where(df["type"] == "Always-Responder", 1, 0),
            },
            label="model_2",
        ),
        Assignment(
            lambda df, rng: _complete_ra_conditions(len(df), ("latino", "white"), rng),
            name="Z",
            label="complete_ra(conditions = latino/white)",
        ),
        reveal_outcomes(
            outcome="Y",
            assignment="Z",
            conditions=("latino", "white"),
            template="{outcome}_Z_{condition}",
        ),
        # R's term "Zwhite:D" is patsy's "Z[T.white]:D".
        Estimator.lm_robust("Y ~ Z * D", term="Z[T.white]:D", inquiry="ATE"),
    )


def declaration_17_3(N: int = 500) -> Design:
    """declaration_17.3: the basic list experiment (3 control items)."""

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        return {
            "control_count": rng.binomial(3, 0.5, n),
            "Y_star": rng.binomial(1, 0.3, n),
        }

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = (
            df["Y_star"].to_numpy() * float(z) + df["control_count"].to_numpy()  # type: ignore[arg-type]
        )
        return result

    return Design(
        Model(n=N, build=build, label="model"),
        potential_outcomes(po, outcome="Y_list"),
        Inquiry("prevalence_rate", lambda df: float(df["Y_star"].mean())),
        Assignment.complete(),
        reveal_outcomes(outcome="Y_list"),
        Estimator.difference_in_means(y="Y_list", inquiry="prevalence_rate"),
    )


def declaration_17_4(N: int = 500, proportion_hiding: float = 0.0) -> Design:
    """declaration_17.4: list experiment vs. direct question under hiding.

    ``proportion_hiding`` of the true positives hide on the direct question
    (W); the direct estimator is the intercept of ``Y_direct ~ 1`` and the
    list estimator the lm_robust coefficient on Z. The book's diagnosis
    sweeps ``proportion_hiding`` in {0, .1, .2, .3} and ``N`` in
    {500, ..., 2500}.
    """

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        u = rng.normal(size=n)  # declared in the R source, unused downstream
        control_count = rng.binomial(3, 0.5, n)
        y_star = rng.binomial(1, 0.3, n)
        # case_when(Y_star == 0 ~ 0L, Y_star == 1 ~ rbinom(N, 1, hiding))
        w = np.where(y_star == 1, rng.binomial(1, proportion_hiding, n), 0)
        return {"U": u, "control_count": control_count, "Y_star": y_star, "W": w}

    def po(df: pd.DataFrame, z: object, rng: np.random.Generator) -> np.ndarray:
        result: np.ndarray = (
            df["Y_star"].to_numpy() * float(z) + df["control_count"].to_numpy()  # type: ignore[arg-type]
        )
        return result

    return Design(
        Model(n=N, build=build, label="model"),
        potential_outcomes(po, outcome="Y_list"),
        Inquiry("prevalence_rate", lambda df: float(df["Y_star"].mean())),
        Assignment.complete(),
        Measurement(
            lambda df, rng: {
                "Y_list": np.where(df["Z"] == 1, df["Y_list1"], df["Y_list0"]),
                "Y_direct": df["Y_star"].to_numpy() - df["W"].to_numpy(),
            },
            label="reveal_outcomes(Y_list) + Y_direct",
        ),
        Estimator.lm_robust("Y_list ~ Z", inquiry="prevalence_rate", label="list"),
        Estimator.lm_robust("Y_direct ~ 1", inquiry="prevalence_rate", label="direct"),
    )


def declaration_17_5(n_subjects: int = 500, n_tasks: int = 3) -> Design:
    """declaration_17.5: forced-choice conjoint with five AMCE inquiries.

    fabricatr's subject/task/profile hierarchy built explicitly: rows are
    ordered subject-major, task, then profile 1 and 2 adjacent;
    ``uij ~ N(0, 1)`` per row. Assignment is rdss's ``conjoint_assignment``
    (complete RA of each attribute over all rows — one Assignment step per
    attribute); measurement is ``conjoint_measurement`` (forced choice on
    the conjectured utility); the estimator is cjoint's ``amce``.
    """
    n_rows = n_subjects * n_tasks * 2

    def build(n: int, rng: np.random.Generator) -> dict[str, object]:
        return {
            "subject": np.repeat(np.arange(1, n_subjects + 1), n_tasks * 2),
            "task": np.tile(np.repeat(np.arange(1, n_tasks + 1), 2), n_subjects),
            "profile": np.tile(np.array([1, 2]), n_subjects * n_tasks),
            "uij": rng.normal(0.0, 1.0, n),
        }

    def make_reader(name: str) -> Callable[[pd.DataFrame], float]:
        col = f"estimand_{name}"
        return lambda df: float(df[col].iloc[0])

    def attribute_step(attr: str) -> Assignment:
        levels = _CONJOINT_LEVELS[attr]
        return Assignment(
            lambda df, rng: _complete_ra_conditions(len(df), levels, rng),
            name=attr,
            label=f"conjoint_assignment({attr})",
        )

    return Design(
        Model(n=n_rows, build=build, label="model"),
        Model(transform=_conjoint_inquiry_prep, label="conjoint_inquiries(prep)"),
        # mypy can't see that the dynamic keywords all land in **named.
        Inquiry(**{name: make_reader(name) for name in _AMCE_NAMES}),  # type: ignore[arg-type]
        attribute_step("gender"),
        attribute_step("party"),
        attribute_step("region"),
        Measurement(_conjoint_measure, label="conjoint_measurement"),
        _AmceEstimator(_CONJOINT_LEVELS, respondent_id="subject"),
    )


def declaration_17_6(deceive: bool = False, n_pairs: int = 200) -> Design:
    """declaration_17.6: the trust game, honest vs. deceptive measurement.

    2·n_pairs subjects with altruism ``a ~ U(0, 1)``; inquiries are the
    behavioral-model averages of investment and return; pairs form by
    ``complete_ra(num_arms = n_pairs)`` + within-pair role assignment, the
    frame pivots wide (one row per pair), player 1 invests, and (after the
    honest ``trusting`` estimate) the measured investment is replaced by
    noise when ``deceive`` — R: ``deceive*runif(N) + (1-deceive)*invested``
  — before computing what player 2 returns.
    """

    def pivot_wider(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        # pivot_wider(id_cols = pair, names_from = role, values_from = c(ID, a))
        wide = df.pivot(index="pair", columns="role", values=["ID", "a"])
        pairs = cast("list[tuple[str, int]]", wide.columns.to_list())
        wide.columns = pd.Index([f"{value}_{role}" for value, role in pairs])
        return wide.reset_index()

    def deceive_measure(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        u = rng.random(len(df))  # runif(N) is drawn unconditionally in R
        invested = float(deceive) * u + (1.0 - float(deceive)) * df["invested"].to_numpy()
        return {"invested": invested, "returned": _returned(invested, df["a_2"].to_numpy())}

    return Design(
        Model(
            n=2 * n_pairs,
            build=lambda n, rng: {"a": rng.random(n)},
            label="model",
        ),
        Inquiry(
            trusting=lambda df: float(_average_invested(df["a"].to_numpy()).mean()),
            trustworthy=lambda df: float(_average_returned(df["a"].to_numpy()).mean()),
        ),
        Assignment(
            lambda df, rng: _complete_ra_conditions(
                len(df), tuple(range(1, n_pairs + 1)), rng
            ),
            name="pair",
            label=f"complete_ra(num_arms = {n_pairs})",
        ),
        Assignment(
            lambda df, rng: 1 + _block_ra(df["pair"], rng=rng),
            name="role",
            label="1 + block_ra(blocks = pair)",
        ),
        Model(transform=pivot_wider, label="pivot_wider"),
        Measurement(
            lambda df, rng: {
                "invested": _invested(df["a_1"].to_numpy(), df["a_2"].to_numpy())
            },
            label="invested",
        ),
        Estimator.lm_robust("invested ~ 1", inquiry="trusting", label="trusting"),
        Measurement(deceive_measure, label="deceive + returned"),
        Estimator.lm_robust("returned ~ 1", inquiry="trustworthy", label="trustworthy"),
    )
