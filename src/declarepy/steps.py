"""Composable design steps: the M, I, D, A of a research design.

A design is an ordered pipeline of steps (DeclareDesign's ``declare_*`` +
``+`` composition). Every step takes **explicit callables and column names**
— there is deliberately no tidy-eval / bare-name capture (see
docs/spec/SEMANTIC_DIFFERENCES.md §10). Potential outcomes are explicit
columns (``Y0``/``Y1`` by default) and revealing them is a visible
``np.where`` switch.

Every stochastic callable receives the run's ``numpy.random.Generator`` so a
whole design run is one continuous, reproducible stream.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd

from . import estimators as _est
from . import ra as _ra

__all__ = [
    "DesignStep",
    "Model",
    "potential_outcomes",
    "Inquiry",
    "Sampling",
    "Assignment",
    "Measurement",
    "reveal_outcomes",
    "Estimator",
    "Design",
]

FrameLike = Union[pd.DataFrame, Mapping[str, object]]


class DesignStep:
    """Base class for design steps; supports ``step + step -> Design``."""

    label: str = ""

    def __add__(self, other: Union["DesignStep", "Design"]) -> "Design":
        if isinstance(other, Design):
            return Design(self, *other.steps)
        if isinstance(other, DesignStep):
            return Design(self, other)
        return NotImplemented


def _as_columns(result: FrameLike, df: pd.DataFrame, what: str) -> pd.DataFrame:
    """Merge a step's returned columns/frame into the running dataframe."""
    if isinstance(result, pd.DataFrame):
        return result.reset_index(drop=True)
    if isinstance(result, Mapping):
        out = df.copy()
        for name, col in result.items():
            out[name] = cast(Any, np.asarray(col) if not np.isscalar(col) else col)
        return out
    raise TypeError(f"{what} must return a DataFrame or a mapping of columns")


class Model(DesignStep):
    """The M of MIDA: create or transform the world's dataframe.

    ``Model(n=100, build=fn)`` creates the base frame from ``fn(n, rng)``
    (returning a DataFrame or a dict of columns). ``Model(n=100)`` alone
    creates an ``ID``-only frame of n units (``declare_model(N = 100)``).
    ``Model(transform=fn)`` applies ``fn(df, rng)`` to the running frame —
    use it for second-stage model steps. ``Model(data=df)`` starts from a
    fixed (empirical) frame.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        build: Optional[Callable[[int, np.random.Generator], FrameLike]] = None,
        transform: Optional[Callable[[pd.DataFrame, np.random.Generator], FrameLike]] = None,
        data: Optional[pd.DataFrame] = None,
        label: str = "model",
    ) -> None:
        given = sum(x is not None for x in (build, transform, data))
        if given > 1:
            raise ValueError("give at most one of build=, transform=, or data=")
        if given == 0 and n is None:
            raise ValueError("give n= (unit count) or one of build=/transform=/data=")
        if build is not None and n is None:
            raise ValueError("build= requires n=")
        self.n = n
        self.build = build
        self.transform = transform
        self.data = data
        self.label = label

    def run(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        if self.data is not None:
            return self.data.copy().reset_index(drop=True)
        if self.transform is not None:
            return _as_columns(self.transform(df, rng), df, "Model(transform=...)")
        assert self.n is not None
        base_df = pd.DataFrame({"ID": np.arange(1, self.n + 1)})
        if self.build is None:
            return base_df
        base = self.build(self.n, rng)
        if isinstance(base, Mapping):
            out = base_df
            for name, col in base.items():
                out[name] = cast(Any, np.asarray(col) if not np.isscalar(col) else col)
            return out
        return base.reset_index(drop=True)


def potential_outcomes(
    fn: Callable[[pd.DataFrame, object, np.random.Generator], object],
    conditions: Sequence[object] = (0, 1),
    outcome: str = "Y",
    template: str = "{outcome}{condition}",
) -> Model:
    """Model step adding one potential-outcome column per condition.

    ``fn(df, z, rng)`` returns the outcome vector had every unit received
    condition ``z``. Default naming is the course convention ``Y0``/``Y1``
    (template ``"{outcome}{condition}"``); use ``"{outcome}_Z_{condition}"``
    for DeclareDesign's naming.

    Mirroring DeclareDesign's ``potential_outcomes()``, ``fn`` is evaluated
    **once per condition**: any randomness drawn inside ``fn`` is drawn
    independently for each condition (this matters for designs like the
    book's declaration 2.1, whose treatment effect is a fresh draw).
    """

    def add_pos(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        for z in conditions:
            name = template.format(outcome=outcome, condition=z)
            out[name] = np.asarray(fn(df, z, rng))
        return out

    return Model(transform=add_pos, label=f"potential_outcomes({outcome})")


class Inquiry(DesignStep):
    """The I of MIDA: named estimands computed on the current frame.

    ``Inquiry("ATE", fn)`` or ``Inquiry(ATE=fn, SATE=fn2)``; each ``fn(df)``
    returns a scalar. Position matters: an inquiry declared before sampling
    is a population estimand, after sampling a sample estimand.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        fn: Optional[Callable[[pd.DataFrame], float]] = None,
        **named: Callable[[pd.DataFrame], float],
    ) -> None:
        fns: dict[str, Callable[[pd.DataFrame], float]] = {}
        if name is not None:
            if fn is None:
                raise ValueError("Inquiry(name, fn): fn is required")
            fns[name] = fn
        fns.update(named)
        if not fns:
            raise ValueError("Inquiry needs at least one named estimand")
        self.fns = fns
        self.label = "inquiry"

    def run(self, df: pd.DataFrame) -> dict[str, float]:
        return {name: float(f(df)) for name, f in self.fns.items()}


class Sampling(DesignStep):
    """The first D of MIDA: who gets into the study.

    ``fn(df, rng)`` returns either a 0/1 inclusion vector (kept as column
    ``S`` and filtered to ``S == 1``, DeclareDesign's default) or an
    already-subset DataFrame.
    """

    def __init__(
        self,
        fn: Callable[[pd.DataFrame, np.random.Generator], object],
        label: str = "sampling",
    ) -> None:
        self.fn = fn
        self.label = label

    @classmethod
    def complete(cls, n: Optional[int] = None, prob: Optional[float] = None) -> "Sampling":
        """Complete random sampling of exactly n (or share prob) units."""
        return cls(
            lambda df, rng: _ra.complete_rs(len(df), n=n, prob=prob, rng=rng),
            label=f"complete_rs(n={n}, prob={prob})",
        )

    @classmethod
    def simple(cls, prob: float) -> "Sampling":
        """Independent Bernoulli(prob) inclusion."""
        return cls(
            lambda df, rng: _ra.simple_rs(len(df), prob=prob, rng=rng),
            label=f"simple_rs(prob={prob})",
        )

    def run(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        res = self.fn(df, rng)
        if isinstance(res, pd.DataFrame):
            return res.reset_index(drop=True)
        mask = np.asarray(res).astype(bool)
        out = df.copy()
        out["S"] = mask.astype(int)
        return out.loc[mask].reset_index(drop=True)


class Assignment(DesignStep):
    """The second D of MIDA: who gets treated.

    ``fn(df, rng)`` returns the assignment vector, stored as column ``name``
    (default ``"Z"``).
    """

    def __init__(
        self,
        fn: Callable[[pd.DataFrame, np.random.Generator], object],
        name: str = "Z",
        label: str = "assignment",
    ) -> None:
        self.fn = fn
        self.name = name
        self.label = label

    @classmethod
    def complete(
        cls, m: Optional[int] = None, prob: Optional[float] = None, name: str = "Z"
    ) -> "Assignment":
        """Complete random assignment of the current frame's units."""
        return cls(
            lambda df, rng: _ra.complete_ra(len(df), m=m, prob=prob, rng=rng),
            name=name,
            label=f"complete_ra(m={m}, prob={prob})",
        )

    @classmethod
    def block(
        cls,
        blocks: str,
        m: Optional[int] = None,
        prob: Optional[float] = None,
        name: str = "Z",
    ) -> "Assignment":
        """Blocked complete random assignment within column ``blocks``."""
        return cls(
            lambda df, rng: _ra.block_ra(df[blocks], m=m, prob=prob, rng=rng),
            name=name,
            label=f"block_ra(blocks={blocks!r})",
        )

    @classmethod
    def simple(cls, prob: float = 0.5, name: str = "Z") -> "Assignment":
        """Independent coin-flip assignment."""
        return cls(
            lambda df, rng: _ra.simple_ra(len(df), prob=prob, rng=rng),
            name=name,
            label=f"simple_ra(prob={prob})",
        )

    def run(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        out = df.copy()
        out[self.name] = np.asarray(self.fn(df, rng))
        return out


class Measurement(DesignStep):
    """The third D of MIDA: what gets recorded.

    ``fn(df, rng)`` returns new/updated columns (dict) or a full DataFrame.
    """

    def __init__(
        self,
        fn: Callable[[pd.DataFrame, np.random.Generator], FrameLike],
        label: str = "measurement",
    ) -> None:
        self.fn = fn
        self.label = label

    def run(self, df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        return _as_columns(self.fn(df, rng), df, "Measurement")


def reveal_outcomes(
    outcome: str = "Y",
    assignment: str = "Z",
    conditions: Sequence[object] = (0, 1),
    template: str = "{outcome}{condition}",
) -> Measurement:
    """Measurement step: observe the potential outcome matching assignment.

    For the binary default this is the visible one-line switch
    ``Y = np.where(Z == 1, Y1, Y0)``.
    """

    def reveal(df: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
        cols = [template.format(outcome=outcome, condition=z) for z in conditions]
        if len(conditions) == 2:
            y = np.where(
                df[assignment] == conditions[1], df[cols[1]], df[cols[0]]
            )
        else:
            choices = [df[c] for c in cols]
            conds = [df[assignment] == z for z in conditions]
            y = np.select(conds, choices)
        return {outcome: y}

    return Measurement(reveal, label=f"reveal_outcomes({outcome} ~ {assignment})")


class Estimator(DesignStep):
    """The A of MIDA: an answer strategy producing estimate rows.

    ``fn(df)`` returns an :class:`~declarepy.estimators.EstimatorResult`, a
    tidy one-or-more-row DataFrame (columns at least ``estimate``), or a
    dict. ``inquiry`` names the estimand this estimator answers (used by
    diagnosis to align estimates with truths); ``label`` names the
    estimator in diagnosis tables.
    """

    def __init__(
        self,
        fn: Callable[[pd.DataFrame], object],
        inquiry: Optional[str] = None,
        label: str = "estimator",
    ) -> None:
        self.fn = fn
        self.inquiry = inquiry
        self.label = label

    @classmethod
    def difference_in_means(
        cls,
        y: str = "Y",
        z: str = "Z",
        blocks: Optional[str] = None,
        inquiry: Optional[str] = None,
        label: str = "estimator",
        alpha: float = 0.05,
    ) -> "Estimator":
        """estimatr-style difference in means (Welch, or blocked)."""
        return cls(
            lambda df: _est.difference_in_means(df, y=y, z=z, blocks=blocks, alpha=alpha),
            inquiry=inquiry,
            label=label,
        )

    @classmethod
    def lm_robust(
        cls,
        formula: str,
        term: Optional[Union[str, Sequence[str]]] = None,
        se_type: str = "HC2",
        inquiry: Optional[str] = None,
        label: str = "estimator",
        alpha: float = 0.05,
    ) -> "Estimator":
        """OLS with robust (HC2-default) SEs; reports ``term``'s row(s).

        ``term=None`` reports the first non-intercept term (DeclareDesign's
        default), or the intercept for an intercept-only model; pass a term
        name, a list of names, or ``"all"``.
        """

        def run(df: pd.DataFrame) -> pd.DataFrame:
            tidy = _est.lm_robust(formula, df, se_type=se_type, alpha=alpha)
            if term == "all":
                return tidy
            if term is None:
                non_int = tidy[tidy["term"] != "Intercept"]
                return (non_int.iloc[:1] if len(non_int) else tidy.iloc[:1]).copy()
            wanted = [term] if isinstance(term, str) else list(term)
            out = tidy[tidy["term"].isin(wanted)].copy()
            if len(out) != len(wanted):
                missing = set(wanted) - set(out["term"])
                raise KeyError(f"terms not in fit: {sorted(missing)}")
            return out

        return cls(run, inquiry=inquiry, label=label)

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        res = self.fn(df)
        if isinstance(res, _est.EstimatorResult):
            rows = res.to_frame()
        elif isinstance(res, pd.DataFrame):
            rows = res.copy()
        elif isinstance(res, Mapping):
            rows = pd.DataFrame([dict(res)])
        else:
            raise TypeError(
                "Estimator fn must return an EstimatorResult, DataFrame, or dict"
            )
        rows["estimator"] = self.label
        rows["inquiry"] = self.inquiry
        return rows


class Design:
    """An ordered pipeline of design steps: model → ... → estimators.

    Compose with ``Design(step1, step2, ...)`` or ``step1 + step2 + ...``;
    ``design + step`` and ``design + design`` also work.
    """

    def __init__(self, *steps: DesignStep) -> None:
        flat: list[DesignStep] = []
        for s in steps:
            if isinstance(s, Design):
                flat.extend(s.steps)
            elif isinstance(s, DesignStep):
                flat.append(s)
            else:
                raise TypeError(f"not a design step: {s!r}")
        self.steps: tuple[DesignStep, ...] = tuple(flat)

    def __add__(self, other: Union[DesignStep, "Design"]) -> "Design":
        if isinstance(other, Design):
            return Design(*self.steps, *other.steps)
        if isinstance(other, DesignStep):
            return Design(*self.steps, other)
        return NotImplemented

    def __iter__(self) -> "Iterable[DesignStep]":
        return iter(self.steps)

    def __repr__(self) -> str:
        names = " + ".join(s.label or type(s).__name__ for s in self.steps)
        return f"Design({names})"
