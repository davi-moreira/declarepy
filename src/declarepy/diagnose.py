"""Run, diagnose, and redesign declared designs.

``run_design`` executes one design run (one simulated study); ``diagnose``
runs it many times and summarizes the diagnosands; ``redesign`` sweeps a
design factory over parameter grids; ``diagnose_all`` diagnoses a whole
sweep into one comparison table.

Reproducibility contract (matching the HONR 46400 course convention): every
``diagnose`` call builds ONE fresh ``numpy.random.default_rng(seed)`` and
threads that single generator through all simulations sequentially — reps
never re-seed individually, and a given ``(design, sims, seed)`` always
yields the identical table.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd

from ._rng import RngLike, resolve_rng
from .steps import Assignment, Design, Estimator, Inquiry, Measurement, Model, Sampling

__all__ = [
    "run_design",
    "draw_data",
    "diagnose",
    "diagnose_all",
    "redesign",
    "Diagnosands",
    "Diagnosis",
    "RunResult",
    "DesignGrid",
]

DiagnosandFn = Callable[[pd.DataFrame], float]

def _coverage(d: pd.DataFrame) -> float:
    """Share of CIs containing the estimand (NaN when there is no estimand)."""
    ok = d["estimand"].notna()
    if not ok.any():
        return float("nan")
    sub = d.loc[ok]
    return float(
        ((sub["conf_low"] <= sub["estimand"]) & (sub["estimand"] <= sub["conf_high"])).mean()
    )


#: DeclareDesign's default diagnosands, computed on the per-sim frame whose
#: columns include estimate, std_error, p_value, conf_low, conf_high, estimand.
DEFAULT_DIAGNOSANDS: dict[str, DiagnosandFn] = {
    "mean_estimand": lambda d: float(d["estimand"].mean()),
    "mean_estimate": lambda d: float(d["estimate"].mean()),
    "bias": lambda d: float((d["estimate"] - d["estimand"]).mean()),
    "sd_estimate": lambda d: float(d["estimate"].std(ddof=1)),
    "rmse": lambda d: float(np.sqrt(((d["estimate"] - d["estimand"]) ** 2).mean())),
    "power": lambda d: float((d["p_value"] <= 0.05).mean()),
    "coverage": _coverage,
}


class Diagnosands:
    """A named set of diagnosand functions over the simulations frame.

    ``Diagnosands(success=fn, ...)`` — each ``fn`` takes the simulations
    DataFrame for one (estimator, inquiry) group and returns a scalar.
    """

    def __init__(self, **named: DiagnosandFn) -> None:
        if not named:
            raise ValueError("Diagnosands needs at least one named function")
        self.fns: dict[str, DiagnosandFn] = dict(named)

    @classmethod
    def default(cls) -> "Diagnosands":
        return cls(**DEFAULT_DIAGNOSANDS)


@dataclass
class RunResult:
    """One design run: the final data, the truths, and the estimates."""

    data: pd.DataFrame
    estimands: dict[str, float]
    estimates: pd.DataFrame


def run_design(design: Design, rng: RngLike = None) -> RunResult:
    """Execute every step of ``design`` once, in declared order."""
    gen = resolve_rng(rng)
    df = pd.DataFrame()
    estimands: dict[str, float] = {}
    estimate_rows: list[pd.DataFrame] = []
    for step in design.steps:
        if isinstance(step, Model):
            df = step.run(df, gen)
        elif isinstance(step, Inquiry):
            estimands.update(step.run(df))
        elif isinstance(step, (Sampling, Assignment, Measurement)):
            df = step.run(df, gen)
        elif isinstance(step, Estimator):
            estimate_rows.append(step.run(df))
        else:
            raise TypeError(f"unknown step type: {type(step).__name__}")
    estimates = (
        pd.concat(estimate_rows, ignore_index=True)
        if estimate_rows
        else pd.DataFrame(columns=["estimator", "term", "estimate", "inquiry"])
    )
    return RunResult(data=df, estimands=estimands, estimates=estimates)


def draw_data(design: Design, rng: RngLike = None) -> pd.DataFrame:
    """Run only the data-generating steps (no inquiries, no estimators)."""
    gen = resolve_rng(rng)
    df = pd.DataFrame()
    for step in design.steps:
        if isinstance(step, (Model, Sampling, Assignment, Measurement)):
            df = step.run(df, gen)
    return df


_GROUP_COLS = ["estimator", "inquiry", "outcome", "term"]
_SIM_COLS = [
    "estimate", "std_error", "statistic", "p_value",
    "conf_low", "conf_high", "df", "estimand",
]


@dataclass
class Diagnosis:
    """The result of :func:`diagnose`: the summary table + raw simulations."""

    diagnosands: pd.DataFrame
    simulations: pd.DataFrame
    sims: int
    seed: Optional[int] = None
    parameters: dict[str, object] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Diagnosis(sims={self.sims}, seed={self.seed})\n"
            + self.diagnosands.round(4).to_string(index=False)
        )


def _simulate(design: Design, sims: int, gen: np.random.Generator) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for i in range(sims):
        run = run_design(design, rng=gen)
        est = run.estimates.copy()
        est["sim"] = i
        est["estimand"] = [
            run.estimands.get(inq) if inq is not None else np.nan
            for inq in est["inquiry"]
        ]
        rows.append(est)
    sims_df = pd.concat(rows, ignore_index=True)
    for col in _SIM_COLS + _GROUP_COLS:
        if col not in sims_df.columns:
            sims_df[col] = np.nan
    return sims_df


def _summarize(
    sims_df: pd.DataFrame,
    diagnosands: Diagnosands,
    bootstrap_sims: int,
    gen: np.random.Generator,
) -> pd.DataFrame:
    out_rows: list[dict[str, object]] = []
    grouped = sims_df.groupby(
        [sims_df[c].fillna("") for c in _GROUP_COLS], sort=False
    )
    for key, grp in grouped:
        row: dict[str, object] = dict(zip(_GROUP_COLS, key))
        for name, fn in diagnosands.fns.items():
            try:
                row[name] = fn(grp)
            except Exception:
                row[name] = np.nan
        if bootstrap_sims:
            sim_ids = grp["sim"].to_numpy()
            uniq = np.unique(sim_ids)
            boots: dict[str, list[float]] = {n: [] for n in diagnosands.fns}
            for _ in range(bootstrap_sims):
                take = gen.choice(uniq, size=len(uniq), replace=True)
                # Resample whole simulations (all rows of each drawn sim).
                counts = pd.Series(take).value_counts()
                parts = [grp[grp["sim"] == s] for s, c in counts.items() for _ in range(int(c))]
                bs_df = pd.concat(parts, ignore_index=True)
                for name, fn in diagnosands.fns.items():
                    try:
                        boots[name].append(fn(bs_df))
                    except Exception:
                        boots[name].append(np.nan)
            for name in diagnosands.fns:
                row[f"se({name})"] = float(np.std(np.asarray(boots[name]), ddof=1))
        row["n_sims"] = int(grp["sim"].nunique())
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def diagnose(
    design: Design,
    sims: int = 1000,
    seed: Optional[int] = 464,
    diagnosands: Optional[Diagnosands] = None,
    bootstrap_sims: int = 0,
) -> Diagnosis:
    """Monte-Carlo diagnosis: run ``design`` ``sims`` times and summarize.

    One fresh ``default_rng(seed)`` drives all simulations sequentially.
    ``bootstrap_sims > 0`` adds bootstrap ``se(diagnosand)`` columns
    (resampling whole simulations, as DeclareDesign does).
    """
    dgs = diagnosands or Diagnosands.default()
    gen = np.random.default_rng(seed)
    sims_df = _simulate(design, sims, gen)
    table = _summarize(sims_df, dgs, bootstrap_sims, gen)
    return Diagnosis(diagnosands=table, simulations=sims_df, sims=sims, seed=seed)


@dataclass
class DesignGrid:
    """A parameter sweep: designs produced by ``redesign`` with their params."""

    designs: list[Design]
    params: list[dict[str, object]]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.designs)

    def __iter__(self) -> "Iterable[Design]":
        return iter(self.designs)


def redesign(
    factory: Callable[..., Design],
    **grids: Union[Sequence[object], object],
) -> DesignGrid:
    """Sweep a design factory over parameter grids (DeclareDesign's redesign).

    ``factory(**params)`` must return a :class:`Design`. Each keyword is a
    sequence of values (scalars are wrapped); the sweep is the Cartesian
    product in keyword order, labeled ``design_1..design_k``.
    """
    lists: dict[str, list[object]] = {}
    for k, v in grids.items():
        if isinstance(v, (str, bytes)) or not isinstance(v, (Sequence, np.ndarray)):
            lists[k] = [v]
        else:
            lists[k] = list(v)
    combos = [
        dict(zip(lists.keys(), values))
        for values in itertools.product(*lists.values())
    ]
    designs = [factory(**combo) for combo in combos]
    labels = [f"design_{i + 1}" for i in range(len(combos))]
    return DesignGrid(designs=designs, params=combos, labels=labels)


def diagnose_all(
    designs: Union[DesignGrid, Mapping[str, Design], Iterable[Design]],
    sims: int = 1000,
    seed: Optional[int] = 464,
    diagnosands: Optional[Diagnosands] = None,
    bootstrap_sims: int = 0,
) -> Diagnosis:
    """Diagnose several designs into one comparison table.

    Accepts a :class:`DesignGrid` (param columns are added to the table), a
    mapping of label → design, or an iterable of designs. Each design gets
    an independent child seed spawned from ``seed`` (deterministic, and
    insensitive to how many designs precede it in the list).
    """
    if isinstance(designs, DesignGrid):
        items = list(zip(designs.labels, designs.designs, designs.params))
    elif isinstance(designs, Mapping):
        items = [(label, d, {}) for label, d in designs.items()]
    else:
        items = [(f"design_{i + 1}", d, {}) for i, d in enumerate(designs)]

    seeds = np.random.SeedSequence(seed).spawn(len(items))
    tables: list[pd.DataFrame] = []
    sim_frames: list[pd.DataFrame] = []
    dgs = diagnosands or Diagnosands.default()
    for (label, design, params), child in zip(items, seeds):
        gen = np.random.default_rng(child)
        sims_df = _simulate(design, sims, gen)
        table = _summarize(sims_df, dgs, bootstrap_sims, gen)
        for col, val in reversed(list(params.items())):
            table.insert(0, col, cast(Any, val))
            sims_df[col] = cast(Any, val)
        table.insert(0, "design", label)
        sims_df["design"] = label
        tables.append(table)
        sim_frames.append(sims_df)
    return Diagnosis(
        diagnosands=pd.concat(tables, ignore_index=True),
        simulations=pd.concat(sim_frames, ignore_index=True),
        sims=sims,
        seed=seed,
    )
