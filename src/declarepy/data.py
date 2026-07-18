"""Bundled datasets from the rdss R package (MIT, with attribution).

See ``src/declarepy/data/README.md`` for provenance. Attribution: datasets
from the ``rdss`` package (Blair, Coppock & Humphreys, MIT License),
companion to *Research Design in the Social Sciences* (2023).
"""

from __future__ import annotations

from importlib import resources

import pandas as pd

__all__ = ["load", "available"]

_DATASETS = {
    "lapop_brazil",
    "la_voter_file",
    "foos_etal",
    "cliningsmith_etal",
    "bonilla_tillery",
}


def available() -> list[str]:
    """Names accepted by :func:`load`."""
    return sorted(_DATASETS)


def load(name: str) -> pd.DataFrame:
    """Load a bundled dataset by name (e.g. ``load("foos_etal")``)."""
    key = name.removesuffix(".csv")
    if key not in _DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    ref = resources.files("declarepy").joinpath("data", f"{key}.csv")
    with resources.as_file(ref) as path:
        return pd.read_csv(path)
