"""Shared fixtures: the course's mentoring-program world builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_world(
    n: int = 100,
    effect: float = 2.0,
    noise: float = 2.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Verbatim course helper (nb04/nb09-nb11): Y0/Y1 world, rounded to 1dp."""
    assert rng is not None
    Y0 = rng.normal(50, 10, n) + rng.normal(0, noise, n)
    return pd.DataFrame({"Y0": Y0.round(1), "Y1": (Y0 + effect).round(1)})


@pytest.fixture
def seed() -> int:
    return 464
