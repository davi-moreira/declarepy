"""Seed/generator plumbing shared by every stochastic function in declarepy.

Every stochastic function takes an ``rng`` argument that may be ``None`` (a
fresh unseeded generator), an ``int`` seed, or an existing
:class:`numpy.random.Generator` (used in place, advancing its stream).
Nothing in declarepy ever touches numpy's global random state.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

RngLike = Union[None, int, np.random.Generator]


def resolve_rng(rng: RngLike = None) -> np.random.Generator:
    """Return a numpy Generator from a seed, an existing generator, or None."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def spawn_rng(seed: Optional[int]) -> np.random.Generator:
    """Fresh generator for a reproducible run (seed may be None)."""
    return np.random.default_rng(seed)
