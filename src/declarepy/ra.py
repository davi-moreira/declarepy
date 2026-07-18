"""Random assignment and random sampling procedures (randomizr-style).

Re-implements the *behavior* of the ``randomizr`` R package's
``complete_ra``, ``block_ra``, ``simple_ra``, ``complete_rs`` and
``simple_rs`` from their published descriptions (Coppock,
declaredesign.org/r/randomizr) — no upstream source code is copied.

Semantics notes
---------------
* ``complete_ra(N, m)`` assigns **exactly** ``m`` of ``N`` units to treatment
  via a seeded permutation of a 0/1 vector — the same construction the
  HONR 46400 course notebooks use inline (``z = zeros(N); z[:m] = 1;
  rng.permutation(z)``), so given the same generator state it reproduces the
  notebooks' assignments bit-for-bit.
* With ``prob`` instead of ``m``, the number treated is ``floor(N * prob)``
  or ``ceil(N * prob)``, choosing the ceiling with probability equal to the
  fractional part (randomizr's rule), so ``prob=0.5, N=100`` treats exactly 50.
* R and NumPy RNG streams are not portable: identical seeds never reproduce
  randomizr's draws digit-for-digit (SEMANTIC_DIFFERENCES §1); parity is
  distributional.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from ._rng import RngLike, resolve_rng

__all__ = [
    "complete_ra",
    "block_ra",
    "simple_ra",
    "complete_rs",
    "simple_rs",
    "block_rs",
]


def _resolve_m(N: int, m: Optional[int], prob: Optional[float], rng: np.random.Generator) -> int:
    """Number of units to treat/sample under randomizr's m/prob rules."""
    if m is not None and prob is not None:
        raise ValueError("specify at most one of m and prob")
    if m is not None:
        if not 0 <= m <= N:
            raise ValueError(f"m must be in [0, {N}], got {m}")
        return int(m)
    if prob is not None:
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        exact = N * prob
        floor = int(np.floor(exact))
        frac = exact - floor
        if frac == 0.0:
            return floor
        return floor + int(rng.random() < frac)
    # Neither given: half, randomizr-style (extra coin only when N is odd).
    if N % 2 == 0:
        return N // 2
    return N // 2 + int(rng.random() < 0.5)


def complete_ra(
    N: int,
    m: Optional[int] = None,
    prob: Optional[float] = None,
    rng: RngLike = None,
) -> np.ndarray:
    """Complete random assignment: exactly-m-of-N treated, as a 0/1 int array.

    ``complete_ra(N)`` treats half (randomizr default); ``m`` fixes the count;
    ``prob`` fixes the expected share with randomizr's floor/ceiling rule.
    """
    gen = resolve_rng(rng)
    m_ = _resolve_m(N, m, prob, gen)
    z = np.zeros(N, dtype=int)
    z[:m_] = 1
    return gen.permutation(z)


def block_ra(
    blocks: Union[Sequence[object], np.ndarray, pd.Series],
    m: Optional[int] = None,
    prob: Optional[float] = None,
    rng: RngLike = None,
) -> np.ndarray:
    """Blocked complete random assignment: complete_ra within each block.

    ``blocks`` is a length-N vector of block labels; the result preserves the
    original unit order. ``m`` (per block) or ``prob`` apply within every
    block; the default is half of each block.
    """
    gen = resolve_rng(rng)
    labels = np.asarray(pd.Series(blocks).to_numpy())
    z = np.zeros(len(labels), dtype=int)
    # Iterate blocks in order of first appearance for deterministic streams.
    seen: dict[object, np.ndarray] = {}
    order: list[object] = []
    for lab in pd.unique(labels):
        idx = np.flatnonzero(labels == lab)
        seen[lab] = idx
        order.append(lab)
    for lab in order:
        idx = seen[lab]
        z[idx] = complete_ra(len(idx), m=m, prob=prob, rng=gen)
    return z


def simple_ra(N: int, prob: float = 0.5, rng: RngLike = None) -> np.ndarray:
    """Simple (coin-flip) random assignment: independent Bernoulli(prob)."""
    gen = resolve_rng(rng)
    return (gen.random(N) < prob).astype(int)


def complete_rs(
    N: int,
    n: Optional[int] = None,
    prob: Optional[float] = None,
    rng: RngLike = None,
) -> np.ndarray:
    """Complete random sampling: exactly-n-of-N inclusion, as a 0/1 int array."""
    return complete_ra(N, m=n, prob=prob, rng=rng)


def simple_rs(N: int, prob: float = 0.1, rng: RngLike = None) -> np.ndarray:
    """Simple random sampling: independent Bernoulli(prob) inclusion mask."""
    gen = resolve_rng(rng)
    return (gen.random(N) < prob).astype(int)


def block_rs(
    blocks: Union[Sequence[object], np.ndarray, pd.Series],
    n: Optional[int] = None,
    prob: Optional[float] = None,
    rng: RngLike = None,
) -> np.ndarray:
    """Blocked complete random sampling: complete_rs within each block."""
    return block_ra(blocks, m=n, prob=prob, rng=rng)
