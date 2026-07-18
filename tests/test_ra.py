"""Assignment/sampling procedures: counts, semantics, stream compatibility."""

from __future__ import annotations

import numpy as np
import pytest

import declarepy as dp


class TestCompleteRa:
    def test_exact_m(self) -> None:
        z = dp.complete_ra(100, m=37, rng=1)
        assert z.sum() == 37 and set(z) == {0, 1} and len(z) == 100

    def test_default_half_even(self) -> None:
        assert dp.complete_ra(100, rng=1).sum() == 50

    def test_default_half_odd_is_floor_or_ceil(self) -> None:
        counts = {int(dp.complete_ra(101, rng=s).sum()) for s in range(30)}
        assert counts <= {50, 51} and len(counts) == 2

    def test_prob_exact_when_integral(self) -> None:
        assert dp.complete_ra(100, prob=0.5, rng=1).sum() == 50
        assert dp.complete_ra(100, prob=0.3, rng=2).sum() == 30

    def test_prob_fractional_floor_or_ceil(self) -> None:
        counts = {int(dp.complete_ra(10, prob=0.55, rng=s).sum()) for s in range(50)}
        assert counts == {5, 6}

    def test_stream_compatible_with_course_inline(self) -> None:
        # Same generator state -> identical vector to the notebooks' inline
        # construction (z = zeros; z[:m] = 1; rng.permutation(z)).
        rng1 = np.random.default_rng(464)
        z_inline = np.zeros(500, dtype=int)
        z_inline[:250] = 1
        z_inline = rng1.permutation(z_inline)
        rng2 = np.random.default_rng(464)
        z_pkg = dp.complete_ra(500, 250, rng=rng2)
        assert (z_inline == z_pkg).all()

    def test_m_and_prob_conflict(self) -> None:
        with pytest.raises(ValueError):
            dp.complete_ra(10, m=5, prob=0.5)

    def test_m_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            dp.complete_ra(10, m=11)


class TestBlockRa:
    def test_exact_within_blocks(self) -> None:
        blocks = np.repeat(["a", "b", "c"], [10, 20, 30])
        z = dp.block_ra(blocks, rng=7)
        for lab, n in [("a", 10), ("b", 20), ("c", 30)]:
            assert z[blocks == lab].sum() == n // 2

    def test_prob_within_blocks(self) -> None:
        blocks = np.repeat([1, 2], [40, 60])
        z = dp.block_ra(blocks, prob=0.25, rng=7)
        assert z[blocks == 1].sum() == 10 and z[blocks == 2].sum() == 15

    def test_preserves_order(self) -> None:
        blocks = np.array(["b", "a", "b", "a"] * 5)
        z = dp.block_ra(blocks, rng=3)
        assert len(z) == 20 and z[blocks == "a"].sum() == 5


class TestSampling:
    def test_complete_rs_exact(self) -> None:
        s = dp.complete_rs(1000, n=150, rng=5)
        assert s.sum() == 150

    def test_simple_rs_bernoulli(self) -> None:
        s = dp.simple_rs(100000, prob=0.1, rng=5)
        assert 0.09 < s.mean() < 0.11

    def test_simple_ra_default_half(self) -> None:
        z = dp.simple_ra(100000, rng=5)
        assert 0.49 < z.mean() < 0.51


class TestRngPlumbing:
    def test_int_seed_reproducible(self) -> None:
        assert (dp.complete_ra(50, rng=42) == dp.complete_ra(50, rng=42)).all()

    def test_generator_advances(self) -> None:
        gen = np.random.default_rng(0)
        z1 = dp.complete_ra(50, rng=gen)
        z2 = dp.complete_ra(50, rng=gen)
        assert not (z1 == z2).all()
