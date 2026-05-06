"""Tests for seedable RNG."""

from poker.rng import RNG


def test_rng_seeded_is_reproducible() -> None:
    rng1 = RNG(seed=42)
    rng2 = RNG(seed=42)
    seq1 = [rng1.randint(1, 100) for _ in range(10)]
    seq2 = [rng2.randint(1, 100) for _ in range(10)]
    assert seq1 == seq2
    items1 = list(range(10))
    items2 = list(range(10))
    rng1 = RNG(seed=42)
    rng1.shuffle(items1)
    rng2 = RNG(seed=42)
    rng2.shuffle(items2)
    assert items1 == items2


def test_rng_different_seeds_differ() -> None:
    rng1 = RNG(seed=42)
    rng2 = RNG(seed=43)
    seq1 = [rng1.randint(1, 1000) for _ in range(100)]
    seq2 = [rng2.randint(1, 1000) for _ in range(100)]
    assert seq1 != seq2
