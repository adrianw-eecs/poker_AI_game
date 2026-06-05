"""Seedable RNG wrapper for reproducibility."""

import random
from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


class RNG:
    """Seedable random number generator wrapper.

    Wraps the standard library's random.Random to ensure reproducibility.
    Same seed produces identical sequences of random numbers across runs.
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize RNG with an optional seed.

        Args:
            seed: If provided, the RNG is seeded deterministically.
                  If None, a random seed is used.
        """
        self._rng = random.Random(seed)
        self.seed = seed

    def shuffle(self, sequence: list[T]) -> None:
        """Shuffle a list in-place.

        Args:
            sequence: The list to shuffle.
        """
        self._rng.shuffle(sequence)

    def choice(self, sequence: Sequence[T]) -> T:
        """Choose a random element from a sequence.

        Args:
            sequence: The sequence to choose from.

        Returns:
            A randomly chosen element.
        """
        return self._rng.choice(sequence)

    def randint(self, a: int, b: int) -> int:
        """Return a random integer N such that a <= N <= b.

        Args:
            a: Lower bound (inclusive).
            b: Upper bound (inclusive).

        Returns:
            A random integer in the range [a, b].
        """
        return self._rng.randint(a, b)

    def random(self) -> float:
        """Return a random float in [0.0, 1.0).

        Returns:
            A random float.
        """
        return self._rng.random()

    def sample(self, population: Sequence[T], k: int) -> list[T]:
        """Return a k-length list of unique elements chosen from the population.

        Args:
            population: The sequence to sample from.
            k: The number of elements to sample.

        Returns:
            A list of k unique elements.
        """
        return self._rng.sample(population, k)

    def choices(self, population: Sequence[T], weights: list[float], k: int = 1) -> list[T]:
        """Return a k-length list of elements chosen from the population with weighted probabilities.

        Args:
            population: The sequence to choose from.
            weights: The weights for each element (must match population length).
            k: The number of elements to choose (default: 1).

        Returns:
            A list of k elements chosen according to weights.
        """
        return self._rng.choices(population, weights=weights, k=k)

    def __repr__(self) -> str:
        """Return a string representation."""
        return f"RNG(seed={self.seed})"
