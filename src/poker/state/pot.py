"""Pot representation for poker games."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Pot:
    """A pot in the game, which may be a main pot or side pot.

    Attributes:
        amount: The total chips in this pot.
        eligible_seats: The set of seat numbers eligible to win from this pot.
    """

    amount: int
    eligible_seats: frozenset[int]
