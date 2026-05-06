"""Tests for Pot."""

import pytest

from poker.state.pot import Pot


def test_pot_construction() -> None:
    pot = Pot(amount=100, eligible_seats=frozenset({0, 1, 2}))
    assert pot.amount == 100
    assert pot.eligible_seats == frozenset({0, 1, 2})


def test_pot_immutability() -> None:
    pot = Pot(amount=100, eligible_seats=frozenset({0, 1}))
    with pytest.raises(AttributeError):
        pot.amount = 200  # type: ignore


def test_pot_equality() -> None:
    p1 = Pot(amount=100, eligible_seats=frozenset({0, 1}))
    p2 = Pot(amount=100, eligible_seats=frozenset({0, 1}))
    p3 = Pot(amount=200, eligible_seats=frozenset({0, 1}))
    assert p1 == p2
    assert p1 != p3
