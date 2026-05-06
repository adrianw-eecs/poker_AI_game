"""Tests for equity calculator."""

import pytest

from poker.domain.card import Card, Rank, Suit
from poker.evaluation.equity import exact_equity_river, monte_carlo_equity
from poker.rng import RNG


def _c(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


@pytest.mark.smoke
def test_aa_has_high_preflop_equity() -> None:
    hole = [_c(Rank.ACE, Suit.HEARTS), _c(Rank.ACE, Suit.DIAMONDS)]
    equity = monte_carlo_equity(hole, [], num_opponents=1, num_trials=2000, rng=RNG(seed=42))
    assert 0.80 < equity < 0.90


def test_dominated_hand_lower_equity() -> None:
    strong = [_c(Rank.ACE, Suit.HEARTS), _c(Rank.ACE, Suit.DIAMONDS)]
    weak = [_c(Rank.SEVEN, Suit.HEARTS), _c(Rank.TWO, Suit.DIAMONDS)]
    eq_strong = monte_carlo_equity(strong, [], num_opponents=1, num_trials=2000, rng=RNG(seed=1))
    eq_weak = monte_carlo_equity(weak, [], num_opponents=1, num_trials=2000, rng=RNG(seed=1))
    assert eq_strong > eq_weak


def test_exact_equity_river() -> None:
    hole = [_c(Rank.ACE, Suit.HEARTS), _c(Rank.KING, Suit.HEARTS)]
    board = [_c(Rank.QUEEN, Suit.HEARTS), _c(Rank.JACK, Suit.SPADES), _c(Rank.TEN, Suit.SPADES),
             _c(Rank.NINE, Suit.SPADES), _c(Rank.EIGHT, Suit.SPADES)]
    opp = [_c(Rank.TWO, Suit.CLUBS), _c(Rank.THREE, Suit.CLUBS)]
    assert exact_equity_river(hole, board, opp) == 1.0

    tie_opp = [_c(Rank.ACE, Suit.CLUBS), _c(Rank.KING, Suit.CLUBS)]
    assert exact_equity_river(hole, board, tie_opp) == 0.5
