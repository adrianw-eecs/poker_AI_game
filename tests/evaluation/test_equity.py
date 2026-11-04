"""Tests for equity calculator."""

import time

import numpy as np
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


def test_board_royal_is_an_exact_chop() -> None:
    """Nobody can beat a royal on the board, so every opponent ties."""
    hole = [_c(Rank.TWO, Suit.CLUBS), _c(Rank.THREE, Suit.DIAMONDS)]
    board = [
        _c(Rank.ACE, Suit.SPADES),
        _c(Rank.KING, Suit.SPADES),
        _c(Rank.QUEEN, Suit.SPADES),
        _c(Rank.JACK, Suit.SPADES),
        _c(Rank.TEN, Suit.SPADES),
    ]
    equity = monte_carlo_equity(hole, board, num_opponents=1, num_trials=20, rng=RNG(seed=7))
    assert equity == pytest.approx(0.5)


def test_unbeatable_quads_have_full_equity() -> None:
    """Four aces on a disconnected rainbow board. No straight flush exists."""
    hole = [_c(Rank.ACE, Suit.HEARTS), _c(Rank.ACE, Suit.DIAMONDS)]
    board = [
        _c(Rank.ACE, Suit.CLUBS),
        _c(Rank.ACE, Suit.SPADES),
        _c(Rank.KING, Suit.DIAMONDS),
        _c(Rank.NINE, Suit.HEARTS),
        _c(Rank.TWO, Suit.DIAMONDS),
    ]
    equity = monte_carlo_equity(hole, board, num_opponents=2, num_trials=30, rng=RNG(seed=8))
    assert equity == pytest.approx(1.0)


def test_same_seed_is_reproducible() -> None:
    hole = [_c(Rank.ACE, Suit.HEARTS), _c(Rank.KING, Suit.DIAMONDS)]
    board = [_c(Rank.QUEEN, Suit.CLUBS), _c(Rank.SEVEN, Suit.SPADES), _c(Rank.TWO, Suit.HEARTS)]
    first = monte_carlo_equity(hole, board, 1, 80, RNG(seed=11))
    second = monte_carlo_equity(hole, board, 1, 80, RNG(seed=11))
    assert first == second


def test_monte_carlo_100_trials_under_3ms() -> None:
    hole = [_c(Rank.ACE, Suit.HEARTS), _c(Rank.ACE, Suit.DIAMONDS)]
    board = [_c(Rank.KING, Suit.CLUBS), _c(Rank.SEVEN, Suit.DIAMONDS), _c(Rank.TWO, Suit.SPADES)]
    monte_carlo_equity(hole, board, 1, 100, RNG(seed=1))
    samples = []
    for i in range(15):
        start = time.perf_counter()
        monte_carlo_equity(hole, board, 1, 100, RNG(seed=i))
        samples.append(time.perf_counter() - start)
    assert sorted(samples)[len(samples) // 2] < 0.003


def test_100_trials_within_5_percent_of_5000() -> None:
    """Mean absolute error of a 100-trial estimate against a 5,000-trial reference."""
    rng = np.random.default_rng(0)
    deck = np.arange(52)
    errors = []
    for i in range(200):
        ids = rng.choice(deck, 7, replace=False)
        n_board = (3, 4, 5)[i % 3]
        hole = _from_ids(ids[:2])
        board = _from_ids(ids[2 : 2 + n_board])
        reference = monte_carlo_equity(hole, board, 1, 5_000, RNG(seed=10_000 + i))
        estimate = monte_carlo_equity(hole, board, 1, 100, RNG(seed=i))
        errors.append(abs(estimate - reference))
    assert float(np.mean(errors)) < 0.05


def _from_ids(ids: np.ndarray) -> list[Card]:
    return [Card(Rank(int(card_id) % 13 + 2), Suit(int(card_id) // 13)) for card_id in ids]
