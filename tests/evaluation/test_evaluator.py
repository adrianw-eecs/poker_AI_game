"""Tests for hand evaluator — one test per hand type."""

import pytest

from poker.domain.card import Card, Rank, Suit
from poker.domain.hand import HandType
from poker.evaluation.evaluator import evaluate
from poker.exceptions import EvaluationError


def _cards(*specs: tuple[Rank, Suit]) -> list[Card]:
    return [Card(r, s) for r, s in specs]


@pytest.mark.smoke
def test_high_card() -> None:
    hand = evaluate(_cards((Rank.ACE, Suit.HEARTS), (Rank.KING, Suit.DIAMONDS),
                           (Rank.QUEEN, Suit.CLUBS), (Rank.JACK, Suit.SPADES),
                           (Rank.NINE, Suit.HEARTS)))
    assert hand.type == HandType.HIGH_CARD
    assert hand.kickers == (14, 13, 12, 11, 9)


@pytest.mark.smoke
def test_pair() -> None:
    hand = evaluate(_cards((Rank.KING, Suit.HEARTS), (Rank.KING, Suit.DIAMONDS),
                           (Rank.ACE, Suit.CLUBS), (Rank.QUEEN, Suit.SPADES),
                           (Rank.JACK, Suit.HEARTS)))
    assert hand.type == HandType.PAIR
    assert hand.kickers == (13, 14, 12, 11)


def test_two_pair() -> None:
    hand = evaluate(_cards((Rank.KING, Suit.HEARTS), (Rank.KING, Suit.DIAMONDS),
                           (Rank.QUEEN, Suit.CLUBS), (Rank.QUEEN, Suit.SPADES),
                           (Rank.ACE, Suit.HEARTS)))
    assert hand.type == HandType.TWO_PAIR
    assert hand.kickers == (13, 12, 14)


def test_three_of_a_kind() -> None:
    hand = evaluate(_cards((Rank.ACE, Suit.HEARTS), (Rank.ACE, Suit.DIAMONDS),
                           (Rank.ACE, Suit.CLUBS), (Rank.KING, Suit.SPADES),
                           (Rank.QUEEN, Suit.HEARTS)))
    assert hand.type == HandType.THREE_OF_A_KIND


def test_straight() -> None:
    hand = evaluate(_cards((Rank.KING, Suit.HEARTS), (Rank.QUEEN, Suit.DIAMONDS),
                           (Rank.JACK, Suit.CLUBS), (Rank.TEN, Suit.SPADES),
                           (Rank.NINE, Suit.HEARTS)))
    assert hand.type == HandType.STRAIGHT
    assert hand.kickers == (13,)

    wheel = evaluate(_cards((Rank.FIVE, Suit.HEARTS), (Rank.FOUR, Suit.DIAMONDS),
                            (Rank.THREE, Suit.CLUBS), (Rank.TWO, Suit.SPADES),
                            (Rank.ACE, Suit.HEARTS)))
    assert wheel.type == HandType.STRAIGHT
    assert wheel.kickers == (5,)


def test_flush() -> None:
    hand = evaluate(_cards((Rank.ACE, Suit.HEARTS), (Rank.KING, Suit.HEARTS),
                           (Rank.QUEEN, Suit.HEARTS), (Rank.JACK, Suit.HEARTS),
                           (Rank.NINE, Suit.HEARTS)))
    assert hand.type == HandType.FLUSH


def test_full_house() -> None:
    hand = evaluate(_cards((Rank.ACE, Suit.HEARTS), (Rank.ACE, Suit.DIAMONDS),
                           (Rank.ACE, Suit.CLUBS), (Rank.KING, Suit.SPADES),
                           (Rank.KING, Suit.HEARTS)))
    assert hand.type == HandType.FULL_HOUSE
    assert hand.kickers == (14, 13)


@pytest.mark.smoke
def test_royal_flush() -> None:
    hand = evaluate(_cards((Rank.ACE, Suit.SPADES), (Rank.KING, Suit.SPADES),
                           (Rank.QUEEN, Suit.SPADES), (Rank.JACK, Suit.SPADES),
                           (Rank.TEN, Suit.SPADES)))
    assert hand.type == HandType.STRAIGHT_FLUSH
    assert hand.kickers == (14,)


def test_seven_card_best_hand() -> None:
    hand = evaluate(_cards(
        (Rank.ACE, Suit.HEARTS), (Rank.ACE, Suit.DIAMONDS), (Rank.ACE, Suit.CLUBS),
        (Rank.KING, Suit.SPADES), (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS), (Rank.JACK, Suit.CLUBS),
    ))
    assert hand.type == HandType.FULL_HOUSE
    assert hand.kickers == (14, 13)


def test_wrong_card_count_raises() -> None:
    with pytest.raises(EvaluationError):
        evaluate([Card(Rank.ACE, Suit.HEARTS)] * 4)
    with pytest.raises(EvaluationError):
        evaluate([Card(Rank.ACE, Suit.HEARTS)] * 8)
