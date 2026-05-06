"""Tests for Card, Rank, Suit."""

import pytest

from poker.domain.card import Card, Rank, Suit
from poker.exceptions import EvaluationError


def test_rank_ordering() -> None:
    assert Rank.TWO < Rank.ACE
    assert Rank.ACE > Rank.KING


def test_card_creation() -> None:
    card = Card(Rank.ACE, Suit.SPADES)
    assert card.rank == Rank.ACE
    assert card.suit == Suit.SPADES


def test_card_equality() -> None:
    assert Card(Rank.ACE, Suit.SPADES) == Card(Rank.ACE, Suit.SPADES)
    assert Card(Rank.ACE, Suit.SPADES) != Card(Rank.ACE, Suit.HEARTS)


def test_card_hashable() -> None:
    c1 = Card(Rank.ACE, Suit.SPADES)
    c2 = Card(Rank.ACE, Suit.SPADES)
    c3 = Card(Rank.KING, Suit.SPADES)
    assert len({c1, c2, c3}) == 2


def test_card_from_string() -> None:
    assert Card.from_string("As") == Card(Rank.ACE, Suit.SPADES)
    assert Card.from_string("2d").rank == Rank.TWO
    with pytest.raises(EvaluationError):
        Card.from_string("Xs")
    with pytest.raises(EvaluationError):
        Card.from_string("As2")


def test_card_round_trip() -> None:
    import re
    strip = lambda s: re.sub(r'\033\[[0-9;]*m', '', s)
    for rank in Rank:
        for suit in Suit:
            card = Card(rank, suit)
            assert Card.from_string(strip(str(card))) == card
