"""Tests for kicker display logic."""

from poker.domain.card import Card, Rank, Suit
from poker.evaluation.evaluator import evaluate_with_best_cards
from poker.evaluation.kicker_display import should_show_kicker_and_card


def _c(r: str, s: str) -> Card:
    rank = {"A": Rank.ACE, "K": Rank.KING, "Q": Rank.QUEEN, "J": Rank.JACK, "T": Rank.TEN,
            "9": Rank.NINE, "8": Rank.EIGHT, "7": Rank.SEVEN, "6": Rank.SIX, "5": Rank.FIVE,
            "4": Rank.FOUR, "3": Rank.THREE, "2": Rank.TWO}[r]
    suit = {"S": Suit.SPADES, "H": Suit.HEARTS, "D": Suit.DIAMONDS, "C": Suit.CLUBS}[s]
    return Card(rank, suit)


def test_kicker_shown_when_differs() -> None:
    p1 = tuple(_c(r, s) for r, s in [("K", "S"), ("K", "H"), ("2", "D"), ("2", "C"), ("Q", "H")])
    p2 = tuple(_c(r, s) for r, s in [("K", "S"), ("K", "H"), ("2", "D"), ("2", "C"), ("J", "H")])
    r1, b1 = evaluate_with_best_cards(p1)
    r2, b2 = evaluate_with_best_cards(p2)
    show, kicker = should_show_kicker_and_card(r1, b1, [(r2, b2)])
    assert show
    assert kicker == 12  # Queen


def test_no_kicker_on_identical_hands() -> None:
    cards = tuple(_c(r, s) for r, s in [("A", "S"), ("A", "H"), ("K", "S"), ("K", "H"), ("Q", "S")])
    r1, b1 = evaluate_with_best_cards(cards)
    r2, b2 = evaluate_with_best_cards(cards)
    show, kicker = should_show_kicker_and_card(r1, b1, [(r2, b2)])
    assert not show
    assert kicker is None
