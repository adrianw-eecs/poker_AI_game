"""Tests for Deck."""

from poker.domain.deck import Deck
from poker.rng import RNG


def test_deck_has_52_unique_cards() -> None:
    deck = Deck(RNG(seed=42))
    cards = deck.deal(52)
    assert len(cards) == 52
    assert len(set(cards)) == 52


def test_deal_reduces_count() -> None:
    deck = Deck(RNG(seed=42))
    deck.deal(5)
    assert deck.remaining() == 47


def test_same_seed_reproducible() -> None:
    cards1 = Deck(RNG(seed=42)).deal(10)
    cards2 = Deck(RNG(seed=42)).deal(10)
    assert cards1 == cards2
    assert Deck(RNG(seed=99)).deal(10) != cards1
