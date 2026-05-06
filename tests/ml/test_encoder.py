"""Tests for ML encoder utilities."""

import pytest

from poker.domain.card import Card, Rank, Suit
from poker.ml.encoder import (
    card_from_index,
    card_to_index,
    cards_to_one_hot,
    dequantize_raise,
    quantize_raise,
)


def test_card_index_round_trip() -> None:
    card = Card(Rank.ACE, Suit.HEARTS)
    index = card_to_index(card)
    recovered = card_from_index(index)
    assert card_to_index(recovered) == index
    for i in range(52):
        card = card_from_index(i)
        assert card_to_index(card) == i


def test_cards_to_one_hot_values() -> None:
    ace_hearts = Card(Rank.ACE, Suit.HEARTS)
    king_hearts = Card(Rank.KING, Suit.HEARTS)
    cards = (ace_hearts, king_hearts)
    one_hot = cards_to_one_hot(cards, num_cards=7)
    assert one_hot.shape == (7 * 52,)
    assert one_hot.dtype.name == "float32"
    ace_idx = card_to_index(ace_hearts)
    king_idx = card_to_index(king_hearts)
    assert one_hot[ace_idx] == 1.0
    assert one_hot[52 + king_idx] == 1.0
    assert (one_hot[:ace_idx] == 0.0).all()
    assert (one_hot[ace_idx + 1 : 52] == 0.0).all()


def test_dequantize_raise_round_trip() -> None:
    min_raise = 20
    max_raise = 1000
    for bucket in range(5):
        amount = dequantize_raise(bucket, min_raise, max_raise)
        assert min_raise <= amount <= max_raise
        recovered_bucket = quantize_raise(amount, min_raise, max_raise)
        assert abs(recovered_bucket - bucket) <= 1
