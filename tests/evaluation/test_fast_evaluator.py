"""Fast evaluator agrees with evaluate() and is cheap enough to call per decision."""

import time

import numpy as np
import pytest

from poker.domain.card import Card, Rank, Suit
from poker.domain.hand import HandType
from poker.evaluation.evaluator import evaluate
from poker.evaluation.fast_evaluator import (
    LOOKUP_SIZE,
    N_MULTISETS,
    best_slots,
    card_index,
    fast_evaluate,
    fast_score,
    intra_category,
    score_hands,
)
from poker.exceptions import EvaluationError
from poker.ml.encoder import card_to_index


def _cards(*specs: tuple[Rank, Suit]) -> list[Card]:
    return [Card(rank, suit) for rank, suit in specs]


def _from_ids(ids: np.ndarray) -> list[Card]:
    return [Card(Rank(int(card_id) % 13 + 2), Suit(int(card_id) // 13)) for card_id in ids]


_EDGE_CASES = [
    _cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.JACK, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
    ),
    _cards(
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.HEARTS),
    ),
    _cards(
        (Rank.KING, Suit.HEARTS),
        (Rank.KING, Suit.DIAMONDS),
        (Rank.QUEEN, Suit.CLUBS),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
    ),
    _cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.HEARTS),
    ),
    _cards(
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.JACK, Suit.CLUBS),
        (Rank.TEN, Suit.SPADES),
        (Rank.NINE, Suit.HEARTS),
    ),
    # Wheel, and the same wheel in one suit.
    _cards(
        (Rank.FIVE, Suit.HEARTS),
        (Rank.FOUR, Suit.DIAMONDS),
        (Rank.THREE, Suit.CLUBS),
        (Rank.TWO, Suit.SPADES),
        (Rank.ACE, Suit.HEARTS),
    ),
    _cards(
        (Rank.FIVE, Suit.HEARTS),
        (Rank.FOUR, Suit.HEARTS),
        (Rank.THREE, Suit.HEARTS),
        (Rank.TWO, Suit.HEARTS),
        (Rank.ACE, Suit.HEARTS),
    ),
    _cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.HEARTS),
        (Rank.JACK, Suit.HEARTS),
        (Rank.NINE, Suit.HEARTS),
    ),
    _cards(
        (Rank.ACE, Suit.SPADES),
        (Rank.KING, Suit.SPADES),
        (Rank.QUEEN, Suit.SPADES),
        (Rank.JACK, Suit.SPADES),
        (Rank.TEN, Suit.SPADES),
    ),
    _cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
    ),
    _cards(
        (Rank.ACE, Suit.HEARTS),
        (Rank.ACE, Suit.DIAMONDS),
        (Rank.ACE, Suit.CLUBS),
        (Rank.KING, Suit.SPADES),
        (Rank.KING, Suit.HEARTS),
        (Rank.QUEEN, Suit.DIAMONDS),
        (Rank.JACK, Suit.CLUBS),
    ),
]


def test_lookup_covers_every_rank_multiset() -> None:
    assert N_MULTISETS == 6_188
    assert LOOKUP_SIZE == 12_376


def test_card_index_matches_encoder() -> None:
    card = Card(Rank.ACE, Suit.SPADES)
    assert card_index(card) == card_to_index(card)


@pytest.mark.smoke
def test_named_straight_and_flush_edges_match() -> None:
    for cards in _EDGE_CASES:
        assert fast_evaluate(cards) == evaluate(cards)


def test_fast_evaluator_matches_reference_on_20000_hands() -> None:
    rng = np.random.default_rng(0)
    deck = np.arange(52)
    for i in range(20_000):
        n = 5 + (i % 3)
        ids = rng.choice(deck, n, replace=False)
        cards = _from_ids(ids)
        got = fast_evaluate(cards)
        ref = evaluate(cards)
        assert got == ref, f"{[c.code() for c in cards]}: {got} != {ref}"


def test_batched_scores_match_scalar_scores() -> None:
    rng = np.random.default_rng(1)
    ids = np.stack([rng.choice(52, 7, replace=False) for _ in range(400)]).astype(np.int16)
    batched = score_hands(ids)
    slots = best_slots(ids)
    assert batched.shape == (400,)
    assert slots.shape == (400,)
    for row, score in zip(ids, batched, strict=True):
        assert int(score) == fast_score(_from_ids(row))


def test_score_order_matches_handrank_order() -> None:
    rng = np.random.default_rng(2)
    deck = np.arange(52)
    previous = None
    previous_cards: list[Card] | None = None
    for _ in range(300):
        cards = _from_ids(rng.choice(deck, 7, replace=False))
        if previous is not None and previous_cards is not None:
            left = evaluate(previous_cards)
            right = evaluate(cards)
            if left > right:
                assert fast_score(previous_cards) > fast_score(cards)
            elif left < right:
                assert fast_score(previous_cards) < fast_score(cards)
            else:
                assert fast_score(previous_cards) == fast_score(cards)
        previous = fast_score(cards)
        previous_cards = cards


def test_intra_category_orders_kickers() -> None:
    wheel = fast_evaluate(_EDGE_CASES[5])
    broadway = fast_evaluate(
        _cards(
            (Rank.ACE, Suit.HEARTS),
            (Rank.KING, Suit.DIAMONDS),
            (Rank.QUEEN, Suit.CLUBS),
            (Rank.JACK, Suit.SPADES),
            (Rank.TEN, Suit.HEARTS),
        )
    )
    assert wheel.type == HandType.STRAIGHT
    assert broadway.type == HandType.STRAIGHT
    assert intra_category(wheel) == pytest.approx(0.0)
    assert intra_category(broadway) == pytest.approx(1.0)


def test_wrong_card_count_raises() -> None:
    with pytest.raises(EvaluationError):
        fast_evaluate(_cards((Rank.ACE, Suit.HEARTS), (Rank.KING, Suit.HEARTS)) * 2)
    with pytest.raises(EvaluationError):
        fast_evaluate(_EDGE_CASES[0] + _EDGE_CASES[1][:3])


def test_batched_seven_card_eval_is_under_10us() -> None:
    rng = np.random.default_rng(3)
    batch = np.stack([rng.choice(52, 7, replace=False) for _ in range(200)]).astype(np.int16)
    score_hands(batch)
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        score_hands(batch)
        samples.append((time.perf_counter() - start) / 200)
    assert sorted(samples)[len(samples) // 2] < 10e-6
