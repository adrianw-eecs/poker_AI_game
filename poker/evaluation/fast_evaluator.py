"""Rank-multiset lookup for 5-, 6-, and 7-card hands.

A 5-card hand is fixed by its sorted rank multiset and whether the suits
match. Multisets of size 5 drawn from 13 ranks are combinations with
repetition, C(13 + 5 - 1, 5) = C(17, 5) = 6,188. The flush bit doubles that
to 12,376 table entries. Each entry stores the ``HandRank`` that
``evaluate`` returns for that multiset, packed into an integer that sorts
the same way ``HandRank`` does.

Six- and seven-card hands take the maximum over the C(n, 5) five-card
subsets. The subsets are integer index tuples, so the hot path never builds
``Card`` objects, ``Counter``s, or sorted rank lists of enum values.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations, combinations_with_replacement
from typing import cast

import numpy as np
import numpy.typing as npt

from poker.domain.card import Card, Rank, Suit
from poker.domain.hand import HandRank, HandType
from poker.evaluation.evaluator import evaluate
from poker.exceptions import EvaluationError

# C(13 + 5 - 1, 5) rank multisets, times the flush flag.
N_MULTISETS = 6_188
LOOKUP_SIZE = N_MULTISETS * 2

_N_CARDS = (5, 6, 7)


def card_index(card: Card) -> int:
    """Encode a card as ``suit * 13 + (rank - 2)``, matching ``card_to_index``."""
    return int(card.suit) * 13 + (int(card.rank) - 2)


def _binomial_rows() -> list[list[int]]:
    """C(n, k) for n = 0..16 and k = 0..5. C(n, k) is 0 when n < k."""
    rows: list[list[int]] = []
    for n in range(17):
        row = [0] * 6
        row[0] = 1
        if n > 0:
            prev = rows[n - 1]
            for k in range(1, 6):
                row[k] = prev[k - 1] + prev[k]
        rows.append(row)
    return rows


_BINOM = _binomial_rows()
_B1 = [row[1] for row in _BINOM]
_B2 = [row[2] for row in _BINOM]
_B3 = [row[3] for row in _BINOM]
_B4 = [row[4] for row in _BINOM]
_B5 = [row[5] for row in _BINOM]
_BINOM_NP = np.asarray(_BINOM, dtype=np.int32)
_OFFSETS = np.arange(5, dtype=np.int16)

# Colex index of a non-decreasing 5-tuple of ranks in 0..12. The map
# b_i = rank_i + i sends it to a 5-combination of 0..16, and there are
# exactly C(17, 5) of those.
def multiset_index(ranks: tuple[int, ...]) -> int:
    """Index a non-decreasing 5-tuple of ranks in ``0..12``."""
    return (
        _B5[ranks[4] + 4]
        + _B4[ranks[3] + 3]
        + _B3[ranks[2] + 2]
        + _B2[ranks[1] + 1]
        + _B1[ranks[0]]
    )


def _combos(n: int) -> tuple[tuple[int, int, int, int, int], ...]:
    return tuple(
        (c[0], c[1], c[2], c[3], c[4]) for c in combinations(range(n), 5)
    )


_COMBOS: dict[int, tuple[tuple[int, int, int, int, int], ...]] = {
    n: _combos(n) for n in _N_CARDS
}
_COMBO_NP: dict[int, npt.NDArray[np.intp]] = {
    n: np.asarray(combos, dtype=np.intp) for n, combos in _COMBOS.items()
}


def _pack(hand: HandRank) -> int:
    """Pack ``(type, kickers)`` so the integer order matches ``HandRank``."""
    score = int(hand.type)
    kickers = hand.kickers
    width = len(kickers)
    for i in range(5):
        score = score * 15 + (kickers[i] if i < width else 0)
    return score


def _cards_for(ranks: tuple[int, ...], flush: bool) -> list[Card]:
    """Build one concrete 5-card hand for a rank multiset.

    Distinct ranks assigned round-robin suits are not a flush. Repeated
    ranks take the next suit, which a flush cannot do (one suit has only
    one card of each rank).
    """
    cards: list[Card] = []
    seen: dict[int, int] = {}
    distinct = len(set(ranks)) == 5
    for i, rank0 in enumerate(ranks):
        if flush:
            suit = Suit.HEARTS
        elif distinct:
            suit = Suit(i % 4)
        else:
            used = seen.get(rank0, 0)
            suit = Suit(used)
            seen[rank0] = used + 1
        cards.append(Card(Rank(rank0 + 2), suit))
    return cards


def _build_tables() -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.int32],
    dict[tuple[int, tuple[int, ...]], float],
]:
    types = np.zeros(LOOKUP_SIZE, dtype=np.uint8)
    lengths = np.zeros(LOOKUP_SIZE, dtype=np.uint8)
    kickers = np.zeros((LOOKUP_SIZE, 5), dtype=np.uint8)
    scores = np.zeros(LOOKUP_SIZE, dtype=np.int32)
    seen: set[int] = set()

    for ranks in combinations_with_replacement(range(13), 5):
        index = multiset_index(ranks)
        if index in seen:
            raise RuntimeError(f"rank multiset {ranks} collided at index {index}")
        seen.add(index)
        highest = max(ranks.count(rank) for rank in set(ranks))
        # Five of a kind cannot be dealt. Leave the slot empty.
        if highest > 4:
            continue
        for flush in (False, True):
            # A flush has five distinct ranks, so a repeated rank has no
            # flush slot that a real deck can reach.
            if flush and highest > 1:
                continue
            dealt = _cards_for(ranks, flush)
            hand = evaluate(dealt)
            is_flush_type = hand.type in (HandType.FLUSH, HandType.STRAIGHT_FLUSH)
            if flush != is_flush_type:
                raise RuntimeError(
                    f"flush flag {flush} disagreed with {hand} for ranks {ranks}"
                )
            slot = (index << 1) | int(flush)
            width = len(hand.kickers)
            types[slot] = int(hand.type)
            lengths[slot] = width
            for i in range(width):
                kickers[slot, i] = hand.kickers[i]
            scores[slot] = _pack(hand)

    if len(seen) != N_MULTISETS:
        raise RuntimeError(f"expected {N_MULTISETS} multisets, indexed {len(seen)}")
    if int(_B5[16]) != 4368:
        raise RuntimeError("binomial table is wrong: C(16, 5) != 4368")

    groups: dict[int, set[tuple[int, ...]]] = {}
    for slot in range(LOOKUP_SIZE):
        hand_type = int(types[slot])
        if hand_type == 0:
            continue
        width = int(lengths[slot])
        key = tuple(int(kickers[slot, j]) for j in range(width))
        groups.setdefault(hand_type, set()).add(key)

    intra: dict[tuple[int, tuple[int, ...]], float] = {}
    for hand_type, keys in groups.items():
        ordered = sorted(keys)
        denom = max(len(ordered) - 1, 1)
        for pos, key in enumerate(ordered):
            intra[(hand_type, key)] = pos / denom
    return types, lengths, kickers, scores, intra


_TYPES, _LENGTHS, _KICKERS, _SCORES_NP, _INTRA = _build_tables()
_SCORE_LIST: list[int] = [int(score) for score in _SCORES_NP]


def _sort5(a: int, b: int, c: int, d: int, e: int) -> tuple[int, int, int, int, int]:
    """Insertion-sort five ranks. No list allocation."""
    if a > b:
        a, b = b, a
    if b > c:
        b, c = c, b
        if a > b:
            a, b = b, a
    if c > d:
        c, d = d, c
        if b > c:
            b, c = c, b
            if a > b:
                a, b = b, a
    if d > e:
        d, e = e, d
        if c > d:
            c, d = d, c
            if b > c:
                b, c = c, b
                if a > b:
                    a, b = b, a
    return a, b, c, d, e


def _winning_slot(ranks: Sequence[int], suits: Sequence[int]) -> int:
    """Table slot of the best 5-card subset."""
    n = len(ranks)
    combos = _COMBOS.get(n)
    if combos is None:
        raise EvaluationError(f"Expected 5-7 cards, got {n}")
    scores = _SCORE_LIST
    b5, b4, b3, b2, b1 = _B5, _B4, _B3, _B2, _B1
    best_score = -1
    best_slot = 0
    for i0, i1, i2, i3, i4 in combos:
        a, b, c, d, e = _sort5(ranks[i0], ranks[i1], ranks[i2], ranks[i3], ranks[i4])
        same_suit = (
            suits[i0] == suits[i1]
            and suits[i0] == suits[i2]
            and suits[i0] == suits[i3]
            and suits[i0] == suits[i4]
        )
        index = b5[e + 4] + b4[d + 3] + b3[c + 2] + b2[b + 1] + b1[a]
        slot = (index << 1) | int(same_suit)
        score = scores[slot]
        if score > best_score:
            best_score = score
            best_slot = slot
    return best_slot


def _split_ids(card_ids: Sequence[int]) -> tuple[list[int], list[int]]:
    ranks = [card_id % 13 for card_id in card_ids]
    suits = [card_id // 13 for card_id in card_ids]
    return ranks, suits


def _hand_at(slot: int) -> HandRank:
    hand_type = int(_TYPES[slot])
    if hand_type == 0:
        raise EvaluationError(f"lookup slot {slot} is not a dealable hand")
    width = int(_LENGTHS[slot])
    kickers = tuple(int(_KICKERS[slot, j]) for j in range(width))
    return HandRank(HandType(hand_type), kickers)


def fast_evaluate(cards: Sequence[Card]) -> HandRank:
    """Best 5-card ``HandRank`` from 5, 6, or 7 cards.

    Agrees with ``evaluate`` on every dealable hand. The reference
    implementation is what filled this table; this function only looks up.
    """
    n = len(cards)
    if n not in _N_CARDS:
        raise EvaluationError(f"Expected 5-7 cards, got {n}")
    ranks = [int(card.rank) - 2 for card in cards]
    suits = [int(card.suit) for card in cards]
    return _hand_at(_winning_slot(ranks, suits))


def fast_score(cards: Sequence[Card]) -> int:
    """Integer strength. Larger is better, and ties match ``HandRank`` ties."""
    n = len(cards)
    if n not in _N_CARDS:
        raise EvaluationError(f"Expected 5-7 cards, got {n}")
    ranks = [int(card.rank) - 2 for card in cards]
    suits = [int(card.suit) for card in cards]
    return _SCORE_LIST[_winning_slot(ranks, suits)]


def intra_category(hand: HandRank) -> float:
    """Where ``hand`` sits inside its category, in ``[0, 1]``.

    0 is the worst kicker tuple of that hand type and 1 is the best, using
    the same lexicographic order as ``HandRank.kickers``.
    """
    return _INTRA.get((int(hand.type), hand.kickers), 0.0)


def best_slots(card_ids: npt.NDArray[np.int16]) -> npt.NDArray[np.int32]:
    """Winning lookup slot for each row of card ids shaped ``(N, 5|6|7)``."""
    cards = np.asarray(card_ids, dtype=np.int16)
    if cards.ndim != 2 or int(cards.shape[1]) not in _N_CARDS:
        raise ValueError(
            f"card_ids must have shape (N, 5|6|7), got {tuple(cards.shape)}"
        )
    combos = _COMBO_NP[int(cards.shape[1])]
    suits = np.floor_divide(cards, 13)
    ranks = np.remainder(cards, 13)
    ordered = np.sort(ranks[:, combos], axis=-1)
    gathered_suits = suits[:, combos]
    bumped = ordered + _OFFSETS
    index = (
        _BINOM_NP[bumped[..., 4], 5]
        + _BINOM_NP[bumped[..., 3], 4]
        + _BINOM_NP[bumped[..., 2], 3]
        + _BINOM_NP[bumped[..., 1], 2]
        + _BINOM_NP[bumped[..., 0], 1]
    )
    flush = np.all(gathered_suits == gathered_suits[..., :1], axis=-1)
    slots = (index.astype(np.int32) << 1) | flush.astype(np.int32)
    scores = _SCORES_NP[slots]
    winner = np.argmax(scores, axis=1)
    chosen = slots[np.arange(cards.shape[0]), winner]
    return cast(npt.NDArray[np.int32], chosen)


def score_hands(card_ids: npt.NDArray[np.int16]) -> npt.NDArray[np.int32]:
    """Batched strength scores. Same order as ``fast_score``."""
    return _SCORES_NP[best_slots(card_ids)]
