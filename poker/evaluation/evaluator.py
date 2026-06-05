"""Hand strength evaluation — best 5-card poker hands."""

from collections import Counter
from collections.abc import Sequence
from itertools import combinations

from poker.domain.card import Card
from poker.domain.hand import HandRank, HandType
from poker.exceptions import EvaluationError


def evaluate(cards: Sequence[Card]) -> HandRank:
    """Evaluate the best 5-card hand from a 5, 6, or 7-card set.

    Args:
        cards: A sequence of 5, 6, or 7 cards.

    Returns:
        The best HandRank achievable.

    Raises:
        EvaluationError: If the input is not 5, 6, or 7 cards.
    """
    if len(cards) not in (5, 6, 7):
        raise EvaluationError(f"Expected 5-7 cards, got {len(cards)}")

    if len(cards) == 5:
        return _evaluate_hand(cards)

    # For 6 or 7 cards, find the best 5-card combination
    best: HandRank | None = None
    for combo in combinations(cards, 5):
        hand = _evaluate_hand(combo)
        if best is None or hand > best:
            best = hand
    assert best is not None  # Guaranteed because C(6,5) >= 1
    return best


def evaluate_with_best_cards(cards: Sequence[Card]) -> tuple[HandRank, tuple[Card, ...]]:
    """Evaluate the best 5-card hand and return both the rank and the actual cards.

    Args:
        cards: A sequence of 5, 6, or 7 cards.

    Returns:
        A tuple of (HandRank, best_5_cards) where best_5_cards is the actual Card objects.

    Raises:
        EvaluationError: If the input is not 5, 6, or 7 cards.
    """
    if len(cards) not in (5, 6, 7):
        raise EvaluationError(f"Expected 5-7 cards, got {len(cards)}")

    if len(cards) == 5:
        return _evaluate_hand(cards), tuple(cards)

    # For 6 or 7 cards, find the best 5-card combination
    best_rank: HandRank | None = None
    best_combo: tuple[Card, ...] | None = None
    for combo in combinations(cards, 5):
        hand = _evaluate_hand(combo)
        if best_rank is None or hand > best_rank:
            best_rank = hand
            best_combo = combo
    assert best_rank is not None  # Guaranteed because C(6,5) >= 1
    assert best_combo is not None
    return best_rank, best_combo


def _evaluate_hand(cards: Sequence[Card]) -> HandRank:
    """Evaluate a 5-card poker hand.

    Args:
        cards: Exactly 5 cards.

    Returns:
        The hand ranking.
    """
    if len(cards) != 5:
        raise EvaluationError(f"_evaluate_hand expects exactly 5 cards, got {len(cards)}")

    # Check for flush and straight
    is_flush = _is_flush(cards)
    straight_high = _get_straight_high(cards)

    if straight_high and is_flush:
        # Straight flush
        return HandRank(HandType.STRAIGHT_FLUSH, (straight_high,))

    # Count rank frequencies
    rank_counts = Counter(card.rank for card in cards)
    counts_list = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)

    # Extract counts as a tuple (for grouping)
    counts_pattern = tuple(count for _, count in counts_list)

    if counts_pattern == (4, 1):
        # Four of a kind
        quads_rank = int(counts_list[0][0])
        kicker = int(counts_list[1][0])
        return HandRank(HandType.FOUR_OF_A_KIND, (quads_rank, kicker))

    if counts_pattern == (3, 2):
        # Full house
        trips_rank = int(counts_list[0][0])
        pair_rank = int(counts_list[1][0])
        return HandRank(HandType.FULL_HOUSE, (trips_rank, pair_rank))

    if is_flush:
        # Flush: kickers are all 5 cards in descending order
        kickers = tuple(sorted((int(card.rank) for card in cards), reverse=True))
        return HandRank(HandType.FLUSH, kickers)

    if straight_high:
        # Straight: kicker is just the high rank
        return HandRank(HandType.STRAIGHT, (straight_high,))

    if counts_pattern == (3, 1, 1):
        # Three of a kind
        trips_rank = int(counts_list[0][0])
        kickers = tuple(
            sorted((int(rank) for rank, _ in counts_list[1:]), reverse=True)
        )
        return HandRank(HandType.THREE_OF_A_KIND, (trips_rank, *kickers))

    if counts_pattern == (2, 2, 1):
        # Two pair
        pair_ranks = sorted(
            (int(counts_list[0][0]), int(counts_list[1][0])), reverse=True
        )
        kicker = int(counts_list[2][0])
        return HandRank(HandType.TWO_PAIR, (pair_ranks[0], pair_ranks[1], kicker))

    if counts_pattern == (2, 1, 1, 1):
        # One pair
        pair_rank = int(counts_list[0][0])
        kickers = tuple(
            sorted((int(rank) for rank, _ in counts_list[1:]), reverse=True)
        )
        return HandRank(HandType.PAIR, (pair_rank, *kickers))

    # High card
    kickers = tuple(sorted((int(card.rank) for card in cards), reverse=True))
    return HandRank(HandType.HIGH_CARD, kickers)


def _is_flush(cards: Sequence[Card]) -> bool:
    """Check if all 5 cards are the same suit."""
    suits = {card.suit for card in cards}
    return len(suits) == 1


def _get_straight_high(cards: Sequence[Card]) -> int | None:
    """Return the high rank of a straight if one exists, else None.

    Handles both ace-high (A-K-Q-J-T) and ace-low (5-4-3-2-A) straights.
    Returns:
        The rank of the high card (14 for ace-high, 5 for wheel), or None.
    """
    ranks = sorted([int(card.rank) for card in cards], reverse=True)

    # Check for regular straight (5 consecutive ranks)
    if ranks[0] - ranks[4] == 4 and len(set(ranks)) == 5:
        return ranks[0]

    # Check for ace-low straight (5-4-3-2-A, called the "wheel")
    # In this case, rank order is [14, 5, 4, 3, 2] (after sorting desc)
    # The high card is 5, not ace (14)
    if set(ranks) == {14, 5, 4, 3, 2}:
        return 5  # Wheel is 5-high

    return None
