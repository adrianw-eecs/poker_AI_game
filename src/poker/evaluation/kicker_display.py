"""Smart kicker display logic for showdown announcements.

Determines whether a kicker should be shown based on whether it was the deciding
factor in a hand comparison.
"""

from poker.domain.card import Card
from poker.domain.hand import HandRank, HandType


def should_show_kicker_and_card(
    hand_rank: HandRank,
    best_5_cards: tuple[Card, ...],
    other_hands: list[tuple[HandRank, tuple[Card, ...]]],
) -> tuple[bool, int | None]:
    """Determine if a kicker should be shown for a hand in showdown.

    A kicker should be shown when:
    1. There are other hands with the exact same hand rank
    2. The kickers differ between this hand and those same-rank hands
    3. Show kicker for all hands involved in the same-rank comparison (winner and loser)

    Args:
        hand_rank: The HandRank of this player's hand.
        best_5_cards: The actual 5 cards that form this hand.
        other_hands: List of (HandRank, best_5_cards) tuples for other players.

    Returns:
        A tuple of (should_show_kicker, kicker_rank) where:
        - should_show_kicker: True if a kicker should be displayed
        - kicker_rank: The rank value of the kicker to show (2-14), or None if no kicker
    """
    # Hands where kickers are not shown: straight, flush, straight flush, full house
    if hand_rank.type in (
        HandType.STRAIGHT,
        HandType.FLUSH,
        HandType.STRAIGHT_FLUSH,
        HandType.FULL_HOUSE,
    ):
        return False, None

    # Find all hands with the same rank as this one
    same_rank_hands = [
        (rank, cards) for rank, cards in other_hands if rank.type == hand_rank.type
    ]

    # If no other hands of the same type, don't show kicker
    if not same_rank_hands:
        return False, None

    # Check if hands are identical
    best_5_card_ranks = tuple(int(card.rank) for card in best_5_cards)

    # Compare with all other same-rank hands to see if any have different kickers
    hands_differ = False
    for other_rank, other_cards in same_rank_hands:
        # Check if this hand differs from the other hand
        # For hands of same type, if the full HandRank comparison shows inequality,
        # then kickers differ
        if hand_rank.kickers != other_rank.kickers:
            # Kickers differ
            hands_differ = True
            break

    # If all hands are identical, don't show kicker (split pot)
    if not hands_differ:
        return False, None

    # Get kickers for this hand based on hand type
    this_kickers = _get_kickers_from_rank(hand_rank, best_5_card_ranks)

    # Find the strongest other hand (highest rank among the same-rank hands)
    max_other_kickers = None
    for other_rank, other_cards in same_rank_hands:
        other_card_ranks = tuple(int(card.rank) for card in other_cards)
        other_kickers = _get_kickers_from_rank(other_rank, other_card_ranks)
        if max_other_kickers is None or other_kickers > max_other_kickers:
            max_other_kickers = other_kickers

    # Show the highest kicker that differs
    if this_kickers is None or max_other_kickers is None:
        return False, None

    # Find the first (highest) kicker that differs
    for this_k, other_k in zip(this_kickers, max_other_kickers):
        if this_k != other_k:
            return True, this_k

    # If all kickers match (shouldn't happen if hands_differ is True), don't show
    return False, None


def _get_kickers_from_rank(
    hand_rank: HandRank, card_ranks: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Extract the kicker(s) from a hand rank based on its type.

    For each hand type, identify which cards in the best 5 are "kickers"
    (not part of the main hand component).

    Args:
        hand_rank: The HandRank object.
        card_ranks: The ranks of the 5-card hand (as integers 2-14).

    Returns:
        A tuple of kicker ranks in descending order, or None if not applicable.
    """
    if hand_rank.type == HandType.PAIR:
        # For a pair, kickers are the 3 remaining cards in descending order
        pair_rank = hand_rank.kickers[0]
        kickers = tuple(sorted([r for r in card_ranks if r != pair_rank], reverse=True))
        return kickers

    if hand_rank.type == HandType.TWO_PAIR:
        # For two pair, the kicker is the 5th card
        high_pair = hand_rank.kickers[0]
        low_pair = hand_rank.kickers[1]
        kicker = tuple(r for r in card_ranks if r != high_pair and r != low_pair)
        return kicker

    if hand_rank.type == HandType.THREE_OF_A_KIND:
        # For trips, kickers are the 2 remaining cards in descending order
        trips_rank = hand_rank.kickers[0]
        kickers = tuple(sorted([r for r in card_ranks if r != trips_rank], reverse=True))
        return kickers

    if hand_rank.type == HandType.FOUR_OF_A_KIND:
        # For quads, the kicker is the 5th card
        quads_rank = hand_rank.kickers[0]
        kicker = tuple(r for r in card_ranks if r != quads_rank)
        return kicker

    if hand_rank.type == HandType.HIGH_CARD:
        # For high card, all 5 cards are kickers in descending order
        return tuple(sorted(card_ranks, reverse=True))

    # Straight, Flush, Full House, Straight Flush: no kickers shown
    return None
