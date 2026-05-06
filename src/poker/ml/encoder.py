"""Card and action encoding utilities for ML."""

from typing import cast

import numpy as np
import numpy.typing as npt

from poker.domain.action import Action, ActionType
from poker.domain.card import Card, Rank, Suit


def card_to_index(card: Card) -> int:
    """Encode a card as a single index 0-51.

    Args:
        card: The card to encode.

    Returns:
        An integer 0-51 where:
        - Suits (0-3): clubs, diamonds, hearts, spades
        - Ranks (2-14 mapped to 0-12): 2 to ace within each suit
        - Index = suit * 13 + (rank - 2)
    """
    suit_offset = card.suit.value * 13
    rank_offset = card.rank.value - 2
    return suit_offset + rank_offset


def card_from_index(index: int) -> Card:
    """Decode a card from an index 0-51.

    Args:
        index: An integer 0-51.

    Returns:
        The corresponding Card.

    Raises:
        ValueError: If index is not in range [0, 51].
    """
    if not 0 <= index <= 51:
        raise ValueError(f"Card index must be in [0, 51], got {index}")
    suit = Suit(index // 13)
    rank = Rank(index % 13 + 2)
    return Card(rank, suit)


def cards_to_one_hot(cards: tuple[Card, ...], num_cards: int = 7) -> npt.NDArray[np.float32]:
    """Encode cards as one-hot vectors concatenated.

    Args:
        cards: Tuple of cards (hole cards + community cards).
        num_cards: Total positions to encode (pad with zeros if needed).

    Returns:
        A one-hot array of shape (num_cards * 52,) with 1.0 at encoded
        positions and 0.0 elsewhere (flattened).
    """
    result = np.zeros(num_cards * 52, dtype=np.float32)
    for i, card in enumerate(cards):
        if i >= num_cards:
            break
        index = card_to_index(card)
        result[i * 52 + index] = 1.0
    return result


def cards_to_indices(cards: tuple[Card, ...], num_cards: int = 7) -> npt.NDArray[np.int32]:
    """Encode cards as a vector of indices.

    Args:
        cards: Tuple of cards.
        num_cards: Total positions to encode (pad with -1 if needed).

    Returns:
        An array of shape (num_cards,) with card indices 0-51 or -1 for unknown.
    """
    result = np.full(num_cards, -1, dtype=np.int32)
    for i, card in enumerate(cards):
        if i >= num_cards:
            break
        result[i] = card_to_index(card)
    return result


def action_type_to_bucket(action_type: ActionType) -> int:
    """Map action type to discrete bucket 0-6.

    This is used for the action space mask and is independent of amounts.

    Mapping:
    - 0: FOLD
    - 1: CHECK/CALL
    - 2-6: RAISE levels (5 buckets)
    - 7: ALL_IN (reserved for actual all-in actions)

    Args:
        action_type: The action type.

    Returns:
        An integer bucket 0-7.
    """
    mapping = {
        ActionType.FOLD: 0,
        ActionType.CHECK: 1,
        ActionType.CALL: 1,
        ActionType.RAISE: 2,  # Will be subdivided by amount in observation
        ActionType.ALL_IN: 7,
    }
    return mapping.get(action_type, 0)


def quantize_raise(amount: int, min_raise: int, max_raise: int, num_buckets: int = 5) -> int:
    """Quantize a raise amount into buckets 0-(num_buckets-1).

    Maps continuous raise amounts to discrete buckets. The buckets are:
    - Bucket 0: min_raise to ~20% of range above min
    - Bucket 1: ~20% to ~40% of range
    - Bucket 2: ~40% to ~60% of range
    - Bucket 3: ~60% to ~80% of range
    - Bucket 4: ~80% to max_raise

    Args:
        amount: The raise amount to quantize.
        min_raise: Minimum legal raise amount.
        max_raise: Maximum legal raise amount (player's stack + committed).

    Returns:
        An integer 0-(num_buckets-1).
    """
    if amount <= min_raise:
        return 0
    if amount >= max_raise:
        return num_buckets - 1

    # Linear interpolation between min and max
    normalized = (amount - min_raise) / (max_raise - min_raise)
    bucket = int(normalized * (num_buckets - 1))
    return min(bucket, num_buckets - 1)


def dequantize_raise(
    bucket: int, min_raise: int, max_raise: int, num_buckets: int = 5
) -> int:
    """Convert a quantized raise bucket back to an amount.

    Args:
        bucket: The raise bucket 0-(num_buckets-1).
        min_raise: Minimum legal raise amount.
        max_raise: Maximum legal raise amount.
        num_buckets: Number of buckets (default 5).

    Returns:
        An estimated raise amount.
    """
    if bucket <= 0:
        return min_raise
    if bucket >= num_buckets - 1:
        return max_raise

    # Inverse of linear interpolation
    normalized = bucket / (num_buckets - 1)
    return min_raise + int(normalized * (max_raise - min_raise))
