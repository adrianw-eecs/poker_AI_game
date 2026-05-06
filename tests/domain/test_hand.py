"""Tests for HandType and HandRank."""

import pytest

from poker.domain.hand import HandRank, HandType


def test_hand_type_ordering() -> None:
    assert HandType.HIGH_CARD < HandType.PAIR < HandType.TWO_PAIR
    assert HandType.TWO_PAIR < HandType.THREE_OF_A_KIND < HandType.STRAIGHT
    assert HandType.STRAIGHT < HandType.FLUSH < HandType.FULL_HOUSE
    assert HandType.FULL_HOUSE < HandType.FOUR_OF_A_KIND < HandType.STRAIGHT_FLUSH


def test_hand_rank_construction() -> None:
    rank = HandRank(type=HandType.PAIR, kickers=(14, 13, 12))
    assert rank.type == HandType.PAIR
    assert rank.kickers == (14, 13, 12)
    with pytest.raises(AttributeError):
        rank.type = HandType.TWO_PAIR  # type: ignore


def test_hand_rank_type_ordering() -> None:
    high_card = HandRank(type=HandType.HIGH_CARD, kickers=(14, 13, 12, 11, 9))
    pair = HandRank(type=HandType.PAIR, kickers=(14, 13, 12, 11))
    sf = HandRank(type=HandType.STRAIGHT_FLUSH, kickers=(14,))
    assert high_card < pair < sf


def test_hand_rank_kicker_ordering() -> None:
    pair_ak = HandRank(type=HandType.PAIR, kickers=(14, 13, 12, 11))
    pair_aq = HandRank(type=HandType.PAIR, kickers=(14, 12, 11, 10))
    pair_kk = HandRank(type=HandType.PAIR, kickers=(13, 12, 11, 10))
    assert pair_ak > pair_aq > pair_kk
