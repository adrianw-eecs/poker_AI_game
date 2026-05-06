"""Tests for pot manager."""

import pytest

from poker.domain.hand import HandRank, HandType
from poker.state.pot import Pot
from poker.state.pot_manager import apply_rake, build_pots, distribute


def test_build_pots_equal_commitment() -> None:
    pots = build_pots({0: 100, 1: 100}, set())
    assert len(pots) == 1
    assert pots[0].amount == 200
    assert pots[0].eligible_seats == frozenset({0, 1})


@pytest.mark.smoke
def test_build_pots_side_pot() -> None:
    pots = build_pots({0: 50, 1: 150, 2: 200}, set())
    assert len(pots) == 3
    assert pots[0].amount == 150   # 50 * 3
    assert pots[0].eligible_seats == frozenset({0, 1, 2})
    assert pots[1].amount == 200   # (150-50) * 2
    assert pots[1].eligible_seats == frozenset({1, 2})
    assert pots[2].amount == 50    # (200-150) * 1
    assert pots[2].eligible_seats == frozenset({2})


def test_distribute_single_winner() -> None:
    pot = Pot(100, frozenset({0, 1}))
    awards = distribute(
        [pot],
        {0: HandRank(HandType.PAIR, (14, 13, 12, 11)),
         1: HandRank(HandType.HIGH_CARD, (14, 13, 12, 11, 10))},
        dealer_seat=0,
    )
    assert awards[0] == 100
    assert awards.get(1, 0) == 0


def test_rake_applied() -> None:
    pot = Pot(1000, frozenset({0, 1}))
    adjusted, rake = apply_rake([pot], rake_percent=5, rake_cap=None)
    assert adjusted[0].amount == 950
    assert rake == 50
