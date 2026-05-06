"""Tests for BlindSchedule configuration."""

import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.exceptions import ConfigError


def test_blind_schedule_construction() -> None:
    levels = [BlindLevel(small=5, big=10), BlindLevel(small=10, big=20)]
    schedule = BlindSchedule(levels=levels, hands_per_level=10)
    assert len(schedule.levels) == 2
    assert schedule.hands_per_level == 10
    assert not schedule.fixed


def test_blind_schedule_level_progression() -> None:
    levels = [BlindLevel(small=5, big=10), BlindLevel(small=10, big=20), BlindLevel(small=25, big=50)]
    schedule = BlindSchedule(levels=levels, hands_per_level=10)
    assert schedule.level_for_hand(0) == levels[0]
    assert schedule.level_for_hand(9) == levels[0]
    assert schedule.level_for_hand(10) == levels[1]
    assert schedule.level_for_hand(20) == levels[2]
    assert schedule.level_for_hand(100) == levels[2]  # clamped to last


def test_blind_schedule_fixed_mode() -> None:
    levels = [BlindLevel(small=5, big=10), BlindLevel(small=10, big=20)]
    schedule = BlindSchedule(levels=levels, hands_per_level=10, fixed=True)
    assert schedule.level_for_hand(0) == levels[0]
    assert schedule.level_for_hand(100) == levels[0]


@pytest.mark.parametrize("levels,hpl,match", [
    ([], 10, "at least one level"),
    ([BlindLevel(small=5, big=10)], 0, "hands_per_level"),
    ([BlindLevel(small=5, big=10)], -1, "hands_per_level"),
])
def test_blind_schedule_invalid(levels, hpl, match) -> None:
    with pytest.raises(ConfigError, match=match):
        BlindSchedule(levels=levels, hands_per_level=hpl)
