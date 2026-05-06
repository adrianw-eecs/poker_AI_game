"""Tests for GameConfig."""

import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.exceptions import ConfigError


def _schedule() -> BlindSchedule:
    return BlindSchedule(levels=[BlindLevel(small=5, big=10)], hands_per_level=10, fixed=True)


def _make_config(**kwargs) -> GameConfig:
    defaults = dict(
        num_players=6, starting_stack=1000, small_blind=5, big_blind=10,
        ante=0, rake_percent=5.0, rake_cap=None, blind_schedule=_schedule(),
    )
    defaults.update(kwargs)
    return GameConfig(**defaults)


def test_game_config_construction() -> None:
    config = _make_config()
    assert config.num_players == 6
    assert config.starting_stack == 1000
    assert config.rake_percent == 5.0


@pytest.mark.parametrize("kwargs,match", [
    ({"num_players": 1}, "num_players"),
    ({"num_players": 11}, "num_players"),
    ({"starting_stack": 15}, "starting_stack"),
    ({"rake_percent": -1.0}, "rake_percent"),
    ({"ante": -1}, "ante"),
])
def test_game_config_invalid(kwargs, match) -> None:
    with pytest.raises(ConfigError, match=match):
        _make_config(**kwargs)


def test_game_config_frozen() -> None:
    config = _make_config()
    with pytest.raises(AttributeError):
        config.num_players = 5  # type: ignore
