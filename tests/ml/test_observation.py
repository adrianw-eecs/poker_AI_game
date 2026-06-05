"""Tests for observation building."""

import numpy as np
import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.card import Card, Rank, Suit
from poker.ml.observation import build_observation, observation_spec
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


@pytest.fixture
def config_3p() -> GameConfig:
    return GameConfig(
        num_players=3,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        ante=0,
        rake_percent=0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(10, 20, 0)],
            hands_per_level=10,
            fixed=True,
        ),
        run_it_twice=False,
    )


def _player(seat: int) -> PlayerState:
    return PlayerState(
        seat=seat,
        name=f"P{seat}",
        stack=1000,
        hole_cards=(Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)),
        committed_this_street=0,
        committed_this_hand=0,
        has_folded=False,
        is_all_in=False,
        is_eliminated=False,
    )


def _state(config: GameConfig, players: list[PlayerState]) -> GameState:
    return GameState(
        hand_number=0,
        street=Street.PREFLOP,
        dealer_seat=0,
        players=tuple(players),
        community_cards=(),
        pots=[Pot(100, frozenset(range(len(players))))],
        current_bet_to_call=0,
        last_raise_size=0,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=52,
        config=config,
        blind_level=config.blind_schedule.levels[0],
        action_on_seat=0,
    )


def test_observation_shape(config_3p: GameConfig) -> None:
    players = [_player(i) for i in range(3)]
    state = _state(config_3p, players)
    obs = build_observation(state, seat=0)
    assert obs.shape == (155,)
    assert obs.dtype == np.float32
    shape, dtype = observation_spec()
    assert shape == (155,)
    assert dtype == "float32"


def test_observation_normalization(config_3p: GameConfig) -> None:
    players = [_player(i) for i in range(3)]
    state = _state(config_3p, players)
    obs = build_observation(state, seat=0)
    assert (obs >= -0.1).all()
    assert (obs <= 1.1).all()


def test_observation_6_player() -> None:
    config = GameConfig(
        num_players=6,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        ante=0,
        rake_percent=0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(10, 20, 0)],
            hands_per_level=10,
            fixed=True,
        ),
        run_it_twice=False,
    )
    players = [_player(i) for i in range(6)]
    state = _state(config, players)
    obs = build_observation(state, seat=0)
    assert obs.shape == (155,)
    assert (obs >= -0.1).all()
    assert (obs <= 1.1).all()
