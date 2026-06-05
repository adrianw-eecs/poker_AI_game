"""Tests for PokerEnv."""

import numpy as np
import pytest

from poker.ml.env import PokerEnv


@pytest.mark.smoke
def test_env_reset_returns_obs() -> None:
    env = PokerEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (155,)
    assert isinstance(info, dict)


@pytest.mark.smoke
def test_env_action_space_is_discrete() -> None:
    env = PokerEnv()
    assert env.action_space.n == 7


@pytest.mark.smoke
def test_env_observation_space_shape() -> None:
    env = PokerEnv()
    assert env.observation_space.shape == (155,)
    assert env.observation_space.dtype == np.float32


def test_env_default_init() -> None:
    env = PokerEnv()
    assert env.num_players == 2
    assert env.starting_stack == 1000
    assert env.big_blind == 20


def test_env_custom_init() -> None:
    env = PokerEnv(num_players=6, starting_stack=5000, small_blind=25, big_blind=50)
    assert env.num_players == 6
    assert env.starting_stack == 5000


@pytest.mark.parametrize("kwargs,exc", [
    ({"num_players": 1}, ValueError),
    ({"num_players": 11}, ValueError),
    ({"starting_stack": 0}, ValueError),
])
def test_env_invalid_params(kwargs, exc) -> None:
    with pytest.raises(exc):
        PokerEnv(**kwargs)
