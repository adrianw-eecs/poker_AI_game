"""Observation v5: big-blind money, a fixed fold-target index, and the tail."""

import numpy as np

from poker.bots.call_bot import CallBot
from poker.ml.action_space import BetScheme
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.observation_v2 import (
    FOLD_COMMITTED_INDEX,
    OBS_SPEC_V2,
    build_observation_v2,
)
from poker.ml.observation_v4 import OBS_V4_DIM, build_observation_v4
from poker.ml.observation_v5 import OBS_V5_DIM, TOURNAMENT_TAIL_DIM, build_observation_v5


def _state(stacks: tuple[int, ...] = (1000, 1000)):
    env = PokerEnv(
        num_players=len(stacks),
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        learning_seat=0,
        opponent_bots=[CallBot() for _ in range(len(stacks) - 1)],
        seed=1,
        obs_version=5,
    )
    env.reset(options={"stacks": stacks, "deck_seed": 3})
    assert env.state is not None and env.state.action_on_seat is not None
    return env


def test_fold_committed_index_is_the_hero_commitment_slot() -> None:
    assert FOLD_COMMITTED_INDEX == OBS_SPEC_V2.seats[0] + 5
    env = _state()
    seat = int(env.state.action_on_seat)
    obs = build_observation_v2(env.state, seat)
    player = env.state.players[seat]
    expected = player.committed_this_hand / env.config.starting_stack
    assert obs[FOLD_COMMITTED_INDEX] == np.float32(expected) or abs(
        float(obs[FOLD_COMMITTED_INDEX]) - expected
    ) < 1e-6


def test_buyin_money_unit_matches_the_default_vector() -> None:
    env = _state()
    seat = int(env.state.action_on_seat)
    default = build_observation_v2(env.state, seat)
    named = build_observation_v2(env.state, seat, "buyin")
    assert np.array_equal(default, named)
    v4_default = build_observation_v4(env.state, seat)
    v4_named = build_observation_v4(env.state, seat, money_unit="buyin")
    assert np.array_equal(v4_default, v4_named)


def test_bb_money_uses_the_hand_stack_not_the_buyin() -> None:
    env = _state((200, 800))
    seat = int(env.state.action_on_seat)
    player = env.state.players[seat]
    buyin = build_observation_v2(env.state, seat, "buyin")
    blinds = build_observation_v2(env.state, seat, "bb")
    stack_at_start = player.stack + player.committed_this_hand
    assert stack_at_start != env.config.starting_stack
    assert abs(float(blinds[FOLD_COMMITTED_INDEX]) - player.committed_this_hand / stack_at_start) < 1e-5
    assert abs(float(buyin[FOLD_COMMITTED_INDEX]) - player.committed_this_hand / 1000) < 1e-5
    assert float(blinds[FOLD_COMMITTED_INDEX]) != float(buyin[FOLD_COMMITTED_INDEX])


def test_v5_appends_a_bounded_tournament_tail() -> None:
    env = _state((500, 500))
    seat = int(env.state.action_on_seat)
    obs = build_observation_v5(env.state, seat, hand_cap=160)
    assert obs.shape == (OBS_V5_DIM,)
    assert OBS_V5_DIM == OBS_V4_DIM + TOURNAMENT_TAIL_DIM
    prefix = build_observation_v4(env.state, seat, money_unit="bb")
    assert np.array_equal(obs[:OBS_V4_DIM], prefix)
    tail = obs[OBS_V4_DIM:]
    assert tail.shape == (TOURNAMENT_TAIL_DIM,)
    assert float(tail.min()) >= 0.0
    assert float(tail.max()) <= 1.0
    # places_paid is 1/field, payout_share is 1 under winner-take-all.
    assert abs(float(tail[14]) - 0.5) < 1e-6
    assert abs(float(tail[15]) - 1.0) < 1e-6
    # Hand 0 of a 160-hand cap has not started the clock.
    assert float(tail[23]) == 0.0


def test_env_v5_observation_matches_the_builder() -> None:
    env = _state()
    seat = int(env.state.action_on_seat)
    assert env.observation_space.shape == (OBS_V5_DIM,)
    assert env.observation_space.high[0] == np.float32(52.0)
    seen = env._observe(seat)
    assert np.array_equal(seen, build_observation_v5(env.state, seat, env.tendencies, env.hand_cap))


def test_v5_network_accepts_the_wider_vector() -> None:
    model = NFSPModel(
        obs_version=5,
        bet_scheme=BetScheme.POT_FRACTION,
        batch_size=8,
        train_every=1,
        q_capacity=32,
        policy_capacity=32,
    )
    assert int(model.q_network.input_dim) == OBS_V5_DIM
    assert model.q_network.equity_head is not None
    env = _state()
    seat = int(env.state.action_on_seat)
    obs = build_observation_v5(env.state, seat)
    action = model.select_action(obs, np.ones(7, dtype=np.int32), training=False)
    assert 0 <= action <= 6
