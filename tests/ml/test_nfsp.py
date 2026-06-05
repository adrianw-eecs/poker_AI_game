"""Tests for NFSP buffers, model, and bot."""

import numpy as np
import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.card import Card, Rank, Suit
from poker.ml.buffers import CircularBuffer, ReservoirBuffer
from poker.ml.models.nfsp_model import NFSPModel
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config() -> GameConfig:
    return GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        ante=0,
        rake_percent=0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(10, 20, 0)],
            hands_per_level=100,
            fixed=True,
        ),
        run_it_twice=False,
    )


def _make_player(seat: int) -> PlayerState:
    return PlayerState(
        seat=seat,
        name=f"P{seat}",
        stack=1000,
        hole_cards=(Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.SPADES)),
        committed_this_street=10,
        committed_this_hand=10,
        has_folded=False,
        is_all_in=False,
        is_eliminated=False,
    )


def _make_state() -> GameState:
    config = _make_config()
    players = tuple(_make_player(i) for i in range(2))
    return GameState(
        hand_number=0,
        street=Street.PREFLOP,
        dealer_seat=0,
        players=players,
        community_cards=(),
        pots=[Pot(20, frozenset({0, 1}))],
        current_bet_to_call=20,
        last_raise_size=20,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=48,
        config=config,
        blind_level=config.blind_schedule.levels[0],
        action_on_seat=0,
    )


# ---------------------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------------------

def test_circular_buffer_add_sample() -> None:
    buf = CircularBuffer(capacity=200, obs_dim=155)
    obs = np.random.rand(155).astype(np.float32)
    next_obs = np.random.rand(155).astype(np.float32)
    for i in range(100):
        buf.add(obs, action=i % 7, reward=float(i), next_obs=next_obs, done=False)

    assert len(buf) == 100
    batch = buf.sample(32)
    assert batch["obs"].shape == (32, 155)
    assert batch["next_obs"].shape == (32, 155)
    assert batch["actions"].shape == (32,)
    assert batch["rewards"].shape == (32,)
    assert batch["dones"].shape == (32,)


def test_reservoir_buffer_uniform() -> None:
    buf = ReservoirBuffer(capacity=5000, obs_dim=155)
    obs = np.zeros(155, dtype=np.float32)
    for i in range(10_000):
        buf.add(obs, action=i % 7)

    assert len(buf) == 5000
    batch = buf.sample(2000)
    counts = np.bincount(batch["actions"], minlength=7)
    # Each action should appear roughly 2000/7 ≈ 286 times; allow ±50 %
    expected = 2000 / 7
    for c in counts:
        assert c > expected * 0.5, f"Action count {c} too low (expected ~{expected:.0f})"
        assert c < expected * 1.5, f"Action count {c} too high (expected ~{expected:.0f})"


# ---------------------------------------------------------------------------
# NFSPModel tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_nfsp_model_select_action_shapes() -> None:
    model = NFSPModel()
    obs = np.random.rand(155).astype(np.float32)
    mask = np.ones(7, dtype=np.int32)
    action = model.select_action(obs, mask, training=False)
    assert isinstance(action, int)
    assert 0 <= action <= 6


@pytest.mark.smoke
def test_nfsp_model_train_step_runs() -> None:
    model = NFSPModel(batch_size=64, train_every=1)
    obs = np.random.rand(155).astype(np.float32)
    next_obs = np.random.rand(155).astype(np.float32)
    mask = np.ones(7, dtype=np.int32)

    # Fill both buffers past the minimum threshold (512)
    for i in range(600):
        model.store_transition(obs, action=i % 7, reward=0.0, next_obs=next_obs, done=False)
        model.policy_buffer.add(obs, action=i % 7)

    result = model.train_step()
    assert "q_loss" in result
    assert "policy_loss" in result
    assert result["q_loss"] is not None
    assert result["policy_loss"] is not None


# ---------------------------------------------------------------------------
# NFSPBot tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_nfsp_bot_act_returns_legal_action() -> None:
    from poker.bots.nfsp_bot import NFSPBot
    from poker.engine.action_validator import legal_actions

    bot = NFSPBot(name="TestNFSP", training=False)
    state = _make_state()
    legal = legal_actions(state, seat=0)
    action = bot.act(state, legal)
    assert action in legal


# ---------------------------------------------------------------------------
# Save / load test
# ---------------------------------------------------------------------------

def test_nfsp_model_save_load(tmp_path) -> None:
    model = NFSPModel()
    obs = np.random.rand(155).astype(np.float32)
    mask = np.ones(7, dtype=np.int32)

    save_file = str(tmp_path / "nfsp_test.pt")
    model.save(save_file)

    model2 = NFSPModel()
    model2.load(save_file)
    action = model2.select_action(obs, mask, training=False)
    assert isinstance(action, int)
    assert 0 <= action <= 6
