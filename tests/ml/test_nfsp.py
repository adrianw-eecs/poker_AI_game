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
    buf = CircularBuffer(capacity=200, obs_dim=98)
    rng = np.random.default_rng(0)
    for i in range(100):
        obs = rng.standard_normal(98).astype(np.float32)
        next_obs = rng.standard_normal(98).astype(np.float32)
        next_mask = np.ones(7, dtype=np.float32)
        buf.add(obs, action=i % 7, reward=float(i), next_obs=next_obs, done=False,
                next_mask=next_mask)

    assert len(buf) == 100
    batch = buf.sample(32)
    assert batch["obs"].shape == (32, 98)
    assert batch["next_obs"].shape == (32, 98)
    assert batch["next_masks"].shape == (32, 7)
    assert batch["actions"].shape == (32,)
    assert batch["rewards"].shape == (32,)
    assert batch["dones"].shape == (32,)


def test_circular_buffer_resume_does_not_sample_zeros() -> None:
    """A capped checkpoint used to keep the old circular cursor."""
    buf = CircularBuffer(capacity=8, obs_dim=2)
    state = {
        "obs": np.ones((3, 2), dtype=np.float32),
        "next_obs": np.ones((3, 2), dtype=np.float32),
        "next_masks": np.ones((3, 7), dtype=np.float32),
        "actions": np.ones(3, dtype=np.int64),
        "rewards": np.ones(3, dtype=np.float32),
        "dones": np.zeros(3, dtype=np.float32),
        "pos": 999,
        "size": 3,
    }
    buf.load_state_dict(state)
    fresh = np.full(2, 5.0, dtype=np.float32)
    buf.add(fresh, action=2, reward=1.0, next_obs=np.zeros(2, dtype=np.float32), done=False)
    assert buf._obs[3, 0] == 5.0
    assert len(buf) == 4
    batch = buf.sample(20)
    assert np.all(np.any(batch["obs"] != 0.0, axis=1))


def test_reservoir_subsample_does_not_slide() -> None:
    """Restored reservoir must not fill empty capacity with only new rows."""
    buf = ReservoirBuffer(capacity=100, obs_dim=4)
    for i in range(50):
        buf.add(np.full(4, float(i), dtype=np.float32), action=i % 7)
    buf._total_seen = 10_000
    buf._size = 50
    for _ in range(200):
        buf.add(np.ones(4, dtype=np.float32), action=1)
    assert len(buf) == 50


def test_reservoir_buffer_uniform() -> None:
    """Reservoir sampling must maintain a uniform distribution over action labels."""
    buf = ReservoirBuffer(capacity=5000, obs_dim=98)
    rng = np.random.default_rng(42)
    for i in range(10_000):
        # Use distinct observations to test realistic reservoir behavior (Fix #25).
        obs = rng.standard_normal(98).astype(np.float32)
        buf.add(obs, action=i % 7)

    assert len(buf) == 5000
    batch = buf.sample(2000)
    counts = np.bincount(batch["actions"], minlength=7)
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
    obs = np.random.rand(98).astype(np.float32)
    mask = np.ones(7, dtype=np.int32)
    action = model.select_action(obs, mask, training=False)
    assert isinstance(action, int)
    assert 0 <= action <= 6


@pytest.mark.smoke
def test_nfsp_model_train_step_runs() -> None:
    """Train step completes and returns non-None losses."""
    # Small q_capacity so _q_min = max(64, 6400//100) = 64 (reachable in this test).
    model = NFSPModel(batch_size=64, train_every=1, q_capacity=6400, policy_capacity=6400)
    rng = np.random.default_rng(1)
    # Use distinct observations so the model trains on a realistic distribution (Fix #25).
    for i in range(600):
        obs = rng.standard_normal(98).astype(np.float32)
        next_obs = rng.standard_normal(98).astype(np.float32)
        next_mask = np.ones(7, dtype=np.float32)
        model.store_transition(obs, action=i % 7, reward=0.0,
                               next_obs=next_obs, done=False, next_mask=next_mask)
        model.policy_buffer.add(obs, action=i % 7)

    result = model.train_step()
    assert result["q_loss"] is not None
    assert result["policy_loss"] is not None


def test_q_network_learns_toward_reward() -> None:
    """Q-network Q-values for a rewarded (obs, action) pair must increase
    after multiple training steps — verifies the TD update is working (Fix #14)."""
    model = NFSPModel(batch_size=32, train_every=1, gamma=0.0,
                      q_capacity=3200, policy_capacity=3200)
    rng = np.random.default_rng(7)

    target_action = 2
    target_obs = rng.standard_normal(98).astype(np.float32)
    terminal_obs = np.zeros(98, dtype=np.float32)
    terminal_mask = np.ones(7, dtype=np.float32)

    # Seed the buffer with 200 transitions: action 2 on target_obs -> reward +5.0
    for _ in range(200):
        noisy = (target_obs + rng.standard_normal(98).astype(np.float32) * 0.05)
        model.store_transition(
            noisy.astype(np.float32), target_action, 5.0, terminal_obs, True,
            next_mask=terminal_mask,
        )
        model.policy_buffer.add(noisy.astype(np.float32), action=target_action)

    # Read initial Q-value for the target action.
    import torch
    obs_t = torch.as_tensor(target_obs, dtype=torch.float32, device=model.device).unsqueeze(0)
    model.q_network.eval()
    with torch.no_grad():
        q_before = model.q_network(obs_t).squeeze(0)[target_action].item()

    # Run 50 training steps.
    for _ in range(50):
        model.train_step()

    with torch.no_grad():
        q_after = model.q_network(obs_t).squeeze(0)[target_action].item()

    assert q_after > q_before, (
        f"Q-value for rewarded action did not increase: before={q_before:.4f}, after={q_after:.4f}"
    )


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

def test_anticipatory_mode_is_sticky_for_the_episode() -> None:
    """Mode is sampled once per episode, not once per action."""
    model = NFSPModel(eta=1.0, epsilon_start=0.0, epsilon_end=0.0)
    model.begin_episode()
    model.eta = 0.0  # a per-action draw would now stay in average-policy mode
    obs = np.zeros(98, dtype=np.float32)
    mask = np.ones(7, dtype=np.int32)
    for _ in range(5):
        model.select_action(obs, mask, training=True)
    assert len(model.policy_buffer) == 5

    model.begin_episode()
    before = len(model.policy_buffer)
    for _ in range(5):
        model.select_action(obs, mask, training=True)
    assert len(model.policy_buffer) == before


def test_policy_imitates_masked_best_response() -> None:
    """Average policy matches reservoir labels over the legal actions only."""
    import torch

    model = NFSPModel(batch_size=32, train_every=1, q_capacity=3200, policy_capacity=3200)
    rng = np.random.default_rng(3)
    obs = rng.standard_normal(98).astype(np.float32)
    legal = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(200):
        noisy = (obs + rng.standard_normal(98).astype(np.float32) * 0.02).astype(np.float32)
        model.policy_buffer.add(noisy, action=1, legal_mask=legal)

    for _ in range(40):
        result = model.train_step()
    assert result["policy_loss"] is not None

    model.policy_network.eval()
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
    with torch.no_grad():
        logits = model.policy_network(obs_t).squeeze(0)
        logits = logits.masked_fill(
            ~torch.as_tensor(legal, dtype=torch.bool, device=model.device),
            float("-inf"),
        )
        probs = torch.softmax(logits, dim=0)
    assert float(probs[1]) > float(probs[0])


def test_nfsp_model_save_load(tmp_path) -> None:
    model = NFSPModel()
    obs = np.random.rand(98).astype(np.float32)
    mask = np.ones(7, dtype=np.int32)

    save_file = str(tmp_path / "nfsp_test.pt")
    model.save(save_file)

    model2 = NFSPModel()
    model2.load(save_file)
    action = model2.select_action(obs, mask, training=False)
    assert isinstance(action, int)
    assert 0 <= action <= 6
    assert model2.obs_version == 1
    assert model2.gamma == pytest.approx(0.99)


def _small_model(**kwargs) -> NFSPModel:
    defaults = {"q_capacity": 400, "policy_capacity": 400, "batch_size": 16}
    defaults.update(kwargs)
    return NFSPModel(**defaults)


def test_float16_buffer_samples_float32() -> None:
    buf = CircularBuffer(capacity=4, obs_dim=3, dtype=np.float16)
    buf.add(
        np.array([1, 2, 3], dtype=np.float32),
        action=1,
        reward=0.5,
        next_obs=np.array([4, 5, 6], dtype=np.float32),
        done=False,
    )
    assert buf._obs.dtype == np.float16
    batch = buf.sample(2)
    assert batch["obs"].dtype == np.float32
    assert batch["obs"][0, 0] == 1.0


def test_per_seat_anticipatory_mode_is_independent() -> None:
    model = _small_model(eta=1.0, epsilon_start=0.0, epsilon_end=0.0)
    model.begin_episode(0)
    model.eta = 0.0
    model.begin_episode(1)
    obs = np.zeros(98, dtype=np.float32)
    mask = np.ones(7, dtype=np.int32)
    model.select_action(obs, mask, training=True, seat=0)
    assert len(model.policy_buffer) == 1
    before = len(model.policy_buffer)
    model.select_action(obs, mask, training=True, seat=1)
    assert len(model.policy_buffer) == before


def test_inv_sqrt_epsilon_decays_toward_the_floor() -> None:
    model = _small_model(
        epsilon_start=0.06,
        epsilon_end=0.01,
        epsilon_decay_steps=100,
        epsilon_schedule="inv_sqrt",
    )
    assert model._current_epsilon() == pytest.approx(0.06)
    model._step = 100
    mid = model._current_epsilon()
    model._step = 10_000
    late = model._current_epsilon()
    assert 0.01 < late < mid < 0.06


def test_two_updates_per_train_event() -> None:
    model = _small_model(train_every=1, updates_per_train=2, batch_size=8, q_capacity=80, policy_capacity=80)
    rng = np.random.default_rng(0)
    for i in range(20):
        obs = rng.standard_normal(98).astype(np.float32)
        model.store_transition(obs, i % 7, 0.0, obs, False, next_mask=np.ones(7, dtype=np.float32))
    model.train_step()
    assert model._q_updates == 2
