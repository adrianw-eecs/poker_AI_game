"""Tests for SD-CFR components."""

import numpy as np
import pytest

from poker.ml.cfr.regret_matching import regret_matching
from poker.ml.buffers import WeightedReservoirBuffer
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.bots.sdcfr_bot import SDCFRBot
from poker.ml.env import PokerEnv
from poker.engine.action_validator import legal_actions


# ---------------------------------------------------------------------------
# regret_matching tests
# ---------------------------------------------------------------------------


def test_regret_matching_all_negative() -> None:
    """All-negative advantages should yield uniform over legal actions."""
    advantages = np.array([-1.0, -2.0, -0.5, -3.0, -1.0, -4.0, -0.1], dtype=np.float32)
    legal_mask = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
    result = regret_matching(advantages, legal_mask)
    num_legal = int(legal_mask.sum())
    expected = legal_mask / num_legal
    np.testing.assert_allclose(result, expected, atol=1e-6)
    assert abs(result.sum() - 1.0) < 1e-6


def test_regret_matching_mixed() -> None:
    """Only positive advantage actions should get probability."""
    # Action 0 has advantage 1.0, rest are zero or negative.
    advantages = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    legal_mask = np.ones(7, dtype=np.float32)
    result = regret_matching(advantages, legal_mask)
    # After positive-regret clipping, only action 0 has regret > 0.
    assert result[0] == pytest.approx(1.0, abs=1e-6)
    assert result[1:].sum() == pytest.approx(0.0, abs=1e-6)


def test_regret_matching_sums_to_one() -> None:
    """Output must sum to 1.0 and be non-negative for any inputs."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        advantages = rng.standard_normal(7).astype(np.float32)
        # Random legal mask with at least 1 legal action
        legal_mask = (rng.random(7) > 0.3).astype(np.float32)
        if legal_mask.sum() == 0:
            legal_mask[0] = 1.0
        result = regret_matching(advantages, legal_mask)
        assert result.shape == (7,)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)
        assert (result >= 0.0).all()
        # Illegal actions must have zero probability
        assert (result[legal_mask == 0] == 0.0).all()


# ---------------------------------------------------------------------------
# SDCFRModel tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sdcfr_model_get_strategy_shape() -> None:
    """get_strategy returns shape (7,), sums to 1, all non-negative."""
    model = SDCFRModel()
    obs = np.random.rand(142).astype(np.float32)
    legal_mask = np.ones(7, dtype=np.float32)
    strategy = model.get_strategy(obs, legal_mask)
    assert strategy.shape == (7,)
    assert strategy.sum() == pytest.approx(1.0, abs=1e-5)
    assert (strategy >= 0.0).all()


# ---------------------------------------------------------------------------
# SDCFRBot tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_sdcfr_bot_returns_legal_action() -> None:
    """SDCFRBot.act() must return an action present in the legal list."""
    env = PokerEnv(num_players=2, learning_seat=0)
    obs, info = env.reset()
    state = env.state
    assert state is not None

    bot = SDCFRBot()

    # The action is on the learning seat (seat 0) after reset.
    seat = state.action_on_seat
    assert seat is not None
    legal = legal_actions(state, seat)

    chosen = bot.act(state, legal)
    legal_types_amounts = [(a.type, a.amount) for a in legal]
    # The chosen action type must be one of the legal action types.
    assert chosen.type in [a.type for a in legal], (
        f"Chosen action type {chosen.type} not in legal types {[a.type for a in legal]}"
    )


# ---------------------------------------------------------------------------
# WeightedReservoirBuffer tests
# ---------------------------------------------------------------------------


def test_weighted_reservoir_buffer() -> None:
    """Sampling from a populated buffer returns correct shapes."""
    obs_dim = 149  # 142 obs + 7 advantages (as used by SDCFRModel)
    buf = WeightedReservoirBuffer(capacity=2000, obs_dim=obs_dim)

    rng = np.random.default_rng(0)
    for i in range(1000):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        weight = float(rng.integers(1, 100))
        buf.add(obs, 0, weight)

    assert len(buf) == 1000

    batch = buf.sample(64)
    assert batch["obs"].shape == (64, obs_dim)
    assert batch["weights"].shape == (64,)
    assert (batch["weights"] > 0).all()
