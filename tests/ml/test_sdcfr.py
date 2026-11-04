"""Tests for SD-CFR components."""

import numpy as np
import pytest

from poker.ml.cfr.regret_matching import regret_matching
from poker.ml.buffers import WeightedCircularBuffer, WeightedReservoirBuffer
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
    advantages = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    legal_mask = np.ones(7, dtype=np.float32)
    result = regret_matching(advantages, legal_mask)
    assert result[0] == pytest.approx(1.0, abs=1e-6)
    assert result[1:].sum() == pytest.approx(0.0, abs=1e-6)


def test_regret_matching_sums_to_one() -> None:
    """Output must sum to 1.0 and be non-negative for any inputs."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        advantages = rng.standard_normal(7).astype(np.float32)
        legal_mask = (rng.random(7) > 0.3).astype(np.float32)
        if legal_mask.sum() == 0:
            legal_mask[0] = 1.0
        result = regret_matching(advantages, legal_mask)
        assert result.shape == (7,)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)
        assert (result >= 0.0).all()
        assert (result[legal_mask == 0] == 0.0).all()


# ---------------------------------------------------------------------------
# SDCFRModel tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_sdcfr_model_get_strategy_shape() -> None:
    """get_strategy returns shape (7,), sums to 1, all non-negative."""
    model = SDCFRModel()
    obs = np.random.rand(98).astype(np.float32)
    legal_mask = np.ones(7, dtype=np.float32)
    strategy = model.get_strategy(obs, legal_mask)
    assert strategy.shape == (7,)
    assert strategy.sum() == pytest.approx(1.0, abs=1e-5)
    assert (strategy >= 0.0).all()


# ---------------------------------------------------------------------------
# ExternalSamplingTraverser tests (Fix #13)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_traversal_returns_float_reward() -> None:
    """traverse() must return a float terminal reward."""
    from poker.config.blind_schedule import BlindLevel, BlindSchedule
    from poker.config.game_config import GameConfig
    from poker.ml.cfr.traversal import ExternalSamplingTraverser
    import torch

    model = SDCFRModel()
    config = GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(25, 50, 0)], hands_per_level=100, fixed=True
        ),
    )
    traverser = ExternalSamplingTraverser(
        config=config,
        advantage_network=model.advantage_network,
        device=model.device,
        regret_buffer=model.regret_buffer,
    )
    reward = traverser.traverse(traverser_seat=0, cfr_iteration=1)
    assert isinstance(reward, float)


def test_traversal_populates_buffer() -> None:
    """After one traversal the regret buffer must contain at least one entry."""
    from poker.config.blind_schedule import BlindLevel, BlindSchedule
    from poker.config.game_config import GameConfig
    from poker.ml.cfr.traversal import ExternalSamplingTraverser
    from poker.bots.random_bot import RandomBot

    model = SDCFRModel()
    config = GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(25, 50, 0)], hands_per_level=100, fixed=True
        ),
    )
    traverser = ExternalSamplingTraverser(
        config=config,
        advantage_network=model.advantage_network,
        device=model.device,
        regret_buffer=model.regret_buffer,
    )

    # Run several traversals to ensure the traverser gets to act at least once.
    for i in range(10):
        traverser.traverse(
            traverser_seat=i % 2,
            cfr_iteration=1,
            opponent_bots=[RandomBot()],
        )

    assert len(model.regret_buffer) > 0, "Regret buffer must be non-empty after traversals"


def test_traversal_buffer_entries_are_well_formed() -> None:
    """Buffer entries must have the correct packed shape and non-trivial observations.

    We intentionally do NOT assert non-zero advantages: a terminal_reward of
    exactly zero (split pot) is valid and produces all-zero advantages, so
    testing for non-zero would flake on split-pot hands.  Instead we verify
    that (a) the obs slice is the correct dimension and has some variation
    (not a constant-zero vector, which would indicate a wiring bug), and
    (b) every weight stored is positive.
    """
    from poker.config.blind_schedule import BlindLevel, BlindSchedule
    from poker.config.game_config import GameConfig
    from poker.ml.cfr.traversal import ExternalSamplingTraverser
    from poker.bots.random_bot import RandomBot
    from poker.ml.models.sdcfr_model import _OBS_DIM, _NUM_ACTIONS

    model = SDCFRModel()
    config = GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(25, 50, 0)], hands_per_level=100, fixed=True
        ),
    )
    traverser = ExternalSamplingTraverser(
        config=config,
        advantage_network=model.advantage_network,
        device=model.device,
        regret_buffer=model.regret_buffer,
    )

    # Run enough traversals to reliably populate the buffer.
    for i in range(50):
        traverser.traverse(traverser_seat=i % 2, cfr_iteration=i + 1,
                           opponent_bots=[RandomBot(seed=i)])

    if len(model.regret_buffer) == 0:
        pytest.skip("Traverser never reached an action decision in 50 traversals")

    batch = model.regret_buffer.sample(min(32, len(model.regret_buffer)))
    packed = batch["obs"]  # (B, _OBS_DIM + _NUM_ACTIONS)

    # Shape check
    assert packed.shape[1] == _OBS_DIM + _NUM_ACTIONS, (
        f"Expected packed width {_OBS_DIM + _NUM_ACTIONS}, got {packed.shape[1]}"
    )

    # Obs slice must not be all-zeros (would mean build_observation returned garbage)
    obs_slice = packed[:, :_OBS_DIM]
    assert np.any(obs_slice != 0), "All observation values are zero — build_observation may be broken"

    # Weights must be strictly positive
    assert (batch["weights"] > 0).all(), "Some buffer weights are zero or negative"


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
    seat = state.action_on_seat
    assert seat is not None
    legal = legal_actions(state, seat)

    chosen = bot.act(state, legal)
    assert chosen.type in [a.type for a in legal], (
        f"Chosen action type {chosen.type} not in legal types {[a.type for a in legal]}"
    )


# ---------------------------------------------------------------------------
# WeightedCircularBuffer tests (Fix #9)
# ---------------------------------------------------------------------------

def test_weighted_circular_buffer_fifo() -> None:
    """WeightedCircularBuffer overwrites oldest entries when full."""
    obs_dim = 105
    buf = WeightedCircularBuffer(capacity=10, obs_dim=obs_dim)
    rng = np.random.default_rng(0)
    for i in range(15):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        buf.add(obs, 0, weight=float(i + 1))
    assert len(buf) == 10  # Capacity capped


def test_weighted_circular_buffer_weight_biased_sampling() -> None:
    """High-weight entries should be sampled more often than low-weight ones."""
    obs_dim = 105
    buf = WeightedCircularBuffer(capacity=1000, obs_dim=obs_dim)
    rng = np.random.default_rng(1)
    for i in range(1000):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        # First 500 entries: weight=1, last 500: weight=10
        weight = 1.0 if i < 500 else 10.0
        buf.add(obs, action=int(i >= 500), weight=weight)

    batch = buf.sample(2000)
    # Action 1 (high-weight entries) should dominate sampling.
    frac_high = float((batch["actions"] == 1).sum()) / 2000
    # Expected fraction: 500*10 / (500*1 + 500*10) = 5000/5500 ≈ 0.909
    assert frac_high > 0.75, f"High-weight fraction {frac_high:.3f} too low (expected >0.75)"


def test_weighted_reservoir_buffer_backward_compat() -> None:
    """WeightedReservoirBuffer still works for backward-compatibility with existing tests."""
    obs_dim = 105
    buf = WeightedReservoirBuffer(capacity=2000, obs_dim=obs_dim)
    rng = np.random.default_rng(0)
    for i in range(1000):
        obs = rng.standard_normal(obs_dim).astype(np.float32)
        buf.add(obs, 0, weight=float(rng.integers(1, 100)))
    assert len(buf) == 1000
    batch = buf.sample(64)
    assert batch["obs"].shape == (64, obs_dim)
    assert batch["weights"].shape == (64,)
    assert (batch["weights"] > 0).all()
