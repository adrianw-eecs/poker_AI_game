"""Tests for the seven architectural improvements to NFSP."""

import numpy as np
import pytest
import torch

from poker.bots.random_bot import RandomBot
from poker.domain.action import ActionType
from poker.ml.action_space import BetScheme
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.nfsp_networks_v3 import (
    PokerNetV3,
    SeatAttention,
    build_hybrid_network_v3,
    build_q_network_v3,
)
from poker.ml.observation_v2 import OBS_V2_DIM, build_observation_v2
from poker.ml.observation_v3 import (
    OBS_V3_DIM,
    build_observation_v3,
    neutral_tendencies,
    seat_occupancy,
)
from poker.ml.tendencies import TENDENCY_DIM, TendencyTracker

MASK = np.ones(7, dtype=np.float32)


def _obs(width=OBS_V3_DIM, seed=0):
    rng = np.random.default_rng(seed)
    o = rng.random(width).astype(np.float32)
    o[:7] = rng.integers(0, 52, 7)
    return o


def _small(**kw):
    base = dict(obs_version=2, batch_size=16, q_capacity=200, policy_capacity=200)
    base.update(kw)
    return NFSPModel(**base)


# ---------------------------------------------- 1. reservoir weighting


@pytest.mark.smoke
def test_reservoir_min_prob_defaults_to_exponential_averaging() -> None:
    """Item 1: uniform reservoir drags the policy backward; floor it."""
    model = NFSPModel(obs_version=2, batch_size=8, q_capacity=64, policy_capacity=64,
                      value_distribution="none", n_step=1)
    assert model.policy_buffer.min_prob == pytest.approx(0.25)


def test_reservoir_min_prob_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        NFSPModel(obs_version=2, reservoir_min_prob=1.0)


def test_exponential_reservoir_biases_toward_recent_rows() -> None:
    """With min_prob>0 the retained sample should skew to later insertions."""
    from poker.ml.buffers import ReservoirBuffer

    def tail_fraction(min_prob: float) -> float:
        buf = ReservoirBuffer(capacity=200, obs_dim=2, min_prob=min_prob)
        for i in range(4000):
            buf.add(np.full(2, float(i), dtype=np.float32), action=i % 7,
                    legal_mask=MASK)
        sample = buf.sample(150)
        # Rows inserted late carry large values; compare against the midpoint.
        return float(np.mean(sample["obs"][:, 0] > 2000))

    uniform = tail_fraction(0.0)
    biased = tail_fraction(0.25)
    assert biased > uniform, (
        f"min_prob did not bias toward recent rows: uniform={uniform:.2f} "
        f"biased={biased:.2f}"
    )


# ---------------------------------------------- 3. tendencies


@pytest.mark.smoke
def test_tendency_tracker_separates_aggressive_and_tight_players() -> None:
    t = TendencyTracker(num_players=6)
    for _ in range(20):
        t.start_hand()
        t.record_action(0, ActionType.RAISE, street_is_preflop=True, facing_bet=True)
        t.record_action(2, ActionType.FOLD, street_is_preflop=True, facing_bet=True)
        t.end_hand()
    f = t.features()
    assert f.shape == (TENDENCY_DIM,)
    assert f[1] > 0.7, "aggressive seat PFR should be high"
    assert f[11] > 0.7, "folding seat fold_to_bet should be high"
    assert (f >= 0).all() and (f <= 1).all()


def test_tendency_prior_is_neutral_not_zero() -> None:
    """An unseen seat must not read as 'never bets, never folds'."""
    f = neutral_tendencies()
    assert f.shape == (TENDENCY_DIM,)
    assert 0.2 < f[0] < 0.5, f"VPIP prior should be near 1/3, got {f[0]}"


def test_env_feeds_tendencies_for_obs_v3() -> None:
    env = PokerEnv(num_players=4, learning_seat=0,
                   opponent_bots=[RandomBot(seed=i) for i in range(3)],
                   small_blind=25, big_blind=50, seed=1,
                   obs_version=3, bet_scheme=BetScheme.POT_FRACTION)
    assert env.tendencies is not None
    for h in range(6):
        env.hand_number = h
        obs, _ = env.reset(options={"deck_seed": 40 + h})
        assert obs.shape[0] == OBS_V3_DIM
        steps = 0
        while env.state is not None and steps < 30:
            if env.state.action_on_seat != 0:
                break
            _, _, done, _, _ = env.step(1)
            steps += 1
            if done:
                break
    assert env.tendencies.hands_seen(0) >= 4


# ---------------------------------------------- 4. n-step


@pytest.mark.smoke
def test_n_step_returns_accumulate_discounted_reward() -> None:
    """With gamma=0.5 and unit rewards, a 3-step window gives 1 + .5 + .25."""
    model = _small(gamma=0.5, n_step=3, value_distribution="none")
    for i in range(3):
        model.store_transition(_obs(OBS_V2_DIM, i), 1, 1.0, _obs(OBS_V2_DIM, i + 1),
                               i == 2, next_mask=MASK, current_mask=MASK)
    model.flush_pending()
    rewards = sorted(set(np.round(model.q_buffer.sample(50)["rewards"], 4)))
    assert any(abs(r - 1.75) < 1e-3 for r in rewards), rewards


def test_n_step_one_preserves_raw_rewards() -> None:
    model = _small(n_step=1, value_distribution="none")
    model.store_transition(_obs(OBS_V2_DIM, 0), 1, 7.0, _obs(OBS_V2_DIM, 1), False,
                           next_mask=MASK, current_mask=MASK)
    assert abs(float(model.q_buffer.sample(4)["rewards"][0]) - 7.0) < 1e-5


def test_end_hand_flushes_pending_window() -> None:
    model = _small(n_step=5, value_distribution="none")
    for i in range(3):
        model.store_transition(_obs(OBS_V2_DIM, i), 1, 1.0, _obs(OBS_V2_DIM, i + 1),
                               False, next_mask=MASK, current_mask=MASK)
    assert len(model.q_buffer) == 0, "window should still be buffered"
    model.end_hand()
    assert len(model.q_buffer) == 3, "end_hand should flush the window"


# ---------------------------------------------- 6. C51


@pytest.mark.smoke
def test_c51_predicts_a_distribution_summing_to_one() -> None:
    model = _small(value_distribution="c51", num_atoms=51)
    assert hasattr(model.q_network, "dist_head")
    assert model.q_network.dist_head.out_features == 7 * 51

    x = torch.as_tensor(_obs(OBS_V2_DIM, 1), device=model.device).unsqueeze(0)
    probs = model._distributional_forward(model.q_network, x)
    assert probs.shape == (1, 7, 51)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1, 7, device=model.device), atol=1e-4)


def test_c51_forward_returns_scalar_values() -> None:
    """Existing call sites read one value per action; C51 must not break them."""
    model = _small(value_distribution="c51")
    x = torch.as_tensor(_obs(OBS_V2_DIM, 1), device=model.device).unsqueeze(0)
    out = model.q_network(x)
    assert out.shape == (1, 7)
    assert torch.isfinite(out).all()


def test_c51_trains_without_error() -> None:
    model = _small(value_distribution="c51", train_every=1, n_step=1)
    for i in range(80):
        model.store_transition(_obs(OBS_V2_DIM, i), 1, 1.0, _obs(OBS_V2_DIM, i + 1),
                               i % 10 == 9, next_mask=MASK, current_mask=MASK)
        model.policy_buffer.add(_obs(OBS_V2_DIM, i), 1, legal_mask=MASK)
    model.flush_pending()
    results = [model.train_step() for _ in range(4)]
    got = [r for r in results if r["q_loss"] is not None]
    assert got, "C51 produced no Q loss"
    assert np.isfinite(got[-1]["q_loss"])


def test_c51_rejects_bad_support() -> None:
    with pytest.raises(ValueError):
        NFSPModel(obs_version=2, value_distribution="c51", vmin=1.0, vmax=1.0)


# ---------------------------------------------- 7. attention / v3 net


@pytest.mark.smoke
def test_v3_network_shapes_and_param_count() -> None:
    net = build_q_network_v3()
    assert isinstance(net, PokerNetV3)
    x = torch.zeros(2, OBS_V3_DIM)
    x[:, 0] = 12
    x[:, 1] = 25
    x[:, 2:7] = 52
    x[:, 43:58] = 1.0
    out = net(x)
    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()
    params = sum(p.numel() for p in net.parameters())
    assert 600_000 < params < 1_200_000


def test_v3_occupancy_masks_padding() -> None:
    net = build_q_network_v3()
    x = torch.zeros(1, OBS_V3_DIM)
    x[:, 0] = 12
    x[:, 1] = 25
    x[:, 2:7] = 52
    x[:, 43:58] = 1.0   # hero
    x[:, 58:73] = 1.0   # one opponent
    occ = net._occupancy(x)
    assert occ.tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]


def test_attention_is_indifferent_to_player_count() -> None:
    """The same occupied seats must give the same output regardless of padding."""
    attn = SeatAttention()
    seats = torch.rand(1, 9, 15)
    two = torch.tensor([[1.0, 1.0] + [0.0] * 7])
    nine = torch.tensor([[1.0] * 9])
    out_two = attn(seats, two)
    # With 9 occupied the pool changes, but the shape must not.
    out_nine = attn(seats, nine)
    assert out_two.shape == out_nine.shape
    assert out_two.shape == (1, attn.out_dim)


def test_v3_observation_keeps_v2_prefix() -> None:
    env = PokerEnv(num_players=6, learning_seat=0,
                   opponent_bots=[RandomBot(seed=i) for i in range(5)],
                   small_blind=25, big_blind=50, seed=1,
                   obs_version=2, bet_scheme=BetScheme.POT_FRACTION)
    env.reset(options={"deck_seed": 3})
    v2 = build_observation_v2(env.state, 0)
    v3 = build_observation_v3(env.state, 0)
    assert v3.shape == (OBS_V3_DIM,)
    assert np.allclose(v3[:OBS_V2_DIM], v2), "v3 must extend v2, not change it"


def test_seat_occupancy_helper() -> None:
    env = PokerEnv(num_players=3, learning_seat=0,
                   opponent_bots=[RandomBot(seed=i) for i in range(2)],
                   small_blind=25, big_blind=50, seed=1,
                   obs_version=2, bet_scheme=BetScheme.POT_FRACTION)
    env.reset(options={"deck_seed": 9})
    occ = seat_occupancy(env.state, 0)
    assert occ.sum() == 3


def test_v3_hybrid_network_has_both_heads() -> None:
    net = build_hybrid_network_v3()
    x = torch.zeros(2, OBS_V3_DIM)
    x[:, 0] = 12
    x[:, 1] = 25
    x[:, 2:7] = 52
    x[:, 43:58] = 1.0
    q, logits = net.forward_both(x)
    assert q.shape == (2, 7) and logits.shape == (2, 7)


def test_v3_rejects_wrong_width() -> None:
    with pytest.raises(ValueError):
        build_q_network_v3(input_dim=OBS_V2_DIM)
