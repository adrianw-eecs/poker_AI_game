"""NFSP v2 network, checkpoint metadata, and multi-seat learning."""

import numpy as np
import pytest
import torch

from poker.bots.nfsp_bot import NFSPBot
from poker.engine.action_validator import legal_actions, validate
from poker.ml.action_space import BetScheme
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.nfsp_networks import (
    build_policy_network_v2,
    build_q_network_v2,
    network_kwargs_from_state_dict,
)
from poker.ml.nfsp_loop import consume_hand
from poker.ml.observation_v2 import OBS_V2_DIM


def _small_v2(**kwargs) -> NFSPModel:
    defaults = {
        "obs_version": 2,
        "q_capacity": 800,
        "policy_capacity": 800,
        "batch_size": 32,
        "train_every": 1,
    }
    defaults.update(kwargs)
    return NFSPModel(**defaults)


def _stable_obs(rng: np.random.Generator, noise: float = 0.0) -> np.ndarray:
    obs = np.full(OBS_V2_DIM, 0.25, dtype=np.float32)
    obs[:7] = np.array([12, 25, 3, 4, 5, 52, 52], dtype=np.float32)
    if noise:
        obs[7:] += rng.standard_normal(OBS_V2_DIM - 7).astype(np.float32) * noise
        np.clip(obs[7:], 0.0, 1.0, out=obs[7:])
    return obs


def test_pokernet_v2_forward_and_round_trip() -> None:
    q_net = build_q_network_v2()
    policy = build_policy_network_v2()
    assert torch.count_nonzero(policy.policy_head.weight) == 0
    params = sum(p.numel() for p in q_net.parameters())
    assert 200_000 < params < 1_200_000

    x = torch.zeros(2, OBS_V2_DIM)
    x[:, 0] = 12
    x[:, 1] = 25
    x[:, 2:] = 0.2
    x[:, 2:7] = 52
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    out = q_net(x, legal_mask=mask)
    assert out.shape == (2, 7)
    assert torch.isfinite(out).all()
    assert policy(x).shape == (2, 7)

    kwargs = network_kwargs_from_state_dict(q_net.state_dict())
    assert kwargs["arch"] == "v2"
    assert kwargs["input_dim"] == OBS_V2_DIM
    kwargs.pop("arch")
    clone = build_q_network_v2(**kwargs)
    clone.load_state_dict(q_net.state_dict())


def test_v2_refuses_wrong_observation_width() -> None:
    model = _small_v2()
    with pytest.raises(ValueError):
        model.select_action(np.zeros(98, dtype=np.float32), np.ones(7, dtype=np.int32), training=False)


def test_v2_q_learns_toward_reward() -> None:
    model = _small_v2(gamma=0.0, q_capacity=400, policy_capacity=400, batch_size=32)
    rng = np.random.default_rng(7)
    target = _stable_obs(rng)
    terminal = np.zeros(OBS_V2_DIM, dtype=np.float32)
    terminal_mask = np.ones(7, dtype=np.float32)
    for _ in range(240):
        noisy = _stable_obs(rng, noise=0.02)
        model.store_transition(noisy, 2, 5.0, terminal, True, next_mask=terminal_mask)
        model.policy_buffer.add(noisy, action=2)

    obs_t = torch.as_tensor(target, dtype=torch.float32, device=model.device).unsqueeze(0)
    model.q_network.eval()
    with torch.no_grad():
        before = model.q_network(obs_t).squeeze(0)[2].item()
    for _ in range(60):
        model.train_step()
    model.q_network.eval()
    with torch.no_grad():
        after = model.q_network(obs_t).squeeze(0)[2].item()
    assert after > before


def test_v2_policy_imitates_masked_best_response() -> None:
    model = _small_v2(q_capacity=400, policy_capacity=400, batch_size=32)
    rng = np.random.default_rng(3)
    obs = _stable_obs(rng)
    legal = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    for _ in range(240):
        model.policy_buffer.add(_stable_obs(rng, noise=0.02), action=1, legal_mask=legal)
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


def test_v2_save_load_metadata_and_mismatch(tmp_path) -> None:
    model = _small_v2()
    assert model.gamma == pytest.approx(1.0)
    assert model.bet_scheme == BetScheme.POT_FRACTION
    assert model.q_loss_name == "huber"
    assert model.q_buffer.dtype == np.dtype(np.float16)

    path = tmp_path / "v2.pt"
    model.save(str(path), training_meta={"generation_count": 3})
    loaded = NFSPModel(q_capacity=800, policy_capacity=800, batch_size=32)
    meta = loaded.load(str(path))
    assert meta["generation_count"] == 3
    assert loaded.obs_version == 2
    assert loaded.bet_scheme == BetScheme.POT_FRACTION
    assert loaded.gamma == pytest.approx(1.0)
    assert loaded.q_loss_name == "huber"
    assert int(loaded.q_network.input_dim) == OBS_V2_DIM
    assert loaded.q_buffer.dtype == np.dtype(np.float16)

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint["model_meta"]["obs_version"] = 1
    bad = tmp_path / "bad.pt"
    torch.save(checkpoint, bad)
    with pytest.raises(ValueError):
        NFSPModel(q_capacity=64, policy_capacity=64, batch_size=8).load(str(bad))


def test_v1_checkpoint_without_metadata_still_loads(tmp_path) -> None:
    model = NFSPModel(q_capacity=32, policy_capacity=32, batch_size=8)
    path = tmp_path / "v1.pt"
    model.save(str(path))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    del checkpoint["model_meta"]
    torch.save(checkpoint, path)
    loaded = NFSPModel(q_capacity=32, policy_capacity=32, batch_size=8)
    loaded.load(str(path))
    assert loaded.obs_version == 1
    assert loaded.bet_scheme == BetScheme.LEGACY


def test_v2_bot_returns_a_legal_action() -> None:
    model = _small_v2(q_capacity=64, policy_capacity=64, batch_size=8)
    env = PokerEnv(
        num_players=2,
        learning_seats={0, 1},
        opponent_bots=[],
        small_blind=25,
        big_blind=50,
        seed=2,
        obs_version=2,
        bet_scheme=BetScheme.POT_FRACTION,
    )
    env.reset_multi()
    assert env.state is not None and env.state.action_on_seat is not None
    seat = env.state.action_on_seat
    legal = legal_actions(env.state, seat)
    action = NFSPBot(model=model, training=False).act(env.state.view_for(seat), legal)
    validate(env.state, seat, action)


def test_multi_seat_hand_stores_each_learners_transition() -> None:
    model = _small_v2(
        eta=1.0,
        epsilon_start=1.0,
        epsilon_end=1.0,
        train_every=10_000,
        q_capacity=4_000,
        policy_capacity=4_000,
    )
    env = PokerEnv(
        num_players=2,
        learning_seats={0, 1},
        opponent_bots=[],
        small_blind=25,
        big_blind=50,
        seed=1,
        obs_version=2,
        bet_scheme=BetScheme.POT_FRACTION,
    )
    both_acted = 0
    for hand in range(12):
        env.hand_number = hand
        before = len(model.q_buffer)
        result = consume_hand(
            model, env, {0, 1}, env.reset_multi({"deck_seed": 2_000 + hand})
        )
        assert result.finished
        gained = len(model.q_buffer) - before
        assert gained == result.decisions
        assert abs(sum(result.rewards.values())) < 0.05
        if gained >= 2:
            both_acted += 1
    assert both_acted >= 1
