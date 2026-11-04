"""Tests for the NFSP + SD-CFR hybrid (shared trunk and distillation)."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from poker.ml.action_space import BetScheme
from poker.ml.cfr.regret_matching import regret_matching
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.nfsp_networks import (
    HybridNetV2,
    build_hybrid_network_v2,
    build_q_network_v2,
)
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.observation_v2 import OBS_V2_DIM

OBS = OBS_V2_DIM
MASK = np.ones(7, dtype=np.float32)

_COMMON = dict(
    obs_version=2,
    bet_scheme=BetScheme.POT_FRACTION,
    batch_size=64,
    train_every=1,
    q_capacity=4_000,
    policy_capacity=4_000,
    eta=1.0,
    epsilon_start=0.0,
    epsilon_end=0.0,
)


def _obs(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    obs = rng.random(OBS).astype(np.float32)
    obs[:7] = rng.integers(0, 52, 7)
    return obs


def _fill(model: NFSPModel, n: int = 400, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    for _ in range(n):
        obs = _obs(int(rng.integers(1, 10_000)))
        nxt = _obs(int(rng.integers(1, 10_000)))
        action = int(rng.integers(0, 7))
        model.store_transition(
            obs, action, 0.0, nxt, False, next_mask=MASK, current_mask=MASK
        )
        model.policy_buffer.add(obs, action, legal_mask=MASK)


def _policy_probs(model: NFSPModel, obs: np.ndarray) -> np.ndarray:
    model.policy_network.eval()
    with torch.no_grad():
        x = torch.as_tensor(obs, dtype=torch.float32, device=model.device).unsqueeze(0)
        return torch.softmax(model.policy_network(x), dim=-1).squeeze(0).cpu().numpy()


# ---------------------------------------------------------------- network


@pytest.mark.smoke
def test_hybrid_network_shares_trunk_and_exposes_both_heads() -> None:
    net = build_hybrid_network_v2()
    assert isinstance(net, HybridNetV2)
    x = torch.zeros(2, OBS)
    x[:, 0] = 12
    x[:, 1] = 25
    x[:, 2:] = 0.2
    x[:, 2:7] = 52
    mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0]] * 2, dtype=torch.bool)

    q_values = net(x, legal_mask=mask)
    assert q_values.shape == (2, 7)

    # Compare in eval mode: dropout is active in train mode, so two separate
    # calls would legitimately differ even when the trunk is shared.
    net.eval()
    q_values = net(x, legal_mask=mask)
    q_both, logits = net.forward_both(x, legal_mask=mask)
    assert torch.allclose(q_values, q_both)
    assert logits.shape == (2, 7)
    assert torch.isfinite(logits).all()

    # The policy head starts uniform.
    assert torch.count_nonzero(net.policy_head.weight) == 0


def test_hybrid_policy_head_receives_gradient_through_shared_trunk() -> None:
    net = build_hybrid_network_v2()
    x = torch.zeros(2, OBS)
    x[:, 0] = 12
    x[:, 1] = 25
    x[:, 2:] = 0.2
    x[:, 2:7] = 52
    net.zero_grad()
    _q, logits = net.forward_both(x)
    logits.sum().backward()
    grad = net.trunk[0].fc1.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_hybrid_adds_only_a_policy_head_over_poker_net_v2() -> None:
    hybrid = build_hybrid_network_v2()
    plain = build_q_network_v2()
    extra = sum(p.numel() for p in hybrid.parameters()) - sum(
        p.numel() for p in plain.parameters()
    )
    # Exactly one Linear(trunk_dim -> num_actions) worth of parameters.
    assert extra == hybrid.trunk_dim * hybrid.num_actions + hybrid.num_actions


# ----------------------------------------------------------------- model


@pytest.mark.smoke
def test_shared_trunk_model_reuses_one_network() -> None:
    model = NFSPModel(shared_trunk=True, **_COMMON)
    assert model.shared_trunk is True
    assert model.q_network is model.policy_network
    assert isinstance(model.q_network, HybridNetV2)


def test_shared_trunk_model_trains_both_heads() -> None:
    model = NFSPModel(shared_trunk=True, **_COMMON)
    _fill(model, seed=3)
    results = [model.train_step() for _ in range(4)]
    last = results[-1]
    assert last["q_loss"] is not None
    assert last["policy_loss"] is not None
    assert model._q_updates > 0
    assert model._p_updates > 0


def test_shared_trunk_requires_obs_v2() -> None:
    with pytest.raises(ValueError):
        NFSPModel(shared_trunk=True, obs_version=1)


def test_kappa_zero_adds_no_distillation_work() -> None:
    model = NFSPModel(distill_kappa=0.0, **_COMMON)
    assert model.distill_kappa == 0.0
    assert model._distill_buffer is None


def test_invalid_kappa_rejected() -> None:
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            NFSPModel(distill_kappa=bad, **_COMMON)


# --------------------------------------------------------- distillation


def test_regret_targets_are_valid_distributions() -> None:
    rng = np.random.default_rng(0)
    for _ in range(100):
        adv = rng.normal(size=7).astype(np.float32)
        mask = (rng.random(7) > 0.3).astype(np.float32)
        if mask.sum() == 0:
            continue
        strat = regret_matching(adv, mask)
        assert abs(float(strat.sum()) - 1.0) < 1e-5
        assert (strat >= 0).all()
        assert (strat[mask == 0] == 0).all()


def test_distillation_loss_is_finite() -> None:
    model = NFSPModel(distill_kappa=0.25, **_COMMON)
    model._ensure_distill_buffer(2_000)
    rng = np.random.default_rng(1)
    for _ in range(300):
        model.add_regret_sample(
            _obs(int(rng.integers(1, 9_000))),
            rng.normal(size=7).astype(np.float32),
            MASK,
        )
    assert len(model._distill_buffer) == 300
    loss = model._distillation_step()
    assert loss is not None and np.isfinite(loss)


@pytest.mark.smoke
def test_distillation_pulls_policy_toward_cfr_target() -> None:
    """The supervised target must actually move the average policy."""
    target_obs = _obs(1)
    advantages = np.array([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)

    control = NFSPModel(distill_kappa=0.0, **_COMMON)
    _fill(control, seed=0)
    for _ in range(30):
        control.train_step()

    treated = NFSPModel(distill_kappa=0.25, **_COMMON)
    _fill(treated, seed=0)
    treated._ensure_distill_buffer(5_000)
    for _ in range(1_500):
        treated.add_regret_sample(target_obs, advantages, MASK)
    for _ in range(80):
        treated.train_step()

    p_control = _policy_probs(control, target_obs)[3]
    p_treated = _policy_probs(treated, target_obs)[3]
    assert p_treated > p_control + 0.05, (
        f"distillation did not move the policy toward the CFR target: "
        f"control={p_control:.4f} treated={p_treated:.4f}"
    )


# ------------------------------------------------------------ SD-CFR v2


@pytest.mark.smoke
def test_sdcfr_defaults_to_v1_and_keeps_legacy_scheme() -> None:
    model = SDCFRModel()
    assert model.obs_version == 1
    assert model.obs_dim == 98
    assert model.bet_scheme is BetScheme.LEGACY


def test_sdcfr_v2_uses_obs_v2_and_pot_fraction() -> None:
    model = SDCFRModel(obs_version=2)
    assert model.obs_dim == OBS
    assert model.bet_scheme is BetScheme.POT_FRACTION
    assert model.regret_buffer.obs_dim == OBS + 7
    strategy = model.get_strategy(np.random.rand(OBS).astype(np.float32), MASK)
    assert abs(float(strategy.sum()) - 1.0) < 1e-5


def test_sdcfr_v2_round_trip(tmp_path) -> None:
    model = SDCFRModel(obs_version=2)
    path = str(tmp_path / "sd_v2.pt")
    model.save(path)

    loaded = SDCFRModel()
    loaded.load(path)
    assert loaded.obs_version == 2
    assert loaded.obs_dim == OBS
    assert loaded.bet_scheme is BetScheme.POT_FRACTION


def test_sdcfr_rejects_bad_obs_version() -> None:
    with pytest.raises(ValueError):
        SDCFRModel(obs_version=3)


def test_sdcfr_bot_follows_model_encoding() -> None:
    from poker.bots.sdcfr_bot import SDCFRBot

    legacy = SDCFRBot(model=SDCFRModel())
    assert legacy.obs_version == 1
    assert legacy.bet_scheme is BetScheme.LEGACY

    modern = SDCFRBot(model=SDCFRModel(obs_version=2))
    assert modern.obs_version == 2
    assert modern.bet_scheme is BetScheme.POT_FRACTION


# ------------------------------------------------- hybrid persistence


@pytest.mark.smoke
def test_shared_trunk_checkpoint_round_trip(tmp_path) -> None:
    """A hybrid checkpoint must reload into a hybrid model, not a plain one."""
    model = NFSPModel(shared_trunk=True, distill_kappa=0.25, **_COMMON)
    _fill(model, seed=2)
    for _ in range(4):
        model.train_step()

    path = str(tmp_path / "hybrid.pt")
    model.save(path)

    loaded = NFSPModel(obs_version=2, batch_size=8, q_capacity=64, policy_capacity=64)
    loaded.load(path)
    assert loaded.shared_trunk is True
    assert loaded.q_network is loaded.policy_network
    assert loaded.distill_kappa == pytest.approx(0.25)

    # Policy output must survive the round trip (eval mode: no dropout).
    target = _obs(1)
    model.policy_network.eval()
    loaded.policy_network.eval()
    with torch.no_grad():
        want = model.policy_network(
            torch.as_tensor(target, device=model.device).unsqueeze(0)
        ).squeeze(0).cpu()
        got = loaded.policy_network(
            torch.as_tensor(target, device=loaded.device).unsqueeze(0)
        ).squeeze(0).cpu()
    assert torch.allclose(want, got, atol=1e-5)


def test_plain_v2_checkpoint_still_loads_as_two_networks(tmp_path) -> None:
    model = NFSPModel(obs_version=2, bet_scheme=BetScheme.POT_FRACTION,
                      batch_size=8, q_capacity=64, policy_capacity=64)
    path = str(tmp_path / "plain.pt")
    model.save(path)

    loaded = NFSPModel(obs_version=2, batch_size=8, q_capacity=64, policy_capacity=64)
    loaded.load(path)
    assert loaded.shared_trunk is False
    assert loaded.q_network is not loaded.policy_network
