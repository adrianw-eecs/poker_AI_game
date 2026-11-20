"""Tests for the three hybrid fixes: target source, early stopping, warm-up gate."""

import numpy as np
import pytest
import torch

from poker.bots.random_bot import RandomBot
from poker.ml.action_space import BetScheme
from poker.ml.cfr.regret_matching import regret_matching
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.observation_v2 import OBS_V2_DIM, build_observation_v2

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


# ------------------------------------------------- 1. target source


def test_distill_from_teacher_uses_network_not_traversal_rows() -> None:
    """The stored target must be the teacher's strategy, with a real mask."""
    from scripts.train_hybrid import distill_from_teacher

    teacher = SDCFRModel(obs_version=2)
    model = NFSPModel(distill_kappa=0.25, **_COMMON)
    model._ensure_distill_buffer(2_000)

    states = [(_obs(i), MASK.copy()) for i in range(40)]
    added = distill_from_teacher(model, teacher, states)
    assert added == 40
    assert len(model._distill_buffer) == 40

    # A stored row must reproduce the teacher's regret-matching distribution.
    row = model._distill_buffer.sample(1)["obs"][0]
    obs, adv = row[:OBS], row[OBS: OBS + 7]
    mask = row[OBS + 7:]
    stored = regret_matching(adv.astype(np.float32), mask.astype(np.float32))

    teacher.advantage_network.eval()
    with torch.no_grad():
        direct = teacher.advantage_network(
            torch.as_tensor(obs, device=teacher.device).unsqueeze(0)
        ).squeeze(0).cpu().numpy().astype(np.float32)
    assert np.allclose(adv, direct, atol=1e-5), "stored advantages are not the network output"
    assert abs(float(stored.sum()) - 1.0) < 1e-5
    assert (stored[mask == 0] == 0).all()


def test_distillation_targets_are_not_degenerate_on_average() -> None:
    """Teacher output must carry information, not collapse to one-hot.

    A single state can legitimately be one-hot (the best action is clear), so
    this checks the average over many real states.
    """
    teacher = SDCFRModel(obs_version=2)
    teacher.advantage_network.eval()

    rng = np.random.RandomState(0)
    states = []
    for h in range(30):
        n = int(rng.randint(2, 7))
        env = PokerEnv(
            num_players=n, learning_seat=0,
            opponent_bots=[RandomBot(seed=int(rng.randint(0, 1000))) for _ in range(n - 1)],
            small_blind=25, big_blind=50, seed=int(rng.randint(0, 10**6)),
            obs_version=2, bet_scheme=BetScheme.POT_FRACTION,
        )
        env.reset()
        if env.state is None or env.state.action_on_seat != 0:
            continue
        states.append((build_observation_v2(env.state, 0),
                       env.get_action_mask().astype(np.float32)))
    assert len(states) >= 10

    obs_b = np.stack([s[0] for s in states])
    with torch.no_grad():
        adv = teacher.advantage_network(
            torch.as_tensor(obs_b, device=teacher.device)
        ).cpu().numpy().astype(np.float32)

    support = []
    for i, (_, mask) in enumerate(states):
        s = regret_matching(adv[i], mask)
        support.append(int((s > 1e-6).sum()))
    mean_support = float(np.mean(support))
    assert mean_support > 1.5, (
        f"teacher output averages {mean_support:.2f} actions with mass; "
        "distillation targets are effectively one-hot"
    )


def test_roster_without_checkpoints_uses_all_three_baselines(tmp_path) -> None:
    from poker.ml.roster import CheckpointRoster

    roster = CheckpointRoster(tmp_path, obs_version=2, teacher=None)
    assert roster.refresh() == 0
    bots = roster.seat(3, np.random.default_rng(0))
    assert {type(bot).__name__ for bot in bots} == {"RandomBot", "FlopBot", "CallBot"}


def test_gate_baseline_cells_are_reported_separately() -> None:
    from poker.ml.gate import evaluate_gate

    model = NFSPModel(
        obs_version=2,
        bet_scheme=BetScheme.POT_FRACTION,
        batch_size=4,
        q_capacity=8,
        policy_capacity=8,
    )
    report = evaluate_gate(model, None, hands=1, fields=(), depths=())
    assert [cell.name for cell in report.baselines] == ["6max_call", "6max_flop", "6max_random"]
    assert report.head_to_head == ()
    assert report.selection_lower == min(cell.share_low for cell in report.baselines)


def test_v5_learner_refuses_a_v2_teacher() -> None:
    from poker.ml.observation_v5 import OBS_V5_DIM
    from scripts.train_hybrid import distill_from_teacher

    teacher = SDCFRModel(obs_version=2)
    learner = NFSPModel(obs_version=5, bet_scheme=BetScheme.POT_FRACTION, batch_size=8, q_capacity=16, policy_capacity=16)
    states = [(np.zeros(OBS_V5_DIM, dtype=np.float32), MASK.copy())]
    with pytest.raises(ValueError, match="Refusing to distill"):
        distill_from_teacher(learner, teacher, states)


def test_tendency_half_life_forgets_early_hands() -> None:
    from poker.domain.action import ActionType
    from poker.ml.tendencies import TendencyTracker

    tracker = TendencyTracker(num_players=2, half_life=1.0)
    for _ in range(30):
        tracker.start_hand()
        tracker.record_action(
            0, ActionType.RAISE, street_is_preflop=True, facing_bet=False
        )
        tracker.end_hand()
    assert tracker.hands_seen(0) < 5
    assert 0.0 < tracker.confidence(0) < 1.0


# ------------------------------------------------- 2. early stopping


def test_hybrid_trainer_has_early_stopping_flags() -> None:
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "scripts/train_hybrid.py", "--help"],
        capture_output=True, text=True,
    ).stdout
    assert "--stop-min-episode" in out
    assert "--stop-patience" in out


# ------------------------------------------------- 3. warm-up gate


@pytest.mark.smoke
def test_warmup_gate_counts_hands_and_opens_early() -> None:
    kwargs = {**_COMMON, "batch_size": 32}
    model = NFSPModel(q_warmup_hands=10_000, **kwargs)
    assert model._q_min == 32 // 4
    assert model._policy_min < model.batch_size

    # Below both thresholds: no training.
    assert not model.ready_to_train_q()

    # A few batches of rows is enough even without reported hands (test-style
    # callers that only store transitions must still train).
    rng = np.random.default_rng(0)
    for i in range(2 * model.batch_size):
        o = _obs(i)
        model.store_transition(o, 1, 0.0, o, False, next_mask=MASK, current_mask=MASK)
    # n-step buffering holds the tail of the window, so flush before reading.
    model.flush_pending()
    assert model.ready_to_train_q(), "row-count fallback did not open the gate"

    # With hands reported, a smaller number of rows suffices.
    hands_model = NFSPModel(q_warmup_hands=5, **kwargs)
    for _ in range(5):
        hands_model.end_hand()
    assert hands_model._hands_completed >= hands_model._q_min_hands
    for i in range(hands_model._q_min):
        o = _obs(i)
        hands_model.store_transition(o, 1, 0.0, o, False, next_mask=MASK, current_mask=MASK)
    hands_model.flush_pending()
    assert hands_model.ready_to_train_q()

    # Hands alone, with no rows, is still not enough.
    bare = NFSPModel(q_warmup_hands=5, **kwargs)
    for _ in range(10):
        bare.end_hand()
    assert not bare.ready_to_train_q()


def test_old_gate_would_have_been_far_larger() -> None:
    """Guard the regression: 1% of a 1M buffer was 10,000 transitions."""
    model = NFSPModel(obs_version=2, batch_size=2048,
                      q_capacity=1_000_000, policy_capacity=1_000_000)
    old_gate = max(2048, 1_000_000 // 100)
    assert old_gate == 10_000
    assert model._q_min < old_gate / 4, "warm-up gate was not lowered"


def test_policy_gate_allows_partial_buffer() -> None:
    model = NFSPModel(**_COMMON)
    assert model._policy_min <= model.batch_size // 4


def test_hand_counter_survives_checkpoint_round_trip(tmp_path) -> None:
    model = NFSPModel(q_warmup_hands=100, **{**_COMMON, "batch_size": 16})
    for _ in range(100):
        model.end_hand()
    path = str(tmp_path / "warm.pt")
    model.save(path)

    loaded = NFSPModel(obs_version=2, batch_size=16, q_capacity=64, policy_capacity=64)
    loaded.load(path)
    # A resumed run must not have to re-earn warm-up it already completed.
    assert loaded._hands_completed >= loaded._q_min_hands
