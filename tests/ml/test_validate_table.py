"""Regression tests for the validate_training.py table fixes.

Two failures were fixed:

1. The env encoding was taken from the learner only, so a table whose learner
   was `random`/`flop` defaulted to obs v1 and any v2 checkpoint at the table
   received a 98-wide observation ("mat1 and mat2 shapes cannot be multiplied
   1x91 and 311x256").
2. A random/flop learner's action was converted to an index with
   action_to_action_index, which assumes LEGACY bucketing, then applied to a
   POT_FRACTION env -> "Invalid action 2: Raise is not legal".
"""

import numpy as np
import pytest

from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.domain.action import Action
from poker.engine.action_validator import legal_actions
from poker.ml.action_space import BetScheme, action_index_to_action, build_action_mask
from poker.ml.env import PokerEnv


def _env(scheme=BetScheme.POT_FRACTION, seed=0, obs_version=2):
    return PokerEnv(
        num_players=4, learning_seat=0,
        opponent_bots=[RandomBot(seed=i) for i in range(3)],
        small_blind=25, big_blind=50, seed=seed,
        obs_version=obs_version, bet_scheme=scheme,
    )


@pytest.mark.smoke
def test_index_for_action_round_trips_under_pot_fraction() -> None:
    """Every bot action must map to an index the POT_FRACTION env accepts."""
    from scripts.validate_training import _index_for_action

    env = _env()
    checked = 0
    for h in range(12):
        env.hand_number = h
        env.reset(options={"deck_seed": 300 + h})
        steps = 0
        while env.state is not None and steps < 30:
            seat = env.state.action_on_seat
            if seat is None:
                break
            st = env.state
            for action in legal_actions(st, seat):
                idx = _index_for_action(action, st, seat, BetScheme.POT_FRACTION)
                reconstructed = action_index_to_action(
                    idx, st, seat, scheme=BetScheme.POT_FRACTION
                )
                # Same kind of action; amount may differ only when the bot's
                # size is not representable by a bucket.
                assert reconstructed.type == action.type, (
                    f"{action.type.name} mapped to {reconstructed.type.name}"
                )
                checked += 1
            if seat != 0:
                break
            try:
                _, _, done, _, _ = env.step(1)
            except ValueError as exc:  # pragma: no cover - regression guard
                pytest.fail(f"env.step(1) raised {exc}")
            steps += 1
            if done:
                break
    assert checked > 20


def test_index_for_action_prefers_exact_amount() -> None:
    from scripts.validate_training import _index_for_action

    env = _env()
    env.reset(options={"deck_seed": 5})
    st = env.state
    assert st is not None and st.action_on_seat == 0

    mask = build_action_mask(st, 0, scheme=BetScheme.POT_FRACTION, dedup=True)
    for idx in np.where(mask)[0]:
        action = action_index_to_action(int(idx), st, 0, scheme=BetScheme.POT_FRACTION)
        back = _index_for_action(action, st, 0, BetScheme.POT_FRACTION)
        again = action_index_to_action(back, st, 0, scheme=BetScheme.POT_FRACTION)
        assert (again.type, again.amount) == (action.type, action.amount)


def test_index_for_action_handles_all_in() -> None:
    from scripts.validate_training import _index_for_action

    env = _env()
    env.reset(options={"deck_seed": 7})
    st = env.state
    assert st is not None
    legal = legal_actions(st, 0)
    all_ins = [a for a in legal if a.type.name == "ALL_IN"]
    if not all_ins:
        pytest.skip("no all-in available in this deal")
    idx = _index_for_action(all_ins[0], st, 0, BetScheme.POT_FRACTION)
    recon = action_index_to_action(idx, st, 0, scheme=BetScheme.POT_FRACTION)
    assert recon.type == all_ins[0].type or recon.amount == all_ins[0].amount


# ------------------------------------------------ table encoding


class _FakeSeat:
    def __init__(self, kind, model=None):
        self.kind = kind
        self.model = model


class _FakeModel:
    def __init__(self, obs_version, bet_scheme):
        self.obs_version = obs_version
        self.bet_scheme = bet_scheme


@pytest.mark.smoke
def test_table_encoding_uses_model_not_learner() -> None:
    """A v2 checkpoint on the table forces a v2 env even if the learner is random."""
    from scripts.validate_training import _table_encoding

    seats = [
        _FakeSeat("random"),
        _FakeSeat("nfsp", _FakeModel(2, BetScheme.POT_FRACTION)),
    ]
    enc = _table_encoding(seats)
    assert enc["obs_version"] == 2
    assert enc["bet_scheme"] is BetScheme.POT_FRACTION


def test_table_encoding_rejects_mixed_versions() -> None:
    from scripts.validate_training import _table_encoding

    seats = [
        _FakeSeat("nfsp", _FakeModel(2, BetScheme.POT_FRACTION)),
        _FakeSeat("sdcfr", _FakeModel(1, BetScheme.LEGACY)),
    ]
    with pytest.raises(SystemExit):
        _table_encoding(seats)


def test_table_encoding_empty_without_models() -> None:
    from scripts.validate_training import _table_encoding

    assert _table_encoding([_FakeSeat("random"), _FakeSeat("flop")]) == {}
