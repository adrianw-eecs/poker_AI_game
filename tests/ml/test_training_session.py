"""Sitting length, bet cap, and checkpoint-ladder selection."""

from dataclasses import replace as dc_replace
from pathlib import Path

import numpy as np
import pytest

from poker.bots.call_bot import CallBot
from poker.domain.action import Action, ActionType
from poker.ml.action_space import action_index_to_action, cap_action
from poker.ml.env import PokerEnv
from scripts.train_nfsp_selfplay import (
    EvalResult,
    append_csv_row,
    eval_history_row,
    ladder_checkpoints,
)


def _heads_up() -> PokerEnv:
    return PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[CallBot()],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=3,
    )


def test_next_hand_carries_stacks_and_rotates_the_button() -> None:
    env = _heads_up()
    env.hand_number = 0
    env.reset()
    _, _, done, _, _ = env.step(0)
    assert done
    assert env.state is not None
    assert env.state.players[0].stack == 975
    assert env.state.players[1].stack == 1025

    opened = env.start_next_hand(seed=11)
    assert opened is not None
    assert env.hand_number == 1
    assert env.state is not None
    assert env.state.dealer_seat == 1
    assert env.player_stacks_at_hand_start[0] == 975
    assert env.player_stacks_at_hand_start[1] == 1025


def test_short_stack_posts_an_all_in_blind_and_the_sitting_continues() -> None:
    """A stack below the big blind is a short all-in, not the end of the sitting."""
    env = PokerEnv(
        num_players=3,
        learning_seat=0,
        opponent_bots=[CallBot(), CallBot()],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=3,
    )
    env.hand_number = 0
    env.reset(options={"stacks": (1000, 1000, 10)})
    assert env.state is not None
    assert env.state.dealer_seat == 0
    # Seat 2 is the big blind and posts the 10 chips it has.
    big_blind = env.state.players[2]
    assert big_blind.is_all_in
    assert big_blind.stack == 0
    assert big_blind.committed_this_street == 10
    assert sum(player.stack + player.committed_this_hand for player in env.state.players) == 2010
    assert env.state.action_on_seat is not None


def test_sitting_ends_when_one_live_seat_remains() -> None:
    env = _heads_up()
    env.hand_number = 2
    env.reset()
    assert env.state is not None
    busted = tuple(
        player.with_stack(0) if player.seat == 1 else player.with_stack(player.stack + env.state.players[1].stack)
        for player in env.state.players
    )
    env.state = dc_replace(env.state, players=busted)
    assert env.start_next_hand(seed=1) is None
    assert env.hand_number == 2
    assert env.state.players[1].is_eliminated
    assert env.state.players[1].stack == 0


def test_max_bet_collapses_raises_to_the_cap() -> None:
    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[CallBot()],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=5,
        max_bet=100,
    )
    env.hand_number = 0
    env.reset()
    assert env.state is not None
    mask = env.get_action_mask()
    amounts = []
    for index in range(2, 7):
        if not mask[index]:
            continue
        action = action_index_to_action(index, env.state, 0, max_bet=100)
        assert action.amount <= 100
        amounts.append(action.amount)
    assert amounts
    assert len(amounts) == len(set(amounts))

    oversized = Action.all_in(1000)
    capped = cap_action(env.state, env.learning_seat, oversized, 100)
    assert capped.type == ActionType.RAISE
    assert capped.amount == 100

    raise_index = int(np.where(mask[2:])[0][-1]) + 2
    _, _, _, _, _ = env.step(raise_index)
    assert env.state is not None
    assert env.state.players[0].stack == 900
    assert env.state.players[1].stack == 900


def test_second_hand_with_a_bet_cap_finishes() -> None:
    """A short call must not drop the live bet and leave the next hand spinning."""
    from poker.domain.action import ActionType

    class _RaiseBot:
        def __init__(self) -> None:
            self.name = "Raise"

        def act(self, state, legal):  # type: ignore[no-untyped-def]
            for action in legal:
                if action.type == ActionType.ALL_IN:
                    return action
            return legal[0]

    env = PokerEnv(
        num_players=9,
        learning_seat=0,
        opponent_bots=[_RaiseBot() for _ in range(8)],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=1,
        max_bet=150,
    )
    env.hand_number = 3
    env.reset()
    for hand in range(2):
        if hand:
            assert env.start_next_hand(seed=4) is not None
        done = False
        steps = 0
        while env.state is not None and env.state.action_on_seat == 0 and not done and steps < 30:
            mask = env.get_action_mask()
            action = 6 if mask[6] else int(np.where(mask)[0][-1])
            _, _, done, _, _ = env.step(action)
            steps += 1
        assert done, f"hand {hand} did not finish"
        assert env.state is not None
        assert sum(player.stack for player in env.state.players) == 9 * 1000


def test_check_is_replaced_when_chips_are_still_owed() -> None:
    from poker.bots.call_bot import CallBot
    from poker.bots.progress import ensure_progressing_action
    from poker.domain.action import Action

    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[CallBot()],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=1,
        max_bet=150,
    )
    env.hand_number = 0
    env.reset()
    assert env.state is not None
    # Hero is the small blind. A check here would leave the big blind unpaid.
    replaced = ensure_progressing_action(
        env.state,
        Action.check(),
        [Action.fold(), Action.check(), Action.call(50), Action.raise_to(100)],
    )
    assert replaced.type.value == "call"


def test_stall_guard_records_a_bot_that_checks_forever() -> None:
    from poker.domain.action import Action

    class _CheckBot:
        name = "CheckForever"

        def act(self, state, legal):  # type: ignore[no-untyped-def]
            del state, legal
            return Action.check()

    counts: dict[str, int] = {}
    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[_CheckBot()],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=2,
        max_bet=150,
        stall_guard=True,
        stall_counts=counts,
    )
    env.hand_number = 0
    env.reset()
    assert env.state is not None
    from dataclasses import replace as dc_replace

    def _stick_on_opponent(state):  # type: ignore[no-untyped-def]
        if state.action_on_seat == 0:
            return dc_replace(state, action_on_seat=1)
        return state

    env._advance_action = _stick_on_opponent  # type: ignore[method-assign]
    done = False
    steps = 0
    while env.state is not None and env.state.action_on_seat == 0 and not done and steps < 5:
        mask = env.get_action_mask()
        action = 6 if mask[6] else int(np.where(mask)[0][-1])
        _, _, done, _, _ = env.step(action)
        steps += 1
    assert counts.get("CheckForever", 0) > 0


def test_max_bet_below_the_big_blind_is_rejected() -> None:
    with pytest.raises(ValueError):
        PokerEnv(max_bet=10, big_blind=50, small_blind=25)


def test_ladder_keeps_every_stride_and_the_best_file(tmp_path: Path) -> None:
    for generation in (0, 1, 5, 6, 10):
        (tmp_path / f"nfsp_gen_{generation}.pt").write_bytes(b"x")
    (tmp_path / "nfsp_best.pt").write_bytes(b"x")
    selected = ladder_checkpoints(tmp_path, stride=5)
    assert [label for label, _ in selected] == ["0", "5", "10", "best"]


def test_eval_history_appends_a_header_once(tmp_path: Path) -> None:
    result = EvalResult(1.5, 0.5, 2.5, 10, 0.2, 0.4)
    row = eval_history_row(0, 3, 0.15, result, result)
    path = tmp_path / "eval_history.csv"
    append_csv_row(path, row)
    append_csv_row(path, row)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("episode,generation,self_play")
    assert len(lines) == 3
    assert "0.400000" in lines[1]
