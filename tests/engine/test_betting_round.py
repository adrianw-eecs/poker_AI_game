"""Tests for BettingRound."""

import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action
from poker.domain.card import Card, Rank, Suit
from poker.engine.betting_round import BettingRound
from poker.logging.logger import NullLogger
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot

_CONFIG = GameConfig(
    num_players=3, starting_stack=1000, small_blind=10, big_blind=20,
    ante=0, rake_percent=0, rake_cap=None,
    blind_schedule=BlindSchedule(levels=[BlindLevel(10, 20, 0)], hands_per_level=10, fixed=True),
)
_BLIND = BlindLevel(10, 20, 0)


def _player(seat: int, stack: int, committed: int = 0, all_in: bool = False) -> PlayerState:
    return PlayerState(
        seat=seat, name=f"P{seat}", stack=stack,
        hole_cards=(Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)),
        committed_this_street=committed, committed_this_hand=0,
        has_folded=False, is_all_in=all_in, is_eliminated=False,
    )


def _state(players, action_on, bet=0, last_raise=0) -> GameState:
    return GameState(
        hand_number=0, street=Street.PREFLOP, dealer_seat=0,
        players=tuple(players), community_cards=(),
        pots=[Pot(0, frozenset(range(len(players))))],
        current_bet_to_call=bet, last_raise_size=last_raise,
        action_history_this_street=[], action_history_this_hand=[],
        deck_remaining_count=52, config=_CONFIG, blind_level=_BLIND,
        action_on_seat=action_on,
    )


def test_all_check_closes_round() -> None:
    state = _state([_player(0, 1000), _player(1, 1000), _player(2, 1000)], action_on=0)
    final = BettingRound(state, lambda seat, s: Action.check(), NullLogger()).run()
    assert final.action_on_seat is None
    assert not any(p.has_folded for p in final.players)


def test_raise_then_call_closes_round() -> None:
    seq = [Action.raise_to(40), Action.call(40)]
    i = 0
    def get(seat, s):
        nonlocal i; a = seq[i] if i < len(seq) else Action.check(); i += 1; return a
    state = _state([_player(0, 1000), _player(1, 1000)], action_on=0)
    final = BettingRound(state, get, NullLogger()).run()
    assert final.action_on_seat is None


def test_all_fold_except_one_closes_round() -> None:
    folds = [0]
    def get(seat, s):
        if folds[0] > 0: folds[0] -= 1; return Action.fold()
        return Action.check()
    folds[0] = 2
    state = _state([_player(0, 1000), _player(1, 1000), _player(2, 1000)], action_on=0)
    final = BettingRound(state, get, NullLogger()).run()
    assert final.action_on_seat is None


def test_no_action_when_only_one_active_player() -> None:
    """Betting round should close when only 1 player can act (not folded/all-in).

    This test reproduces and verifies the fix for the Hand 47 bug where
    players who are folded or all-in should not trigger further actions.
    """
    # Player 0 folded, Player 1 all-in, only Player 2 can act
    players = [
        _player(0, 1000, committed=20, all_in=False).with_folded(True),
        _player(1, 50, committed=50, all_in=True),
        _player(2, 1000, committed=100, all_in=False),
    ]
    state = _state(players, action_on=2, bet=100)

    action_count = [0]
    def get_action(seat, s):
        action_count[0] += 1
        if action_count[0] > 2:
            raise RuntimeError(f"Too many actions! Got {action_count[0]} when only 1 active player")
        return Action.fold()

    final = BettingRound(state, get_action, NullLogger()).run()
    assert final.action_on_seat is None, "Betting round should close with only 1 active player"
    assert action_count[0] <= 1, f"Should have 0-1 actions, got {action_count[0]}"


def test_no_cycling_back_to_same_player() -> None:
    """Action should not cycle back to same player when others are all-in/folded."""
    # Create state where players 0 and 2 are all-in, only player 1 can act
    players = [
        _player(0, 5, committed=5, all_in=True),
        _player(1, 1000, committed=100, all_in=False),
        _player(2, 50, committed=50, all_in=True),
    ]
    state = _state(players, action_on=1, bet=100)

    # Advance action from player 1
    betting_round = BettingRound(state, lambda s, st: Action.fold(), NullLogger())
    advanced = betting_round._advance_action_on_seat(state, 1)

    # Should return None (no next player), not cycle back to player 1
    assert advanced.action_on_seat is None, \
        "Should return None when only 1 player can act, not cycle to next"


def test_raise_is_recorded_in_action_history() -> None:
    """RAISE actions must appear in action_history_this_street (regression)."""
    state = _state([_player(0, 1000), _player(1, 1000)], action_on=0, bet=0)
    final = BettingRound(
        state,
        lambda seat, s: Action.raise_to(50) if seat == 0 else Action.call(50),
        NullLogger(),
    ).run()
    action_types = [a.type.value for _, a in final.action_history_this_street]
    assert "raise" in action_types
    assert len(final.action_history_this_street) == 2


def test_multiple_raises_without_double_actions() -> None:
    """Multiple raises should not cause the same player to act consecutively."""
    players = [
        _player(0, 1000, committed=0),
        _player(1, 1000, committed=0),
    ]
    state = _state(players, action_on=0, bet=0)

    actions_taken = []

    def get_action(seat, s):
        actions_taken.append(seat)
        if len(actions_taken) == 1:
            # Player 0 raises to 50
            return Action.raise_to(50)
        elif len(actions_taken) == 2:
            # Player 1 re-raises to 150
            return Action.raise_to(150)
        elif len(actions_taken) == 3:
            # Player 0 calls
            return Action.call(150)
        else:
            return Action.fold()

    final = BettingRound(state, get_action, NullLogger()).run()

    # Verify the sequence is: 0, 1, 0 (not 0, 1, 0, 1, ... looping)
    assert actions_taken == [0, 1, 0], f"Expected [0, 1, 0], got {actions_taken}"
    assert final.action_on_seat is None


@pytest.mark.smoke
def test_hand_47_scenario_no_double_action_after_all_in() -> None:
    """Reproduces Hand 47 bug: Seat 1 raises, others go all-in, Seat 1 shouldn't act again.

    Original bug sequence:
    - Seat 1 raises to 1124
    - Seat 3 calls 881 (all-in with 881 chips)
    - Seat 7 calls 969 (all-in with 969 chips)
    - Seat 0 calls 5 (all-in with 5 chips)
    - BUG: Seat 1 calls 1124 again (WRONG - shouldn't act after all are all-in)

    Fixed behavior:
    - Betting round should close before Seat 1 gets to act again
    """
    # Simulate the exact scenario
    players = [
        _player(0, 5, committed=5, all_in=True),    # Called for 5
        _player(1, 1000, committed=1124, all_in=False),  # Raised to 1124
        _player(2, 1000, committed=0, all_in=False),     # Folded (but still in for testing)
        _player(3, 881, committed=881, all_in=True),  # Called for 881
        _player(4, 1000, committed=0, all_in=False),  # Folded
        _player(5, 1000, committed=0, all_in=False),  # Folded
        _player(6, 1000, committed=0, all_in=False),  # Folded
        _player(7, 969, committed=969, all_in=True),  # Called for 969
    ]

    # Mark folded players
    players = list(players)
    players[2] = players[2].with_folded(True)
    players[4] = players[4].with_folded(True)
    players[5] = players[5].with_folded(True)
    players[6] = players[6].with_folded(True)

    state = _state(players, action_on=1, bet=1124)

    # Count how many times player 1 gets to act
    p1_action_count = [0]

    def get_action(seat, s):
        if seat == 1:
            p1_action_count[0] += 1
            if p1_action_count[0] > 1:
                raise RuntimeError("Player 1 is acting again after all others are all-in!")
        return Action.fold()

    final = BettingRound(state, get_action, NullLogger()).run()

    # Player 1 should not get to act since all others are folded or all-in
    assert p1_action_count[0] == 0, \
        f"Player 1 should not act when only player who can act after all go all-in, but acted {p1_action_count[0]} times"
    assert final.action_on_seat is None, "Betting round should be closed"
