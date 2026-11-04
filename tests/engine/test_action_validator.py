"""Tests for action validator."""

import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action, ActionType
from poker.domain.card import Card, Rank, Suit
from poker.engine.action_validator import legal_actions, validate
from poker.exceptions import IllegalActionError
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot

_CONFIG = GameConfig(
    num_players=2, starting_stack=1000, small_blind=10, big_blind=20,
    ante=0, rake_percent=0, rake_cap=None,
    blind_schedule=BlindSchedule(levels=[BlindLevel(10, 20, 0)], hands_per_level=10, fixed=True),
)
_BLIND = BlindLevel(10, 20, 0)


def _player(seat: int, stack: int, committed: int = 0, folded: bool = False) -> PlayerState:
    return PlayerState(
        seat=seat, name=f"P{seat}", stack=stack,
        hole_cards=(Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.HEARTS)),
        committed_this_street=committed, committed_this_hand=0,
        has_folded=folded, is_all_in=False, is_eliminated=False,
    )


def _state(players, action_on: int, bet: int = 0, last_raise: int = 0) -> GameState:
    return GameState(
        hand_number=0, street=Street.PREFLOP, dealer_seat=0,
        players=tuple(players), community_cards=(),
        pots=[Pot(0, frozenset(range(len(players))))],
        current_bet_to_call=bet, last_raise_size=last_raise,
        action_history_this_street=[], action_history_this_hand=[],
        deck_remaining_count=52, config=_CONFIG, blind_level=_BLIND,
        action_on_seat=action_on,
    )


def test_fold_always_legal() -> None:
    state = _state([_player(0, 1000), _player(1, 1000)], action_on=0)
    types = [a.type for a in legal_actions(state, 0)]
    assert ActionType.FOLD in types


def test_big_blind_can_check_when_limped_to() -> None:
    """A player who has already matched the bet checks. They do not call zero."""
    state = _state(
        [_player(0, 980, committed=20), _player(1, 990, committed=10)],
        action_on=0,
        bet=20,
        last_raise=20,
    )
    types = [a.type for a in legal_actions(state, 0)]
    assert ActionType.CHECK in types
    assert ActionType.CALL not in types
    validate(state, 0, Action.check())


def test_check_only_without_bet() -> None:
    state_no_bet = _state([_player(0, 1000), _player(1, 1000)], action_on=0, bet=0)
    assert ActionType.CHECK in [a.type for a in legal_actions(state_no_bet, 0)]

    with pytest.raises(IllegalActionError):
        validate(_state([_player(0, 1000), _player(1, 1000)], action_on=0, bet=20), 0, Action.check())


def test_raise_boundaries() -> None:
    state = _state([_player(0, 1000), _player(1, 1000, committed=50)], action_on=0, bet=50, last_raise=30)
    raise_actions = [a for a in legal_actions(state, 0) if a.type == ActionType.RAISE]
    assert raise_actions
    assert raise_actions[0].amount >= 50 + max(20, 30)


def test_short_stack_goes_all_in() -> None:
    state = _state([_player(0, 10), _player(1, 1000, committed=20)], action_on=0, bet=20)
    types = [a.type for a in legal_actions(state, 0)]
    assert ActionType.ALL_IN in types
    assert ActionType.RAISE not in types


def test_short_allin_does_not_reopen_for_a_player_who_acted() -> None:
    """A player who already bet may only call a short all-in, not raise over it."""
    from dataclasses import replace

    acted = _player(0, 500, committed=100)
    short = _player(1, 0, committed=140).with_all_in(True)
    waiting = _player(2, 500, committed=0)
    state = _state([acted, short, waiting], action_on=0, bet=140, last_raise=100)
    state = replace(
        state,
        street=Street.FLOP,
        action_history_this_street=[
            (0, Action.raise_to(100)),
            (1, Action.all_in(140)),
        ],
    )
    types = [action.type for action in legal_actions(state, 0)]
    assert ActionType.FOLD in types
    assert ActionType.CALL in types
    assert ActionType.RAISE not in types
    assert ActionType.ALL_IN not in types
    with pytest.raises(IllegalActionError, match="re-open"):
        validate(state, 0, Action.raise_to(240))
    with pytest.raises(IllegalActionError, match="re-open"):
        validate(state, 0, Action.all_in(600))

    still_to_act = replace(state, action_on_seat=2)
    waiting_types = [action.type for action in legal_actions(still_to_act, 2)]
    assert ActionType.RAISE in waiting_types


def test_opening_short_allin_can_still_be_raised() -> None:
    """The first wager of a street is not an incomplete raise."""
    from dataclasses import replace

    checker = _player(0, 500, committed=0)
    short = _player(1, 0, committed=15).with_all_in(True)
    state = _state([checker, short], action_on=0, bet=15, last_raise=0)
    state = replace(
        state,
        street=Street.FLOP,
        action_history_this_street=[
            (0, Action.check()),
            (1, Action.all_in(15)),
        ],
    )
    types = [action.type for action in legal_actions(state, 0)]
    assert ActionType.RAISE in types


def test_full_allin_raise_reopens() -> None:
    from dataclasses import replace

    acted = _player(0, 500, committed=100)
    full = _player(1, 0, committed=200).with_all_in(True)
    state = _state([acted, full], action_on=0, bet=200, last_raise=100)
    state = replace(
        state,
        street=Street.FLOP,
        action_history_this_street=[
            (0, Action.raise_to(100)),
            (1, Action.all_in(200)),
        ],
    )
    types = [action.type for action in legal_actions(state, 0)]
    assert ActionType.RAISE in types
