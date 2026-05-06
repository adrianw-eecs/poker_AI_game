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
