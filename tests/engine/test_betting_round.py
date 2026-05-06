"""Tests for BettingRound."""

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
