"""Tests for FlopBot."""

import pytest

from poker.bots.flop_bot import FlopBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action, ActionType
from poker.domain.card import Card
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


def _state(street=Street.PREFLOP, bet=10, hole_cards=(), community=()) -> GameState:
    sched = BlindSchedule(levels=[BlindLevel(5, 10)], hands_per_level=10, fixed=True)
    config = GameConfig(num_players=2, starting_stack=1000, small_blind=5, big_blind=10,
                        ante=0, rake_percent=0.0, rake_cap=None, blind_schedule=sched)
    players = tuple(PlayerState(
        seat=i, name=f"P{i}", stack=1000,
        hole_cards=hole_cards if i == 0 else (),
        committed_this_street=0, committed_this_hand=0,
        has_folded=False, is_all_in=False, is_eliminated=False,
    ) for i in range(2))
    blind_level = config.blind_schedule.level_for_hand(0)
    return GameState(
        hand_number=0, street=street, dealer_seat=0,
        players=players, community_cards=community,
        pots=[Pot(15, frozenset({0, 1}))],
        current_bet_to_call=bet, last_raise_size=10,
        action_history_this_street=[], action_history_this_hand=[],
        deck_remaining_count=48, config=config, blind_level=blind_level,
        action_on_seat=0,
    )


@pytest.mark.smoke
def test_preflop_never_raises() -> None:
    bot = FlopBot()
    state = _state(street=Street.PREFLOP)
    legal = [Action.fold(), Action.call(10), Action.raise_to(20), Action.all_in(1000)]
    for _ in range(50):
        assert bot.act(state, legal).type != ActionType.RAISE


def test_flop_raises_with_trips() -> None:
    bot = FlopBot()
    hole = (Card.from_string("9d"), Card.from_string("2s"))
    community = (Card.from_string("9s"), Card.from_string("9h"), Card.from_string("Kc"))
    state = _state(street=Street.FLOP, bet=0, hole_cards=hole, community=community)
    legal = [Action.check(), Action.raise_to(20), Action.all_in(1000)]
    action = bot.act(state, legal)
    assert action.type == ActionType.RAISE


def test_flop_folds_no_pair_facing_bet() -> None:
    bot = FlopBot()
    hole = (Card.from_string("As"), Card.from_string("Kd"))
    community = (Card.from_string("2d"), Card.from_string("3h"), Card.from_string("7c"))
    state = _state(street=Street.FLOP, bet=10, hole_cards=hole, community=community)
    legal = [Action.fold(), Action.call(10), Action.raise_to(20)]
    assert bot.act(state, legal).type == ActionType.FOLD
