"""Tests for HumanBot."""

from unittest.mock import patch

from poker.bots.human_bot import HumanBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action, ActionType
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


def _state() -> GameState:
    sched = BlindSchedule(levels=[BlindLevel(5, 10)], hands_per_level=10, fixed=True)
    config = GameConfig(num_players=2, starting_stack=1000, small_blind=5, big_blind=10,
                        ante=0, rake_percent=0.0, rake_cap=None, blind_schedule=sched)
    players = tuple(PlayerState(
        seat=i, name=f"P{i}", stack=1000, hole_cards=(),
        committed_this_street=0, committed_this_hand=0,
        has_folded=False, is_all_in=False, is_eliminated=False,
    ) for i in range(2))
    blind_level = config.blind_schedule.level_for_hand(0)
    return GameState(
        hand_number=0, street=Street.PREFLOP, dealer_seat=0,
        players=players, community_cards=(),
        pots=[Pot(15, frozenset({0, 1}))],
        current_bet_to_call=10, last_raise_size=10,
        action_history_this_street=[], action_history_this_hand=[],
        deck_remaining_count=48, config=config, blind_level=blind_level,
        action_on_seat=0,
    )


def test_valid_input_returns_action() -> None:
    bot = HumanBot(seat=0, name="Alice")
    legal = [Action.fold(), Action.call(10)]
    with patch("builtins.input", return_value="f"), patch("builtins.print"):
        action = bot.act(_state(), legal)
    assert action.type == ActionType.FOLD


def test_observe_result_silent() -> None:
    bot = HumanBot(seat=0)
    with patch("builtins.print") as mock_print:
        bot.observe_result(_state(), 0.25)
    mock_print.assert_not_called()
