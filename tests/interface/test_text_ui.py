"""Tests for text UI renderer and input parser."""

from unittest.mock import patch

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action, ActionType
from poker.domain.card import Card, Rank, Suit
from poker.interface.text_ui import _parse_input, render
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


def _make_config() -> GameConfig:
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=10,
        fixed=True,
    )
    return GameConfig(
        num_players=6,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
    )


def _make_state(street: Street = Street.PREFLOP) -> GameState:
    config = _make_config()
    players = tuple(
        PlayerState(
            seat=i,
            name=f"Player{i}",
            stack=1000,
            hole_cards=(
                (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
                if i == 0
                else ()
            ),
            committed_this_street=0,
            committed_this_hand=0,
            has_folded=False,
            is_all_in=False,
            is_eliminated=False,
        )
        for i in range(6)
    )
    blind_level = config.blind_schedule.level_for_hand(0)
    return GameState(
        hand_number=0,
        street=street,
        dealer_seat=0,
        players=players,
        community_cards=(),
        pots=[Pot(amount=15, eligible_seats=frozenset(range(6)))],
        current_bet_to_call=0,
        last_raise_size=0,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=52,
        config=config,
        blind_level=blind_level,
        action_on_seat=0,
    )


def test_render_produces_non_empty_string() -> None:
    state = _make_state()
    output = render(state, viewer_seat=0)
    assert isinstance(output, str)
    assert len(output) > 0


def test_parse_fold_returns_fold_action() -> None:
    legal = [Action.fold(), Action.call(10)]
    result = _parse_input("f", legal)
    assert result is not None
    assert result.type == ActionType.FOLD


def test_parse_raise_with_valid_amount() -> None:
    legal = [Action.fold(), Action.call(10), Action.raise_to(20), Action.all_in(200)]
    result = _parse_input("r 50", legal)
    assert result is not None
    assert result.type == ActionType.RAISE
    assert result.amount == 50


def test_parse_invalid_input_returns_none() -> None:
    legal = [Action.fold(), Action.call(10)]
    assert _parse_input("xyz", legal) is None
    assert _parse_input("r abc", legal) is None
    assert _parse_input("", legal) is None
