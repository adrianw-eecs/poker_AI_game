"""Tests for GameState."""

import pytest

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.card import Card, Rank, Suit
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


def _make_config() -> GameConfig:
    schedule = BlindSchedule(levels=[BlindLevel(small=5, big=10)], hands_per_level=10, fixed=True)
    return GameConfig(
        num_players=2, starting_stack=1000, small_blind=5, big_blind=10,
        ante=0, rake_percent=0.0, rake_cap=None, blind_schedule=schedule,
    )


def _make_player(seat: int, stack: int = 1000, **kwargs) -> PlayerState:
    defaults = dict(
        seat=seat, name=f"P{seat}", stack=stack, hole_cards=(),
        committed_this_street=0, committed_this_hand=0,
        has_folded=False, is_all_in=False, is_eliminated=False,
    )
    defaults.update(kwargs)
    return PlayerState(**defaults)


def _make_state(config: GameConfig, players, **kwargs) -> GameState:
    blind_level = config.blind_schedule.level_for_hand(0)
    defaults = dict(
        hand_number=0, street=Street.PREFLOP, dealer_seat=0,
        players=tuple(players), community_cards=(),
        pots=[Pot(amount=0, eligible_seats=frozenset(range(len(players))))],
        current_bet_to_call=0, last_raise_size=0,
        action_history_this_street=[], action_history_this_hand=[],
        deck_remaining_count=52, config=config, blind_level=blind_level,
        action_on_seat=0,
    )
    defaults.update(kwargs)
    return GameState(**defaults)


def test_game_state_construction() -> None:
    config = _make_config()
    state = _make_state(config, [_make_player(0), _make_player(1)])
    assert state.hand_number == 0
    assert state.street == Street.PREFLOP
    assert len(state.players) == 2


def test_game_state_frozen() -> None:
    config = _make_config()
    state = _make_state(config, [_make_player(0), _make_player(1)])
    with pytest.raises(AttributeError):
        state.hand_number = 1  # type: ignore


def test_game_state_view_for_hides_opponent_cards() -> None:
    config = _make_config()
    cards_alice = (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
    cards_bob = (Card(Rank.TWO, Suit.CLUBS), Card(Rank.THREE, Suit.DIAMONDS))
    players = [_make_player(0, hole_cards=cards_alice), _make_player(1, hole_cards=cards_bob)]
    state = _make_state(config, players)

    view = state.view_for(0)
    assert view.players[0].hole_cards == cards_alice
    assert view.players[1].hole_cards == ()
