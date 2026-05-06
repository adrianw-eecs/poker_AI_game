"""Tests for RandomBot."""

import pytest

from poker.bots.random_bot import RandomBot
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


@pytest.mark.smoke
def test_always_returns_legal_action() -> None:
    bot = RandomBot(seed=7)
    state = _state()
    legal = [Action.fold(), Action.call(10), Action.raise_to(20), Action.all_in(1000)]
    legal_types = {a.type for a in legal}
    for _ in range(100):
        assert bot.act(state, legal).type in legal_types


def test_fold_only_returns_fold() -> None:
    bot = RandomBot(seed=1)
    state = _state()
    for _ in range(30):
        assert bot.act(state, [Action.fold()]).type == ActionType.FOLD


def test_same_seed_reproducible() -> None:
    legal = [Action.fold(), Action.call(10), Action.raise_to(20)]
    state = _state()
    bot1, bot2 = RandomBot(seed=42), RandomBot(seed=42)
    for _ in range(50):
        a1, a2 = bot1.act(state, legal), bot2.act(state, legal)
        assert a1.type == a2.type and a1.amount == a2.amount


def test_all_action_types_reachable() -> None:
    bot = RandomBot(seed=13)
    state = _state()
    legal = [Action.fold(), Action.call(10), Action.raise_to(20), Action.all_in(1000)]
    seen = set()
    for _ in range(2000):
        seen.add(bot.act(state, legal).type)
        if len(seen) == len(legal):
            break
    assert len(seen) == len(legal)
