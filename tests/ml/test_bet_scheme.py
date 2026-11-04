"""Pot-fraction bet buckets and legacy duplicate masking."""

import numpy as np

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import ActionType
from poker.domain.card import Card, Rank, Suit
from poker.engine.action_validator import validate
from poker.ml.action_space import (
    BetScheme,
    action_index_to_action,
    build_action_mask,
)
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


def _state(bet: int = 20, commit: tuple[int, int] = (20, 20), street: Street = Street.FLOP) -> GameState:
    config = GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        ante=0,
        rake_percent=0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(10, 20, 0)],
            hands_per_level=10,
            fixed=True,
        ),
        run_it_twice=False,
    )
    players = []
    for seat, committed in enumerate(commit):
        players.append(
            PlayerState(
                seat=seat,
                name=f"P{seat}",
                stack=1000 - committed,
                hole_cards=(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS)),
                committed_this_street=committed,
                committed_this_hand=committed,
                has_folded=False,
                is_all_in=False,
                is_eliminated=False,
            )
        )
    return GameState(
        hand_number=0,
        street=street,
        dealer_seat=0,
        players=tuple(players),
        community_cards=(),
        pots=[Pot(sum(commit), frozenset({0, 1}))],
        current_bet_to_call=bet,
        last_raise_size=20,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=48,
        config=config,
        blind_level=BlindLevel(10, 20, 0),
        action_on_seat=0,
    )


def test_legacy_dedup_drops_identical_raise_buckets() -> None:
    state = _state()
    raw = build_action_mask(state, 0, scheme=BetScheme.LEGACY, dedup=False)
    deduped = build_action_mask(state, 0, scheme=BetScheme.LEGACY, dedup=True)
    assert int(raw[2:].sum()) >= int(deduped[2:].sum())
    amounts = []
    for index in range(2, 7):
        if not deduped[index]:
            continue
        action = action_index_to_action(index, state, 0, scheme=BetScheme.LEGACY)
        amounts.append((action.type, action.amount))
    assert len(amounts) == len(set(amounts))
    assert amounts


def test_pot_fraction_amounts_are_legal_and_unique() -> None:
    state = _state(bet=20, commit=(0, 20), street=Street.PREFLOP)
    state = GameState(
        hand_number=state.hand_number,
        street=Street.PREFLOP,
        dealer_seat=0,
        players=state.players,
        community_cards=(),
        pots=state.pots,
        current_bet_to_call=20,
        last_raise_size=20,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=48,
        config=state.config,
        blind_level=state.blind_level,
        action_on_seat=0,
    )
    mask = build_action_mask(state, 0, scheme=BetScheme.POT_FRACTION)
    amounts = []
    for index in np.where(mask)[0]:
        action = action_index_to_action(int(index), state, 0, scheme=BetScheme.POT_FRACTION)
        validate(state, 0, action)
        if action.type in (ActionType.RAISE, ActionType.ALL_IN):
            amounts.append(action.amount)
    assert len(amounts) == len(set(amounts))
    assert amounts
    open_to = action_index_to_action(2, state, 0, scheme=BetScheme.POT_FRACTION)
    assert open_to.amount == 50
