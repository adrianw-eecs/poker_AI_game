"""Observation v2 shape, range, and hero-relative rotation."""

import numpy as np

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action
from poker.domain.card import Card, Rank, Suit
from poker.ml.observation import observation_spec
from poker.ml.observation_v2 import (
    EMPTY_CARD,
    OBS_SPEC_V2,
    OBS_V2_DIM,
    build_observation_v2,
    load_preflop_equity,
    preflop_combo_index,
)
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


def _config(n: int) -> GameConfig:
    return GameConfig(
        num_players=n,
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


def _player(seat: int, stack: int, hole: tuple[Card, Card]) -> PlayerState:
    return PlayerState(
        seat=seat,
        name=f"P{seat}",
        stack=stack,
        hole_cards=hole,
        committed_this_street=20,
        committed_this_hand=20,
        has_folded=False,
        is_all_in=False,
        is_eliminated=False,
    )


def _state() -> GameState:
    holes = (
        (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)),
        (Card(Rank.QUEEN, Suit.HEARTS), Card(Rank.QUEEN, Suit.DIAMONDS)),
        (Card(Rank.TEN, Suit.CLUBS), Card(Rank.NINE, Suit.CLUBS)),
    )
    players = [_player(i, 800 + 50 * i, holes[i]) for i in range(3)]
    return GameState(
        hand_number=1,
        street=Street.FLOP,
        dealer_seat=0,
        players=tuple(players),
        community_cards=(
            Card(Rank.TWO, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.DIAMONDS),
            Card(Rank.JACK, Suit.CLUBS),
        ),
        pots=[Pot(60, frozenset(range(3)))],
        current_bet_to_call=20,
        last_raise_size=20,
        action_history_this_street=[(1, Action.raise_to(40))],
        action_history_this_hand=[(1, Action.raise_to(40))],
        deck_remaining_count=47,
        config=_config(3),
        blind_level=BlindLevel(10, 20, 0),
        action_on_seat=2,
        street_history_ends=(),
        pots_entering_street=(0, 60),
    )


def test_observation_v2_shape_and_range() -> None:
    obs = build_observation_v2(_state(), seat=0)
    assert obs.shape == (OBS_V2_DIM,)
    assert obs.dtype == np.float32
    shape, dtype = observation_spec(2)
    assert shape == (OBS_V2_DIM,)
    assert dtype == "float32"
    ids = obs[OBS_SPEC_V2.card_ids[0] : OBS_SPEC_V2.card_ids[1]]
    assert ids.min() >= 0
    assert ids.max() <= EMPTY_CARD
    rest = np.concatenate(
        [
            obs[: OBS_SPEC_V2.card_ids[0]],
            obs[OBS_SPEC_V2.card_ids[1] :],
        ]
    )
    assert rest.min() >= -1e-6
    assert rest.max() <= 1.0 + 1e-6


def test_hero_relative_rotation() -> None:
    state = _state()
    spec = OBS_SPEC_V2
    first = build_observation_v2(state, 0)[spec.seats[0] : spec.seats[1]].reshape(9, spec.seat_dim)
    second = build_observation_v2(state, 1)[spec.seats[0] : spec.seats[1]].reshape(9, spec.seat_dim)
    # Slot k from seat 1 is the player one seat further than slot k from seat 0.
    np.testing.assert_allclose(second[0], first[1])
    np.testing.assert_allclose(second[1], first[2])
    np.testing.assert_allclose(second[2], first[0])
    assert second[3].sum() == 0


def test_preflop_combo_index_covers_169_classes() -> None:
    seen = set()
    for rank_a in Rank:
        for suit_a in Suit:
            for rank_b in Rank:
                for suit_b in Suit:
                    if rank_a == rank_b and suit_a == suit_b:
                        continue
                    seen.add(preflop_combo_index(Card(rank_a, suit_a), Card(rank_b, suit_b)))
    assert seen == set(range(169))
    pair = preflop_combo_index(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS))
    suited = preflop_combo_index(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES))
    offsuit = preflop_combo_index(Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
    assert pair == 12
    assert 13 <= suited <= 90
    assert 91 <= offsuit <= 168


def test_preflop_equity_table_ranks_aces_above_seven_deuce() -> None:
    table = load_preflop_equity()
    assert table is not None and table.shape == (169, 8)
    aces = preflop_combo_index(Card(Rank.ACE, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS))
    seven_two = preflop_combo_index(Card(Rank.SEVEN, Suit.SPADES), Card(Rank.TWO, Suit.HEARTS))
    assert table[aces, 0] > 0.8
    assert table[aces, 0] > table[seven_two, 0]

    state = _state()
    preflop = GameState(
        hand_number=0,
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
    obs = build_observation_v2(preflop, seat=0)
    equity = obs[OBS_SPEC_V2.hero[0] + 13]
    assert equity > 0.5
