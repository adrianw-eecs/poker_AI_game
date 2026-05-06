"""Tests for Session management."""

import pytest
from dataclasses import replace

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.engine.session import Session, SessionConfig
from poker.logging.logger import NullLogger


@pytest.fixture
def session_2p():
    sched = BlindSchedule(levels=[BlindLevel(small=5, big=10)], hands_per_level=10, fixed=True)
    config = GameConfig(num_players=2, starting_stack=1000, small_blind=5, big_blind=10,
                        ante=0, rake_percent=0.0, rake_cap=None, blind_schedule=sched)
    return Session(config=config, blind_schedule=sched,
                   session_config=SessionConfig(duration_hands=None), logger=NullLogger())


@pytest.fixture
def session_3p():
    sched = BlindSchedule(levels=[BlindLevel(small=5, big=10)], hands_per_level=10, fixed=True)
    config = GameConfig(num_players=3, starting_stack=1000, small_blind=5, big_blind=10,
                        ante=0, rake_percent=0.0, rake_cap=None, blind_schedule=sched)
    return Session(config=config, blind_schedule=sched,
                   session_config=SessionConfig(duration_hands=None), logger=NullLogger())


@pytest.mark.smoke
def test_chip_conservation(session_3p):
    state = session_3p.create_initial_state(3)
    assert sum(p.stack for p in state.players) == 3000
    state2 = session_3p.advance_to_next_hand(
        replace(state, players=(
            replace(state.players[0], stack=1500),
            replace(state.players[1], stack=800),
            replace(state.players[2], stack=700),
        ))
    )
    assert sum(p.stack for p in state2.players) == 3000


@pytest.mark.smoke
def test_button_rotation(session_3p):
    s1 = session_3p.create_initial_state(3)
    assert s1.dealer_seat == 0
    s2 = session_3p.advance_to_next_hand(s1)
    assert s2.dealer_seat == 1
    s3 = session_3p.advance_to_next_hand(s2)
    assert s3.dealer_seat == 2
    s4 = session_3p.advance_to_next_hand(s3)
    assert s4.dealer_seat == 0


@pytest.mark.smoke
def test_session_ends_when_one_player_remains(session_2p):
    state = session_2p.create_initial_state(2)
    state_elim = replace(state, players=(
        state.players[0],
        replace(state.players[1], is_eliminated=True),
    ))
    assert session_2p.is_session_over(state_elim)
    assert not session_2p.is_session_over(state)


def test_hand_limit_ends_session(session_2p):
    sched = BlindSchedule(levels=[BlindLevel(small=5, big=10)], hands_per_level=10, fixed=True)
    config = GameConfig(num_players=2, starting_stack=1000, small_blind=5, big_blind=10,
                        ante=0, rake_percent=0.0, rake_cap=None, blind_schedule=sched)
    session = Session(config=config, blind_schedule=sched,
                      session_config=SessionConfig(duration_hands=10), logger=NullLogger())
    state = session.create_initial_state(2)
    assert not session.is_session_over(replace(state, hand_number=9))
    assert session.is_session_over(replace(state, hand_number=10))
