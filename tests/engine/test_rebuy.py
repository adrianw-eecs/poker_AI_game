"""Tests for rebuy game feature."""

import pytest
from dataclasses import replace

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.engine.session import Session, SessionConfig
from poker.logging.logger import NullLogger
from poker.state.game_state import GameState


@pytest.fixture
def session_2p_rebuy():
    """2-player game with rebuy enabled."""
    sched = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=2,
        starting_stack=100,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=sched
    )
    return Session(
        config=config,
        blind_schedule=sched,
        session_config=SessionConfig(duration_hands=None, rebuy_enabled=True),
        logger=NullLogger()
    )


@pytest.mark.smoke
def test_rebuy_resets_eliminated_player_stack(session_2p_rebuy):
    """Unit: _apply_rebuys() resets eliminated player to starting_stack."""
    state = session_2p_rebuy.create_initial_state(2)

    # Eliminate Player1
    new_players = (
        state.players[0],
        replace(state.players[1], stack=0, is_eliminated=True)
    )
    state_elim = replace(state, players=new_players)

    # Apply rebuy
    state_after_rebuy = session_2p_rebuy._apply_rebuys(state_elim)

    # Verify Player1 is reset
    assert state_after_rebuy.players[1].stack == 100  # starting_stack
    assert not state_after_rebuy.players[1].is_eliminated


@pytest.mark.smoke
def test_rebuy_only_applied_to_eliminated():
    """Unit: _apply_rebuys() only affects eliminated players."""
    sched = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=2,
        starting_stack=100,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=sched
    )
    session = Session(
        config=config,
        blind_schedule=sched,
        session_config=SessionConfig(duration_hands=None, rebuy_enabled=True),
        logger=NullLogger()
    )

    state = session.create_initial_state(2)

    # Player0 has 50 (reduced stack), Player1 eliminated
    new_players = (
        replace(state.players[0], stack=50),
        replace(state.players[1], stack=0, is_eliminated=True)
    )
    state_mixed = replace(state, players=new_players)

    # Apply rebuy
    state_after = session._apply_rebuys(state_mixed)

    # Verify only eliminated player reset
    assert state_after.players[0].stack == 50  # unchanged
    assert not state_after.players[0].is_eliminated
    assert state_after.players[1].stack == 100  # reset
    assert not state_after.players[1].is_eliminated


@pytest.mark.smoke
def test_rebuy_disabled_by_default():
    """Unit: Rebuy disabled when rebuy_enabled=False."""
    sched = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=2,
        starting_stack=100,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=sched
    )
    session = Session(
        config=config,
        blind_schedule=sched,
        session_config=SessionConfig(duration_hands=None, rebuy_enabled=False),
        logger=NullLogger()
    )

    state = session.create_initial_state(2)

    # Eliminate Player1
    new_players = (
        state.players[0],
        replace(state.players[1], stack=0, is_eliminated=True)
    )
    state_elim = replace(state, players=new_players)

    # Apply rebuy (should do nothing)
    state_after = session._apply_rebuys(state_elim)

    # Verify no change (still eliminated)
    assert state_after.players[1].stack == 0
    assert state_after.players[1].is_eliminated


@pytest.mark.smoke
def test_custom_rebuy_stack():
    """Unit: Custom rebuy_stack amount is used."""
    sched = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=2,
        starting_stack=100,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=sched
    )
    session = Session(
        config=config,
        blind_schedule=sched,
        session_config=SessionConfig(duration_hands=None, rebuy_enabled=True, rebuy_stack=200),
        logger=NullLogger()
    )

    state = session.create_initial_state(2)

    # Eliminate Player1
    new_players = (
        state.players[0],
        replace(state.players[1], stack=0, is_eliminated=True)
    )
    state_elim = replace(state, players=new_players)

    # Apply rebuy
    state_after = session._apply_rebuys(state_elim)

    # Verify custom rebuy_stack used
    assert state_after.players[1].stack == 200  # custom rebuy_stack
    assert not state_after.players[1].is_eliminated
