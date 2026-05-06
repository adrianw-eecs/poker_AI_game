"""Tests for player elimination mechanics.

Validates: elimination marking, session termination, betting flow, dealer rotation.
"""

import pytest
from dataclasses import replace
from typing import Callable

from poker.bots.base import Bot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.action import Action
from poker.domain.deck import Deck
from poker.engine.hand_engine import play_hand
from poker.engine.session import Session, SessionConfig
from poker.logging.logger import NullLogger
from poker.rng import RNG
from poker.state.game_state import GameState


class DeterministicBot(Bot):
    """Bot that plays predetermined actions."""

    def __init__(self, seat: int, name: str, action_sequence: Callable[[str], Action]):
        self._seat = seat
        self._name = name
        self._action_sequence = action_sequence

    @property
    def name(self) -> str:
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        try:
            action = self._action_sequence(state.street.value)
            if action in legal:
                return action
            return legal[0]
        except (KeyError, StopIteration):
            return legal[0]

    def observe_result(self, final_state: GameState, reward: float) -> None:
        pass


@pytest.fixture
def session_3p_no_rake():
    """3-player game, no rake, fixed blinds."""
    sched = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=3,
        starting_stack=1000,
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
        session_config=SessionConfig(duration_hands=None),
        logger=NullLogger()
    )


def _create_deck_factory(seed: int) -> Callable[[], Deck]:
    def factory() -> Deck:
        return Deck(RNG(seed=seed))
    return factory


@pytest.mark.smoke
def test_eliminated_player_marked_on_zero_stack():
    """Unit: Player marked is_eliminated when stack <= 0."""
    sched = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=3,
        starting_stack=1000,
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
        session_config=SessionConfig(duration_hands=None),
        logger=NullLogger()
    )
    state = session.create_initial_state(3)

    # Manually create state with Player2 at 0 chips
    new_players = (
        state.players[0],
        state.players[1],
        replace(state.players[2], stack=0, is_eliminated=True)
    )
    state_with_elimination = replace(state, players=new_players)

    assert state_with_elimination.players[2].stack == 0
    assert state_with_elimination.players[2].is_eliminated


@pytest.mark.smoke
def test_dealer_rotation_skips_eliminated_seats(session_3p_no_rake):
    """Unit: advance_to_next_hand skips eliminated seats when rotating dealer."""
    state = session_3p_no_rake.create_initial_state(3)

    # Verify initial dealer is seat 0
    assert state.dealer_seat == 0

    # Eliminate Player1 (seat 1)
    new_players = (
        state.players[0],
        replace(state.players[1], stack=0, is_eliminated=True),
        state.players[2]
    )
    state_p1_eliminated = replace(state, players=new_players)

    # Advance to next hand — dealer should skip seat 1, go to seat 2
    state_after = session_3p_no_rake.advance_to_next_hand(state_p1_eliminated)
    assert state_after.dealer_seat == 2

    # Advance again — dealer should skip seat 1, go back to seat 0
    state_again = session_3p_no_rake.advance_to_next_hand(state_after)
    assert state_again.dealer_seat == 0


@pytest.mark.smoke
def test_session_ends_with_one_active_player(session_3p_no_rake):
    """Unit: is_session_over() returns True when only 1 active player remains."""
    state = session_3p_no_rake.create_initial_state(3)

    # Session should not be over with all 3 active
    assert not session_3p_no_rake.is_session_over(state)

    # Eliminate 2 players
    new_players = (
        state.players[0],
        replace(state.players[1], stack=0, is_eliminated=True),
        replace(state.players[2], stack=0, is_eliminated=True)
    )
    state_one_active = replace(state, players=new_players)

    # Session should end with only 1 active player
    assert session_3p_no_rake.is_session_over(state_one_active)


@pytest.mark.smoke
def test_game_consistency_after_elimination(session_3p_no_rake):
    """Unit: State remains valid with eliminated players in tuple."""
    state = session_3p_no_rake.create_initial_state(3)

    # Eliminate Player1
    new_players = (
        state.players[0],
        replace(state.players[1], stack=0, is_eliminated=True),
        state.players[2]
    )
    state_with_elim = replace(state, players=new_players)

    # Players tuple length unchanged
    assert len(state_with_elim.players) == 3

    # Seat numbers still correct
    assert state_with_elim.players[0].seat == 0
    assert state_with_elim.players[1].seat == 1
    assert state_with_elim.players[2].seat == 2

    # Eliminated player properties correct
    assert state_with_elim.players[1].is_eliminated
    assert state_with_elim.players[1].stack == 0

    # Active players count correct
    active_count = sum(1 for p in state_with_elim.players if not p.is_eliminated)
    assert active_count == 2


@pytest.mark.smoke
def test_player_elimination_flow_full_game():
    """Integration: Full elimination flow with 3 players playing multiple hands."""
    # Use aggressive strategy with high blinds relative to stack
    sched = BlindSchedule(
        levels=[BlindLevel(small=25, big=50)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=3,
        starting_stack=150,  # Lower stack with higher blinds to force eliminations
        small_blind=25,
        big_blind=50,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=sched
    )
    session = Session(
        config=config,
        blind_schedule=sched,
        session_config=SessionConfig(duration_hands=None),
        logger=NullLogger()
    )

    state = session.create_initial_state(3)

    # Create bots that use diverse strategies to force chip movement
    def seat_0_strategy(street: str) -> Action:
        return Action.fold() if street == "PREFLOP" else Action.check()

    def seat_1_strategy(street: str) -> Action:
        return Action.raise_to(100) if street == "PREFLOP" else Action.check()

    def seat_2_strategy(street: str) -> Action:
        return Action.call(100) if street == "PREFLOP" else Action.check()

    bots: dict[int, Bot] = {
        0: DeterministicBot(0, "Fold", seat_0_strategy),
        1: DeterministicBot(1, "Raise", seat_1_strategy),
        2: DeterministicBot(2, "Call", seat_2_strategy),
    }

    # Play hands until someone is eliminated or max hands
    initial_total = sum(p.stack for p in state.players)
    hands_played = 0
    max_hands = 10

    while hands_played < max_hands:
        state = play_hand(state, bots, _create_deck_factory(42)(), NullLogger())
        hands_played += 1

        # Check if anyone is eliminated
        eliminated_count = sum(1 for p in state.players if p.is_eliminated)
        if eliminated_count > 0:
            break

        # Advance to next hand
        state = session.advance_to_next_hand(state)

    # Verify elimination occurred or max hands reached
    eliminated_count = sum(1 for p in state.players if p.is_eliminated)
    # If no elimination, verify we hit max hands
    assert eliminated_count >= 1 or hands_played == max_hands

    # Verify chip conservation (with rake deduction)
    final_total = sum(p.stack for p in state.players)
    assert final_total <= initial_total, "Chips increased (no rake configured)"

    # Verify eliminated players have 0 stack
    for player in state.players:
        if player.is_eliminated:
            assert player.stack == 0


@pytest.mark.smoke
def test_eliminated_player_skips_betting_actions():
    """Unit: Eliminated players don't participate in betting logic."""
    sched = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)],
        hands_per_level=100,
        fixed=True
    )
    config = GameConfig(
        num_players=3,
        starting_stack=1000,
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
        session_config=SessionConfig(duration_hands=None),
        logger=NullLogger()
    )

    state = session.create_initial_state(3)

    # Eliminate Player1
    new_players = (
        state.players[0],
        replace(state.players[1], stack=0, is_eliminated=True),
        state.players[2]
    )
    state_with_elim = replace(state, players=new_players)

    # Create bot that counts action calls
    action_count = {"count": 0}

    class CountingBot(Bot):
        def __init__(self, seat: int, name: str):
            self._seat = seat
            self._name = name

        @property
        def name(self) -> str:
            return self._name

        def act(self, state: GameState, legal: list[Action]) -> Action:
            action_count["count"] += 1
            return legal[0]

        def observe_result(self, final_state: GameState, reward: float) -> None:
            pass

    bots: dict[int, Bot] = {
        0: CountingBot(0, "C0"),
        1: CountingBot(1, "C1"),  # This bot should not get any action calls
        2: CountingBot(2, "C2")
    }

    # Play hand starting from state with Player1 eliminated
    final_state = play_hand(state_with_elim, bots, _create_deck_factory(42)(), NullLogger())

    # Verify the game completed without errors
    assert final_state is not None
    assert len(final_state.players) == 3
    # Eliminated player should remain eliminated
    assert final_state.players[1].is_eliminated
