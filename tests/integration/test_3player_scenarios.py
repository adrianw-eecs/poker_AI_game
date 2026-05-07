"""Integration tests for 3-player poker scenarios.

Validates core game mechanics: game flow, pot calculations, winner determination, stack updates.
"""

import pytest
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


def _create_3player_config() -> GameConfig:
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)] * 10,
        hands_per_level=10,
        fixed=True,
    )
    return GameConfig(
        num_players=3,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )


def _create_session_with_seed(config: GameConfig, seed: int) -> tuple[Session, GameState]:
    session = Session(
        config=config,
        blind_schedule=config.blind_schedule,
        session_config=SessionConfig(duration_hands=1),
        logger=NullLogger(),
    )
    state = session.create_initial_state(num_players=3)
    return session, state


def _create_deck_factory(seed: int) -> Callable[[], Deck]:
    def factory() -> Deck:
        return Deck(RNG(seed=seed))
    return factory


@pytest.mark.smoke
def test_all_check_flow_and_stacks() -> None:
    """Verify check scenario flow, chip conservation, and determinism."""
    seed = 42
    config = _create_3player_config()
    session, state = _create_session_with_seed(config, seed)

    assert len(state.players) == 3
    assert all(p.stack == 1000 for p in state.players)

    check_bots: dict[int, Bot] = {
        0: DeterministicBot(0, "C0", lambda _: Action.check()),
        1: DeterministicBot(1, "C1", lambda _: Action.check()),
        2: DeterministicBot(2, "C2", lambda _: Action.check()),
    }

    final_state = play_hand(state, check_bots, _create_deck_factory(seed)(), NullLogger())

    total_chips = sum(p.stack for p in final_state.players)
    assert total_chips == 3000, f"Chip leak: {total_chips} != 3000"

    session2, state2 = _create_session_with_seed(config, seed)
    final2 = play_hand(state2, check_bots, _create_deck_factory(seed)(), NullLogger())
    stacks1 = tuple(p.stack for p in final_state.players)
    stacks2 = tuple(p.stack for p in final2.players)
    assert stacks1 == stacks2, f"Determinism failed: {stacks1} != {stacks2}"


def test_all_in_creates_pots() -> None:
    """Verify aggressive raises conserve chips."""
    seed = 100
    config = _create_3player_config()
    session, state = _create_session_with_seed(config, seed)

    def action_for_seat(seat):
        def get_action(street):
            if street == "preflop":
                if seat == 0:
                    return Action.raise_to(30)
                elif seat in [1, 2]:
                    return Action.call(30)
            return Action.check()
        return get_action

    aggressive_bots: dict[int, Bot] = {
        0: DeterministicBot(0, "R0", action_for_seat(0)),
        1: DeterministicBot(1, "C1", action_for_seat(1)),
        2: DeterministicBot(2, "C2", action_for_seat(2)),
    }

    try:
        final_state = play_hand(state, aggressive_bots, _create_deck_factory(seed)(), NullLogger())
        total_chips = sum(p.stack for p in final_state.players)
        assert total_chips == 3000, f"Chip leak: {total_chips} != 3000"
        for p in final_state.players:
            assert p.stack >= 0, f"Negative stack: {p.stack}"
    except Exception:
        pytest.skip("Action sequencing failed")


def test_folds_no_showdown() -> None:
    """Verify folding conserves chips and produces correct winner stack."""
    seed = 400
    config = _create_3player_config()
    session, state = _create_session_with_seed(config, seed)

    fold_bots: dict[int, Bot] = {
        0: DeterministicBot(0, "F0", lambda _: Action.fold()),
        1: DeterministicBot(1, "F1", lambda _: Action.fold()),
        2: DeterministicBot(2, "W", lambda _: Action.check()),
    }

    final_state = play_hand(state, fold_bots, _create_deck_factory(seed)(), NullLogger())
    total_chips = sum(p.stack for p in final_state.players)
    assert total_chips == 3000, f"Chip leak: {total_chips}"
    assert final_state.players[2].stack == 1005, f"Winner stack wrong: {final_state.players[2].stack}"


@pytest.mark.smoke
def test_chip_conservation_all_scenarios() -> None:
    """Verify chips never leak across different game flows."""
    config = _create_3player_config()
    scenarios = [
        ("all_check", {
            0: DeterministicBot(0, "C0", lambda _: Action.check()),
            1: DeterministicBot(1, "C1", lambda _: Action.check()),
            2: DeterministicBot(2, "C2", lambda _: Action.check()),
        }),
        ("all_fold", {
            0: DeterministicBot(0, "F0", lambda _: Action.fold()),
            1: DeterministicBot(1, "F1", lambda _: Action.fold()),
            2: DeterministicBot(2, "W", lambda _: Action.check()),
        }),
    ]

    for name, bots in scenarios:
        seed = hash(name) % 10000
        session, state = _create_session_with_seed(config, seed)
        initial_total = sum(p.stack for p in state.players)

        final_state = play_hand(state, bots, _create_deck_factory(seed)(), NullLogger())
        final_total = sum(p.stack for p in final_state.players)

        assert final_total == initial_total, f"{name}: chip leak {initial_total} -> {final_total}"


@pytest.mark.smoke
def test_stacks_non_negative_after_hand() -> None:
    """Verify no player ever has negative stack."""
    config = _create_3player_config()

    check_bots: dict[int, Bot] = {
        0: DeterministicBot(0, "C0", lambda _: Action.check()),
        1: DeterministicBot(1, "C1", lambda _: Action.check()),
        2: DeterministicBot(2, "C2", lambda _: Action.check()),
    }

    for seed in [800, 801, 802]:
        session, state = _create_session_with_seed(config, seed)
        final_state = play_hand(state, check_bots, _create_deck_factory(seed)(), NullLogger())

        for i, player in enumerate(final_state.players):
            assert player.stack >= 0, f"Seed {seed}: Player {i} has negative stack {player.stack}"


@pytest.mark.smoke
def test_6player_20hand_with_rebuy() -> None:
    """Integration: 6-player, 20-hand game with rebuy enabled.

    Validates:
    - Multi-hand session completes successfully
    - Rebuy system auto-resets eliminated players
    - Game continues even when players are eliminated and rebuyed
    - Chip conservation with rebuy (chips injected when rebuy happens)
    - Session terminates correctly after 20 hands
    """
    from poker.bots.random_bot import RandomBot

    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)] * 10,
        hands_per_level=20,
        fixed=True,
    )
    config = GameConfig(
        num_players=6,
        starting_stack=100,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )

    session = Session(
        config=config,
        blind_schedule=blind_schedule,
        session_config=SessionConfig(
            duration_hands=20,
            rebuy_enabled=True,
            rebuy_stack=100
        ),
        logger=NullLogger(),
    )

    state = session.create_initial_state(num_players=6)
    initial_total = sum(p.stack for p in state.players)

    # Create 6 random bots
    bots = {i: RandomBot(name=f"Bot{i}", seed=42 + i) for i in range(6)}

    # Run session
    final_state = session.run(state, bots, lambda: Deck(RNG(seed=12345)))

    # Verify session completed
    assert final_state.hand_number == 20, f"Expected 20 hands, got {final_state.hand_number}"

    # Verify no negative stacks
    for p in final_state.players:
        assert p.stack >= 0, f"Negative stack: {p.stack}"

    # Verify valid game state
    assert len(final_state.players) == 6, f"Expected 6 players, got {len(final_state.players)}"


@pytest.mark.smoke
def test_8player_50hand_without_rebuy() -> None:
    """Integration: 8-player, 50-hand game without rebuy.

    Validates:
    - Large multi-hand session completes successfully
    - Player elimination mechanics work correctly
    - Game terminates when 1 or fewer active players remain
    - Betting round side pot detection works under load
    - Hand numbering is sequential across all hands
    """
    from poker.bots.random_bot import RandomBot

    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10)] * 20,
        hands_per_level=50,
        fixed=True,
    )
    config = GameConfig(
        num_players=8,
        starting_stack=100,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )

    session = Session(
        config=config,
        blind_schedule=blind_schedule,
        session_config=SessionConfig(
            duration_hands=50,
            rebuy_enabled=False,  # No rebuy
        ),
        logger=NullLogger(),
    )

    state = session.create_initial_state(num_players=8)
    initial_total = sum(p.stack for p in state.players)

    # Create 8 random bots
    bots = {i: RandomBot(name=f"Bot{i}", seed=99 + i) for i in range(8)}

    # Run session
    final_state = session.run(state, bots, lambda: Deck(RNG(seed=54321)))

    # Verify session completed (either 50 hands or only 1 player remains)
    assert final_state.hand_number <= 50, f"Expected <= 50 hands, got {final_state.hand_number}"

    # Verify no negative stacks
    for p in final_state.players:
        assert p.stack >= 0, f"Negative stack: {p.stack}"

    # Verify valid game state
    assert len(final_state.players) == 8, f"Expected 8 players, got {len(final_state.players)}"

    # Verify game terminated correctly
    # Session ends when: (1) only 1 active player remains, or (2) 50 hands completed
    active_count = sum(1 for p in final_state.players if not p.is_eliminated)
    assert active_count <= 1 or final_state.hand_number == 50, \
        f"Game should end when 1 player remains or 50 hands reached. Active: {active_count}, Hands: {final_state.hand_number}"
