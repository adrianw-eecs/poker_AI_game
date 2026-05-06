"""Multi-hand session management with blind progression."""

from dataclasses import dataclass, replace
from typing import Callable, Optional

from poker.bots.base import Bot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.deck import Deck
from poker.engine.action_handler import ActionHandler
from poker.engine.hand_engine import play_hand
from poker.logging.logger import Logger
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState


@dataclass(frozen=True)
class SessionConfig:
    """Configuration for a session.

    Attributes:
        duration_hands: Maximum number of hands to play (or None for unlimited).
        duration_seconds: Maximum session duration in seconds (or None for unlimited).
        rebuy_enabled: Enable automatic rebuy when players hit 0 chips (default False).
        rebuy_stack: Stack amount to rebuy to (default to starting_stack if None).
    """

    duration_hands: int | None = None
    duration_seconds: float | None = None
    rebuy_enabled: bool = False
    rebuy_stack: int | None = None


@dataclass
class Session:
    """Multi-hand session management with blind progression.

    Manages the progression of multiple hands, including:
    - Button rotation: (dealer_seat + 1) % num_players
    - Blind progression: Updates blinds and antes per blind schedule
    - Player elimination: Removes players with zero stack
    - Stack tracking: Maintains long-term chip counts
    - Event logging: Logs hand start/end events
    - Action handling: Validates bot actions with retries and logging

    Attributes:
        config: The game configuration.
        blind_schedule: The blind schedule for progression.
        session_config: Session constraints (duration).
        logger: Logger for events.
        action_handler: Optional ActionHandler for bot action validation.
    """

    config: GameConfig
    blind_schedule: BlindSchedule
    session_config: SessionConfig
    logger: Logger
    action_handler: Optional[ActionHandler] = None

    def create_initial_state(self, num_players: int) -> GameState:
        """Create the initial game state for the session.

        All players start with the configured starting_stack.
        Dealer button is at seat 0.
        First blind level is used.

        Args:
            num_players: Number of players for this session.

        Returns:
            Initial GameState ready for the first hand.
        """
        if num_players != self.config.num_players:
            raise ValueError(
                f"Expected {self.config.num_players} players, got {num_players}"
            )

        blind_level = self.blind_schedule.level_for_hand(0)

        # Initialize players with full stacks
        players = tuple(
            PlayerState(
                seat=seat,
                name=f"Player{seat + 1}",
                stack=self.config.starting_stack,
                hole_cards=(),
                committed_this_street=0,
                committed_this_hand=0,
                has_folded=False,
                is_all_in=False,
                is_eliminated=False,
            )
            for seat in range(num_players)
        )

        return GameState(
            hand_number=0,
            street=Street.PREFLOP,
            dealer_seat=0,
            players=players,
            community_cards=(),
            pots=[],
            current_bet_to_call=0,
            last_raise_size=0,
            action_history_this_street=[],
            action_history_this_hand=[],
            deck_remaining_count=52,
            config=self.config,
            blind_level=blind_level,
            action_on_seat=None,
        )

    def advance_to_next_hand(self, state: GameState) -> GameState:
        """Advance to the next hand after the current one finishes.

        Updates:
        - hand_number: Increments by 1
        - dealer_seat: Rotates to next player
        - blind_level: Gets level for new hand number
        - players: Resets fold/all-in flags, clears hole cards
        - street: Reset to PREFLOP
        - Removes eliminated players from the players tuple

        Args:
            state: The final game state from the previous hand.

        Returns:
            Initial GameState for the next hand.
        """
        next_hand_number = state.hand_number + 1

        # Rotate button
        next_dealer_seat = (state.dealer_seat + 1) % len(state.players)

        # Get blind level for next hand
        blind_level = self.blind_schedule.level_for_hand(next_hand_number)

        # Reset players: clear fold/all-in flags, clear hole cards
        # NOTE: Do NOT filter out eliminated players - keep them with their original seat numbers
        # This keeps the bots dictionary synchronized with player seats
        reset_players = tuple(
            replace(
                player,
                has_folded=False,
                is_all_in=False,
                hole_cards=(),
                committed_this_street=0,
                committed_this_hand=0,
            )
            for player in state.players
        )

        # Get count of active (non-eliminated) players for dealer rotation
        active_players = sum(1 for p in reset_players if not p.is_eliminated)

        # Dealer seat rotation: find next active player
        if active_players > 1:
            new_dealer_seat = next_dealer_seat
            while new_dealer_seat < len(reset_players):
                if not reset_players[new_dealer_seat].is_eliminated:
                    break
                new_dealer_seat += 1
            if new_dealer_seat >= len(reset_players):
                new_dealer_seat = 0
                # Find first active player
                while new_dealer_seat < len(reset_players):
                    if not reset_players[new_dealer_seat].is_eliminated:
                        break
                    new_dealer_seat += 1
            next_dealer_seat = new_dealer_seat
        else:
            next_dealer_seat = 0

        return GameState(
            hand_number=next_hand_number,
            street=Street.PREFLOP,
            dealer_seat=next_dealer_seat,
            players=reset_players,
            community_cards=(),
            pots=[],
            current_bet_to_call=0,
            last_raise_size=0,
            action_history_this_street=[],
            action_history_this_hand=[],
            deck_remaining_count=52,
            config=self.config,
            blind_level=blind_level,
            action_on_seat=None,
        )

    def is_session_over(self, state: GameState, elapsed_seconds: float = 0.0) -> bool:
        """Check if the session should end.

        Returns True if:
        - Only 1 player remains (others eliminated)
        - duration_hands limit reached
        - duration_seconds limit exceeded

        Args:
            state: The current game state.
            elapsed_seconds: Elapsed time in seconds (only checked if duration_seconds is set).

        Returns:
            True if session should end, False otherwise.
        """
        # Check for only 1 player remaining
        active_players = sum(1 for p in state.players if not p.is_eliminated)
        if active_players <= 1:
            return True

        # Check hand count limit
        if (
            self.session_config.duration_hands is not None
            and state.hand_number >= self.session_config.duration_hands
        ):
            return True

        # Check time limit
        if (
            self.session_config.duration_seconds is not None
            and elapsed_seconds >= self.session_config.duration_seconds
        ):
            return True

        return False

    def _apply_rebuys(self, state: GameState) -> GameState:
        """Reset eliminated players' stacks to starting_stack (rebuy).

        Args:
            state: Current game state.

        Returns:
            Updated game state with rebuyed players.
        """
        if not self.session_config.rebuy_enabled:
            return state

        players_list = list(state.players)
        for seat in range(len(state.players)):
            player = state.players[seat]
            if player.is_eliminated and player.stack == 0:
                rebuy_amount = self.session_config.rebuy_stack or self.config.starting_stack
                players_list[seat] = player.with_stack(rebuy_amount).with_eliminated(False)

        return replace(state, players=tuple(players_list))

    def run(
        self,
        state: GameState,
        bots: dict[int, Bot],
        deck_factory: Callable[[], Deck],
    ) -> GameState:
        """Run the session loop, playing hands until termination.

        Args:
            state: The initial game state (from create_initial_state or loaded).
            bots: Dict mapping seat → Bot for action selection.
            deck_factory: Callable to create a new shuffled Deck for each hand.

        Returns:
            Final GameState after session ends.
        """
        import time

        start_time = time.time()

        while not self.is_session_over(state, time.time() - start_time):
            # Play one hand
            deck = deck_factory()
            state = play_hand(state, bots, deck, self.logger, self.action_handler)

            # Reset action handler attempt counters for next hand
            if self.action_handler:
                self.action_handler.reset()

            # Check if session should end
            if self.is_session_over(state, time.time() - start_time):
                break

            # Apply rebuys before advancing to next hand
            if self.session_config.rebuy_enabled:
                state = self._apply_rebuys(state)

            # Advance to next hand
            state = self.advance_to_next_hand(state)

        # Flush and close action handler
        if self.action_handler:
            self.action_handler.flush_and_close()

        return state

    def get_final_results(self, state: GameState) -> dict[int, int]:
        """Get final stack sizes at the end of the session.

        Args:
            state: The final game state.

        Returns:
            Dict mapping seat → final stack size.
        """
        return {player.seat: player.stack for player in state.players}

    def get_eliminations(self, state: GameState) -> list[tuple[int, int]]:
        """Get eliminations list (seat, hand_number).

        Args:
            state: The final game state.

        Returns:
            List of (seat, hand_number_eliminated) tuples.
        """
        # Note: To fully implement this, we'd need to track elimination events.
        # For now, return empty list (would be populated by tracking HandEnded events).
        return []
