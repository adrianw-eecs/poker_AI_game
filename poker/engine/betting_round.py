"""Single street betting round orchestration."""

import time
from collections.abc import Callable
from dataclasses import replace

from poker.domain.action import Action, ActionType
from poker.logging.events import ActionTaken, PlayerOut, TimingEvent
from poker.logging.logger import Logger
from poker.logging.state_snapshot import position_label
from poker.engine.rules import (
    advance_action,
    apply_action,
    count_active_players,
    focus_action,
    reopens_betting,
    round_is_closed,
)
from poker.state.game_state import GameState
from poker.state.pot_manager import rebuild_pots, return_uncalled_bet


def _action_label(action: Action, had_bet: bool) -> str:
    """Map an engine action to a hand-history verb.

    The first voluntary raise on a street with no bet yet is a bet.
    The stored action_type stays ``raise``.
    """
    if action.type == ActionType.RAISE and not had_bet:
        return "bet"
    return action.type.value


class BettingRound:
    """Orchestrates a single street's betting until it closes.

    A betting round ends when all active (non-folded) players have:
    1. Had an opportunity to act at least once
    2. All matched the highest bet (or gone all-in, or folded)

    The half-raise rule is enforced: an all-in shorter than a full minimum raise
    does not re-open action for players who have already acted.
    """

    def __init__(
        self,
        state: GameState,
        get_action: Callable[[int, GameState], Action],
        logger: Logger,
        names: dict[int, str] | None = None,
    ) -> None:
        """Initialize a betting round.

        Args:
            state: The starting GameState for this street.
            get_action: Callable that takes (seat, state) and returns an Action.
                        Called once per player per street until round closes.
            logger: Logger to emit ActionTaken events.
            names: Optional seat → display name. Falls back to PlayerState.name.
        """
        self.state = state
        self.get_action = get_action
        self.logger = logger
        self.names = names or {}
        # Track which players have acted at least once in this round
        self.players_acted_this_round: set[int] = set()
        # Track the highest bet to know when all players have matched
        self.max_commitment: int = max(
            (p.committed_this_street for p in state.players),
            default=0
        )
        self.action_count_this_street: int = 0

    def run(self) -> GameState:
        """Execute the betting round until it closes.

        Returns:
            Updated GameState after the round completes.
        """
        # If only 1 or fewer active players, round is already closed
        if self._count_active_players(self.state) <= 1:
            return replace(self.state, action_on_seat=None)

        # Ensure action starts with a player who can act (not folded/eliminated/all-in)
        self.state = self._set_initial_action_on_seat(self.state)

        # Main betting loop
        while not self._round_is_closed():
            if self.state.action_on_seat is None:
                # All remaining players are all-in or folded
                break

            seat = self.state.action_on_seat
            actor = self.state.players[seat]
            committed_before = actor.committed_this_hand
            stack_before = actor.stack
            to_call_before = self.state.current_bet_to_call
            had_bet = to_call_before > 0
            t0 = time.perf_counter()
            action = self.get_action(seat, self.state)
            elapsed_us = int((time.perf_counter() - t0) * 1_000_000)
            self.logger.log_event(TimingEvent(
                phase="bot_action",
                seat=seat,
                elapsed_us=elapsed_us,
                hand_number=self.state.hand_number,
            ))

            # Track this player as having acted this round
            self.players_acted_this_round.add(seat)

            # Apply the action to the state. Reopen checks use the minimum
            # that was in force before a short all-in can rewrite it.
            prior_max = self.max_commitment
            prior_min = max(self.state.config.big_blind, self.state.last_raise_size)
            self.state = self._apply_action(self.state, seat, action)
            self.state = rebuild_pots(self.state)
            chips_added = (
                self.state.players[seat].committed_this_hand - committed_before
            )
            pot_after = sum(p.amount for p in self.state.pots)
            self.action_count_this_street += 1

            # Update the max commitment (highest bet seen so far)
            self.max_commitment = max(
                self.max_commitment,
                max((p.committed_this_street for p in self.state.players), default=0)
            )

            # Check if this action re-opens betting (half-raise rule)
            reopens = self._reopens_betting(action, prior_max, prior_min)
            if reopens:
                # This is a full raise. Reset acted set so all others must act again.
                self.players_acted_this_round = {seat}  # Only raiser has acted

            num_players = len(self.state.players)
            name = self.names.get(seat, self.state.players[seat].name)
            self.logger.log_event(
                ActionTaken(
                    hand_number=self.state.hand_number,
                    street=self.state.street.value,
                    seat=seat,
                    action_type=action.type.value,
                    amount=action.amount,
                    chips_added=chips_added,
                    pot_after=pot_after,
                    action_num=self.action_count_this_street,
                    label=_action_label(action, had_bet),
                    to_call_before=to_call_before,
                    stack_before=stack_before,
                    stack_after=self.state.players[seat].stack,
                    all_in=self.state.players[seat].is_all_in,
                    reopens=reopens,
                    name=name,
                    position=position_label(seat, self.state.dealer_seat, num_players),
                )
            )
            if action.type == ActionType.FOLD:
                self.logger.log_event(
                    PlayerOut(
                        hand_number=self.state.hand_number,
                        seat=seat,
                        reason="fold",
                        street=self.state.street.value,
                        stack=self.state.players[seat].stack,
                        name=name,
                        position=position_label(seat, self.state.dealer_seat, num_players),
                    )
                )

        # Return chips no opponent matched before the pot is built.
        self.state = return_uncalled_bet(self.state)
        self.state = rebuild_pots(self.state)
        return replace(self.state, action_on_seat=None)

    def _round_is_closed(self) -> bool:
        """Check if the betting round is closed."""
        return round_is_closed(self.state, self.players_acted_this_round, self.max_commitment)

    def _apply_action(self, state: GameState, seat: int, action: Action) -> GameState:
        """Apply an action and move action to the next player who can bet."""
        state = apply_action(state, seat, action)
        return advance_action(state, seat)

    def _advance_action_on_seat(self, state: GameState, current_seat: int) -> GameState:
        """Advance to the next player who needs to act."""
        return advance_action(state, current_seat)

    def _set_initial_action_on_seat(self, state: GameState) -> GameState:
        """Set action to the first player from the current seat who can still bet."""
        return focus_action(state)

    def _count_active_players(self, state: GameState) -> int:
        """Count active (non-folded, non-eliminated) players."""
        return count_active_players(state)

    def _reopens_betting(self, action: Action, prior_max: int, prior_min: int) -> bool:
        """Check the half-raise rule against the minimum in force before the action."""
        return reopens_betting(action, prior_max, prior_min)
