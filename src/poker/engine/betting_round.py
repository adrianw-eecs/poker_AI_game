"""Single street betting round orchestration."""

import time
from collections.abc import Callable
from dataclasses import replace

from poker.domain.action import Action, ActionType
from poker.logging.events import ActionTaken, TimingEvent
from poker.logging.logger import Logger
from poker.state.game_state import GameState


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
    ) -> None:
        """Initialize a betting round.

        Args:
            state: The starting GameState for this street.
            get_action: Callable that takes (seat, state) and returns an Action.
                        Called once per player per street until round closes.
            logger: Logger to emit ActionTaken events.
        """
        self.state = state
        self.get_action = get_action
        self.logger = logger
        # Track which players have acted at least once in this round
        self.players_acted_this_round: set[int] = set()
        # Track the highest bet to know when all players have matched
        self.max_commitment: int = max(
            (p.committed_this_street for p in state.players),
            default=0
        )

    def run(self) -> GameState:
        """Execute the betting round until it closes.

        Returns:
            Updated GameState after the round completes.
        """
        # If only 1 or fewer active players, round is already closed
        if self._count_active_players(self.state) <= 1:
            return replace(self.state, action_on_seat=None)

        # Main betting loop
        while not self._round_is_closed():
            if self.state.action_on_seat is None:
                # All remaining players are all-in or folded
                break

            seat = self.state.action_on_seat
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

            # Apply the action to the state
            self.state = self._apply_action(self.state, seat, action)

            # Update the max commitment (highest bet seen so far)
            self.max_commitment = max(
                self.max_commitment,
                max((p.committed_this_street for p in self.state.players), default=0)
            )

            # Check if this action re-opens betting (half-raise rule)
            if self._reopens_betting(action):
                # This is a full raise. Reset acted set so all others must act again.
                self.players_acted_this_round = {seat}  # Only raiser has acted

            # Emit ActionTaken event
            self.logger.log_event(
                ActionTaken(
                    hand_number=self.state.hand_number,
                    street=self.state.street.value,
                    seat=seat,
                    action_type=action.type.value,
                    amount=action.amount,
                )
            )

        # Round is closed - clear action_on_seat
        return replace(self.state, action_on_seat=None)

    def _round_is_closed(self) -> bool:
        """Check if the betting round is closed.

        A round is closed when all players who can act (non-folded, non-all-in)
        have acted and matched the highest bet. All-in players are excluded from
        this check since they cannot act further.

        Returns:
            True if round is closed, False otherwise.
        """
        # Only count players who can still act (not folded, not all-in, not eliminated)
        can_act_seats = [
            i for i, p in enumerate(self.state.players)
            if not p.has_folded and not p.is_eliminated and not p.is_all_in
        ]

        # If 1 or fewer players can act, round is closed
        if len(can_act_seats) <= 1:
            return True

        # All players who can act must have acted at least once
        for seat in can_act_seats:
            if seat not in self.players_acted_this_round:
                return False

        # All players who can act must have matched the highest bet
        for seat in can_act_seats:
            player = self.state.players[seat]
            if player.committed_this_street < self.max_commitment:
                return False

        return True

    def _apply_action(self, state: GameState, seat: int, action: Action) -> GameState:
        """Apply an action to the state and return the updated state.

        Action amounts represent the TOTAL chips committed by the player,
        not the additional amount.

        Args:
            state: Current GameState.
            seat: Seat number of the player taking the action.
            action: The action to apply.

        Returns:
            Updated GameState after the action.
        """
        player = state.players[seat]
        players_list = list(state.players)

        if action.type == ActionType.FOLD:
            # Fold: player is marked as folded
            players_list[seat] = player.with_folded(True)

        elif action.type == ActionType.CHECK:
            # Check: no change to player state
            pass

        elif action.type == ActionType.CALL:
            # Call: action.amount is the total to call
            # Calculate additional amount beyond what's already committed
            additional_to_commit = action.amount - player.committed_this_street
            new_stack = player.stack - additional_to_commit
            new_committed_street = action.amount
            new_committed_hand = player.committed_this_hand + additional_to_commit

            updated_player = (
                player.with_stack(new_stack)
                .with_committed_this_street(new_committed_street)
                .with_committed_this_hand(new_committed_hand)
            )

            if new_stack == 0:
                updated_player = updated_player.with_all_in(True)

            players_list[seat] = updated_player

        elif action.type == ActionType.RAISE:
            # Raise: action.amount is the total raise to
            amount_to_raise = action.amount - state.current_bet_to_call
            additional_to_commit = action.amount - player.committed_this_street

            new_stack = player.stack - additional_to_commit
            new_committed_street = action.amount
            new_committed_hand = player.committed_this_hand + additional_to_commit

            updated_player = (
                player.with_stack(new_stack)
                .with_committed_this_street(new_committed_street)
                .with_committed_this_hand(new_committed_hand)
            )

            if new_stack == 0:
                updated_player = updated_player.with_all_in(True)

            players_list[seat] = updated_player

            # Update game state: new bet to call and last raise size
            state = replace(
                state,
                current_bet_to_call=action.amount,
                last_raise_size=amount_to_raise,
                players=tuple(players_list),
            )
            players_list = list(state.players)

            return self._advance_action_on_seat(state, seat)

        elif action.type == ActionType.ALL_IN:
            # All-in: action.amount is the total all-in to
            additional_to_commit = action.amount - player.committed_this_street
            new_stack = player.stack - additional_to_commit
            new_committed_street = action.amount
            new_committed_hand = player.committed_this_hand + additional_to_commit

            updated_player = (
                player.with_stack(new_stack)
                .with_committed_this_street(new_committed_street)
                .with_committed_this_hand(new_committed_hand)
                .with_all_in(True)
            )

            players_list[seat] = updated_player

            # If all-in amount > current bet, it's a raise
            if action.amount > state.current_bet_to_call:
                amount_to_raise = action.amount - state.current_bet_to_call
                state = replace(
                    state,
                    current_bet_to_call=action.amount,
                    last_raise_size=amount_to_raise,
                    players=tuple(players_list),
                )
                players_list = list(state.players)

        # Update game state with new players
        state = replace(state, players=tuple(players_list))

        # Advance to next player
        state = self._advance_action_on_seat(state, seat)

        # Update action history
        state = replace(
            state,
            action_history_this_street=[*state.action_history_this_street, (seat, action)],
            action_history_this_hand=[*state.action_history_this_hand, (seat, action)],
        )

        return state

    def _advance_action_on_seat(self, state: GameState, current_seat: int) -> GameState:
        """Advance to the next player who needs to act.

        Args:
            state: Current GameState.
            current_seat: The seat that just acted.

        Returns:
            Updated GameState with action_on_seat pointing to next actor.
        """
        num_players = len(state.players)
        next_seat = (current_seat + 1) % num_players

        # Count how many players can act (not folded, not eliminated, not all-in)
        can_act_count = 0
        for i in range(num_players):
            player = state.players[i]
            if not player.has_folded and not player.is_eliminated and not player.is_all_in:
                can_act_count += 1

        # If only 1 or fewer players can act, betting round should close
        # Return None to signal no more actions needed
        if can_act_count <= 1:
            return replace(state, action_on_seat=None)

        # Skip players until we find one who can act
        attempts = 0
        while attempts < num_players:
            player = state.players[next_seat]

            # Skip folded and eliminated players
            if player.has_folded or player.is_eliminated:
                next_seat = (next_seat + 1) % num_players
                attempts += 1
                continue

            # Skip all-in players (they've posted all they can)
            if player.is_all_in:
                next_seat = (next_seat + 1) % num_players
                attempts += 1
                continue

            # This player can act
            return replace(state, action_on_seat=next_seat)

        # No more players can act
        return replace(state, action_on_seat=None)

    def _count_active_players(self, state: GameState) -> int:
        """Count the number of active (non-folded, non-eliminated) players.

        Args:
            state: Current GameState.

        Returns:
            The count of active players.
        """
        return sum(
            1
            for p in state.players
            if not p.has_folded and not p.is_eliminated
        )

    def _reopens_betting(self, action: Action) -> bool:
        """Check if an action re-opens betting for other players (half-raise rule).

        A RAISE always re-opens. An ALL_IN re-opens only if it qualifies as
        a "full raise" (raise_amount >= min_raise_increment). Shorter all-ins
        are treated as "short all-ins" and do NOT re-open (half-raise rule).

        Returns:
            True if the action re-opens betting, False otherwise.
        """
        if action.type == ActionType.RAISE:
            # A RAISE always re-opens betting
            return True

        if action.type == ActionType.ALL_IN:
            # All-in re-opens only if it's a "full raise"
            # Calculate the raise amount (how much beyond the bet to call)
            raise_amount = action.amount - self.max_commitment
            min_raise_increment = max(
                self.state.config.big_blind, self.state.last_raise_size
            )
            # Re-opens if raise_amount >= min_raise_increment
            return raise_amount >= min_raise_increment

        # CHECK, CALL, FOLD do not re-open
        return False
