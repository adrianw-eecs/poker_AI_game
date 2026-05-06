"""Action legality validator for poker game state."""

from poker.domain.action import Action, ActionType
from poker.exceptions import IllegalActionError
from poker.state.game_state import GameState


def legal_actions(state: GameState, seat: int) -> list[Action]:
    """Compute the set of legal actions for a player.

    Returns action templates with min/max raise ranges. All returned actions
    are guaranteed to be legal given the current GameState.

    Args:
        state: The current game state.
        seat: The seat number of the player to compute actions for.

    Returns:
        A list of legal Action objects. Raise actions will have amount set to
        the minimum valid raise (caller should raise more up to their full stack).

    Raises:
        IllegalActionError: If seat is not the action_on_seat.
    """
    if state.action_on_seat != seat:
        raise IllegalActionError(
            f"Action is on seat {state.action_on_seat}, not seat {seat}"
        )

    player = state.players[seat]
    actions: list[Action] = []

    # Fold is always available
    actions.append(Action.fold())

    # No bet to call scenario (can check or open raise if player has chips)
    if state.current_bet_to_call == 0:
        if player.stack > 0:
            actions.append(Action.check())
            min_open_raise = state.config.big_blind
            max_raise_to = player.stack + player.committed_this_street
            actions.append(Action.raise_to(min(min_open_raise, max_raise_to)))

    # Bet to call scenario
    else:
        # Call is legal if player can match the bet (possibly all-in)
        if player.stack > 0:
            call_amount = min(state.current_bet_to_call, player.stack)
            actions.append(Action.call(call_amount))

            # Raise is legal only if player has chips beyond the call amount
            if player.stack > state.current_bet_to_call:
                min_raise_increment = max(state.config.big_blind, state.last_raise_size)
                min_raise_to = state.current_bet_to_call + min_raise_increment
                max_raise_to = player.stack + player.committed_this_street
                actions.append(Action.raise_to(min(min_raise_to, max_raise_to)))

    # All-in is available if player has chips and it's not already an available action
    if player.stack > 0:
        all_in_amount = player.stack + player.committed_this_street
        if not any(a.type == ActionType.ALL_IN for a in actions):
            actions.append(Action.all_in(all_in_amount))

    return actions


def validate(state: GameState, seat: int, action: Action) -> None:
    """Validate that a concrete action is legal.

    Args:
        state: The current game state.
        seat: The seat number taking the action.
        action: The action to validate.

    Raises:
        IllegalActionError: If the action is not legal.
    """
    if state.action_on_seat != seat:
        raise IllegalActionError(
            f"Action is on seat {state.action_on_seat}, not seat {seat}"
        )

    player = state.players[seat]

    # Fold is always legal
    if action.type == ActionType.FOLD:
        return

    # Check is legal only if no bet to call
    if action.type == ActionType.CHECK:
        if state.current_bet_to_call != 0:
            raise IllegalActionError("Cannot check when facing a bet")
        return

    # Call is legal only if there's a bet to call
    if action.type == ActionType.CALL:
        if state.current_bet_to_call == 0:
            raise IllegalActionError("Cannot call when there's no bet")
        # Call amount must be at least the minimum (either the full bet or all remaining chips)
        min_call_amount = min(state.current_bet_to_call, player.stack)
        max_call_amount = player.stack + player.committed_this_street
        if action.amount < min_call_amount or action.amount > max_call_amount:
            raise IllegalActionError(
                f"Call amount {action.amount} is invalid. Must be between {min_call_amount} and {max_call_amount}"
            )
        return

    # Raise validation
    if action.type == ActionType.RAISE:
        if action.amount <= state.current_bet_to_call:
            raise IllegalActionError(
                f"Raise amount {action.amount} must exceed bet to call {state.current_bet_to_call}"
            )
        if action.amount > player.stack + player.committed_this_street:
            raise IllegalActionError(
                f"Cannot raise to {action.amount} with only {player.stack} chips in stack"
            )
        return

    # All-in validation (can be any amount up to total stack)
    if action.type == ActionType.ALL_IN:
        if action.amount <= 0:
            raise IllegalActionError(
                f"All-in amount must be positive, got {action.amount}"
            )
        if action.amount > player.stack + player.committed_this_street:
            raise IllegalActionError(
                f"Cannot go all-in for {action.amount} with only {player.stack} chips in stack"
            )
        return

    raise IllegalActionError(f"Unknown action type: {action.type}")
