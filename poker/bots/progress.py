"""Turn a bot action into one that can close the betting round."""

from poker.domain.action import Action, ActionType
from poker.state.game_state import GameState


def chips_still_owed(state: GameState | None) -> bool:
    """True when the player to act has not matched the highest street commitment."""
    if state is None or state.action_on_seat is None:
        return False
    seat = state.action_on_seat
    player = state.players[seat]
    highest = max((p.committed_this_street for p in state.players), default=0)
    live = max(highest, state.current_bet_to_call)
    return player.committed_this_street < live and player.stack > 0


def ensure_progressing_action(
    state: GameState | None,
    action: Action,
    legal: list[Action],
) -> Action:
    """Replace a check that leaves chips unpaid with a call or a fold.

    Bots that prefer check will otherwise answer the same decision forever
    when a short call has left the live bet below the chips already in the pot.
    """
    if not legal:
        return Action.fold()

    by_type: dict[ActionType, Action] = {}
    for candidate in legal:
        by_type.setdefault(candidate.type, candidate)

    if action.type == ActionType.CHECK and chips_still_owed(state):
        if ActionType.CALL in by_type:
            return by_type[ActionType.CALL]
        if ActionType.FOLD in by_type:
            return by_type[ActionType.FOLD]
        if ActionType.ALL_IN in by_type:
            return by_type[ActionType.ALL_IN]
        return next(item for item in legal if item.type != ActionType.CHECK)

    if action.type in by_type:
        listed = by_type[action.type]
        if action.type in (ActionType.FOLD, ActionType.CHECK):
            return listed
        if action.type == ActionType.CALL:
            return listed
        return action
    if ActionType.FOLD in by_type:
        return by_type[ActionType.FOLD]
    return legal[0]
