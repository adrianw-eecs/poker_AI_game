"""CallBot - Always calls, never folds or raises (simple calling station).

This bot is deliberately simple and predictable - it forces the learning agent
to exploit weak play patterns (always calling). Combined with RandomBot and
FlopBot, it provides diverse opponent types for training.
"""

from poker.bots.progress import ensure_progressing_action
from poker.domain.action import Action, ActionType


class CallBot:
    """Calling station bot - calls every bet, never raises or folds voluntarily.

    Strategy:
    - CHECK: when there is no bet to call
    - CALL: any bet, any amount
    - FOLD: only when check and call are both illegal
    - RAISE: never

    This creates a weak but predictable opponent that forces the learning agent
    to learn value betting (exploit weak calling).
    """

    def __init__(self, seed: int | None = None, name: str = "CallBot"):
        self.seed = seed
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def act(self, state, legal_actions: list) -> Action:
        """Return a check or a call from the legal Action list.

        Args:
            state: Game state (unused, for interface compatibility)
            legal_actions: Legal Action objects for this seat

        Returns:
            The check or call action. Fold only when neither is legal.
        """
        by_type: dict[ActionType, Action] = {}
        for action in legal_actions:
            if isinstance(action, Action):
                by_type.setdefault(action.type, action)

        if ActionType.CHECK in by_type:
            chosen = by_type[ActionType.CHECK]
        elif ActionType.CALL in by_type:
            chosen = by_type[ActionType.CALL]
        elif ActionType.FOLD in by_type:
            chosen = by_type[ActionType.FOLD]
        elif not legal_actions:
            chosen = Action.fold()
        else:
            chosen = legal_actions[0]
        return ensure_progressing_action(state, chosen, legal_actions)
