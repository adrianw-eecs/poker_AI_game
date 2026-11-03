"""Seat hook for a neural network bot that is not implemented yet."""

from poker.bots.progress import ensure_progressing_action
from poker.domain.action import Action, ActionType
from poker.state.game_state import GameState


class NNBot:
    """Bot protocol shell for the upcoming neural network.

    ``load_model`` is the only integration point. Fill that in when the
    network exists; ``act`` already matches the other model bots.
    """

    def __init__(self, name: str = "NNBot", model: object | None = None) -> None:
        self._name = name
        self.model = model

    @property
    def name(self) -> str:
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Answer with a call or a check. This bot has no network yet.

        Raising here used to stop the hand with no action at all.
        """
        by_type = {action.type: action for action in legal}
        if ActionType.CALL in by_type:
            chosen = by_type[ActionType.CALL]
        elif ActionType.CHECK in by_type:
            chosen = by_type[ActionType.CHECK]
        elif ActionType.FOLD in by_type:
            chosen = by_type[ActionType.FOLD]
        else:
            chosen = legal[0] if legal else Action.fold()
        return ensure_progressing_action(state, chosen, legal)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        return None


def load_model(path: str) -> object:
    """Load a future neural-network checkpoint.

    Raises:
        NotImplementedError: Always, until the network is implemented.
    """
    raise NotImplementedError(
        "Neural network bot is not implemented yet. "
        f"Cannot load '{path}'."
    )
