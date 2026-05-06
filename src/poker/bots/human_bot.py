"""Human player adapter implementing the Bot protocol."""

import poker.interface.text_ui as text_ui
from poker.domain.action import Action
from poker.state.game_state import GameState


class HumanBot:
    """Adapter that lets a human play through the Bot protocol.

    Renders the current game state to the terminal and then prompts
    the human for an action. Satisfies the Bot Protocol via structural typing
    (no explicit inheritance required).
    """

    def __init__(self, seat: int, name: str = "Human") -> None:
        """Initialize the human bot.

        Args:
            seat: The seat number this human occupies (used for rendering
                  the correct card view).
            name: The display name for this player.
        """
        self._seat = seat
        self._name = name

    @property
    def name(self) -> str:
        """Return the player's name."""
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Render the game state, prompt the human, and return their action.

        Args:
            state: The current game state (typically state.view_for(self._seat)).
            legal: List of legal actions available to this player.

        Returns:
            The action chosen by the human.
        """
        print(text_ui.render(state, self._seat))
        return text_ui.prompt(state, legal)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """Observe the hand outcome (silent for human players).

        The hand result is already displayed by display_showdown(), so we don't
        print an additional reward line for human players. This avoids duplicate
        output in the game log.

        Args:
            final_state: The final game state after the hand.
            reward: The normalized chip delta (stack_change / starting_stack).
        """
        # Silent: humans already see the result through display_showdown()
        pass
