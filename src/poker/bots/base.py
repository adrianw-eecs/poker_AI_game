"""Bot interface protocol."""

from typing import Protocol

from poker.domain.action import Action
from poker.state.game_state import GameState


class Bot(Protocol):
    """Protocol for poker bot implementations.

    Any object implementing this protocol can be used as a player in the game.
    All bots receive a view of the game state with their own hole cards visible
    but other players' hole cards hidden (enforced by GameState.view_for).
    """

    @property
    def name(self) -> str:
        """Return the bot's name.

        Returns:
            A string identifier for this bot.
        """
        ...

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Choose an action given the current game state.

        Args:
            state: The current game state (always called with state.view_for(self.seat)).
            legal: List of legal actions available to this player.

        Returns:
            The chosen action (must be in the legal list).
        """
        ...

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """Observe the hand outcome and reward.

        Called at the end of each hand so the bot can learn from the result.
        The reward is the change in stack normalized to the starting stack.

        Args:
            final_state: The final game state after the hand (with all cards revealed).
            reward: The normalized chip delta (stack_change / starting_stack).
        """
        ...
