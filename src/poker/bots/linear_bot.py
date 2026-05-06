"""Linear regression bot for poker."""

from poker.domain.action import Action
from poker.ml.features.handcrafted import extract_handcrafted_features
from poker.ml.models.linear_q import LinearQModel
from poker.state.game_state import GameState


class LinearBot:
    """Poker bot using linear Q-learning model.

    Uses handcrafted features (15-dim) and a linear regression model
    to predict Q-values and choose actions.
    """

    def __init__(self, name: str = "LinearBot", model: LinearQModel | None = None) -> None:
        """Initialize the bot.

        Args:
            name: Display name for the bot.
            model: Optional pre-trained LinearQModel. If None, uses unfitted model.
        """
        self._name = name
        self.model = model or LinearQModel()

    @property
    def name(self) -> str:
        """Return the bot's name."""
        return self._name

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Choose an action using the linear Q-model.

        Args:
            state: Current game state (observed from this bot's perspective).
            legal: List of legal actions available.

        Returns:
            The chosen action.
        """
        if not legal:
            raise ValueError("No legal actions available")

        # Get this bot's seat from the state
        # Since this is called with state.view_for(seat), we need to find our seat
        # The state should have our hole cards visible, so find the player with non-empty hole cards
        seat = None
        for i, p in enumerate(state.players):
            if len(p.hole_cards) > 0:
                seat = i
                break

        if seat is None:
            raise RuntimeError("Cannot determine bot's seat from state")

        # Extract features
        features = extract_handcrafted_features(state, seat)

        # Build legal action mask
        from poker.ml.action_space import action_to_action_index

        mask = [0] * 7
        for action in legal:
            action_idx = action_to_action_index(action, state, seat)
            mask[action_idx] = 1

        # Predict best action
        import numpy as np

        best_action_idx = self.model.predict_best_action(features, np.array(mask, dtype=np.int32))

        # Convert back to concrete action
        from poker.ml.action_space import action_index_to_action

        return action_index_to_action(best_action_idx, state, seat)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """Observe the hand outcome.

        Args:
            final_state: The final game state after the hand.
            reward: The normalized chip delta.
        """
        # Linear model does not learn online; training happens offline via fit()
        pass
