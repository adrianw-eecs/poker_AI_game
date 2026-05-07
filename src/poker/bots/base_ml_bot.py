"""Base class for ML-based poker bots with common functionality."""

from typing import Any

import numpy as np
import numpy.typing as npt

from poker.domain.action import Action
from poker.ml.action_space import action_index_to_action, action_to_action_index
from poker.ml.features.handcrafted import extract_handcrafted_features
from poker.state.game_state import GameState


class BaseMLBot:
    """Base class for machine learning poker bots.

    Provides shared functionality for linear, tree, and deep bots:
    - Feature extraction and caching
    - Legal action mask building
    - Seat detection
    - Type hints and clean API
    """

    def __init__(self, name: str, model: Any) -> None:
        """Initialize the ML bot.

        Args:
            name: Display name for the bot.
            model: Pre-trained ML model with predict_best_action method.
        """
        self._name = name
        self.model = model
        self._seat_cache: int | None = None
        self._last_state_id: int | None = None

    @property
    def name(self) -> str:
        """Return the bot's name."""
        return self._name

    def _find_seat(self, state: GameState) -> int:
        """Find bot's seat from game state.

        Uses cached seat if available to avoid repeated lookups.

        Args:
            state: Current game state.

        Returns:
            Bot's seat number (0-indexed).

        Raises:
            RuntimeError: If seat cannot be determined.
        """
        # Use cache to avoid repeated lookups in same hand
        state_id = id(state)
        if self._last_state_id == state_id and self._seat_cache is not None:
            return self._seat_cache

        # Find seat by looking for non-empty hole cards
        for i, player in enumerate(state.players):
            if len(player.hole_cards) > 0:
                self._seat_cache = i
                self._last_state_id = state_id
                return i

        raise RuntimeError("Cannot determine bot's seat from state")

    def _build_action_mask(
        self, state: GameState, legal_actions: list[Action], seat: int
    ) -> npt.NDArray[np.int32]:
        """Build legal action mask for model inference.

        Args:
            state: Current game state.
            legal_actions: List of legal actions available.
            seat: Bot's seat number.

        Returns:
            Binary mask of shape (7,) indicating legal actions.
        """
        mask = np.zeros(7, dtype=np.int32)
        for action in legal_actions:
            action_idx = action_to_action_index(action, state, seat)
            mask[action_idx] = 1
        return mask

    def act(self, state: GameState, legal: list[Action]) -> Action:
        """Choose an action using the ML model.

        Args:
            state: Current game state (observed from this bot's perspective).
            legal: List of legal actions available.

        Returns:
            The chosen action.

        Raises:
            ValueError: If no legal actions available.
            RuntimeError: If bot's seat cannot be determined.
        """
        if not legal:
            raise ValueError("No legal actions available")

        seat = self._find_seat(state)

        # Extract features
        features = extract_handcrafted_features(state, seat)

        # Build legal action mask
        mask = self._build_action_mask(state, legal, seat)

        # Get best action from model
        best_action_idx = self.model.predict_best_action(features, mask)

        # Convert back to concrete action
        return action_index_to_action(best_action_idx, state, seat)

    def observe_result(self, final_state: GameState, reward: float) -> None:
        """Observe the hand outcome.

        Args:
            final_state: The final game state after the hand.
            reward: The normalized chip delta.
        """
        # ML models train offline; no online learning
        pass
