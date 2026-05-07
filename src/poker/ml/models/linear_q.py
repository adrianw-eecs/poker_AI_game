"""Linear regression Q-learning model."""

from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge

from poker.ml.features.handcrafted import extract_handcrafted_features
from poker.state.game_state import GameState


class LinearQModel:
    """Multi-output linear regression for Q-value estimation.

    Trains one Ridge regressor per action to predict Q(state, action).
    At inference, picks the action with highest Q value (masked by legality).
    """

    def __init__(self, num_actions: int = 7, alpha: float = 1.0) -> None:
        """Initialize the model.

        Args:
            num_actions: Number of discrete actions (default 7 for poker).
            alpha: Ridge regularization strength (default 1.0).
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.models: list[Ridge] = [Ridge(alpha=alpha) for _ in range(num_actions)]
        self.is_fitted = False

    def fit(
        self,
        X: npt.NDArray[np.float32],
        actions: npt.NDArray[np.int32],
        rewards: npt.NDArray[np.float32],
        legal_masks: npt.NDArray[np.int32] | None = None,
    ) -> None:
        """Train the model on collected data.

        Args:
            X: Feature matrix of shape (N, 15).
            actions: Action indices of shape (N,).
            rewards: Reward targets of shape (N,).
            legal_masks: Optional legal action masks of shape (N, 7).
                        Only train on legal actions if provided.
        """
        self.fitted_actions = set()
        for action in range(self.num_actions):
            # Get indices where this action was taken (and legal if mask provided)
            action_mask = actions == action
            if legal_masks is not None:
                action_mask = action_mask & (legal_masks[:, action] == 1)

            if np.sum(action_mask) > 0:
                X_action = X[action_mask]
                y_action = rewards[action_mask]
                self.models[action].fit(X_action, y_action)
                self.fitted_actions.add(action)

        self.is_fitted = True

    def predict_q_values(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Predict Q-values for all actions.

        Args:
            X: Feature matrix of shape (N, 15) or (15,).

        Returns:
            Q-values of shape (N, 7) or (7,).
        """
        if not self.is_fitted:
            # Return zeros if not fitted
            if X.ndim == 1:
                return np.zeros(self.num_actions, dtype=np.float32)
            return np.zeros((X.shape[0], self.num_actions), dtype=np.float32)

        single_input = X.ndim == 1
        if single_input:
            X = X.reshape(1, -1)

        q_values = np.zeros((X.shape[0], self.num_actions), dtype=np.float32)
        for action in range(self.num_actions):
            if action in getattr(self, 'fitted_actions', set(range(self.num_actions))):
                q_values[:, action] = self.models[action].predict(X).astype(np.float32)
            # else: leave as 0 for unfitted actions

        if single_input:
            q_values = q_values[0]

        return q_values

    def predict_best_action(
        self, X: npt.NDArray[np.float32], mask: npt.NDArray[np.int32] | None = None
    ) -> int:
        """Predict best action given features.

        Args:
            X: Feature vector of shape (15,).
            mask: Optional legal action mask of shape (7,). Defaults to all 1s.

        Returns:
            Best legal action index.
        """
        q_values = self.predict_q_values(X)
        if mask is None:
            # Fast path: no masking needed
            return int(np.argmax(q_values))

        # Apply mask in-place by converting to float and using masked assignment
        # This is more efficient than copy + assignment pattern
        masked_q = np.where(mask, q_values, -np.inf)
        return int(np.argmax(masked_q))

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save to (e.g., 'models/linear_q.npz').
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump((self.models, self.num_actions, self.alpha, self.is_fitted, getattr(self, 'fitted_actions', set())), f)
        print(f"Saved LinearQModel to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load from.
        """
        import pickle

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            if len(data) == 4:
                # Old format without fitted_actions
                self.models, self.num_actions, self.alpha, self.is_fitted = data
                self.fitted_actions = set(range(self.num_actions))
            else:
                self.models, self.num_actions, self.alpha, self.is_fitted, self.fitted_actions = data
        print(f"Loaded LinearQModel from {filepath}")
