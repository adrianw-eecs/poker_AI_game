"""Deep Q-learning neural network model."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from poker.state.game_state import GameState

# Check for mixed precision support (available in CUDA-capable GPUs)
try:
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False


class QNetwork(nn.Module):
    """Neural network for Q-value prediction.

    Multi-output network that predicts Q(state, action) for all actions simultaneously.
    Architecture: Input (15) -> Hidden (128) -> Hidden (64) -> Output (7 actions)
    """

    def __init__(self, input_size: int = 15, hidden_size: int = 128, num_actions: int = 7) -> None:
        """Initialize the network.

        Args:
            input_size: Dimension of handcrafted features (default 15).
            hidden_size: Hidden layer dimension (default 128).
            num_actions: Number of discrete actions (default 7 for poker).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (batch_size, 15) or (15,).

        Returns:
            Q-values of shape (batch_size, 7) or (7,).
        """
        return self.net(x)


class DeepQModel:
    """Deep Q-learning model using neural networks.

    Trains a single neural network to predict Q(state, action) for all actions.
    Uses supervised learning to fit Q-values from collected data.
    """

    def __init__(
        self, input_size: int = 15, hidden_size: int = 128, num_actions: int = 7, lr: float = 0.001
    ) -> None:
        """Initialize the model.

        Args:
            input_size: Dimension of handcrafted features (default 15).
            hidden_size: Hidden layer dimension (default 128).
            num_actions: Number of discrete actions (default 7 for poker).
            lr: Learning rate for optimizer (default 0.001).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.network = QNetwork(input_size, hidden_size, num_actions).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.is_fitted = False

        # Mixed precision training support for NVIDIA GPUs (RTX 3080)
        self.use_amp = HAS_AMP and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            if torch.cuda.is_available():
                print(f"Using mixed precision training on {torch.cuda.get_device_name(0)}")

        # Cache for single prediction to avoid repeated tensor creation
        self._x_cache: torch.Tensor | None = None

    def fit(
        self,
        X: npt.NDArray[np.float32],
        actions: npt.NDArray[np.int32],
        rewards: npt.NDArray[np.float32],
        legal_masks: npt.NDArray[np.int32] | None = None,
        epochs: int = 100,
        batch_size: int = 16,
    ) -> None:
        """Train the model on collected data with mixed precision support.

        Args:
            X: Feature matrix of shape (N, 15).
            actions: Action indices of shape (N,).
            rewards: Reward targets of shape (N,).
            legal_masks: Optional legal action masks of shape (N, 7). Unused for now.
            epochs: Number of training epochs (default 100).
            batch_size: Batch size for training (default 16).
        """
        # Transfer to GPU with optimal dtypes
        X_tensor = torch.FloatTensor(X).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)

        num_samples = len(X)
        self.network.train()

        for epoch in range(epochs):
            # Shuffle indices for mini-batch training
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0

            # Train on batches with mixed precision
            for batch_start in range(0, num_samples, batch_size):
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = indices[batch_start:batch_end]

                X_batch = X_tensor[batch_indices]
                actions_batch = actions_tensor[batch_indices]
                rewards_batch = rewards_tensor[batch_indices]

                self.optimizer.zero_grad()

                # Use mixed precision (fp16) for forward pass on NVIDIA GPUs
                if self.use_amp:
                    with autocast(dtype=torch.float16):
                        q_values = self.network(X_batch)
                        action_q_values = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
                        loss = self.loss_fn(action_q_values, rewards_batch)

                    # Scale loss and backward pass
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Fallback to fp32 training on CPU or non-CUDA devices
                    q_values = self.network(X_batch)
                    action_q_values = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze(1)
                    loss = self.loss_fn(action_q_values, rewards_batch)

                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                device_info = f" on {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}{device_info}")

        self.network.eval()
        self.is_fitted = True

    def predict_q_values(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Predict Q-values for all actions with GPU optimization.

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

        self.network.eval()
        with torch.no_grad():
            # Use fp16 on GPU for faster inference (RTX 3080 Tensor Core optimization)
            if torch.cuda.is_available() and self.use_amp:
                with autocast(dtype=torch.float16):
                    X_tensor = torch.FloatTensor(X).to(self.device)
                    if X.ndim == 1:
                        q_values = self.network(X_tensor.unsqueeze(0)).squeeze(0)
                    else:
                        q_values = self.network(X_tensor)
            else:
                X_tensor = torch.FloatTensor(X).to(self.device)
                if X.ndim == 1:
                    q_values = self.network(X_tensor.unsqueeze(0)).squeeze(0)
                else:
                    q_values = self.network(X_tensor)

            # Convert back to numpy with minimal overhead
            return q_values.float().cpu().numpy().astype(np.float32)

    def predict_best_action(
        self, X: npt.NDArray[np.float32], mask: npt.NDArray[np.int32] | None = None
    ) -> int:
        """Predict best action given features with efficient masking.

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

        # Apply mask using vectorized where operation (more efficient than copy + assignment)
        masked_q = np.where(mask, q_values, -np.inf)
        return int(np.argmax(masked_q))

    def save(self, filepath: str) -> None:
        """Save model to disk.

        Args:
            filepath: Path to save to (e.g., 'models/deep_q.pt').
        """
        torch.save(
            {
                "network_state": self.network.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_actions": self.num_actions,
                "is_fitted": self.is_fitted,
            },
            filepath,
        )
        print(f"Saved DeepQModel to {filepath}")

    def load(self, filepath: str) -> None:
        """Load model from disk.

        Args:
            filepath: Path to load from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.input_size = checkpoint["input_size"]
        self.hidden_size = checkpoint["hidden_size"]
        self.num_actions = checkpoint["num_actions"]
        self.network = QNetwork(self.input_size, self.hidden_size, self.num_actions).to(self.device)
        self.network.load_state_dict(checkpoint["network_state"])
        self.is_fitted = checkpoint["is_fitted"]
        print(f"Loaded DeepQModel from {filepath}")
