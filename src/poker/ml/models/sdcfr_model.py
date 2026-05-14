"""SD-CFR (Single Deep Counterfactual Regret Minimization) model."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from poker.ml.buffers import WeightedReservoirBuffer
from poker.ml.cfr.regret_matching import regret_matching
from poker.ml.models.nfsp_networks import PokerNetwork

_OBS_DIM = 155
_NUM_ACTIONS = 7
# Buffer entries pack [obs (155) | advantages (7)] = 162 features.
_BUFFER_OBS_DIM = _OBS_DIM + _NUM_ACTIONS


class SDCFRModel:
    """Single Deep CFR model for poker.

    Maintains an advantage network trained via regret-matching targets
    collected during external-sampling CFR traversals.

    The regret_buffer stores packed entries of shape (162,) where the first
    155 elements are the observation and the last 7 are advantage targets.
    """

    def __init__(
        self,
        lr: float = 0.0005,
        batch_size: int = 32_000,
        sgd_steps_per_iteration: int = 16_000,
        capacity: int = 10_000_000,
        device: torch.device | None = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.sgd_steps_per_iteration = sgd_steps_per_iteration
        self.cfr_iteration = 0

        self.advantage_network = PokerNetwork(
            input_dim=_OBS_DIM, num_actions=_NUM_ACTIONS
        ).to(self.device)

        # Buffer stores packed (obs || advantages) vectors.
        self.regret_buffer = WeightedReservoirBuffer(
            capacity=capacity, obs_dim=_BUFFER_OBS_DIM
        )

        self.optimizer = optim.Adam(self.advantage_network.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_strategy(
        self,
        obs: npt.NDArray[np.float32],
        legal_mask: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Return a mixed strategy (probability distribution) for a given state.

        Args:
            obs: Observation vector of shape (155,).
            legal_mask: Binary mask of shape (7,).

        Returns:
            Probability distribution of shape (7,).
        """
        self.advantage_network.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            advantages = (
                self.advantage_network(x).squeeze(0).cpu().numpy().astype(np.float32)
            )
        return regret_matching(advantages, legal_mask)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_iteration(self) -> dict[str, Any]:
        """Run one SD-CFR training iteration.

        Reinitialises the advantage network and trains it on regret targets
        sampled from the weighted reservoir buffer.

        Returns:
            Dict with keys 'iteration', 'loss', 'buffer_size'.
            Returns early dict (loss=-1) when buffer has fewer than batch_size samples.
        """
        if len(self.regret_buffer) < self.batch_size:
            return {
                "iteration": self.cfr_iteration,
                "loss": -1.0,
                "buffer_size": len(self.regret_buffer),
            }

        # Reinitialise network weights (SD-CFR resets each iteration)
        self.advantage_network = PokerNetwork(
            input_dim=_OBS_DIM, num_actions=_NUM_ACTIONS
        ).to(self.device)
        self.optimizer = optim.Adam(self.advantage_network.parameters(), lr=self.lr)

        num_steps = min(
            self.sgd_steps_per_iteration,
            (len(self.regret_buffer) // self.batch_size) * 10,
        )
        num_steps = max(1, num_steps)

        self.advantage_network.train()
        total_loss = 0.0

        for _ in range(num_steps):
            batch = self.regret_buffer.sample(self.batch_size)
            packed_obs: npt.NDArray[np.float32] = batch["obs"]  # (B, 149)
            weights: npt.NDArray[np.float32] = batch["weights"]  # (B,)

            obs_batch = packed_obs[:, :_OBS_DIM]  # (B, 142)
            target_batch = packed_obs[:, _OBS_DIM:]  # (B, 7)

            # Normalise targets
            mean = target_batch.mean(axis=0)
            std = target_batch.std(axis=0) + 1e-8
            target_norm = (target_batch - mean) / std

            obs_t = torch.tensor(obs_batch, dtype=torch.float32, device=self.device)
            target_t = torch.tensor(target_norm, dtype=torch.float32, device=self.device)
            w_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad()
            pred = self.advantage_network(obs_t)  # (B, 7)
            diff = pred - target_t
            loss = (w_t.unsqueeze(1) * diff * diff).sum() / w_t.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(self.advantage_network.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.advantage_network.eval()
        self.cfr_iteration += 1

        return {
            "iteration": self.cfr_iteration,
            "loss": total_loss / num_steps,
            "buffer_size": len(self.regret_buffer),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Save model state to disk.

        Args:
            filepath: Destination file path (e.g. 'models/sdcfr.pt').
        """
        torch.save(
            {
                "network_state": self.advantage_network.state_dict(),
                "cfr_iteration": self.cfr_iteration,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "sgd_steps_per_iteration": self.sgd_steps_per_iteration,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load model state from disk.

        Args:
            filepath: Source file path.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.lr = checkpoint.get("lr", self.lr)
        self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self.sgd_steps_per_iteration = checkpoint.get(
            "sgd_steps_per_iteration", self.sgd_steps_per_iteration
        )
        self.cfr_iteration = checkpoint.get("cfr_iteration", 0)
        self.advantage_network = PokerNetwork(
            input_dim=_OBS_DIM, num_actions=_NUM_ACTIONS
        ).to(self.device)
        self.advantage_network.load_state_dict(checkpoint["network_state"])
        self.advantage_network.eval()
        self.optimizer = optim.Adam(self.advantage_network.parameters(), lr=self.lr)
