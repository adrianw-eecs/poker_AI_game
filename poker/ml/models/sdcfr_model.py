"""SD-CFR (Single Deep Counterfactual Regret Minimization) model."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from poker.ml.action_space import BetScheme
from poker.ml.buffers import WeightedCircularBuffer
from poker.ml.cfr.regret_matching import regret_matching
from poker.ml.models.nfsp_networks import (
    PokerNetwork,
    build_q_network_v2,
    network_kwargs_from_state_dict,
)
from poker.ml.observation_v2 import OBS_V2_DIM

_NUM_ACTIONS = 7

# Observation widths per encoder version. v1 is the legacy 98-wide flat vector;
# v2 is the 318-wide hero-relative vector NFSP v2 uses. Both are supported so
# the two models can be trained on the same representation (see HYBRID_PLAN.md).
_OBS_DIM_V1 = 98
_OBS_DIM_V2 = OBS_V2_DIM
_OBS_DIMS = {1: _OBS_DIM_V1, 2: _OBS_DIM_V2}
# Backwards-compatible alias for callers that imported the old constant.
_OBS_DIM = _OBS_DIM_V1

# Max buffer entries persisted per checkpoint (avoids large files).
_MAX_CHECKPOINT_BUFFER = 100_000


class SDCFRModel:
    """Single Deep CFR model for poker.

    Maintains an advantage network trained via regret-matching targets
    collected during external-sampling CFR traversals.

    The regret_buffer stores packed entries of shape ``(obs_dim + 7,)``: the
    observation followed by the 7 advantage targets. ``obs_dim`` is 98 under
    ``obs_version=1`` and 318 under ``obs_version=2``.
    """

    def __init__(
        self,
        lr: float = 0.0005,
        batch_size: int = 2_048,
        sgd_steps_per_iteration: int = 1_000,
        capacity: int = 500_000,
        device: torch.device | None = None,
        obs_version: int = 1,
        bet_scheme: BetScheme | None = None,
    ) -> None:
        if obs_version not in _OBS_DIMS:
            raise ValueError(f"obs_version must be 1 or 2, got {obs_version}")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.sgd_steps_per_iteration = sgd_steps_per_iteration
        self.cfr_iteration = 0
        self.num_actions = _NUM_ACTIONS
        self.obs_version = obs_version
        self.obs_dim = _OBS_DIMS[obs_version]
        if bet_scheme is None:
            bet_scheme = BetScheme.POT_FRACTION if obs_version == 2 else BetScheme.LEGACY
        elif not isinstance(bet_scheme, BetScheme):
            bet_scheme = BetScheme(bet_scheme)
        self.bet_scheme = bet_scheme

        self.advantage_network = self._build_network()

        # FIFO buffer with weight-biased sampling (Fix #9: replaces conflicted
        # WeightedReservoirBuffer which used uniform insertion + biased sampling).
        self.regret_buffer = WeightedCircularBuffer(
            capacity=capacity, obs_dim=self.obs_dim + _NUM_ACTIONS
        )

        self.optimizer = optim.Adam(self.advantage_network.parameters(), lr=self.lr)

    def _build_network(self) -> nn.Module:
        """Advantage net for the configured encoder.

        v1 keeps the original ``PokerNetwork`` (residual MLP on the flat 98-wide
        vector). v2 reuses ``PokerNetV2``'s trunk with a dueling-style head so
        SD-CFR shares NFSP's representation, which is the prerequisite for
        distilling CFR targets into the NFSP policy head.
        """
        if self.obs_version == 2:
            return build_q_network_v2(
                input_dim=self.obs_dim, num_actions=_NUM_ACTIONS
            ).to(self.device)
        return PokerNetwork(input_dim=self.obs_dim, num_actions=_NUM_ACTIONS).to(self.device)

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
            obs: Observation vector of shape (OBS_DIM,).
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
        self.advantage_network = self._build_network()
        self.optimizer = optim.Adam(self.advantage_network.parameters(), lr=self.lr)

        # Allow at least 32 passes over the available data before capping at
        # sgd_steps_per_iteration. Using *10 gave only 10 steps when the buffer
        # was barely larger than batch_size (Fix #20).
        num_steps = min(
            self.sgd_steps_per_iteration,
            max(32, (len(self.regret_buffer) // self.batch_size) * 32),
        )
        num_steps = max(1, num_steps)

        self.advantage_network.train()
        total_loss = 0.0

        for _ in range(num_steps):
            batch = self.regret_buffer.sample(self.batch_size)
            packed_obs: npt.NDArray[np.float32] = batch["obs"]  # (B, OBS_DIM + NUM_ACTIONS)
            weights: npt.NDArray[np.float32] = batch["weights"]  # (B,)

            obs_batch = packed_obs[:, : self.obs_dim]
            target_batch = packed_obs[:, self.obs_dim:]

            # Clip advantages to a fixed range instead of per-batch normalization.
            # Per-batch normalization collapses the sparse advantage signal: most
            # unsampled actions have advantage=0, which gets shifted by the batch
            # mean and divided by the batch std, making zero no longer mean "no
            # information" (Fix #10).
            target_norm = np.clip(target_batch, -50.0, 50.0)

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
        """Save model state to disk, including a capped buffer snapshot.

        Args:
            filepath: Destination file path (e.g. 'models/sdcfr.pt').
        """
        buf_state = self.regret_buffer.state_dict()
        # Cap buffer snapshot to avoid large checkpoint files (matches NFSP cap).
        if buf_state["size"] > _MAX_CHECKPOINT_BUFFER:
            n = _MAX_CHECKPOINT_BUFFER
            idx = np.random.choice(buf_state["size"], n, replace=False)
            buf_state["obs"] = buf_state["obs"][idx]
            buf_state["actions"] = buf_state["actions"][idx]
            buf_state["weights"] = buf_state["weights"][idx]
            buf_state["pos"] = n
            buf_state["size"] = n

        torch.save(
            {
                "network_state": self.advantage_network.state_dict(),
                "cfr_iteration": self.cfr_iteration,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "sgd_steps_per_iteration": self.sgd_steps_per_iteration,
                "obs_version": self.obs_version,
                "bet_scheme": self.bet_scheme.value,
                "obs_dim": self.obs_dim,
                "regret_buffer": buf_state,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        """Load model state from disk, restoring buffer if present.

        Older checkpoints predate the obs_version field; they are inferred from
        the saved network's input width so existing files keep loading.
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.lr = checkpoint.get("lr", self.lr)
        self.batch_size = checkpoint.get("batch_size", self.batch_size)
        self.sgd_steps_per_iteration = checkpoint.get(
            "sgd_steps_per_iteration", self.sgd_steps_per_iteration
        )
        self.cfr_iteration = checkpoint.get("cfr_iteration", 0)

        state = checkpoint["network_state"]
        if "obs_version" in checkpoint:
            self.obs_version = int(checkpoint["obs_version"])
        else:
            # Infer from the checkpoint itself: a v2 PokerNetV2 has embeddings.
            self.obs_version = 2 if "rank_emb.weight" in state else 1
        self.obs_dim = _OBS_DIMS[self.obs_version]
        if checkpoint.get("bet_scheme"):
            self.bet_scheme = BetScheme(checkpoint["bet_scheme"])
        elif self.obs_version == 2:
            self.bet_scheme = BetScheme.POT_FRACTION

        self.advantage_network = self._rebuild_from_state(state)
        self.advantage_network.load_state_dict(state)
        self.advantage_network.eval()
        self.optimizer = optim.Adam(self.advantage_network.parameters(), lr=self.lr)
        if "regret_buffer" in checkpoint:
            stored = checkpoint["regret_buffer"]
            expected = self.obs_dim + _NUM_ACTIONS
            if stored.get("obs") is not None and stored["obs"].shape[-1] == expected:
                if self.regret_buffer.obs_dim != expected:
                    self.regret_buffer = WeightedCircularBuffer(
                        capacity=self.regret_buffer.capacity, obs_dim=expected
                    )
                self.regret_buffer.load_state_dict(stored)

    def _rebuild_from_state(self, state: dict) -> nn.Module:
        """Construct the network whose shapes match a saved state dict."""
        if "rank_emb.weight" in state:
            kwargs = network_kwargs_from_state_dict(state)
            kwargs.pop("arch", None)
            return build_q_network_v2(**kwargs).to(self.device)
        return PokerNetwork(
            input_dim=self.obs_dim, num_actions=_NUM_ACTIONS
        ).to(self.device)
