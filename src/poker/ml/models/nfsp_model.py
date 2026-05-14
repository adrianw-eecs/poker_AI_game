"""Neural Fictitious Self-Play (NFSP) model."""

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from poker.ml.buffers import CircularBuffer, ReservoirBuffer
from poker.ml.models.nfsp_networks import (
    build_policy_network,
    build_q_network,
    make_target_network,
    sync_target_network,
)

_Q_MIN = 512       # Q-buffer fills at every step — 512 is reasonable
_POLICY_MIN = 64   # Policy buffer fills at rate eta (~10%) — use lower threshold


class NFSPModel:
    """Neural Fictitious Self-Play agent.

    Maintains two networks:
    - Q-network: Trained with DQN to approximate best-response values.
    - Policy network: Trained via supervised learning on best-response actions,
      converging toward an approximate Nash equilibrium policy.

    At each decision the agent mixes between the two strategies with
    probability eta (best-response) and 1-eta (average policy).
    """

    def __init__(
        self,
        eta: float = 0.15,
        epsilon_start: float = 0.10,
        epsilon_end: float = 0.005,
        epsilon_decay_steps: int = 5_000_000,
        gamma: float = 0.99,
        batch_size: int = 2048,
        lr: float = 0.0005,
        q_capacity: int = 1_000_000,
        policy_capacity: int = 5_000_000,
        target_sync_every: int = 500,
        train_every: int = 16,
        device: str | None = None,
    ) -> None:
        self.eta = eta
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every
        self.train_every = train_every

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.q_network = build_q_network().to(self.device)
        self.policy_network = build_policy_network().to(self.device)
        self.target_network = make_target_network(self.q_network).to(self.device)

        self.q_buffer = CircularBuffer(capacity=q_capacity, obs_dim=155)
        self.policy_buffer = ReservoirBuffer(capacity=policy_capacity, obs_dim=155)

        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)

        self._step = 0

        # Store config for save/load
        self._config = {
            "eta": eta,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay_steps": epsilon_decay_steps,
            "gamma": gamma,
            "batch_size": batch_size,
            "lr": lr,
            "q_capacity": q_capacity,
            "policy_capacity": policy_capacity,
            "target_sync_every": target_sync_every,
            "train_every": train_every,
        }

    def _current_epsilon(self) -> float:
        decay = (self.epsilon_start - self.epsilon_end) * min(
            1.0, self._step / self.epsilon_decay_steps
        )
        return max(self.epsilon_end, self.epsilon_start - decay)

    def _apply_mask(self, logits: torch.Tensor, legal_mask: npt.NDArray[np.int32]) -> torch.Tensor:
        mask_t = torch.as_tensor(legal_mask, dtype=torch.bool, device=self.device)
        return logits.masked_fill(~mask_t, float("-inf"))

    def select_action(
        self,
        obs: npt.NDArray[np.float32],
        legal_mask: npt.NDArray[np.int32],
        training: bool = True,
    ) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        if not training:
            # Pure average policy (Nash strategy)
            self.policy_network.eval()
            with torch.no_grad():
                logits = self._apply_mask(self.policy_network(obs_t).squeeze(0), legal_mask)
                probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1).item())

        use_best_response = np.random.random() < self.eta

        if use_best_response:
            # Epsilon-greedy with Q-network
            epsilon = self._current_epsilon()
            if np.random.random() < epsilon:
                legal_indices = np.where(legal_mask)[0]
                action = int(np.random.choice(legal_indices))
            else:
                self.q_network.eval()
                with torch.no_grad():
                    q_vals = self._apply_mask(self.q_network(obs_t).squeeze(0), legal_mask)
                action = int(q_vals.argmax().item())
            # Record best-response action for policy supervised learning
            self.policy_buffer.add(obs, action)
            return action
        else:
            # Average policy
            self.policy_network.eval()
            with torch.no_grad():
                logits = self._apply_mask(self.policy_network(obs_t).squeeze(0), legal_mask)
                probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1).item())

    def store_transition(
        self,
        obs: npt.NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: npt.NDArray[np.float32],
        done: bool,
    ) -> None:
        self.q_buffer.add(obs, action, reward, next_obs, done)

    def train_step(self) -> dict[str, float | None]:
        self._step += 1

        if self._step % self.train_every != 0:
            return {"q_loss": None, "policy_loss": None}

        q_ready = len(self.q_buffer) >= _Q_MIN
        policy_ready = len(self.policy_buffer) >= _POLICY_MIN
        if not (q_ready and policy_ready):
            return {"q_loss": None, "policy_loss": None}

        # --- Q-network update ---
        self.q_network.train()
        batch = self.q_buffer.sample(self.batch_size)
        obs_t = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_q = self.target_network(next_obs_t).max(dim=1).values
            td_targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        q_vals = self.q_network(obs_t)
        predicted_q = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        q_loss = nn.functional.mse_loss(predicted_q, td_targets)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # --- Policy-network update ---
        self.policy_network.train()
        pbatch = self.policy_buffer.sample(self.batch_size)
        pobs_t = torch.as_tensor(pbatch["obs"], dtype=torch.float32, device=self.device)
        plabels_t = torch.as_tensor(pbatch["actions"], dtype=torch.int64, device=self.device)

        logits = self.policy_network(pobs_t)
        policy_loss = nn.functional.cross_entropy(logits, plabels_t)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # --- Sync target network ---
        if self._step % self.target_sync_every == 0:
            sync_target_network(self.q_network, self.target_network)

        return {"q_loss": q_loss.item(), "policy_loss": policy_loss.item()}

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "policy_network": self.policy_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "step": self._step,
                "config": self._config,
            },
            filepath,
        )

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self._step = checkpoint["step"]
        self._config = checkpoint["config"]
