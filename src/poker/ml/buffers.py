"""Replay buffers for NFSP training."""

import threading
from typing import Any

import numpy as np
import numpy.typing as npt


class CircularBuffer:
    """Fixed-capacity FIFO buffer for RL transitions.

    Overwrites oldest entries when full. Used for the Q-network's
    experience replay buffer.
    """

    def __init__(self, capacity: int, obs_dim: int = 142) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._lock = threading.Lock()

    def add(
        self,
        obs: npt.NDArray[np.float32],
        action: int,
        reward: float,
        next_obs: npt.NDArray[np.float32],
        done: bool,
    ) -> None:
        with self._lock:
            idx = self._pos % self.capacity
            self._obs[idx] = obs
            self._actions[idx] = action
            self._rewards[idx] = reward
            self._next_obs[idx] = next_obs
            self._dones[idx] = float(done)
            self._pos += 1
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, npt.NDArray[Any]]:
        with self._lock:
            indices = np.random.randint(0, self._size, size=batch_size)
            return {
                "obs": self._obs[indices].copy(),
                "actions": self._actions[indices].copy(),
                "rewards": self._rewards[indices].copy(),
                "next_obs": self._next_obs[indices].copy(),
                "dones": self._dones[indices].copy(),
            }

    def __len__(self) -> int:
        with self._lock:
            return self._size


class ReservoirBuffer:
    """Reservoir sampling buffer for behavioral cloning.

    Maintains a uniform sample over all seen data regardless of insertion
    order. Used for the policy network's supervised learning buffer.
    """

    def __init__(self, capacity: int, obs_dim: int = 142) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._size = 0
        self._total_seen = 0
        self._lock = threading.Lock()

    def add(self, obs: npt.NDArray[np.float32], action: int) -> None:
        with self._lock:
            self._total_seen += 1
            if self._size < self.capacity:
                idx = self._size
                self._obs[idx] = obs
                self._actions[idx] = action
                self._size += 1
            else:
                idx = np.random.randint(0, self._total_seen)
                if idx < self.capacity:
                    self._obs[idx] = obs
                    self._actions[idx] = action

    def sample(self, batch_size: int) -> dict[str, npt.NDArray[Any]]:
        with self._lock:
            indices = np.random.randint(0, self._size, size=batch_size)
            return {
                "obs": self._obs[indices].copy(),
                "actions": self._actions[indices].copy(),
            }

    def __len__(self) -> int:
        with self._lock:
            return self._size


class WeightedReservoirBuffer:
    """Reservoir buffer with importance weights for prioritized sampling.

    Extends ReservoirBuffer with per-sample weights that can be used for
    importance-weighted updates.
    """

    def __init__(self, capacity: int, obs_dim: int = 142) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._weights = np.ones(capacity, dtype=np.float32)
        self._size = 0
        self._total_seen = 0
        self._lock = threading.Lock()

    def add(self, obs: npt.NDArray[np.float32], action: int, weight: float = 1.0) -> None:
        with self._lock:
            self._total_seen += 1
            if self._size < self.capacity:
                idx = self._size
                self._obs[idx] = obs
                self._actions[idx] = action
                self._weights[idx] = weight
                self._size += 1
            else:
                idx = np.random.randint(0, self._total_seen)
                if idx < self.capacity:
                    self._obs[idx] = obs
                    self._actions[idx] = action
                    self._weights[idx] = weight

    def sample(self, batch_size: int) -> dict[str, npt.NDArray[Any]]:
        with self._lock:
            probs = self._weights[: self._size]
            probs = probs / probs.sum()
            indices = np.random.choice(self._size, size=batch_size, p=probs, replace=True)
            return {
                "obs": self._obs[indices].copy(),
                "actions": self._actions[indices].copy(),
                "weights": self._weights[indices].copy(),
            }

    def __len__(self) -> int:
        with self._lock:
            return self._size
