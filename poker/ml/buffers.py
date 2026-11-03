"""Replay buffers for NFSP and SD-CFR training."""

import threading
from typing import Any

import numpy as np
import numpy.typing as npt

_NUM_ACTIONS = 7  # Discrete action space width


class CircularBuffer:
    """Fixed-capacity FIFO buffer for RL transitions.

    Stores next-state legal masks alongside each transition so that TD
    targets can mask out illegal actions in the next state (Fix #1).
    """

    def __init__(
            self,
            capacity: int,
            obs_dim: int = 98,
            dtype: type[np.generic] | np.dtype = np.float32,
        ) -> None:
            self.capacity = capacity
            self.obs_dim = obs_dim
            self.dtype = np.dtype(dtype)
            self._obs = np.zeros((capacity, obs_dim), dtype=self.dtype)
            self._next_obs = np.zeros((capacity, obs_dim), dtype=self.dtype)
            self._next_masks = np.ones((capacity, _NUM_ACTIONS), dtype=np.float32)
            self._current_masks = np.ones((capacity, _NUM_ACTIONS), dtype=np.float32)
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
        next_mask: npt.NDArray[np.float32] | None = None,
        current_mask: npt.NDArray[np.float32] | None = None,
    ) -> None:
        with self._lock:
            idx = self._pos % self.capacity
            self._obs[idx] = obs
            self._actions[idx] = action
            self._rewards[idx] = reward
            self._next_obs[idx] = next_obs
            self._dones[idx] = float(done)
            # Store next-state legal mask; default to all-legal if not supplied.
            self._next_masks[idx] = next_mask if next_mask is not None else 1.0
            # Store current-state legal mask; default to all-legal if not supplied.
            self._current_masks[idx] = current_mask if current_mask is not None else 1.0
            self._pos += 1
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, npt.NDArray[Any]]:
        with self._lock:
            indices = np.random.randint(0, self._size, size=batch_size)
            return {
                "obs": self._obs[indices].astype(np.float32, copy=True),
                "actions": self._actions[indices].copy(),
                "rewards": self._rewards[indices].copy(),
                "next_obs": self._next_obs[indices].astype(np.float32, copy=True),
                "next_masks": self._next_masks[indices].copy(),
                "dones": self._dones[indices].copy(),
                "current_masks": self._current_masks[indices].copy(),
            }

    def state_dict(self) -> dict[str, Any]:
        """Serialize buffer contents for checkpointing."""
        with self._lock:
            s = self._size
            return {
                "obs": self._obs[:s].copy(),
                "next_obs": self._next_obs[:s].copy(),
                "next_masks": self._next_masks[:s].copy(),
                "current_masks": self._current_masks[:s].copy(),
                "actions": self._actions[:s].copy(),
                "rewards": self._rewards[:s].copy(),
                "dones": self._dones[:s].copy(),
                "pos": self._pos,
                "size": s,
            }

    def recent_state_dict(self, max_entries: int) -> dict[str, Any]:
        """Most recent entries in insertion order, with the write cursor reset.

        A capped checkpoint must not keep the live circular cursor. That cursor
        points outside the saved prefix, so the next writes land on zeroed rows
        that are then sampled as real transitions.
        """
        with self._lock:
            s = self._size
            n = min(max_entries, s)
            if n <= 0:
                return {
                    "obs": np.zeros((0, self.obs_dim), dtype=self.dtype),
                    "next_obs": np.zeros((0, self.obs_dim), dtype=self.dtype),
                    "next_masks": np.zeros((0, _NUM_ACTIONS), dtype=np.float32),
                    "current_masks": np.zeros((0, self.obs_dim), dtype=self.dtype),
                    "actions": np.zeros(0, dtype=np.int64),
                    "rewards": np.zeros(0, dtype=np.float32),
                    "dones": np.zeros(0, dtype=np.float32),
                    "pos": 0,
                    "size": 0,
                }
            if s < self.capacity:
                idx = np.arange(s - n, s)
            else:
                oldest = self._pos % self.capacity
                idx = (oldest + np.arange(s - n, s)) % self.capacity
            return {
                "obs": self._obs[idx].copy(),
                "next_obs": self._next_obs[idx].copy(),
                "next_masks": self._next_masks[idx].copy(),
                "current_masks": self._current_masks[idx].copy(),
                "actions": self._actions[idx].copy(),
                "rewards": self._rewards[idx].copy(),
                "dones": self._dones[idx].copy(),
                "pos": n,
                "size": n,
            }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        with self._lock:
            s = d["size"]
            self._obs[:s] = d["obs"]
            self._next_obs[:s] = d["next_obs"]
            self._next_masks[:s] = d.get(
                "next_masks", np.ones((s, _NUM_ACTIONS), dtype=np.float32)
            )
            self._current_masks[:s] = d.get(
                "current_masks", np.ones((s, _NUM_ACTIONS), dtype=np.float32)
            )
            self._actions[:s] = d["actions"]
            self._rewards[:s] = d["rewards"]
            self._dones[:s] = d["dones"]
            self._size = s
            # Unwrapped buffers are a prefix. A capped save used to persist the
            # old circular cursor, which then wrote into zeroed slots.
            if s < self.capacity:
                self._pos = s
            else:
                self._pos = int(d["pos"])

    def keep_recent_fraction(self, fraction: float, min_size: int = 4096) -> int:
        """Drop oldest entries; keep the most recent `fraction` of the buffer."""
        with self._lock:
            if self._size == 0:
                return 0
            keep = max(min_size, int(self._size * fraction))
            keep = min(keep, self._size)
            if keep >= self._size:
                return 0
            dropped = self._size - keep

            if self._size < self.capacity:
                order = list(range(self._size))
            else:
                order = [(self._pos + i) % self.capacity for i in range(self._size)]
            keep_idx = np.array(order[-keep:], dtype=np.int64)

            self._obs[:keep] = self._obs[keep_idx]
            self._next_obs[:keep] = self._next_obs[keep_idx]
            self._next_masks[:keep] = self._next_masks[keep_idx]
            self._current_masks[:keep] = self._current_masks[keep_idx]
            self._actions[:keep] = self._actions[keep_idx]
            self._rewards[:keep] = self._rewards[keep_idx]
            self._dones[:keep] = self._dones[keep_idx]
            self._size = keep
            self._pos = keep
            return dropped

    def __len__(self) -> int:
        with self._lock:
            return self._size


class ReservoirBuffer:
    """Reservoir sampling buffer for the NFSP average-policy memory.

    While the buffer is filling and every seen row is stored, new rows append.
    After that, each new row replaces a uniform slot with probability
    ``max(width / n_entries_seen, min_prob)`` (Steinberger ActionBufferFLAT).
    ``min_prob=0`` is vanilla reservoir sampling. A positive floor is
    exponential averaging, which the NFSP paper found unstable at small scale.

    A capped checkpoint keeps ``total_seen`` from the full stream. New rows
    then replace inside the restored sample instead of appending into the
    unused capacity. Appending would turn the memory into a sliding window,
    which diverges NFSP.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int = 98,
        min_prob: float = 0.0,
        dtype: type[np.generic] | np.dtype = np.float32,
    ) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.min_prob = min_prob
        self.dtype = np.dtype(dtype)
        self._obs = np.zeros((capacity, obs_dim), dtype=self.dtype)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._masks = np.ones((capacity, _NUM_ACTIONS), dtype=np.float32)
        self._size = 0
        self._total_seen = 0
        self._lock = threading.Lock()

    def add(
        self,
        obs: npt.NDArray[np.float32],
        action: int,
        legal_mask: npt.NDArray[np.float32] | None = None,
    ) -> None:
        mask = (
            np.ones(_NUM_ACTIONS, dtype=np.float32)
            if legal_mask is None
            else np.asarray(legal_mask, dtype=np.float32)
        )
        with self._lock:
            # Append only while the stream and the stored prefix are the same
            # set. A restored subsample has total_seen > size and must not fill
            # the empty tail with only the newest rows.
            if self._size < self.capacity and self._total_seen == self._size:
                self._write(self._size, obs, action, mask)
                self._size += 1
            else:
                seen = max(self._total_seen, 1)
                width = self._size if self._size < self.capacity else self.capacity
                prob_add = max(width / seen, self.min_prob)
                if width > 0 and np.random.random() < prob_add:
                    slot = int(np.random.randint(0, width))
                    self._write(slot, obs, action, mask)
            self._total_seen += 1

    def _write(
        self,
        idx: int,
        obs: npt.NDArray[np.float32],
        action: int,
        mask: npt.NDArray[np.float32],
    ) -> None:
        self._obs[idx] = obs
        self._actions[idx] = action
        self._masks[idx] = mask

    def sample(self, batch_size: int) -> dict[str, npt.NDArray[Any]]:
        with self._lock:
            indices = np.random.randint(0, self._size, size=batch_size)
            return {
                "obs": self._obs[indices].astype(np.float32, copy=True),
                "actions": self._actions[indices].copy(),
                "masks": self._masks[indices].copy(),
            }

    def state_dict(self) -> dict[str, Any]:
        with self._lock:
            s = self._size
            return {
                "obs": self._obs[:s].copy(),
                "actions": self._actions[:s].copy(),
                "masks": self._masks[:s].copy(),
                "size": s,
                "total_seen": self._total_seen,
            }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        with self._lock:
            s = d["size"]
            self._obs[:s] = d["obs"]
            self._actions[:s] = d["actions"]
            if "masks" in d:
                self._masks[:s] = d["masks"]
            else:
                self._masks[:s] = 1.0
            self._size = s
            self._total_seen = d["total_seen"]

    def __len__(self) -> int:
        with self._lock:
            return self._size


class WeightedCircularBuffer:
    """FIFO buffer with weight-biased sampling for SD-CFR regret replay.

    Insertion is FIFO so the buffer always contains the most recent traversal
    data. Sampling is biased by stored weights so that later CFR iterations
    (carrying higher sqrt-iteration weights) dominate gradient updates.

    This replaces WeightedReservoirBuffer for SD-CFR, which had a fundamental
    contradiction: reservoir insertion ignores weights (uniform retention) but
    sampling then biases by weight — the two distributions conflict.
    """

    def __init__(self, capacity: int, obs_dim: int = 98) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._weights = np.ones(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._lock = threading.Lock()

    def add(self, obs: npt.NDArray[np.float32], action: int, weight: float = 1.0) -> None:
        with self._lock:
            idx = self._pos % self.capacity
            self._obs[idx] = obs
            self._actions[idx] = action
            self._weights[idx] = max(weight, 1e-8)
            self._pos += 1
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, npt.NDArray[Any]]:
        with self._lock:
            w = self._weights[: self._size]
            probs = w / w.sum()
            indices = np.random.choice(self._size, size=batch_size, p=probs, replace=True)
            return {
                "obs": self._obs[indices].copy(),
                "actions": self._actions[indices].copy(),
                "weights": self._weights[indices].copy(),
            }

    def state_dict(self) -> dict[str, Any]:
        """Serialize buffer contents for checkpointing."""
        with self._lock:
            s = self._size
            return {
                "obs": self._obs[:s].copy(),
                "actions": self._actions[:s].copy(),
                "weights": self._weights[:s].copy(),
                "pos": self._pos,
                "size": s,
            }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        with self._lock:
            s = d["size"]
            self._obs[:s] = d["obs"]
            self._actions[:s] = d["actions"]
            self._weights[:s] = d["weights"]
            self._pos = d["pos"]
            self._size = s

    def __len__(self) -> int:
        with self._lock:
            return self._size


class WeightedReservoirBuffer:
    """Reservoir buffer with importance weights.

    Kept for backward compatibility with existing tests. For new SD-CFR
    training use WeightedCircularBuffer which has consistent insertion and
    sampling distributions.
    """

    def __init__(self, capacity: int, obs_dim: int = 98) -> None:
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
