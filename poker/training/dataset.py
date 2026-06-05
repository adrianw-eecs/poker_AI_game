"""Replay buffer and self-play data collection."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class Experience:
    """A single decision point (state, action, reward, legal_mask)."""

    observation: npt.NDArray[np.float32]  # (142,)
    action: int  # 0-6
    reward: float  # normalized stack change
    legal_mask: npt.NDArray[np.int32]  # (7,) binary mask
    seat: int
    hand_id: int


class ReplayBuffer:
    """In-memory replay buffer for Monte-Carlo Q-learning data.

    Stores experience tuples from self-play. One experience per decision point.
    All decisions in a hand get the same end-of-hand reward (Monte-Carlo return).
    """

    def __init__(self, max_size: int = 1_000_000) -> None:
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of experiences to store.
        """
        self.max_size = max_size
        self.observations = np.zeros((max_size, 142), dtype=np.float32)
        self.actions = np.zeros(max_size, dtype=np.int32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.legal_masks = np.zeros((max_size, 7), dtype=np.int32)
        self.seats = np.zeros(max_size, dtype=np.int32)
        self.hand_ids = np.zeros(max_size, dtype=np.int32)
        self.size = 0

    def add(self, experience: Experience) -> None:
        """Add one experience to the buffer.

        Args:
            experience: The experience to add.

        Raises:
            RuntimeError: If buffer is full.
        """
        if self.size >= self.max_size:
            raise RuntimeError(f"Replay buffer full ({self.max_size} experiences)")

        idx = self.size
        self.observations[idx] = experience.observation
        self.actions[idx] = experience.action
        self.rewards[idx] = experience.reward
        self.legal_masks[idx] = experience.legal_mask
        self.seats[idx] = experience.seat
        self.hand_ids[idx] = experience.hand_id
        self.size += 1

    def sample_batch(self, batch_size: int) -> dict[str, npt.NDArray]:
        """Sample a random batch from the buffer.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            Dictionary with keys: 'observations', 'actions', 'rewards', 'legal_masks'.
        """
        if self.size == 0:
            raise RuntimeError("Cannot sample from empty buffer")

        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)

        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "legal_masks": self.legal_masks[indices],
        }

    def save(self, filepath: str) -> None:
        """Save buffer to disk as .npz file.

        Args:
            filepath: Path to save to (e.g., 'data/replay_buffer.npz').
        """
        np.savez_compressed(
            filepath,
            observations=self.observations[: self.size],
            actions=self.actions[: self.size],
            rewards=self.rewards[: self.size],
            legal_masks=self.legal_masks[: self.size],
            seats=self.seats[: self.size],
            hand_ids=self.hand_ids[: self.size],
        )
        print(f"Saved {self.size} experiences to {filepath}")

    def load(self, filepath: str) -> None:
        """Load buffer from disk.

        Args:
            filepath: Path to load from.
        """
        data = np.load(filepath)
        n = len(data["observations"])

        if n > self.max_size:
            print(f"Warning: Truncating loaded data from {n} to {self.max_size}")
            n = self.max_size

        self.observations[:n] = data["observations"][:n]
        self.actions[:n] = data["actions"][:n]
        self.rewards[:n] = data["rewards"][:n]
        self.legal_masks[:n] = data["legal_masks"][:n]
        self.seats[:n] = data["seats"][:n]
        self.hand_ids[:n] = data["hand_ids"][:n]
        self.size = n
        print(f"Loaded {n} experiences from {filepath}")

    def stats(self) -> dict[str, float]:
        """Return buffer statistics.

        Returns:
            Dictionary with stats like mean_reward, min_reward, max_reward, etc.
        """
        if self.size == 0:
            return {}

        rewards = self.rewards[: self.size]
        return {
            "num_experiences": self.size,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
        }
