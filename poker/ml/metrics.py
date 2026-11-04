"""Shared training / validation metrics."""

import numpy as np

from poker.ml.env import REWARD_SCALE

DEFAULT_STACK = 1000
DEFAULT_BB = 50


def reward_to_bb_per_hand(
    reward: float,
    stack: int = DEFAULT_STACK,
    big_blind: int = DEFAULT_BB,
    scale: float = REWARD_SCALE,
) -> float:
    """Convert scaled env reward to big blinds won/lost per hand."""
    stack_change = (reward / scale) * stack
    return stack_change / big_blind


def reward_to_bb100(
    reward: float,
    stack: int = DEFAULT_STACK,
    big_blind: int = DEFAULT_BB,
    scale: float = REWARD_SCALE,
) -> float:
    """Convert scaled env reward to BB/100."""
    return reward_to_bb_per_hand(reward, stack, big_blind, scale) * 100.0


def bb100_interval(rewards: list[float]) -> tuple[float, float, float, int]:
    """Mean BB/100 and the 95% interval for one reward per hand.

    The interval is mean ± 1.96 standard errors. Fewer than two hands
    returns the mean as both bounds.
    """
    if not rewards:
        return 0.0, 0.0, 0.0, 0
    samples = np.array([reward_to_bb100(reward) for reward in rewards], dtype=float)
    mean = float(samples.mean())
    n = int(samples.size)
    if n < 2:
        return mean, mean, mean, n
    stderr = float(samples.std(ddof=1) / np.sqrt(n))
    margin = 1.96 * stderr
    return mean, mean - margin, mean + margin, n
