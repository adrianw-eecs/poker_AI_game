"""Shared training / validation metrics."""

import math

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


def win_rate_interval(wins: int, hands: int, z: float = 1.96) -> tuple[float, float, float, int]:
    """Win rate and its Wilson 95% interval.

    Returns ``(rate, lower, upper, hands)``. An empty sample is all zeros.
    The lower bound is what the gate compares across opponents.
    """
    if hands <= 0:
        return 0.0, 0.0, 0.0, 0
    wins = max(0, min(int(wins), int(hands)))
    proportion = wins / hands
    z2 = z * z
    denom = 1.0 + z2 / hands
    centre = (proportion + z2 / (2.0 * hands)) / denom
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z2 / (4.0 * hands)) / hands) / denom
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return proportion, lower, upper, hands


def equity_share_delta(
    deltas: list[float],
    rng: np.random.Generator | None = None,
    draws: int = 1000,
) -> tuple[float, float, float, int]:
    """Mean prize-share change and a paired bootstrap 95% interval.

    Each sample is one hand's change in winner-take-all equity. The interval
    is the 2.5 and 97.5 percentiles of the resampled means.
    """
    if not deltas:
        return 0.0, 0.0, 0.0, 0
    samples = np.asarray(deltas, dtype=float)
    mean = float(samples.mean())
    count = int(samples.size)
    if count < 2:
        return mean, mean, mean, count
    generator = rng if rng is not None else np.random.default_rng(0)
    indices = generator.integers(0, count, size=(draws, count))
    means = samples[indices].mean(axis=1)
    means.sort()
    lower = float(means[int(0.025 * draws)])
    upper = float(means[min(draws - 1, int(0.975 * draws))])
    return mean, lower, upper, count
