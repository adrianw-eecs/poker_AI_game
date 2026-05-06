"""Tests for action space utilities."""

import numpy as np
import pytest

from poker.ml.action_space import ActionSpace


def test_action_space_discrete() -> None:
    space = ActionSpace()
    assert space.num_actions == 7
    assert space.contains(0)
    assert space.contains(6)
    assert not space.contains(7)


def test_action_space_sample() -> None:
    space = ActionSpace()
    mask = np.array([1, 1, 0, 0, 0, 0, 0], dtype=np.int32)
    for _ in range(10):
        assert space.sample(mask) in [0, 1]


def test_action_space_sample_all_legal() -> None:
    space = ActionSpace()
    mask = np.ones(7, dtype=np.int32)
    sampled = set(space.sample(mask) for _ in range(100))
    assert len(sampled) > 1
