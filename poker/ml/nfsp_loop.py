"""One-hand NFSP transition writer for a shared multi-seat model."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel


@dataclass(frozen=True)
class HandResult:
    """Chip-delta rewards for seats that acted, and whether the hand finished."""

    rewards: dict[int, float]
    finished: bool
    decisions: int


def consume_hand(
    model: NFSPModel,
    env: PokerEnv,
    learning_seats: set[int],
    first: tuple[int, npt.NDArray[np.float32], npt.NDArray[np.int32]],
    max_steps: int = 200,
) -> HandResult:
    """Play one dealt hand and write a transition for every learner decision.

    ``first`` is the ``(seat, obs, mask)`` tuple from ``reset_multi`` or
    ``start_next_hand_multi``. Anticipatory mode is sampled once per learning
    seat. A seat's transition is stored at the end of the hand with the
    full-hand return (terminal chip delta, γ=1, intermediate rewards 0).
    """
    for seat in learning_seats:
        model.begin_episode(seat)

    actor, obs, mask = first
    # One entry per learner decision, not per seat. A seat that acts several
    # times in a hand contributes several transitions; keying by seat alone
    # silently dropped all but its last decision.
    pending: list[tuple[int, npt.NDArray[np.float32], int, npt.NDArray[np.float32]]] = []
    rewards: dict[int, float] = {}
    if actor < 0:
        return HandResult(rewards, True, 0)

    done = False
    steps = 0
    while not done and steps < max_steps:
        action = model.select_action(obs, mask, training=True, seat=actor)
        pending.append(
            (
                actor,
                np.array(obs, dtype=np.float32, copy=True),
                int(action),
                np.asarray(mask, dtype=np.float32),
            )
        )
        actor, obs, mask, step_rewards, done = env.step_multi(action)
        model.train_step()
        steps += 1
        if done:
            terminal = np.zeros(int(model.q_network.input_dim), dtype=np.float32)
            ones = np.ones(model.num_actions, dtype=np.float32)
            for seat, prev_obs, prev_action, prev_mask in pending:
                hand_return = float(step_rewards.get(seat, 0.0))
                model.store_transition(
                    prev_obs,
                    prev_action,
                    hand_return,
                    terminal,
                    True,
                    next_mask=ones,
                    current_mask=prev_mask,
                )
                rewards[seat] = hand_return
            return HandResult(rewards, True, steps)
    return HandResult(rewards, False, steps)
