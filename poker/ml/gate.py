"""Head-to-head and baseline cells used to keep a checkpoint.

The learner and the previous best share one table, so the comparison is paired
by the deck. Baseline cells are separate: 6-max, equal stacks, against Call,
Flop, and Random. The checkpoint is chosen on the minimum equity-share lower
bound across those three baseline cells. A big win against one of them cannot
hide a loss to another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.action_space import BetScheme
from poker.ml.env import REWARD_SCALE, PokerEnv
from poker.ml.metrics import equity_share_delta, win_rate_interval
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.sitting import (
    TOURNAMENT_LEVELS,
    buyin_chips,
    chip_pool,
    nearest_level_index,
    schedule_from_level,
)
from poker.state.game_state import Street

_BASELINE_FIELD = 6
_HEAD_TO_HEAD_FIELDS = (2, 4, 6, 9)
_HEAD_TO_HEAD_DEPTHS = (30, 10)


@dataclass(frozen=True)
class CellResult:
    """One gate cell."""

    name: str
    wins: int
    hands: int
    win_rate: float
    win_low: float
    win_high: float
    share_mean: float
    share_low: float
    share_high: float
    bb_per_hand: float


@dataclass(frozen=True)
class GateReport:
    """Every cell from one gate run. Selection reads ``baselines`` only."""

    baselines: tuple[CellResult, ...]
    head_to_head: tuple[CellResult, ...]

    @property
    def selection_lower(self) -> float:
        if not self.baselines:
            return 0.0
        return min(cell.share_low for cell in self.baselines)


def _share_delta(reward: float) -> float:
    return float(reward) / REWARD_SCALE if REWARD_SCALE else float(reward)


def _cell(name: str, rewards: list[float], total_chips: int, big_blind: int, rng: np.random.Generator) -> CellResult:
    wins = sum(1 for reward in rewards if reward > 1e-8)
    rate, low, high, hands = win_rate_interval(wins, len(rewards))
    deltas = [_share_delta(reward) for reward in rewards]
    mean, share_low, share_high, _count = equity_share_delta(deltas, rng)
    bb_per_hand = mean * total_chips / max(big_blind, 1)
    return CellResult(
        name=name,
        wins=wins,
        hands=hands,
        win_rate=rate,
        win_low=low,
        win_high=high,
        share_mean=mean,
        share_low=share_low,
        share_high=share_high,
        bb_per_hand=bb_per_hand,
    )


def _baseline_bots(kind: str, count: int, rng: np.random.Generator) -> list:
    bots = []
    for index in range(count):
        if kind == "call":
            bots.append(CallBot(name=f"Call_{index}"))
        elif kind == "flop":
            bots.append(FlopBot(name=f"Flop_{index}"))
        else:
            bots.append(RandomBot(name=f"Random_{index}", seed=int(rng.integers(0, 2**31 - 1))))
    return bots


def _play_single(
    model: NFSPModel,
    *,
    num_players: int,
    stacks: tuple[int, ...],
    big_blind: int,
    small_blind: int,
    ante: int,
    opponents: list,
    seed: int,
    hands: int,
) -> list[float]:
    """One fresh hand per deck. The learner is seat 0."""
    rewards: list[float] = []
    level_index = nearest_level_index(big_blind)
    schedule = schedule_from_level(level_index, hands_per_level=10**6)
    # Force the posted blind to the requested level even if snapping moved it.
    schedule.levels[0] = type(schedule.levels[0])(small_blind, big_blind, ante)
    for hand in range(hands):
        env = PokerEnv(
            num_players=num_players,
            starting_stack=max(stacks),
            learning_seat=0,
            opponent_bots=opponents,
            seed=seed + hand,
            obs_version=int(model.obs_version),
            bet_scheme=BetScheme.POT_FRACTION,
            blind_schedule=schedule,
        )
        obs, _info = env.reset(options={"stacks": stacks, "deck_seed": seed + hand})
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 200:
            if (
                env.state is None
                or env.state.action_on_seat != 0
                or env.state.street == Street.SHOWDOWN
            ):
                break
            action = model.select_action(obs, env.get_action_mask(), training=False)
            obs, reward, done, _, _ = env.step(action)
            steps += 1
        rewards.append(float(reward))
    return rewards


def _equal_stacks(field: int, depth_bb: int) -> tuple[tuple[int, ...], int, int, int, int]:
    """Equal stacks whose average depth is about ``depth_bb`` on the snapped level."""
    total = chip_pool(field)
    target = max(1, int(round(total / (field * depth_bb))))
    index = nearest_level_index(target)
    level = TOURNAMENT_LEVELS[index]
    base, extra = divmod(total, field)
    stacks = tuple(base + (1 if seat < extra else 0) for seat in range(field))
    return stacks, level.small, level.big, level.ante, total


def _play_paired(
    learner: NFSPModel,
    previous: NFSPModel,
    *,
    num_players: int,
    stacks: tuple[int, ...],
    small_blind: int,
    big_blind: int,
    ante: int,
    seed: int,
    hands: int,
    rng: np.random.Generator,
) -> list[float]:
    """Learner in seat 0, previous best in seat 1, one shared deck per hand."""
    rewards: list[float] = []
    level_index = nearest_level_index(big_blind)
    schedule = schedule_from_level(level_index, hands_per_level=10**6)
    schedule.levels[0] = type(schedule.levels[0])(small_blind, big_blind, ante)
    others = num_players - 2
    for hand in range(hands):
        opponents = _baseline_bots("random", others, rng) if others else []
        env = PokerEnv(
            num_players=num_players,
            starting_stack=max(stacks),
            learning_seat=0,
            learning_seats={0, 1},
            opponent_bots=opponents,
            seed=seed + hand,
            obs_version=int(learner.obs_version),
            bet_scheme=BetScheme.POT_FRACTION,
            blind_schedule=schedule,
        )
        actor, obs, mask = env.reset_multi(
            {"stacks": stacks, "deck_seed": seed + 10_000 + hand}
        )
        models = {0: learner, 1: previous}
        done = False
        reward = 0.0
        steps = 0
        while not done and steps < 200 and actor >= 0:
            action = models[actor].select_action(obs, mask, training=False, seat=actor)
            actor, obs, mask, step_rewards, done = env.step_multi(action)
            steps += 1
            if done:
                reward = float(step_rewards.get(0, 0.0))
        rewards.append(reward)
    return rewards


def evaluate_gate(
    model: NFSPModel,
    previous: NFSPModel | None,
    *,
    hands: int,
    seed: int = 17_001,
    fields: tuple[int, ...] = _HEAD_TO_HEAD_FIELDS,
    depths: tuple[int, ...] = _HEAD_TO_HEAD_DEPTHS,
) -> GateReport:
    """Run the baseline cells and, when a previous best exists, the paired cells."""
    rng = np.random.default_rng(seed)
    model.q_network.eval()
    model.policy_network.eval()
    baselines: list[CellResult] = []
    buyin = buyin_chips()
    equal = tuple(buyin for _ in range(_BASELINE_FIELD))
    level = TOURNAMENT_LEVELS[0]
    for kind in ("call", "flop", "random"):
        opponents = _baseline_bots(kind, _BASELINE_FIELD - 1, rng)
        rewards = _play_single(
            model,
            num_players=_BASELINE_FIELD,
            stacks=equal,
            big_blind=level.big,
            small_blind=level.small,
            ante=level.ante,
            opponents=opponents,
            seed=seed + {"call": 11, "flop": 22, "random": 33}[kind],
            hands=hands,
        )
        baselines.append(
            _cell(
                f"6max_{kind}",
                rewards,
                total_chips=buyin * _BASELINE_FIELD,
                big_blind=level.big,
                rng=rng,
            )
        )

    paired: list[CellResult] = []
    if previous is not None and int(previous.obs_version) == int(model.obs_version):
        previous.q_network.eval()
        previous.policy_network.eval()
        for field in fields:
            if field < 2:
                continue
            for depth in depths:
                stacks, small, big, ante, total = _equal_stacks(field, depth)
                rewards = _play_paired(
                    model,
                    previous,
                    num_players=field,
                    stacks=stacks,
                    small_blind=small,
                    big_blind=big,
                    ante=ante,
                    seed=seed + field * 100 + depth,
                    hands=hands,
                    rng=rng,
                )
                paired.append(
                    _cell(
                        f"h2h_{field}max_{depth}bb",
                        rewards,
                        total_chips=total,
                        big_blind=big,
                        rng=rng,
                    )
                )
    model.q_network.train()
    model.policy_network.train()
    return GateReport(baselines=tuple(baselines), head_to_head=tuple(paired))
