"""Tournament sittings: field size, stack profile, and blind schedule.

The environment deals and settles hands. This module only decides who sits
down, with how many chips, and which blind level that chip pool corresponds
to. A sitting is one table from the first hand until one stack remains or
the hand cap cuts it off.

The chip pool is ``field_size * 100`` big blinds of the first schedule
level. The posted big blind is then chosen so the average stack is about
100bb, 30bb, or 12bb, and snapped to the nearest level. Profiles and blinds
therefore agree by construction, instead of 8bb spots showing up only when
a deep table happens to run long.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from poker.config.blind_schedule import BlindLevel, BlindSchedule

# 25/50 through 500/1000. Antes start at 100/200 so a sitting that reaches
# the middle of the schedule actually posts them.
TOURNAMENT_LEVELS: tuple[BlindLevel, ...] = (
    BlindLevel(25, 50, 0),
    BlindLevel(50, 100, 0),
    BlindLevel(75, 150, 0),
    BlindLevel(100, 200, 25),
    BlindLevel(150, 300, 50),
    BlindLevel(200, 400, 50),
    BlindLevel(300, 600, 75),
    BlindLevel(500, 1000, 100),
)

PROFILE_DEPTH_BB: dict[str, int] = {
    "level_start": 100,
    "middle": 30,
    "end": 12,
}

BUYIN_BB = 100


@dataclass(frozen=True)
class SittingSpec:
    """One table: stacks, the blinds it starts at, and when to stop."""

    field_size: int
    starting_stacks: tuple[int, ...]
    schedule: BlindSchedule
    hand_cap: int
    profile: str
    button_seat: int
    level_index: int
    target_depth_bb: int


@dataclass(frozen=True)
class SittingResult:
    """What a finished sitting owes a later multi-table driver.

    ``finish_position`` is 1 for the unique chip leader and higher as more
    seats cover the hero. ``censored`` is set when the hand cap stopped the
    table before a winner.
    """

    finish_position: int
    hands_played: int
    final_stacks: tuple[int, ...]
    censored: bool


def chip_pool(field_size: int) -> int:
    """Total chips at the table: ``field_size`` buy-ins of 100 opening big blinds."""
    return field_size * BUYIN_BB * TOURNAMENT_LEVELS[0].big


def buyin_chips() -> int:
    """Nominal per-player buy-in. The env uses this as ``starting_stack``."""
    return BUYIN_BB * TOURNAMENT_LEVELS[0].big


def nearest_level_index(target_big_blind: int) -> int:
    """Index of the schedule level whose big blind is closest to ``target_big_blind``.

    A tie prefers the smaller blind, which is the earlier level.
    """
    return min(
        range(len(TOURNAMENT_LEVELS)),
        key=lambda index: (abs(TOURNAMENT_LEVELS[index].big - target_big_blind), index),
    )


def schedule_from_level(level_index: int, hands_per_level: int) -> BlindSchedule:
    """Escalating schedule whose first level is ``TOURNAMENT_LEVELS[level_index]``."""
    index = min(max(level_index, 0), len(TOURNAMENT_LEVELS) - 1)
    return BlindSchedule(
        levels=list(TOURNAMENT_LEVELS[index:]),
        hands_per_level=hands_per_level,
        fixed=False,
    )


def finish_position(stacks: tuple[int, ...] | list[int], hero_seat: int) -> int:
    """1-based place of ``hero_seat``. Ties take the better place."""
    hero = stacks[hero_seat]
    return 1 + sum(1 for stack in stacks if stack > hero)


def is_censored(
    hands_played: int,
    hand_cap: int,
    live_players: int,
    *,
    incomplete: bool,
) -> bool:
    """True when the table was cut off before a winner.

    A busted hero is a real finish for that seat. Losing the tournament does
    not censor the sitting. ``incomplete`` is reserved for a hand that failed
    to reach showdown, which is not a result that can be ranked.
    """
    if incomplete:
        return True
    return live_players >= 2 and hands_played >= hand_cap


def make_result(
    stacks: tuple[int, ...] | list[int],
    hero_seat: int,
    hands_played: int,
    hand_cap: int,
    *,
    live_players: int,
    incomplete: bool = False,
) -> SittingResult:
    """Build a result from the stacks left at the table."""
    frozen = tuple(int(stack) for stack in stacks)
    return SittingResult(
        finish_position=finish_position(frozen, hero_seat),
        hands_played=hands_played,
        final_stacks=frozen,
        censored=is_censored(
            hands_played, hand_cap, live_players, incomplete=incomplete
        ),
    )


def draw_sitting(
    rng: np.random.Generator,
    *,
    min_players: int = 2,
    max_players: int = 9,
    profiles: tuple[str, ...] = ("level_start", "middle", "end"),
    weights: tuple[float, ...] = (0.4, 0.4, 0.2),
    hands_per_level: tuple[int, int] = (6, 10),
    hand_cap: int = 160,
) -> SittingSpec:
    """Draw one sitting.

    ``end`` uses 3 to 6 seats when that range fits inside the requested
    field. ``hands_per_level`` is an inclusive ``(low, high)`` draw.
    """
    if min_players < 2 or max_players > 9 or min_players > max_players:
        raise ValueError(f"field must be inside 2..9, got {min_players}..{max_players}")
    if len(profiles) != len(weights) or not profiles:
        raise ValueError("profiles and weights must be the same non-empty length")
    unknown = [name for name in profiles if name not in PROFILE_DEPTH_BB]
    if unknown:
        raise ValueError(f"unknown profile(s): {unknown}")
    if hand_cap < 1:
        raise ValueError(f"hand_cap must be positive, got {hand_cap}")
    low, high = hands_per_level
    if low > high:
        low, high = high, low
    if low < 1:
        raise ValueError(f"hands_per_level must be positive, got {hands_per_level}")

    weight = np.asarray(weights, dtype=np.float64)
    if np.any(weight < 0) or float(weight.sum()) <= 0:
        raise ValueError(f"profile weights must be non-negative and sum > 0, got {weights}")
    weight = weight / weight.sum()
    profile = str(rng.choice(list(profiles), p=weight))
    field_size = _field_size(rng, profile, min_players, max_players)
    depth = PROFILE_DEPTH_BB[profile]
    total = chip_pool(field_size)
    target_bb = max(1, int(round(total / (field_size * depth))))
    level_index = nearest_level_index(target_bb)
    level = TOURNAMENT_LEVELS[level_index]
    per_level = int(rng.integers(low, high + 1))
    stacks = _stacks_for(profile, field_size, total, level.big, rng)
    button = int(rng.integers(0, field_size))
    return SittingSpec(
        field_size=field_size,
        starting_stacks=stacks,
        schedule=schedule_from_level(level_index, per_level),
        hand_cap=hand_cap,
        profile=profile,
        button_seat=button,
        level_index=level_index,
        target_depth_bb=depth,
    )


def _field_size(
    rng: np.random.Generator, profile: str, min_players: int, max_players: int
) -> int:
    low, high = min_players, max_players
    if profile == "end":
        low = max(low, 3)
        high = min(high, 6)
        if low > high:
            low, high = min_players, max_players
    return int(rng.integers(low, high + 1))


def _stacks_for(
    profile: str,
    field_size: int,
    total: int,
    big_blind: int,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    if profile == "level_start":
        return tuple(_split_equal(total, field_size))
    if profile == "end":
        return _end_stacks(field_size, total, big_blind, rng)
    return _middle_stacks(field_size, total, big_blind, rng)


def _split_equal(total: int, field_size: int) -> list[int]:
    base, extra = divmod(total, field_size)
    return [base + (1 if seat < extra else 0) for seat in range(field_size)]


def _draw_depth(
    rng: np.random.Generator, low: int, high: int, spare_bb: int
) -> int:
    """A depth in ``[low, high]`` that does not spend more than ``spare_bb``."""
    cap = min(high, max(low, spare_bb))
    if cap <= low:
        return low
    return int(rng.integers(low, cap + 1))


def _middle_stacks(
    field_size: int, total: int, big_blind: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """One larger stack, up to two stacks at 8-15bb, the rest at 30-60bb.

    The bands are reserved before anyone is allowed to grow. Proportional
    scaling used to pull the 30-60bb seats out of that band so the shape
    never showed up at 6-max.
    """
    if field_size < 2 or big_blind <= 0:
        return tuple(_split_equal(total, field_size))
    pool_bb = total // big_blind
    n_short = min(2, field_size - 1)
    n_mid = field_size - 1 - n_short
    if n_mid == 0:
        max_short = min(15, max(8, pool_bb - 30))
        depth = _draw_depth(rng, 8, max_short, max_short)
        depths = [depth, pool_bb - depth]
    else:
        floor = 8 * n_short + 30 * n_mid
        big_floor = 60 if floor + 60 <= pool_bb else max(1, pool_bb - floor)
        if floor + big_floor > pool_bb:
            return tuple(_split_equal(total, field_size))
        spare = pool_bb - floor - big_floor
        shorts = []
        for _ in range(n_short):
            grow = int(rng.integers(0, min(7, spare) + 1))
            shorts.append(8 + grow)
            spare -= grow
        mids = []
        for _ in range(n_mid):
            grow = int(rng.integers(0, min(30, spare) + 1))
            mids.append(30 + grow)
            spare -= grow
        depths = shorts + mids + [big_floor + spare]
    chips = [depth * big_blind for depth in depths]
    chips[int(np.argmax(chips))] += total - sum(chips)
    rng.shuffle(chips)
    return tuple(int(stack) for stack in chips)


def _end_stacks(
    field_size: int, total: int, big_blind: int, rng: np.random.Generator
) -> tuple[int, ...]:
    """One covering stack and the rest between one and 14 big blinds.

    A stack below one big blind is eliminated by the ante or the blind on the
    first hand, which silently shrinks the field. Every seat therefore starts
    with at least one big blind.
    """
    if field_size < 2 or big_blind <= 0:
        return tuple(_split_equal(total, field_size))
    pool_bb = total // big_blind
    n_short = field_size - 1
    # The cover has to stay strictly larger than every short stack.
    max_short = max(1, min(14, (pool_bb - 1) // field_size))
    shorts: list[int] = []
    spare = pool_bb
    for index in range(n_short):
        reserve = (n_short - index - 1) + (max_short + 1)
        room = min(max_short, spare - reserve)
        depth = _draw_depth(rng, 1, max_short, max(1, room))
        shorts.append(depth)
        spare -= depth
    cover = spare
    if cover <= max(shorts):
        # Pull chips off the largest shorts until the cover is strictly ahead.
        while cover <= max(shorts) and max(shorts) > 1:
            donor = int(np.argmax(shorts))
            shorts[donor] -= 1
            cover += 1
    depths = shorts + [cover]
    chips = [depth * big_blind for depth in depths]
    chips[int(np.argmax(chips))] += total - sum(chips)
    rng.shuffle(chips)
    return tuple(int(stack) for stack in chips)
