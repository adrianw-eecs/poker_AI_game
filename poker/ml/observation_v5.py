"""Observation v5: v4 with money in big blinds, plus a tournament tail.

The v4 prefix keeps the same offsets. Only the money features inside it change
scale (commitment, pot). The tail is 26 floats in ``[0, 1]`` that describe
the sitting: orbit cost, blind pressure, stack rank, and how soon the hero
posts a blind. ``places_paid`` and ``payout_share`` are winner-take-all
values, so a different payout is a value change rather than a new width.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from poker.ml.observation_v2 import _clip01, _ratio
from poker.ml.observation_v4 import OBS_V4_DIM, build_observation_v4
from poker.ml.tendencies import TendencyTracker
from poker.state.game_state import GameState

TOURNAMENT_TAIL_DIM = 26
OBS_V5_DIM = OBS_V4_DIM + TOURNAMENT_TAIL_DIM

# Winner-take-all. The slots stay so a 50/30/20 table is a value change.
WTA_PLACES_PAID = 1
WTA_PAYOUT_SHARE = 1.0


def build_observation_v5(
    state: GameState,
    seat: int,
    tendencies: TendencyTracker | None = None,
    hand_cap: int = 160,
) -> npt.NDArray[np.float32]:
    """Build the v5 vector. Money in the prefix is in big blinds."""
    base = build_observation_v4(state, seat, tendencies, money_unit="bb")
    tail = np.asarray(_tournament_tail(state, seat, hand_cap), dtype=np.float32)
    if tail.shape[0] != TOURNAMENT_TAIL_DIM:
        raise RuntimeError(
            f"observation v5 tail is {tail.shape[0]} floats, expected {TOURNAMENT_TAIL_DIM}"
        )
    np.clip(tail, 0.0, 1.0, out=tail)
    out = np.empty(OBS_V5_DIM, dtype=np.float32)
    out[:OBS_V4_DIM] = base
    out[OBS_V4_DIM:] = tail
    return out


def _chips(player) -> int:
    return int(player.stack) + int(player.committed_this_hand)


def _tournament_tail(state: GameState, hero: int, hand_cap: int) -> list[float]:
    players = state.players
    field = max(len(players), 1)
    hero_player = players[hero]
    hero_chips = max(_chips(hero_player), 0)
    live = [seat for seat, player in enumerate(players) if not player.is_eliminated and _chips(player) > 0]
    left = max(len(live), 1)
    total = max(sum(_chips(player) for player in players if not player.is_eliminated), 1)
    bb = max(state.config.big_blind, 1)
    sb = max(state.config.small_blind, 0)
    ante = max(state.config.ante, 0)
    orbit = sb + bb + ante * field
    stacks = [_chips(players[seat]) for seat in live] or [0]
    average = sum(stacks) / max(len(stacks), 1)
    largest = max(stacks) if stacks else 1
    ordered = sorted(stacks)
    median = ordered[len(ordered) // 2]
    denom = max(field - 1, 1)
    covers = sum(1 for stack in stacks if stack < hero_chips)
    covered = sum(1 for stack in stacks if stack > hero_chips)
    short_10 = sum(1 for stack in stacks if stack <= 10 * bb)
    short_20 = sum(1 for stack in stacks if stack <= 20 * bb)

    schedule = state.config.blind_schedule
    n_levels = max(len(schedule.levels), 1)
    if schedule.fixed:
        level_index = 0
        hands_left = 1.0
    else:
        level_index = min(state.hand_number // schedule.hands_per_level, n_levels - 1)
        into = state.hand_number % schedule.hands_per_level
        hands_left = (schedule.hands_per_level - into) / schedule.hands_per_level
    initial_bb = max(schedule.levels[0].big, 1)
    cap = max(int(hand_cap), 1)

    behind, hero_last = _players_behind(state, hero)
    until_sb = _hands_until_blind(state, hero, "sb")
    until_bb = _hands_until_blind(state, hero, "bb")

    return [
        _ratio(orbit, max(hero_chips, 1)),
        _clip01((hero_chips / orbit) / 30.0) if orbit > 0 else 0.0,
        _ratio(ante, bb),
        _ratio(ante, max(hero_chips, 1)),
        _ratio(level_index, max(n_levels - 1, 1)),
        _clip01(bb / initial_bb / 8.0),
        _clip01(hands_left),
        _clip01((hero_chips / bb) / 200.0),
        _ratio(hero_chips, max(average, 1)),
        _ratio(hero_chips, max(largest, 1)),
        _ratio(hero_chips, total),
        _ratio(left, field),
        _ratio(field, 9),
        _ratio(max(left - 1, 0), field),
        _ratio(WTA_PLACES_PAID, field),
        _clip01(WTA_PAYOUT_SHARE),
        _ratio(covers, denom),
        _ratio(covered, denom),
        _clip01(median / max(hero_chips, 1)),
        _clip01(until_bb),
        _clip01(until_sb),
        _ratio(short_10, left),
        _ratio(short_20, left),
        _clip01(state.hand_number / cap),
        _ratio(behind, field),
        hero_last,
    ]


def _players_behind(state: GameState, hero: int) -> tuple[int, float]:
    """Players who can still act after the hero, and whether the hero is last."""
    acted = {seat for seat, _action in state.action_history_this_street}
    n = len(state.players)
    behind = 0
    seat = (hero + 1) % n
    for _ in range(n - 1):
        player = state.players[seat]
        if (
            not player.has_folded
            and not player.is_eliminated
            and not player.is_all_in
            and seat not in acted
        ):
            behind += 1
        seat = (seat + 1) % n
    hero_player = state.players[hero]
    hero_can = (
        not hero_player.has_folded
        and not hero_player.is_eliminated
        and not hero_player.is_all_in
    )
    return behind, 1.0 if hero_can and behind == 0 else 0.0


def _live_seats(state: GameState) -> list[int]:
    return [
        seat
        for seat, player in enumerate(state.players)
        if not player.is_eliminated and _chips(player) > 0
    ]


def _button_after(live: list[int], button: int, steps: int, n: int) -> int:
    start = button % n
    for _ in range(n):
        if start in live:
            break
        start = (start + 1) % n
    index = live.index(start)
    return live[(index + steps) % len(live)]


def _blind_seats_from_button(live: list[int], button: int) -> tuple[int, int]:
    index = live.index(button)
    if len(live) == 2:
        return button, live[(index + 1) % 2]
    small = live[(index + 1) % len(live)]
    big = live[(index + 2) % len(live)]
    return small, big


def _hands_until_blind(state: GameState, hero: int, which: str) -> float:
    """Fraction of an orbit until the hero posts the requested blind."""
    live = _live_seats(state)
    if hero not in live or len(live) < 2:
        return 0.0
    n = len(state.players)
    button = _button_after(live, state.dealer_seat, 0, n)
    for step in range(len(live)):
        btn = _button_after(live, button, step, n)
        small, big = _blind_seats_from_button(live, btn)
        posted = big if which == "bb" else small
        if posted == hero:
            return step / len(live)
    return 1.0
