"""Sitting generator: depths, chip pool, blinds, and a carried-stack table."""

from dataclasses import replace as dc_replace

import numpy as np
import pytest

from poker.bots.call_bot import CallBot
from poker.config.blind_schedule import BlindLevel
from poker.ml.env import PokerEnv
from poker.ml.sitting import (
    TOURNAMENT_LEVELS,
    buyin_chips,
    chip_pool,
    draw_sitting,
    finish_position,
    is_censored,
    nearest_level_index,
)
from poker.state.game_state import Street


def _finish(env: PokerEnv) -> None:
    if env.state is not None and env.state.street == Street.SHOWDOWN:
        return
    done = False
    steps = 0
    while not done and steps < 80:
        if env.state is None or env.state.action_on_seat is None:
            break
        if env.state.street == Street.SHOWDOWN:
            return
        mask = env.get_action_mask()
        action = 1 if mask[1] else int(np.flatnonzero(mask)[0])
        _, _, done, _, _ = env.step(action)
        steps += 1
    assert done or (env.state is not None and env.state.street == Street.SHOWDOWN)


def test_level_start_is_equal_hundred_big_blinds() -> None:
    spec = draw_sitting(
        np.random.default_rng(0),
        profiles=("level_start",),
        weights=(1.0,),
        min_players=6,
        max_players=6,
        hands_per_level=(8, 8),
    )
    assert spec.field_size == 6
    assert spec.profile == "level_start"
    assert sum(spec.starting_stacks) == chip_pool(6)
    assert len(set(spec.starting_stacks)) == 1
    level = spec.schedule.levels[0]
    assert level == TOURNAMENT_LEVELS[0]
    assert spec.starting_stacks[0] // level.big == 100
    assert 0 <= spec.button_seat < 6


def test_middle_and_end_keep_the_chip_pool_and_snap_the_blind() -> None:
    rng = np.random.default_rng(1)
    for profile, depth in (("middle", 30), ("end", 12)):
        spec = draw_sitting(
            rng,
            profiles=(profile,),
            weights=(1.0,),
            min_players=2,
            max_players=9,
        )
        assert spec.profile == profile
        assert spec.target_depth_bb == depth
        assert sum(spec.starting_stacks) == chip_pool(spec.field_size)
        assert all(stack > 0 for stack in spec.starting_stacks)
        if profile == "end":
            assert 3 <= spec.field_size <= 6
        total = chip_pool(spec.field_size)
        target = round(total / (spec.field_size * depth))
        assert spec.level_index == nearest_level_index(target)
        assert spec.schedule.levels[0] == TOURNAMENT_LEVELS[spec.level_index]
        # The suffix still escalates into an ante level, or already has one.
        assert any(level.ante > 0 for level in spec.schedule.levels)


def test_middle_profile_keeps_short_and_medium_bands() -> None:
    """The 8-15bb and 30-60bb seats survive contact with the chip pool."""
    rng = np.random.default_rng(11)
    for field in (2, 6, 9):
        band_hits = 0
        seats = 0
        for _ in range(40):
            spec = draw_sitting(
                rng,
                profiles=("middle",),
                weights=(1.0,),
                min_players=field,
                max_players=field,
            )
            blind = spec.schedule.levels[0].big
            depths = [stack / blind for stack in spec.starting_stacks]
            assert min(spec.starting_stacks) >= blind
            largest = max(depths)
            outsiders = 0
            for depth in depths:
                if abs(depth - largest) < 1e-6:
                    outsiders += 1
                    continue
                assert 8 - 0.05 <= depth <= 15.05 or 30 - 0.05 <= depth <= 60.05
                band_hits += 1
            seats += field - outsiders
        assert band_hits == seats


def test_end_profile_never_starts_below_one_big_blind() -> None:
    spec = draw_sitting(
        np.random.default_rng(2),
        profiles=("end",),
        weights=(1.0,),
        min_players=6,
        max_players=6,
    )
    bb = spec.schedule.levels[0].big
    ordered = sorted(spec.starting_stacks)
    cover = ordered[-1]
    assert cover > ordered[-2]
    assert ordered[0] >= bb
    assert all(stack <= 15 * bb for stack in ordered[:-1])
    rng = np.random.default_rng(9)
    for _ in range(40):
        sample = draw_sitting(
            rng, profiles=("end",), weights=(1.0,), min_players=3, max_players=6
        )
        blind = sample.schedule.levels[0].big
        assert min(sample.starting_stacks) >= blind


def test_profile_mix_and_button_are_spread_across_seats() -> None:
    rng = np.random.default_rng(3)
    profiles: list[str] = []
    buttons = [0, 0, 0, 0, 0, 0]
    for _ in range(200):
        spec = draw_sitting(
            rng,
            min_players=6,
            max_players=6,
            profiles=("level_start", "middle", "end"),
            weights=(0.4, 0.4, 0.2),
        )
        profiles.append(spec.profile)
        if spec.field_size == 6:
            buttons[spec.button_seat] += 1
    assert profiles.count("level_start") > 50
    assert profiles.count("middle") > 50
    assert profiles.count("end") > 20
    # A pinned button would leave most seats at 0. Every seat should open some sittings.
    assert min(buttons) > 10


def test_a_bust_is_not_censored_and_a_hand_cap_is() -> None:
    assert is_censored(18, 160, live_players=3, incomplete=False) is False
    assert is_censored(160, 160, live_players=3, incomplete=False) is True
    assert is_censored(40, 160, live_players=1, incomplete=False) is False
    assert is_censored(4, 160, live_players=1, incomplete=True) is True


def test_finish_position_counts_stacks_that_cover_the_hero() -> None:
    assert finish_position((0, 500, 100), 1) == 1
    assert finish_position((0, 500, 100), 2) == 2
    assert finish_position((0, 500, 100), 0) == 3


def test_sitting_conserves_chips_and_rotates_the_button() -> None:
    spec = draw_sitting(
        np.random.default_rng(4),
        profiles=("level_start",),
        weights=(1.0,),
        min_players=6,
        max_players=6,
        hands_per_level=(4, 4),
        hand_cap=4,
    )
    env = PokerEnv(
        num_players=spec.field_size,
        starting_stack=buyin_chips(),
        learning_seat=0,
        opponent_bots=[CallBot() for _ in range(spec.field_size - 1)],
        seed=4,
        blind_schedule=spec.schedule,
    )
    total = sum(spec.starting_stacks)
    env.reset(
        options={
            "stacks": spec.starting_stacks,
            "deck_seed": 11,
            "dealer_seat": spec.button_seat,
        }
    )
    env._dealer_override = None
    assert env.state is not None
    assert env.state.dealer_seat == spec.button_seat
    assert env.config.big_blind == spec.schedule.levels[0].big
    seen_buttons = [env.state.dealer_seat]
    for hand in range(spec.hand_cap):
        if hand:
            opened = env.start_next_hand(seed=20 + hand)
            if opened is None:
                break
            assert env.state is not None
            seen_buttons.append(env.state.dealer_seat)
        assert sum(
            player.stack + player.committed_this_hand for player in env.state.players
        ) == total
        _finish(env)
        assert env.state is not None
        assert sum(player.stack for player in env.state.players) == total
        assert all(player.stack >= 0 for player in env.state.players)
    # No eliminations on four equal 100bb hands, so the button walks one seat each hand.
    assert seen_buttons[0] == spec.button_seat
    if len(seen_buttons) > 1:
        assert seen_buttons[1] == (spec.button_seat + 1) % spec.field_size


def test_short_profile_survives_an_ante_level() -> None:
    """A 12bb sitting starts near an ante level and must not mint chips."""
    spec = draw_sitting(
        np.random.default_rng(5),
        profiles=("end",),
        weights=(1.0,),
        min_players=4,
        max_players=4,
        hands_per_level=(2, 2),
        hand_cap=3,
    )
    assert spec.schedule.levels[0].ante > 0 or any(
        level.ante > 0 for level in spec.schedule.levels[:2]
    )
    env = PokerEnv(
        num_players=spec.field_size,
        starting_stack=buyin_chips(),
        learning_seat=0,
        opponent_bots=[CallBot() for _ in range(spec.field_size - 1)],
        seed=5,
        blind_schedule=spec.schedule,
    )
    total = sum(spec.starting_stacks)
    env.reset(options={"stacks": spec.starting_stacks, "deck_seed": 7, "dealer_seat": 0})
    env._dealer_override = None
    for hand in range(3):
        if hand:
            opened = env.start_next_hand(seed=30 + hand)
            if opened is None:
                break
        _finish(env)
        assert env.state is not None
        assert sum(player.stack for player in env.state.players) == total
        assert all(player.stack >= 0 for player in env.state.players)


def test_nearest_level_prefers_the_lower_blind_on_a_tie() -> None:
    # 250 is equidistant from 200 and 300. The earlier level wins the tie.
    assert nearest_level_index(250) == 3
    assert TOURNAMENT_LEVELS[3] == BlindLevel(100, 200, 25)


@pytest.mark.parametrize("bad", [0, -3])
def test_hand_cap_must_be_positive(bad: int) -> None:
    with pytest.raises(ValueError):
        draw_sitting(np.random.default_rng(0), hand_cap=bad)


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        draw_sitting(np.random.default_rng(0), profiles=("cash",), weights=(1.0,))


def test_busted_stack_is_not_required_for_the_generator() -> None:
    """Generator stacks are all live. Busting happens during play, not at the draw."""
    spec = draw_sitting(np.random.default_rng(8), min_players=2, max_players=9, hand_cap=1)
    assert min(spec.starting_stacks) > 0
    # dc_replace is imported so a future edit that freezes the spec still type-checks
    # the stacks as a tuple the env can carry.
    assert dc_replace(spec, hand_cap=2).hand_cap == 2
