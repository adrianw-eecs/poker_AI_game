"""Tests for PokerEnv."""

import numpy as np
import pytest

from poker.ml.env import PokerEnv


@pytest.mark.smoke
def test_env_reset_returns_obs() -> None:
    env = PokerEnv()
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (98,)
    assert isinstance(info, dict)


@pytest.mark.smoke
def test_env_action_space_is_discrete() -> None:
    env = PokerEnv()
    assert env.action_space.n == 7


@pytest.mark.smoke
def test_env_observation_space_shape() -> None:
    env = PokerEnv()
    assert env.observation_space.shape == (98,)
    assert env.observation_space.dtype == np.float32


def test_env_default_init() -> None:
    env = PokerEnv()
    assert env.num_players == 2
    assert env.starting_stack == 1000
    assert env.big_blind == 20


def test_env_custom_init() -> None:
    env = PokerEnv(num_players=6, starting_stack=5000, small_blind=25, big_blind=50)
    assert env.num_players == 6
    assert env.starting_stack == 5000


@pytest.mark.parametrize("kwargs,exc", [
    ({"num_players": 1}, ValueError),
    ({"num_players": 11}, ValueError),
    ({"starting_stack": 0}, ValueError),
])
def test_env_invalid_params(kwargs, exc) -> None:
    with pytest.raises(exc):
        PokerEnv(**kwargs)


# ---------------------------------------------------------------------------
# End-to-end episode tests (Fix #22)
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_env_episode_completes_and_chip_conservation() -> None:
    """Run a full episode and verify no chips are created or destroyed."""
    from poker.bots.random_bot import RandomBot

    starting_stack = 1000
    num_players = 2
    env = PokerEnv(
        num_players=num_players,
        starting_stack=starting_stack,
        small_blind=10,
        big_blind=20,
        opponent_bots=[RandomBot(seed=42)],
        seed=0,
    )
    obs, _ = env.reset()
    total_chips_start = num_players * starting_stack

    done = False
    steps = 0
    final_reward = 0.0
    while not done and steps < 200:
        mask = env.get_action_mask()
        # Choose random legal action.
        legal = np.where(mask)[0]
        action = int(np.random.choice(legal))
        obs, reward, done, _, _ = env.step(action)
        if done:
            final_reward = reward
        steps += 1

    assert done, "Episode did not terminate within 200 steps"
    # After a hand the env returns stack_change relative to starting_stack.
    # The terminal reward should be finite (not NaN/inf).
    assert np.isfinite(final_reward), f"Terminal reward is not finite: {final_reward}"


def test_terminal_reward_is_the_prize_share_delta() -> None:
    """Winning the opponent's stack moves winner-take-all equity by that share."""
    from poker.bots.call_bot import CallBot
    from poker.ml.env import REWARD_SCALE

    env = PokerEnv(
        num_players=2,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        learning_seat=0,
        opponent_bots=[CallBot()],
        seed=1,
    )
    env.hand_number = 0
    env.reset()
    _, _, done, _, _ = env.step(0)
    assert done and env.state is not None
    before = env.player_stacks_at_hand_start
    total = sum(before.values())
    after = env.state.players[0].stack
    expected = REWARD_SCALE * (after / total - before[0] / total)
    assert env._compute_reward(0) == pytest.approx(expected)
    assert abs(env._compute_reward(0) + env._compute_reward(1)) < 1e-6


def test_env_episode_reward_range() -> None:
    """Terminal reward should be within [-1, 1] (normalised stack delta)."""
    from poker.bots.random_bot import RandomBot

    env = PokerEnv(
        num_players=2,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        opponent_bots=[RandomBot(seed=7)],
        seed=99,
    )
    obs, _ = env.reset()
    done = False
    steps = 0
    final_reward = 0.0
    while not done and steps < 200:
        mask = env.get_action_mask()
        legal = np.where(mask)[0]
        action = int(np.random.choice(legal))
        obs, reward, done, _, _ = env.step(action)
        if done:
            final_reward = reward
        steps += 1

    assert done
    assert -1.5 <= final_reward <= 1.5, (
        f"Terminal reward {final_reward:.4f} outside expected range [-1.5, 1.5]"
    )


@pytest.mark.smoke
def test_env_six_player_episode() -> None:
    """6-player episode must terminate without errors."""
    from poker.bots.random_bot import RandomBot

    num_players = 6
    env = PokerEnv(
        num_players=num_players,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        opponent_bots=[RandomBot(seed=i) for i in range(num_players - 1)],
        seed=123,
    )
    obs, _ = env.reset()
    assert obs.shape == (98,)

    done = False
    steps = 0
    while not done and steps < 300:
        mask = env.get_action_mask()
        legal = np.where(mask)[0]
        action = int(np.random.choice(legal))
        obs, reward, done, _, _ = env.step(action)
        steps += 1

    assert done, "6-player episode did not terminate within 300 steps"


def test_folding_the_small_blind_is_a_loss() -> None:
    """Folding the small blind loses that fraction of the chips in play."""
    from poker.bots.call_bot import CallBot

    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[CallBot()],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=3,
    )
    env.hand_number = 0
    env.reset()
    _, reward, done, _, _ = env.step(0)
    assert done
    assert env.state is not None
    assert env.state.players[0].stack == 975
    # 25 chips out of a 2,000-chip pool, scaled by REWARD_SCALE.
    assert reward == pytest.approx(-0.00625)


def test_limp_reaches_a_flop_with_no_live_bet() -> None:
    from poker.bots.call_bot import CallBot
    from poker.state.game_state import Street

    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[CallBot()],
        small_blind=25,
        big_blind=50,
        seed=4,
    )
    env.hand_number = 0
    env.reset()
    _, _, done, _, _ = env.step(1)
    assert not done
    assert env.state is not None
    assert env.state.street == Street.FLOP
    assert env.state.current_bet_to_call == 0
    assert len(env.state.community_cards) == 3
    assert not any(player.has_folded for player in env.state.players)
    assert env.state.action_history_this_hand


def test_chip_conservation_from_two_to_nine_players() -> None:
    from poker.bots.random_bot import RandomBot

    for num_players in range(2, 10):
        env = PokerEnv(
            num_players=num_players,
            starting_stack=500,
            small_blind=10,
            big_blind=20,
            opponent_bots=[RandomBot(seed=num_players) for _ in range(num_players - 1)],
            seed=num_players,
        )
        env.reset(seed=num_players)
        done = False
        steps = 0
        while not done and steps < 400:
            mask = env.get_action_mask()
            action = int(np.random.default_rng(steps + num_players).choice(np.where(mask)[0]))
            _, _, done, _, _ = env.step(action)
            steps += 1
        assert done
        assert env.state is not None
        assert sum(player.stack for player in env.state.players) == num_players * 500


def test_env_matches_play_hand_on_the_same_deck() -> None:
    from poker.bots.call_bot import CallBot
    from poker.domain.deck import Deck
    from poker.engine.hand_engine import play_hand
    from poker.logging.logger import NullLogger
    from poker.rng import RNG
    from poker.state.game_state import GameState, Street
    from poker.state.player_state import PlayerState
    from poker.state.pot import Pot

    class _Station:
        name = "Station"

        def act(self, state, legal):  # type: ignore[no-untyped-def]
            return CallBot().act(state, legal)

        def observe_result(self, final_state, reward):  # type: ignore[no-untyped-def]
            return None

    stacks = (150, 400, 1000)
    config_env = PokerEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        opponent_bots=[CallBot(), CallBot()],
        seed=1,
    )
    config = config_env.config
    players = tuple(
        PlayerState(
            seat=seat,
            name=f"P{seat}",
            stack=stack,
            hole_cards=(),
            committed_this_street=0,
            committed_this_hand=0,
            has_folded=False,
            is_all_in=False,
            is_eliminated=False,
        )
        for seat, stack in enumerate(stacks)
    )
    state = GameState(
        hand_number=0,
        street=Street.PREFLOP,
        dealer_seat=0,
        players=players,
        community_cards=(),
        pots=[Pot(0, frozenset(range(3)))],
        current_bet_to_call=0,
        last_raise_size=0,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=52,
        config=config,
        blind_level=config.blind_schedule.levels[0],
        action_on_seat=None,
    )
    final = play_hand(
        state,
        {seat: _Station() for seat in range(3)},
        Deck(RNG(seed=5)),
        NullLogger(),
    )
    env = PokerEnv(
        num_players=3,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        opponent_bots=[CallBot(), CallBot()],
        seed=1,
    )
    env.hand_number = 0
    env.reset(options={"deck_seed": 5, "stacks": stacks})
    done = False
    steps = 0
    while not done and steps < 40:
        _, _, done, _, _ = env.step(1)
        steps += 1
    assert done
    assert env.state is not None
    assert [player.stack for player in env.state.players] == [player.stack for player in final.players]
    assert [card.code() for card in env.state.community_cards] == [
        card.code() for card in final.community_cards
    ]


def test_multi_seat_api_records_both_learners() -> None:
    from poker.bots.call_bot import CallBot

    env = PokerEnv(
        num_players=2,
        learning_seats={0, 1},
        opponent_bots=[],
        small_blind=25,
        big_blind=50,
        starting_stack=1000,
        seed=2,
    )
    seat, obs, mask = env.reset_multi()
    assert seat in (0, 1)
    assert obs.shape == (98,)
    assert mask.shape == (7,)
    seen = {seat}
    done = False
    rewards = {}
    for _ in range(12):
        action = 1 if mask[1] else int(np.where(mask)[0][0])
        seat, obs, mask, rewards, done = env.step_multi(action)
        if done:
            break
        seen.add(seat)
    assert done
    assert set(rewards) == {0, 1}
    assert abs(sum(rewards.values())) < 1e-6


def test_half_raise_does_not_reopen() -> None:
    from poker.domain.action import Action
    from poker.engine.rules import reopens_betting

    assert reopens_betting(Action.raise_to(200), prior_max=100, prior_min=100)
    assert not reopens_betting(Action.all_in(150), prior_max=100, prior_min=100)
    assert reopens_betting(Action.all_in(200), prior_max=100, prior_min=100)


def _call_hand_to_showdown(env: PokerEnv) -> None:
    """Finish the current hand by calling whenever that is legal."""
    from poker.state.game_state import Street

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


def test_ante_clamps_a_short_stack_and_conserves_chips() -> None:
    from poker.bots.call_bot import CallBot
    from poker.config.blind_schedule import BlindLevel, BlindSchedule

    schedule = BlindSchedule(
        levels=[BlindLevel(10, 20, 25)],
        hands_per_level=10,
        fixed=True,
    )
    env = PokerEnv(
        num_players=3,
        starting_stack=100,
        learning_seat=0,
        opponent_bots=[CallBot(), CallBot()],
        seed=1,
        blind_schedule=schedule,
    )
    env.reset(options={"stacks": (100, 10, 100)})
    assert env.state is not None
    short = env.state.players[1]
    assert short.stack >= 0
    assert short.is_all_in
    assert short.stack + short.committed_this_hand == 10
    assert sum(player.stack + player.committed_this_hand for player in env.state.players) == 210


def test_chip_conservation_across_a_sitting_with_antes() -> None:
    """Stacks sum to the same total from 2 to 9 seats, across carried hands."""
    from poker.bots.call_bot import CallBot
    from poker.config.blind_schedule import BlindLevel, BlindSchedule

    schedule = BlindSchedule(
        levels=[
            BlindLevel(10, 20, 0),
            BlindLevel(25, 50, 10),
            BlindLevel(100, 200, 25),
        ],
        hands_per_level=2,
        fixed=False,
    )
    for num_players in range(2, 10):
        raw = tuple(80 if seat % 2 == 0 else 35 for seat in range(num_players))
        total = sum(raw)
        env = PokerEnv(
            num_players=num_players,
            starting_stack=200,
            learning_seat=0,
            opponent_bots=[CallBot() for _ in range(num_players - 1)],
            seed=num_players + 3,
            blind_schedule=schedule,
        )
        env.reset(options={"stacks": raw, "deck_seed": num_players})
        assert env.state is not None
        assert sum(
            player.stack + player.committed_this_hand for player in env.state.players
        ) == total
        for hand in range(6):
            if hand:
                opened = env.start_next_hand(seed=100 + hand + num_players)
                if opened is None:
                    break
                assert env.state is not None
                assert sum(
                    player.stack + player.committed_this_hand for player in env.state.players
                ) == total
            _call_hand_to_showdown(env)
            assert env.state is not None
            assert all(player.stack >= 0 for player in env.state.players)
            assert sum(player.stack for player in env.state.players) == total
            assert all(
                player.is_eliminated for player in env.state.players if player.stack <= 0
            )


def test_busted_seat_stays_eliminated_and_the_button_skips_it() -> None:
    from dataclasses import replace as dc_replace

    from poker.bots.call_bot import CallBot

    env = PokerEnv(
        num_players=3,
        starting_stack=300,
        small_blind=10,
        big_blind=20,
        learning_seat=0,
        opponent_bots=[CallBot(), CallBot()],
        seed=1,
    )
    env.hand_number = 0
    env.reset()
    _call_hand_to_showdown(env)
    assert env.state is not None
    assert sum(player.stack for player in env.state.players) == 900
    assert env.state.dealer_seat == 0
    moved = env.state.players[1].stack
    players = list(env.state.players)
    players[0] = players[0].with_stack(players[0].stack + moved)
    players[1] = players[1].with_stack(0)
    env.state = dc_replace(env.state, players=tuple(players))
    opened = env.start_next_hand(seed=5)
    assert opened is not None
    assert env.state is not None
    assert env.state.players[1].is_eliminated
    assert env.state.players[1].stack == 0
    assert env.state.dealer_seat == 2
    _call_hand_to_showdown(env)
    assert env.state.players[1].is_eliminated
    assert env.state.players[1].stack == 0
    assert sum(player.stack for player in env.state.players) == 900


def test_blinds_advance_past_half_the_buyin() -> None:
    from poker.bots.call_bot import CallBot
    from poker.config.blind_schedule import BlindLevel, BlindSchedule

    schedule = BlindSchedule(
        levels=[
            BlindLevel(10, 20, 0),
            BlindLevel(50, 100, 10),
            BlindLevel(200, 400, 40),
        ],
        hands_per_level=1,
        fixed=False,
    )
    env = PokerEnv(
        num_players=2,
        starting_stack=150,
        learning_seat=0,
        opponent_bots=[CallBot()],
        seed=4,
        blind_schedule=schedule,
    )
    env.reset(options={"stacks": (400, 400)})
    assert env.config.big_blind == 20
    _call_hand_to_showdown(env)
    assert env.start_next_hand(seed=2) is not None
    assert env.state is not None
    assert env.hand_number == 1
    assert env.config.big_blind == 100
    assert env.state.config.big_blind == 100
    assert env.state.config.ante == 10
    _call_hand_to_showdown(env)
    assert env.start_next_hand(seed=3) is not None
    assert env.config.big_blind == 400
    assert env.config.starting_stack == 150
    assert env.state is not None
    assert env.state.config.ante == 40
    assert all(player.stack >= 0 for player in env.state.players)
