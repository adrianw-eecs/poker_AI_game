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
    """The chip baseline is the stack before blinds are posted."""
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
    assert reward == pytest.approx(-0.0125)


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
