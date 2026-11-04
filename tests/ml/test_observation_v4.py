"""Observation v4: v3 prefix, kickers, equity, and pot structure."""

import time

import numpy as np
import pytest
import torch

from poker.bots.random_bot import RandomBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.card import Card, Rank, Suit
from poker.ml.action_space import BetScheme
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.nfsp_networks_v3 import build_q_network_v3
from poker.ml.observation_v2 import OBS_SPEC_V2
from poker.ml.observation_v3 import OBS_V3_DIM, build_observation_v3
from poker.ml.observation_v4 import (
    EQ_BOARD_CLASS,
    EQ_EFFECTIVE,
    EQ_EQUITY,
    EQ_KICKER,
    EQ_OVERPAIR,
    OBS_V4_DIM,
    ST_ALL_IN,
    ST_ELIGIBLE,
    ST_SIDE,
    build_observation_v4,
)
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


def _config(n: int) -> GameConfig:
    return GameConfig(
        num_players=n,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        ante=0,
        rake_percent=0,
        rake_cap=None,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(10, 20, 0)],
            hands_per_level=10,
            fixed=True,
        ),
        run_it_twice=False,
    )


def _player(
    seat: int,
    stack: int,
    hole: tuple[Card, Card],
    *,
    committed: int = 20,
    all_in: bool = False,
) -> PlayerState:
    return PlayerState(
        seat=seat,
        name=f"P{seat}",
        stack=stack,
        hole_cards=hole,
        committed_this_street=committed,
        committed_this_hand=committed,
        has_folded=False,
        is_all_in=all_in,
        is_eliminated=False,
    )


def _flop(
    holes: tuple[tuple[Card, Card], ...],
    board: tuple[Card, ...],
    *,
    stacks: list[int] | None = None,
    pots: list[Pot] | None = None,
    all_in: set[int] | None = None,
) -> GameState:
    n = len(holes)
    all_in = all_in or set()
    stacks = stacks if stacks is not None else [800] * n
    players = tuple(
        _player(i, stacks[i], holes[i], all_in=i in all_in) for i in range(n)
    )
    if pots is None:
        pots = [Pot(20 * n, frozenset(range(n)))]
    return GameState(
        hand_number=1,
        street=Street.FLOP,
        dealer_seat=0,
        players=players,
        community_cards=board,
        pots=pots,
        current_bet_to_call=20,
        last_raise_size=20,
        action_history_this_street=[],
        action_history_this_hand=[],
        deck_remaining_count=52 - 2 * n - len(board),
        config=_config(n),
        blind_level=BlindLevel(10, 20, 0),
        action_on_seat=0,
    )


def _c(rank: Rank, suit: Suit) -> Card:
    return Card(rank, suit)


@pytest.mark.smoke
def test_v4_keeps_v3_prefix_byte_for_byte() -> None:
    env = PokerEnv(
        num_players=6,
        learning_seat=0,
        opponent_bots=[RandomBot(seed=i) for i in range(5)],
        small_blind=25,
        big_blind=50,
        seed=1,
        obs_version=4,
        bet_scheme=BetScheme.POT_FRACTION,
    )
    env.reset(options={"deck_seed": 3})
    assert env.state is not None
    v3 = build_observation_v3(env.state, 0, env.tendencies)
    v4 = build_observation_v4(env.state, 0, env.tendencies)
    assert v4.shape == (OBS_V4_DIM,)
    assert OBS_V4_DIM == 372
    assert np.array_equal(v4[:OBS_V3_DIM], v3)
    assert np.array_equal(v4, build_observation_v4(env.state, 0, env.tendencies))
    tail = v4[OBS_V3_DIM:]
    assert np.isfinite(tail).all()
    assert ((tail >= 0.0) & (tail <= 1.0)).all()


def test_kicker_splits_top_pair_from_top_pair() -> None:
    """v2's hero block cannot tell an ace kicker from a trey. v4 can."""
    board = (
        _c(Rank.KING, Suit.SPADES),
        _c(Rank.SEVEN, Suit.DIAMONDS),
        _c(Rank.TWO, Suit.CLUBS),
    )
    strong = _flop(
        (
            (_c(Rank.ACE, Suit.HEARTS), _c(Rank.KING, Suit.DIAMONDS)),
            (_c(Rank.QUEEN, Suit.CLUBS), _c(Rank.JACK, Suit.CLUBS)),
        ),
        board,
    )
    weak = _flop(
        (
            (_c(Rank.KING, Suit.HEARTS), _c(Rank.THREE, Suit.DIAMONDS)),
            (_c(Rank.QUEEN, Suit.CLUBS), _c(Rank.JACK, Suit.CLUBS)),
        ),
        board,
    )
    hero = slice(*OBS_SPEC_V2.hero)
    strong_v4 = build_observation_v4(strong, 0)
    weak_v4 = build_observation_v4(weak, 0)
    assert np.array_equal(strong_v4[hero], weak_v4[hero])
    assert strong_v4[EQ_KICKER] > weak_v4[EQ_KICKER]
    assert strong_v4[EQ_OVERPAIR] == 0.0
    assert strong_v4[EQ_BOARD_CLASS] == pytest.approx(1.0)


def test_board_class_separates_top_middle_and_bottom_pair() -> None:
    board = (
        _c(Rank.ACE, Suit.SPADES),
        _c(Rank.SEVEN, Suit.DIAMONDS),
        _c(Rank.TWO, Suit.CLUBS),
    )
    top = _flop(
        (
            (_c(Rank.ACE, Suit.HEARTS), _c(Rank.THREE, Suit.DIAMONDS)),
            (_c(Rank.KING, Suit.CLUBS), _c(Rank.QUEEN, Suit.CLUBS)),
        ),
        board,
    )
    middle = _flop(
        (
            (_c(Rank.SEVEN, Suit.HEARTS), _c(Rank.THREE, Suit.DIAMONDS)),
            (_c(Rank.KING, Suit.CLUBS), _c(Rank.QUEEN, Suit.CLUBS)),
        ),
        board,
    )
    bottom = _flop(
        (
            (_c(Rank.TWO, Suit.HEARTS), _c(Rank.THREE, Suit.DIAMONDS)),
            (_c(Rank.KING, Suit.CLUBS), _c(Rank.QUEEN, Suit.CLUBS)),
        ),
        board,
    )
    top_obs = build_observation_v4(top, 0)
    middle_obs = build_observation_v4(middle, 0)
    bottom_obs = build_observation_v4(bottom, 0)
    assert top_obs[EQ_BOARD_CLASS] == pytest.approx(1.0)
    assert middle_obs[EQ_BOARD_CLASS] == pytest.approx(0.5)
    assert bottom_obs[EQ_BOARD_CLASS] == pytest.approx(0.25)
    assert top_obs[EQ_OVERPAIR] == bottom_obs[EQ_OVERPAIR] == 0.0


def test_overpair_is_marked() -> None:
    state = _flop(
        (
            (_c(Rank.ACE, Suit.HEARTS), _c(Rank.ACE, Suit.DIAMONDS)),
            (_c(Rank.KING, Suit.CLUBS), _c(Rank.QUEEN, Suit.CLUBS)),
        ),
        (
            _c(Rank.SEVEN, Suit.SPADES),
            _c(Rank.FOUR, Suit.DIAMONDS),
            _c(Rank.TWO, Suit.CLUBS),
        ),
    )
    obs = build_observation_v4(state, 0)
    assert obs[EQ_OVERPAIR] == pytest.approx(1.0)
    assert obs[EQ_BOARD_CLASS] == pytest.approx(1.0)


def test_effective_equity_divides_by_opponent_count() -> None:
    state = _flop(
        (
            (_c(Rank.ACE, Suit.HEARTS), _c(Rank.KING, Suit.SPADES)),
            (_c(Rank.QUEEN, Suit.CLUBS), _c(Rank.JACK, Suit.CLUBS)),
            (_c(Rank.TEN, Suit.DIAMONDS), _c(Rank.NINE, Suit.DIAMONDS)),
        ),
        (
            _c(Rank.TWO, Suit.HEARTS),
            _c(Rank.SEVEN, Suit.CLUBS),
            _c(Rank.EIGHT, Suit.SPADES),
        ),
    )
    obs = build_observation_v4(state, 0)
    assert obs[EQ_EFFECTIVE] == pytest.approx(float(obs[EQ_EQUITY]) / 2.0)

    heads_up = _flop(
        (
            (_c(Rank.ACE, Suit.HEARTS), _c(Rank.KING, Suit.SPADES)),
            (_c(Rank.QUEEN, Suit.CLUBS), _c(Rank.JACK, Suit.CLUBS)),
        ),
        (
            _c(Rank.TWO, Suit.HEARTS),
            _c(Rank.SEVEN, Suit.CLUBS),
            _c(Rank.EIGHT, Suit.SPADES),
        ),
    )
    hu = build_observation_v4(heads_up, 0)
    assert hu[EQ_EFFECTIVE] == pytest.approx(float(hu[EQ_EQUITY]))


def test_side_pot_exposure_excludes_pots_hero_cannot_win() -> None:
    holes = (
        (_c(Rank.ACE, Suit.HEARTS), _c(Rank.KING, Suit.SPADES)),
        (_c(Rank.QUEEN, Suit.CLUBS), _c(Rank.QUEEN, Suit.DIAMONDS)),
        (_c(Rank.JACK, Suit.CLUBS), _c(Rank.TEN, Suit.CLUBS)),
    )
    state = _flop(
        holes,
        (
            _c(Rank.TWO, Suit.HEARTS),
            _c(Rank.SEVEN, Suit.DIAMONDS),
            _c(Rank.NINE, Suit.CLUBS),
        ),
        stacks=[0, 400, 400],
        pots=[
            Pot(300, frozenset({0, 1, 2})),
            Pot(800, frozenset({1, 2})),
        ],
        all_in={0, 1},
    )
    obs = build_observation_v4(state, 0)
    assert obs[ST_ELIGIBLE] == pytest.approx(300 / 1100)
    assert obs[ST_SIDE] == pytest.approx(800 / 1100)
    assert obs[ST_ALL_IN] == pytest.approx(1 / 8)


def test_build_observation_v4_stays_under_2ms() -> None:
    holes = tuple(
        (
            _c(Rank(14 - i), Suit.HEARTS),
            _c(Rank(8 - i), Suit.DIAMONDS),
        )
        for i in range(6)
    )
    state = _flop(
        holes,
        (
            _c(Rank.KING, Suit.SPADES),
            _c(Rank.SEVEN, Suit.CLUBS),
            _c(Rank.TWO, Suit.SPADES),
        ),
    )
    build_observation_v4(state, 0)
    samples = []
    for _ in range(12):
        start = time.perf_counter()
        build_observation_v4(state, 0)
        samples.append(time.perf_counter() - start)
    assert sorted(samples)[len(samples) // 2] < 0.002


def _batch() -> torch.Tensor:
    x = torch.zeros(2, OBS_V4_DIM)
    x[:, 0] = 12
    x[:, 1] = 25
    x[:, 2:7] = 52
    x[:, 43:58] = 1.0
    return x


def test_v4_network_reads_the_extra_block() -> None:
    net = build_q_network_v3(input_dim=OBS_V4_DIM)
    assert net.equity_head is not None
    assert net.equity_head.out_features == 1
    assert bool(torch.all(net.equity_head.weight == 0).item())
    out = net(_batch())
    assert out.shape == (2, 7)
    first = net.table_mlp[0]
    assert isinstance(first, torch.nn.Linear)
    assert first.in_features == 230


def test_v4_short_training_has_finite_losses() -> None:
    model = NFSPModel(
        obs_version=4,
        bet_scheme=BetScheme.POT_FRACTION,
        batch_size=16,
        q_capacity=256,
        policy_capacity=256,
        train_every=1,
        n_step=1,
        q_warmup_hands=1,
        eta=1.0,
        value_distribution="c51",
        num_atoms=11,
    )
    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[RandomBot(seed=1)],
        small_blind=25,
        big_blind=50,
        seed=2,
        obs_version=4,
        bet_scheme=BetScheme.POT_FRACTION,
    )
    q_losses: list[float] = []
    p_losses: list[float] = []
    for hand in range(24):
        obs, _ = env.reset(options={"deck_seed": 100 + hand})
        model.begin_episode()
        guard = 0
        while (
            env.state is not None
            and env.state.action_on_seat == env.learning_seat
            and guard < 12
        ):
            mask = env.get_action_mask()
            action = model.select_action(np.asarray(obs), mask, training=True)
            next_obs, reward, done, _, _ = env.step(action)
            model.store_transition(
                np.asarray(obs),
                action,
                float(reward) if done else 0.0,
                np.asarray(next_obs),
                done,
                mask,
                mask,
            )
            obs = next_obs
            guard += 1
            if done:
                break
        model.end_hand()
        result = model.train_step()
        if result["q_loss"] is not None:
            q_losses.append(float(result["q_loss"]))
        if result["policy_loss"] is not None:
            p_losses.append(float(result["policy_loss"]))
    assert q_losses, "Q update never fired"
    assert p_losses, "policy update never fired"
    assert all(np.isfinite(loss) for loss in q_losses)
    assert all(np.isfinite(loss) for loss in p_losses)


def test_v4_checkpoint_roundtrip(tmp_path) -> None:
    model = NFSPModel(
        obs_version=4,
        batch_size=8,
        q_capacity=32,
        policy_capacity=32,
        value_distribution="c51",
        num_atoms=11,
        n_step=1,
    )
    path = tmp_path / "v4.pt"
    model.save(str(path))
    loaded = NFSPModel(
        obs_version=1,
        batch_size=8,
        q_capacity=32,
        policy_capacity=32,
        value_distribution="none",
        n_step=1,
    )
    loaded.load(str(path))
    assert loaded.obs_version == 4
    assert int(loaded.q_network.input_dim) == OBS_V4_DIM
    assert getattr(loaded.q_network, "equity_head", None) is not None
    with torch.no_grad():
        out = loaded.q_network(_batch()[:1].to(loaded.device))
    assert out.shape == (1, 7)
    assert torch.isfinite(out).all()


def test_env_observation_is_v4() -> None:
    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[RandomBot(seed=0)],
        small_blind=25,
        big_blind=50,
        seed=4,
        obs_version=4,
        bet_scheme=BetScheme.POT_FRACTION,
    )
    obs, _ = env.reset(options={"deck_seed": 9})
    assert obs.shape == (OBS_V4_DIM,)
    assert env.tendencies is not None
    # Preflop equity comes from the 169 table, so it is deterministic and non-zero
    # for a dealt hand.
    assert 0.0 <= float(obs[EQ_EQUITY]) <= 1.0
