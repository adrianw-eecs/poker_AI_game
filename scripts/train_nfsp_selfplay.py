#!/usr/bin/env python
"""NFSP Self-Play Training: variable player count with adaptive self-play.

Each episode is a short sitting of hands (default 10). Stacks carry and the
button rotates, but each hand is its own NFSP episode: anticipatory mode is
sampled again and the hand ends with a terminal reward. The sitting stops
early when any player cannot post the big blind.

The first button and the hero seat are random. Opponent types are shuffled
into the remaining seats, and each RandomBot draws a fresh seed. Deck seeds
are drawn per hand. An optional --seed makes that stream repeatable.

Each episode samples a player count in [--min-players, --max-players]. The
observation encoding uses fixed MAX_PLAYERS=9 slots so the network input
is constant regardless of how many seats are active.

Opponent composition per episode:
    SD-CFR is seated first: 40% of opponent seats (at least one) are one
    frozen SDCFRModel, shared across every SDCFRBot and never trained.
    Self-play then rises on a fixed schedule inside the remaining seats,
    up to 90%, using frozen NFSP copies. Whatever is left rotates
    RandomBot, FlopBot, and CallBot.

    The self-play fraction steps up 5% per eval period only when the
    preflop fold rate is at least 15% and the SD-CFR 95% lower bound is
    above -50 BB/100. Training stops after several evals fail to improve
    that lower bound, and the saved model is the best bound so far.

Resume from the 98-feature self-play checkpoint only:
  python scripts/train_nfsp_selfplay.py --checkpoint models/nfsp_prod/nfsp_selfplay.pt --sdcfr-opponent models/sdcfr_prod/sdcfr_ckpt_500.pt

  python scripts/train_nfsp_selfplay.py
  python scripts/train_nfsp_selfplay.py --min-players 2 --max-players 2
"""

import argparse
import csv
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.bots.sdcfr_bot import SDCFRBot
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.env import PokerEnv
from poker.ml.metrics import bb100_interval
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.population import PopulationManager


# ---------------------------------------------------------------------------
# Self-play scheduler
# ---------------------------------------------------------------------------

class SelfPlayScheduler:
    """Steps the self-play fraction on a fixed schedule when play is sane.

    Each qualifying eval adds RAMP_STEP, capped at MAX_FRACTION. The step
    is held while preflop folds stay under MIN_PREFLOP_FOLD or the SD-CFR
    95% lower bound is at or below SDCFR_RAMP_FLOOR_BB100.
    """

    MIN_PREFLOP_FOLD = 0.15
    SDCFR_RAMP_FLOOR_BB100 = -50.0
    RAMP_STEP = 0.05
    MAX_FRACTION = 0.90

    def __init__(self) -> None:
        self.fraction: float = 0.0

    def apply_scheduled_step(
        self,
        preflop_fold: float | None,
        sdcfr_lower: float,
    ) -> str:
        """Advance one schedule step, or explain why it was held."""
        if preflop_fold is None or preflop_fold < self.MIN_PREFLOP_FOLD:
            shown = "n/a" if preflop_fold is None else f"{preflop_fold:.0%}"
            return f"hold: preflop fold {shown} under {self.MIN_PREFLOP_FOLD:.0%}"
        if sdcfr_lower <= self.SDCFR_RAMP_FLOOR_BB100:
            return (
                f"hold: SD-CFR lower bound {sdcfr_lower:+.1f} "
                f"<= {self.SDCFR_RAMP_FLOOR_BB100:+.1f}"
            )
        if self.fraction >= self.MAX_FRACTION - 1e-9:
            return f"cap: self-play already {self.fraction:.0%}"
        self.fraction = min(self.MAX_FRACTION, self.fraction + self.RAMP_STEP)
        return f"ramp: self-play now {self.fraction:.0%}"

    @property
    def status(self) -> str:
        return f"self_play={self.fraction:.0%}"

    def state_dict(self) -> dict:
        return {"fraction": self.fraction}

    @classmethod
    def from_state_dict(cls, state: dict) -> "SelfPlayScheduler":
        sched = cls()
        sched.fraction = float(state.get("fraction", 0.0))
        return sched


# ---------------------------------------------------------------------------
# Opponent builder
# ---------------------------------------------------------------------------

# Baseline bots include CallBot so the policy sees stations, not only raise/fold lines.
_BASELINE_BOT_TYPES = (RandomBot, FlopBot, CallBot)
_SELFPLAY_GEN_LOOKBACK = 5   # rotate self-play opponents across recent generations
_Q_BUFFER_KEEP_ON_RAMP = 0.75
_HIGH_SP_FRACTION = 0.50
_HIGH_SP_Q_LR = 0.00025
_OBS_DIM = 98
# Share of opponent seats reserved for the frozen SD-CFR model before self-play.
_SDCFR_SEAT_FRACTION = 0.40
# Hands per table size before an SD-CFR BB/100 claim is used to ramp or stop.
_TRUST_HANDS_PER_SIZE = 400
_RANDOM_EVAL_HANDS = 100
_EVAL_SEED = 17_001
_HANDS_PER_EPISODE = 10
_LADDER_STRIDE = 5
_STOP_PATIENCE = 5
_STOP_MIN_EPISODE = 20_000


def opponent_seat_counts(num_opponents: int, self_play_fraction: float) -> tuple[int, int, int]:
    """Reserve SD-CFR first, then ramp self-play only inside the seats left.

    Returns (n_sdcfr, n_self, n_baseline). Heads-up has one opponent, so that
    seat is SD-CFR. At 6-max (5 opponents) this is 2 SD-CFR seats every hand.
    """
    if num_opponents <= 0:
        return 0, 0, 0
    n_sdcfr = min(num_opponents, max(1, int(round(num_opponents * _SDCFR_SEAT_FRACTION))))
    n_self = int(round((num_opponents - n_sdcfr) * self_play_fraction))
    n_baseline = num_opponents - n_sdcfr - n_self
    return n_sdcfr, n_self, n_baseline


def build_opponents(
    num_opponents: int,
    self_play_fraction: float,
    pop_manager: PopulationManager,
    generation_count: int,
    profiler_timed_bot: type | None = None,
    sdcfr_model: SDCFRModel | None = None,
) -> list:
    """Build a mixed opponent list.

    SD-CFR seats come first and all share the same frozen SDCFRModel.
    Self-play opponents sample from the last N saved generations, capped at
    90% of the seats that are not SD-CFR. Baseline seats rotate RandomBot,
    FlopBot, and CallBot.

    If profiler_timed_bot is provided (the _TimedBot class), wraps each bot
    to collect timing statistics.
    """
    if num_opponents == 0:
        return []

    n_sdcfr, n_selfplay, n_baseline = opponent_seat_counts(num_opponents, self_play_fraction)
    if n_sdcfr and sdcfr_model is None:
        raise RuntimeError("SD-CFR seats requested but no frozen SDCFRModel was loaded")

    bots: list = []
    for i in range(n_sdcfr):
        bots.append(SDCFRBot(name=f"SDCFR_{i}", model=sdcfr_model))

    for i in range(n_baseline):
        bot_cls = _BASELINE_BOT_TYPES[i % len(_BASELINE_BOT_TYPES)]
        if bot_cls is RandomBot:
            bots.append(
                RandomBot(
                    name=f"Random_{i}",
                    seed=int(np.random.randint(0, 2**31 - 1)),
                )
            )
        elif bot_cls is FlopBot:
            bots.append(FlopBot(name=f"Flop_{i}"))
        else:
            bots.append(CallBot())

    for i in range(n_selfplay):
        loaded = False
        if generation_count > 0:
            lookback = min(generation_count, _SELFPLAY_GEN_LOOKBACK)
            gen_id = generation_count - 1 - (i % lookback)
            try:
                prev_model = pop_manager.load_generation(gen_id)
                bots.append(NFSPBot(name=f"Self_{gen_id}_{i}", model=prev_model, training=False))
                loaded = True
            except FileNotFoundError:
                pass
        if not loaded:
            bots.append(
                RandomBot(
                    name=f"SelfFallback_{i}",
                    seed=int(np.random.randint(0, 2**31 - 1)),
                )
            )

    # Wrap bots with profiler if enabled
    if profiler_timed_bot is not None:
        bots = [profiler_timed_bot(bot) for bot in bots]

    return bots


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# Evaluate at these player counts (deduped/clamped later).
_EVAL_PLAYER_COUNTS = [2, 4, 6, 9]


@dataclass(frozen=True)
class EvalResult:
    """BB/100 with a 95% interval, plus the preflop fold rate of the hero."""

    mean_bb100: float
    lower_bb100: float
    upper_bb100: float
    n: int
    preflop_fold_rate: float | None
    win_rate: float


def _format_bb100(mean: float, lower: float, upper: float, n: int) -> str:
    return f"{mean:+.1f} BB/100  [{lower:+.1f}, {upper:+.1f}] n={n}"


def _win_rate(rewards: list[float]) -> float:
    """Fraction of hands with a positive chip result. Ties are not wins."""
    if not rewards:
        return 0.0
    return sum(1 for reward in rewards if reward > 0) / len(rewards)


def _env_encoding(model: NFSPModel) -> dict:
    """Observation version and bet scheme the loaded model was trained with."""
    version = int(getattr(model, "obs_version", 1))
    encoding: dict = {"obs_version": version}
    scheme = getattr(model, "bet_scheme", None)
    if scheme is not None:
        encoding["bet_scheme"] = scheme
    return encoding


def _play_scored_hand(
    model: NFSPModel,
    env: PokerEnv,
) -> tuple[float, int, int] | None:
    """Play one hand. Returns (reward, preflop folds, preflop decisions)."""
    obs, _ = env.reset()
    if env.state is None or env.state.action_on_seat != env.learning_seat:
        return None
    folds = 0
    decisions = 0
    done = False
    reward = 0.0
    while not done:
        street = env.state.street.value if env.state is not None else ""
        mask = env.get_action_mask()
        action = model.select_action(obs, mask, training=False)
        if street == "preflop":
            decisions += 1
            if action == 0:
                folds += 1
        obs, reward, done, _, _ = env.step(action)
    return reward, folds, decisions


def _target_counts(min_players: int, max_players: int) -> list[int]:
    counts = sorted(set(n for n in _EVAL_PLAYER_COUNTS if min_players <= n <= max_players))
    return counts or [min_players]


def evaluate(
    model: NFSPModel,
    num_hands_per_size: int = 20,
    min_players: int = 2,
    max_players: int = 9,
    seed: int = _EVAL_SEED,
    max_bet: int = 0,
    stall_guard: bool = True,
    stall_counts: dict[str, int] | None = None,
) -> EvalResult:
    """Evaluate at multiple table sizes vs Random and Random+Flop.

    Decks come from `seed`, so successive checkpoints see the same hands.
    Each scored hand is a single hand with fresh stacks.
    """
    rng = np.random.RandomState(seed)
    all_rewards: list[float] = []
    folds = 0
    decisions = 0

    for n in _target_counts(min_players, max_players):
        size_rewards: list[float] = []
        for config in range(2):
            for h in range(num_hands_per_size):
                env_seed = int(rng.randint(0, 2**31 - 1))
                if config == 0:
                    opponents = [
                        RandomBot(seed=int(rng.randint(0, 2**31 - 1)))
                        for _ in range(n - 1)
                    ]
                else:
                    opponents = [
                        FlopBot() if i % 2 == 0 else RandomBot(seed=int(rng.randint(0, 2**31 - 1)))
                        for i in range(n - 1)
                    ]
                env = PokerEnv(
                    num_players=n,
                    learning_seat=h % n,
                    opponent_bots=opponents,
                    small_blind=25,
                    big_blind=50,
                    seed=env_seed,
                    max_bet=max_bet,
                    stall_guard=stall_guard,
                    stall_counts=stall_counts,
                    **_env_encoding(model),
                )
                env.hand_number = h
                played = _play_scored_hand(model, env)
                if played is None:
                    continue
                reward, hand_folds, hand_decisions = played
                size_rewards.append(reward)
                all_rewards.append(reward)
                folds += hand_folds
                decisions += hand_decisions
        if size_rewards:
            mean, lower, upper, count = bb100_interval(size_rewards)
            print(
                f"    {n:2d}-player: {_format_bb100(mean, lower, upper, count)}"
                f" win={_win_rate(size_rewards):.0%}",
                flush=True,
            )

    mean, lower, upper, count = bb100_interval(all_rewards)
    rate = None if decisions == 0 else folds / decisions
    return EvalResult(mean, lower, upper, count, rate, _win_rate(all_rewards))


def evaluate_vs_sdcfr(
    model: NFSPModel,
    sdcfr_model: SDCFRModel,
    num_hands_per_size: int = 20,
    min_players: int = 2,
    max_players: int = 9,
    seed: int = _EVAL_SEED,
    max_bet: int = 0,
    stall_guard: bool = True,
    stall_counts: dict[str, int] | None = None,
) -> EvalResult:
    """Evaluate against a full table of the frozen SD-CFR bot.

    The same `seed` is reused every call so generation comparisons are paired.
    """
    rng = np.random.RandomState(seed + 1)
    all_rewards: list[float] = []
    folds = 0
    decisions = 0

    for n in _target_counts(min_players, max_players):
        size_rewards: list[float] = []
        for h in range(num_hands_per_size):
            env_seed = int(rng.randint(0, 2**31 - 1))
            # New bot objects each hand. SDCFRBot caches its seat by state id,
            # and the same object must not be moved to another seat.
            opp_list = [
                SDCFRBot(name=f"SDCFR_eval_{i}", model=sdcfr_model)
                for i in range(n - 1)
            ]
            env = PokerEnv(
                num_players=n,
                learning_seat=h % n,
                opponent_bots=opp_list,
                small_blind=25,
                big_blind=50,
                seed=env_seed,
                max_bet=max_bet,
                stall_guard=stall_guard,
                stall_counts=stall_counts,
                **_env_encoding(model),
            )
            env.hand_number = h
            played = _play_scored_hand(model, env)
            if played is None:
                continue
            reward, hand_folds, hand_decisions = played
            size_rewards.append(reward)
            all_rewards.append(reward)
            folds += hand_folds
            decisions += hand_decisions
        if size_rewards:
            mean, lower, upper, count = bb100_interval(size_rewards)
            print(
                f"    {n:2d}-player vs SD-CFR: {_format_bb100(mean, lower, upper, count)}"
                f" win={_win_rate(size_rewards):.0%}",
                flush=True,
            )

    mean, lower, upper, count = bb100_interval(all_rewards)
    rate = None if decisions == 0 else folds / decisions
    return EvalResult(mean, lower, upper, count, rate, _win_rate(all_rewards))


def _bot_label(bot: object) -> str:
    """Display name for a training opponent, including profiler wrappers."""
    inner = getattr(bot, "bot", bot)
    name = getattr(inner, "name", None)
    return str(name) if name else type(inner).__name__


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    def __init__(self, log_file: str = "training_selfplay.txt") -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"NFSP SELF-PLAY TRAINING SESSION\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n")

    def log(self, msg: str) -> None:
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Maximum steps per hand before force-break; 9 players × 4 streets × some raises.
_MAX_STEPS_PER_HAND = 200


# ---------------------------------------------------------------------------
# Game logging
# ---------------------------------------------------------------------------

def log_game(
    model: NFSPModel,
    num_players: int,
    num_hands: int,
    games_dir: Path,
    learning_seat: int = 0,
    max_bet: int = 0,
    stall_guard: bool = True,
    stall_counts: dict[str, int] | None = None,
) -> None:
    """Play and log sample games with detailed transcript format.

    Creates detailed session logs with:
    - Street-by-street action tracking with amounts, pots, positions
    - Community card reveals at each street
    - Player state snapshots (stacks, commitments, status)
    - Hole cards revealed at showdown
    - Hand action replay summary
    - Chip distribution tracking
    """
    from poker.state.game_state import Street

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = games_dir / f"session_{timestamp}.txt"
    position_names = {0: "Button", 1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO"}

    def get_position_name(dealer_seat: int, seat: int, num_players: int) -> str:
        """Get position name for a seat."""
        relative_pos = (seat - dealer_seat) % num_players
        return position_names.get(relative_pos, f"Pos{relative_pos}")

    def cards_to_str(cards) -> str:
        """Format cards as string with suit symbols."""
        import re
        if not cards:
            return "[-]"
        # Remove ANSI color codes and format cards
        card_strs = []
        for c in cards:
            card_str = str(c)
            # Remove ANSI color codes
            card_str = re.sub(r'\x1b\[[0-9;]*m', '', card_str)
            card_strs.append(card_str)
        return "[" + " ".join(card_strs) + "]"

    # Write header
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== POKER SESSION TRANSCRIPT ===\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n")

    total_reward = 0.0
    hands_played = 0
    action_type_names = ["FOLD", "CHECK", "CALL", "RAISE"]

    for hand_idx in range(num_hands):
        # Create game with randomized seeds
        opponents = [
            RandomBot(seed=int(np.random.randint(0, 2**31 - 1))) if i % 2 == 0 else FlopBot()
            for i in range(num_players - 1)
        ]
        env = PokerEnv(
            num_players=num_players,
            learning_seat=learning_seat,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
            seed=int(np.random.randint(0, 2**31 - 1)),
            max_bet=max_bet,
            stall_guard=stall_guard,
            stall_counts=stall_counts,
            **_env_encoding(model),
        )

        obs, _ = env.reset()
        state = env.state
        if state is None or state.action_on_seat != learning_seat:
            continue

        hands_played += 1
        hand_reward = 0.0
        action_log_lines = []
        street_snapshots = {}

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Hand #{hands_played} started | dealer_seat={state.dealer_seat} | blinds=25/50\n")
            f.write(f"{'='*60}\n\n")

        done = False
        while not done:
            state = env.state
            if state is None:
                break

            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=False)
            obs, reward, done, _, _ = env.step(action)
            if done:
                hand_reward = reward

        total_reward += hand_reward

        # Write final state
        if env.state is not None:
            state = env.state
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[End of {state.street.name.upper()}]\n")
                f.write(f"=== Hand #{hands_played}  |  Street: {state.street.name.upper()} ===\n\n")

                # Board
                if state.community_cards:
                    f.write(f"  Board: {cards_to_str(state.community_cards)}\n")
                else:
                    f.write(f"  Board: [ - ]\n")

                # Pot (sum of all pots)
                total_pot = sum(p.amount for p in state.pots) if state.pots else 0
                f.write(f"  Pot: {total_pot}\n")

                # Players at showdown
                f.write(f"\n  Players:\n")
                for player in state.players:
                    pos = get_position_name(state.dealer_seat, player.seat, num_players)
                    status = ""
                    if player.has_folded:
                        status = "(FOLDED)"
                    elif player.is_all_in:
                        status = "(ALL-IN)"

                    # Show hole cards at showdown
                    if state.street == Street.SHOWDOWN and player.hole_cards:
                        hole_str = cards_to_str(player.hole_cards)
                    else:
                        hole_str = "[?? ??]"

                    f.write(f"    [{pos:<6}] Player{player.seat} stack={player.stack} {hole_str} {status}\n")

                f.write(f"\n")

                # Chip distribution
                chip_dist = " ".join(f"seat{p.seat}={p.stack}" for p in state.players)
                f.write(f"Hand ended | chip_distribution: {chip_dist}\n\n")

    # Write summary
    if hands_played > 0:
        avg_reward = total_reward / hands_played
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{'='*60}\n")
            f.write(f"SESSION SUMMARY\n")
            f.write(f"{'='*60}\n")
            f.write(f"Hands played: {hands_played}\n")
            f.write(f"Total reward: {total_reward:+.4f}\n")
            f.write(f"Average reward: {avg_reward:+.4f}\n")
            f.write(f"{'='*60}\n")


def run_checkpoint_validation(
    model: NFSPModel,
    sdcfr_model: SDCFRModel,
    min_players: int,
    max_players: int,
    random_hands: int,
    trust_hands: int,
    max_bet: int = 0,
    seed: int = _EVAL_SEED,
    stall_guard: bool = True,
    stall_counts: dict[str, int] | None = None,
) -> tuple[str, EvalResult, EvalResult]:
    """Score one checkpoint with the paired single-hand eval used each epoch."""
    print("  [EVAL] Random+Flop", flush=True)
    random_eval = evaluate(
        model,
        num_hands_per_size=random_hands,
        min_players=min_players,
        max_players=max_players,
        seed=seed,
        max_bet=max_bet,
        stall_guard=stall_guard,
        stall_counts=stall_counts,
    )
    print(
        f"  [EVAL] SD-CFR ({trust_hands} hands/size, seed {seed})",
        flush=True,
    )
    sdcfr_eval = evaluate_vs_sdcfr(
        model,
        sdcfr_model,
        num_hands_per_size=trust_hands,
        min_players=min_players,
        max_players=max_players,
        seed=seed,
        max_bet=max_bet,
        stall_guard=stall_guard,
        stall_counts=stall_counts,
    )
    summary = (
        f"random={_format_bb100(random_eval.mean_bb100, random_eval.lower_bb100, random_eval.upper_bb100, random_eval.n)}"
        f" win={random_eval.win_rate:.0%} | "
        f"sdcfr={_format_bb100(sdcfr_eval.mean_bb100, sdcfr_eval.lower_bb100, sdcfr_eval.upper_bb100, sdcfr_eval.n)}"
        f" win={sdcfr_eval.win_rate:.0%}"
    )
    return summary, random_eval, sdcfr_eval


def append_csv_row(path: Path, row: dict[str, object]) -> None:
    """Append one row, writing the header when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def eval_history_row(
    episode: int,
    generation: int,
    self_play: float,
    random_eval: EvalResult,
    sdcfr_eval: EvalResult,
) -> dict[str, object]:
    """One epoch of paired BB/100 and win rate."""
    random_fold = (
        "" if random_eval.preflop_fold_rate is None else f"{random_eval.preflop_fold_rate:.6f}"
    )
    sdcfr_fold = (
        "" if sdcfr_eval.preflop_fold_rate is None else f"{sdcfr_eval.preflop_fold_rate:.6f}"
    )
    return {
        "episode": episode,
        "generation": generation,
        "self_play": f"{self_play:.4f}",
        "random_bb100": f"{random_eval.mean_bb100:.4f}",
        "random_lower": f"{random_eval.lower_bb100:.4f}",
        "random_upper": f"{random_eval.upper_bb100:.4f}",
        "random_win_rate": f"{random_eval.win_rate:.6f}",
        "random_n": random_eval.n,
        "random_preflop_fold": random_fold,
        "sdcfr_bb100": f"{sdcfr_eval.mean_bb100:.4f}",
        "sdcfr_lower": f"{sdcfr_eval.lower_bb100:.4f}",
        "sdcfr_upper": f"{sdcfr_eval.upper_bb100:.4f}",
        "sdcfr_win_rate": f"{sdcfr_eval.win_rate:.6f}",
        "sdcfr_n": sdcfr_eval.n,
        "sdcfr_preflop_fold": sdcfr_fold,
    }


def format_stall_summary(counts: dict[str, int]) -> str:
    """Bots whose move was repeated without changing the betting round."""
    if not counts:
        return "none"
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name} x{count}" for name, count in ranked)


def ladder_checkpoints(models_dir: Path, stride: int) -> list[tuple[str, Path]]:
    """Generation files whose index is a multiple of ``stride``, then the best model."""
    found: list[tuple[int, Path]] = []
    for path in models_dir.glob("nfsp_gen_*.pt"):
        match = re.search(r"nfsp_gen_(\d+)", path.name)
        if match is None:
            continue
        generation = int(match.group(1))
        if stride > 0 and generation % stride == 0:
            found.append((generation, path))
    found.sort(key=lambda item: item[0])
    rows = [(str(generation), path) for generation, path in found]
    best = models_dir / "nfsp_best.pt"
    if best.is_file():
        rows.append(("best", best))
    return rows


def _infer_next_generation(checkpoint_path: str, models_dir: Path) -> int:
    """Guess the next generation index when resuming a legacy checkpoint."""
    match = re.search(r"nfsp_gen_(\d+)", Path(checkpoint_path).name)
    if match:
        return int(match.group(1)) + 1
    existing = sorted(
        int(m.group(1))
        for p in models_dir.glob("nfsp_gen_*.pt")
        if (m := re.search(r"nfsp_gen_(\d+)", p.name))
    )
    return (existing[-1] + 1) if existing else 0


def _build_training_meta(
    sp_scheduler: SelfPlayScheduler,
    generation_count: int,
    episode: int,
    best_lower: float = -1e18,
    stalls: int = 0,
) -> dict:
    return {
        "self_play": sp_scheduler.state_dict(),
        "generation_count": generation_count,
        "episodes_completed": episode,
        "best_lower_bb100": best_lower,
        "sdcfr_stalls": stalls,
    }


def _restore_training_state(
    checkpoint_path: str,
    models_dir: Path,
    training_meta: dict,
) -> tuple[SelfPlayScheduler, int]:
    """Restore scheduler + generation counter from checkpoint metadata."""
    if training_meta:
        sp_scheduler = SelfPlayScheduler.from_state_dict(
            training_meta.get("self_play", {})
        )
        generation_count = int(training_meta.get("generation_count", 0))
        print(
            f"[RESUME] self_play={sp_scheduler.fraction:.0%} | "
            f"next_gen={generation_count} | "
            f"prior_episodes={training_meta.get('episodes_completed', '?')}"
        )
        return sp_scheduler, generation_count

    generation_count = _infer_next_generation(checkpoint_path, models_dir)
    sp_scheduler = SelfPlayScheduler()
    sp_scheduler.fraction = SelfPlayScheduler.MAX_FRACTION
    print(
        f"[RESUME] legacy checkpoint (no metadata): "
        f"self_play={sp_scheduler.fraction:.0%} | next_gen={generation_count}"
    )
    return sp_scheduler, generation_count


def _on_self_play_ramp(
    model: NFSPModel,
    old_fraction: float,
    new_fraction: float,
    q_lr_reduced: bool,
) -> bool:
    """React to a self-play increase: trim stale Q data and lower Q lr once."""
    if new_fraction <= old_fraction + 1e-9:
        return q_lr_reduced
    dropped = model.trim_q_buffer(_Q_BUFFER_KEEP_ON_RAMP)
    if dropped:
        print(f"  [BUFFER] Trimmed {dropped} stale Q transitions (SP {old_fraction:.0%}->{new_fraction:.0%})")
    if not q_lr_reduced and new_fraction >= _HIGH_SP_FRACTION:
        model.set_q_learning_rate(_HIGH_SP_Q_LR)
        model.target_sync_every = min(model.target_sync_every, 1000)
        print(
            f"  [LR] Q-network lr -> {_HIGH_SP_Q_LR}, "
            f"target sync every {model.target_sync_every} Q updates "
            f"(SP >= {_HIGH_SP_FRACTION:.0%})"
        )
        return True
    return q_lr_reduced


def score_checkpoint_ladder(
    models_dir: Path,
    sdcfr_model: SDCFRModel,
    min_players: int,
    max_players: int,
    random_hands: int,
    trust_hands: int,
    max_bet: int,
    stride: int,
    batch_size: int,
    train_every: int,
) -> None:
    """Score every ``stride``-th generation, plus the best checkpoint, on one seed."""
    selected = ladder_checkpoints(models_dir, stride)
    print(f"\n{'='*70}")
    print(f"CHECKPOINT LADDER  |  every {stride} generations  |  seed {_EVAL_SEED}")
    print(f"{'='*70}")
    if not selected:
        print("  No nfsp_gen_*.pt checkpoints to score.")
        print(f"{'='*70}\n")
        return

    ladder_path = models_dir / "checkpoint_ladder.csv"
    if ladder_path.exists():
        ladder_path.unlink()

    for label, path in selected:
        print(f"\n  [LADDER] {label}  {path.name}", flush=True)
        probe = NFSPModel(batch_size=batch_size, train_every=train_every)
        try:
            probe.load(str(path))
        except Exception as exc:
            print(f"  [LADDER] skip {path.name}: {exc}")
            continue
        probe.q_network.eval()
        probe.policy_network.eval()
        summary, random_eval, sdcfr_eval = run_checkpoint_validation(
            probe,
            sdcfr_model,
            min_players,
            max_players,
            random_hands=random_hands,
            trust_hands=trust_hands,
            max_bet=max_bet,
        )
        generation = -1 if label == "best" else int(label)
        row = eval_history_row(0, generation, 0.0, random_eval, sdcfr_eval)
        row = {"checkpoint": label, "file": path.name, **row}
        append_csv_row(ladder_path, row)
        print(f"  [LADDER] {label}: {summary}", flush=True)

    print(f"\n  Ladder file: {ladder_path}")
    print(f"{'='*70}\n")


def main() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
    else:
        print("[WARNING] No GPU detected — training will be slow")

    parser = argparse.ArgumentParser(description="NFSP Self-Play Training")
    parser.add_argument("--model-name",       type=str, default="nfsp_v1",
                        help="Model folder name (saved to models/MODEL_NAME/)")
    parser.add_argument("--checkpoint",       type=str, default=None,
                        help="Resume from nfsp_v2 checkpoint (must use nfsp_v2 directory)")
    parser.add_argument("--sdcfr-opponent",   type=str,
                        default="models/sdcfr_prod/sdcfr_ckpt_500.pt",
                        help="Frozen SD-CFR checkpoint seated on a fixed share of opponents")
    parser.add_argument("--episodes",         type=int, default=100_000)
    parser.add_argument("--eval-every",       type=int, default=5_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--log-file",         type=str, default="training_selfplay.txt")
    parser.add_argument("--min-players",      type=int, default=2)
    parser.add_argument("--max-players",      type=int, default=9)
    parser.add_argument("--batch-size",       type=int, default=2048,
                        help="Training batch size (default 2048)")
    parser.add_argument("--train-every",      type=int, default=10,
                        help="Train every N env steps (default 10)")
    parser.add_argument("--no-validation",    action="store_true",
                        help="Skip validation on checkpoint")
    parser.add_argument("--sample-games",     type=int, default=100,
                        help="Sample N games to log on checkpoint validation (default 100)")
    parser.add_argument("--trust-hands",      type=int, default=_TRUST_HANDS_PER_SIZE,
                        help="SD-CFR hands per table size for the ramp and early-stop eval")
    parser.add_argument("--random-eval-hands", type=int, default=_RANDOM_EVAL_HANDS,
                        help="Random+Flop hands per table size at each eval")
    parser.add_argument("--hands-per-episode", type=int, default=_HANDS_PER_EPISODE,
                        help="Hands in one sitting. Stacks carry and the button rotates.")
    parser.add_argument("--max-bet",          type=int, default=0,
                        help="Max chips committed on a street. 0 leaves no-limit sizing.")
    parser.add_argument("--stall-guard",      action=argparse.BooleanOptionalAction, default=True,
                        help="Record bots that repeat a move without changing the round, then force the hand on.")
    parser.add_argument("--seed",             type=int, default=None,
                        help="Seed the training RNG. Each hand still draws a new deck seed.")
    parser.add_argument("--ladder-stride",    type=int, default=_LADDER_STRIDE,
                        help="After training, score every Nth saved generation")
    parser.add_argument("--skip-ladder",      action="store_true",
                        help="Skip the end-of-training generation ladder")
    parser.add_argument("--profile",          action="store_true",
                        help="Enable performance profiling (records timing for each component)")
    parser.add_argument("--eta",              type=float, default=0.1,
                        help="Best-response mix fraction (default 0.1 = 90% avg policy)")
    parser.add_argument("--updates-per-train", type=int, default=1,
                        help="Number of updates per training step (default 1)")
    args = parser.parse_args()

    if not (2 <= args.min_players <= args.max_players <= 9):
        parser.error("Require 2 <= min-players <= max-players <= 9")
    if args.hands_per_episode < 1:
        parser.error("--hands-per-episode must be >= 1")
    if args.max_bet < 0 or (args.max_bet > 0 and args.max_bet < 50):
        parser.error("--max-bet must be 0 or at least the big blind (50)")
    if args.random_eval_hands < 1:
        parser.error("--random-eval-hands must be >= 1")
    if args.ladder_stride < 1:
        parser.error("--ladder-stride must be >= 1")
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[SEED] {args.seed}")

    # Initialize profiler if enabled
    _TimedBot = None
    _print_report = None
    _P = None
    if args.profile:
        print("[PROFILER] Enabling performance profiling...")
        # Import profiler components
        from scripts.profile_training import (
            _P, _TimedBot, _timed_build_obs, _profiled_train_step, print_report
        )
        import poker.ml.observation as _obs_mod
        import poker.ml.env as _env_mod

        # Apply monkey patches (NFSPModel is already imported at top)
        _obs_mod.build_observation = _timed_build_obs
        _env_mod.build_observation = _timed_build_obs
        NFSPModel.train_step = _profiled_train_step
        _print_report = print_report
        print("[PROFILER] Profiling enabled - timing data will be collected\n")

    # Resume writes back next to the checkpoint so frozen NFSP generations
    # (nfsp_gen_*.pt) are the copies seated in self-play. An explicit
    # --model-name other than the default keeps that folder instead.
    if args.checkpoint:
        models_dir = Path(args.checkpoint).resolve().parent
    else:
        models_dir = Path("models") / "nfsp_v2"
    models_dir.mkdir(parents=True, exist_ok=True)

    games_dir = Path("games") / "AI_games"
    games_dir.mkdir(parents=True, exist_ok=True)

    save_path = models_dir / "nfsp_v2.pt"
    log_file = models_dir / args.log_file
    history_path = models_dir / "eval_history.csv"

    logger = TrainingLogger(str(log_file))

    # Set up compilation flag
    pop_manager = PopulationManager(models_dir=str(models_dir))

    model = NFSPModel(
        batch_size=args.batch_size,
        train_every=args.train_every,
        eta=args.eta,
        updates_per_train=args.updates_per_train
    )
    training_meta: dict = {}
    if args.checkpoint:
        print(f"[LOAD] {args.checkpoint}")
        training_meta = model.load(args.checkpoint)
        obs_dim = int(model.q_network.input_dim)
        if obs_dim != _OBS_DIM:
            raise SystemExit(
                f"Refusing checkpoint {args.checkpoint}: "
                f"q_network.input_dim={obs_dim}, expected {_OBS_DIM}. "
                f"Resume only from models/nfsp_prod/nfsp_selfplay.pt."
            )
        sp_scheduler, generation_count = _restore_training_state(
            args.checkpoint, save_path.parent, training_meta
        )
    else:
        print("[INIT] Training new model from scratch")
        sp_scheduler = SelfPlayScheduler()
        # Start at max self-play if profiling (test at realistic conditions)
        if args.profile:
            sp_scheduler.fraction = SelfPlayScheduler.MAX_FRACTION
            print(f"[PROFILER] Starting self-play at {sp_scheduler.fraction:.0%} for realistic profiling")
        generation_count = 0

    sdcfr_path = Path(args.sdcfr_opponent)
    if not sdcfr_path.is_file():
        raise SystemExit(f"SD-CFR opponent checkpoint not found: {sdcfr_path}")
    print(f"[LOAD] {sdcfr_path} (frozen SD-CFR, once)")
    sdcfr_model = SDCFRModel()
    sdcfr_model.load(str(sdcfr_path))
    sdcfr_model.advantage_network.eval()
    print(f"[OBS] q_network.input_dim={int(model.q_network.input_dim)}")

    # Print actual compilation status based on model's _compile_enabled flag
    compile_status = "yes" if hasattr(model, "_compile_enabled") and model._compile_enabled else "no"
    print(f"[COMPILE] Networks compiled: {compile_status}")
    print()

    print(f"\n{'='*70}")
    print(f"NFSP SELF-PLAY TRAINING")
    print(f"{'='*70}")
    print(f"  Model name:       {args.model_name}")
    print(f"  Players/episode:  {args.min_players}-{args.max_players} (6p weighted 2x)")
    print(f"  Hands/episode:    {args.hands_per_episode} (stacks carry, new deck and button each hand)")
    print(f"  Episodes:         {args.episodes:,}")
    print(f"  Blinds:           25/50  |  Stack: 1000")
    print(f"  Max bet:          {'none' if args.max_bet == 0 else args.max_bet}")
    print(f"  Stall guard:      {'on' if args.stall_guard else 'off'}")
    print(f"  Batch size:       {args.batch_size:,}  |  Train every: {args.train_every} steps")
    print(f"  SD-CFR seats:     {_SDCFR_SEAT_FRACTION:.0%} of opponents (min 1), frozen")
    print(f"  SD-CFR file:      {sdcfr_path}")
    print(f"  Self-play schedule: +{SelfPlayScheduler.RAMP_STEP:.0%} per eval, cap {SelfPlayScheduler.MAX_FRACTION:.0%} (resume={sp_scheduler.fraction:.0%})")
    print(f"  Ramp requires:    preflop fold >= {SelfPlayScheduler.MIN_PREFLOP_FOLD:.0%} and SD-CFR lower bound > {SelfPlayScheduler.SDCFR_RAMP_FLOOR_BB100:+.0f} BB/100")
    print(f"  Early stop:       {_STOP_PATIENCE} evals without a better SD-CFR lower bound, after episode {_STOP_MIN_EPISODE}")
    print(f"  Trusted SD-CFR:   {args.trust_hands} hands per size, paired seed {_EVAL_SEED}")
    print(f"  Random+Flop eval: {args.random_eval_hands} hands per size")
    ladder_note = "skipped" if args.skip_ladder else f"every {args.ladder_stride} generations"
    print(f"  Checkpoint ladder:{ladder_note}")
    print(f"  Next generation:  {generation_count}")
    print(f"  Eval metric:      BB/100 vs Random+Flop and vs frozen SD-CFR")
    print(f"  Eval player sizes:{_EVAL_PLAYER_COUNTS}")
    print(f"  Model folder:     {models_dir}")
    print(f"  Game logs:        {games_dir}")
    print(f"  Auto-validation:  {'enabled' if not args.no_validation else 'disabled'}")
    print()

    last_q_loss: float | None = None
    last_policy_loss: float | None = None
    start_time = time.time()
    episode_times: list[float] = []
    q_lr_reduced = (
        args.checkpoint is not None
        and sp_scheduler.fraction >= _HIGH_SP_FRACTION
    )
    if q_lr_reduced:
        model.set_q_learning_rate(_HIGH_SP_Q_LR)

    logged_opponent_field = False
    checked_obs_width = False
    best_lower = float(training_meta.get("best_lower_bb100", -1e18)) if training_meta else -1e18
    stalls = int(training_meta.get("sdcfr_stalls", 0)) if training_meta else 0
    stop_training = False
    best_path: Path | None = None
    stall_counts: dict[str, int] = {}

    for episode in range(args.episodes):
        ep_start = time.time()

        # Sample player count with 2x weight on 6-player to improve 6p performance
        player_counts = list(range(args.min_players, args.max_players + 1))
        weights = np.array([2.0 if n == 6 else 1.0 for n in player_counts], dtype=float)
        weights = weights / weights.sum()
        num_players = int(np.random.choice(player_counts, p=weights))

        num_opponents = num_players - 1
        learning_seat = int(np.random.randint(0, num_players))

        opponents = build_opponents(
            num_opponents=num_opponents,
            self_play_fraction=sp_scheduler.fraction,
            pop_manager=pop_manager,
            generation_count=generation_count,
            profiler_timed_bot=_TimedBot,
            sdcfr_model=sdcfr_model,
        )
        np.random.shuffle(opponents)
        if not logged_opponent_field and num_players == 6:
            names = [_bot_label(bot) for bot in opponents]
            sdcfr_ids = {id(bot.model) for bot in opponents if isinstance(bot, SDCFRBot)}
            field_line = (
                f"[OPPONENTS] 6-max: {names} | "
                f"sdcfr_model_objects={len(sdcfr_ids)}"
            )
            print(field_line, flush=True)
            logger.log(field_line)
            logged_opponent_field = True

        env = PokerEnv(
            num_players=num_players,
            learning_seat=learning_seat,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
            seed=int(np.random.randint(0, 2**31 - 1)),
            max_bet=args.max_bet,
            stall_guard=args.stall_guard,
            stall_counts=stall_counts,
        )
        # Button is hand_number % seats. A random start, then +1 each hand.
        env.hand_number = int(np.random.randint(0, num_players))
        session_deltas: list[float] = []

        for hand_i in range(args.hands_per_episode):
            deck_seed = int(np.random.randint(0, 2**31 - 1))
            if _P is not None:
                _P._step_bot_s = 0.0
                _P._step_obs_s = 0.0
                reset_t0 = time.perf_counter()
            if hand_i == 0:
                env.rng = np.random.Generator(np.random.PCG64(deck_seed))
                obs, _ = env.reset()
            else:
                opened = env.start_next_hand(seed=deck_seed)
                if opened is None:
                    break
                obs, _ = opened
            if _P is not None:
                reset_elapsed = time.perf_counter() - reset_t0
                _P.record("env.reset() [total]", reset_elapsed)
                _P.record("env.reset() [game_logic]",
                          reset_elapsed - _P._step_bot_s - _P._step_obs_s)

            if not checked_obs_width:
                obs_width = int(np.asarray(obs).shape[-1])
                if obs_width != int(model.q_network.input_dim):
                    raise SystemExit(
                        f"Observation width {obs_width} does not match "
                        f"q_network.input_dim={int(model.q_network.input_dim)}"
                    )
                print(f"[OBS] forward width={obs_width}", flush=True)
                checked_obs_width = True

            if env.state is None:
                break
            if env.state.action_on_seat is None:
                env.state = env._resolve_hand()
                session_deltas.append(env._compute_reward(learning_seat))
                continue
            if env.state.action_on_seat != learning_seat:
                break

            model.begin_episode()
            done = False
            hand_step = 0
            reward = 0.0

            while not done:
                # Record get_action_mask() timing if profiling
                if _P is not None:
                    mask_t0 = time.perf_counter()
                mask = env.get_action_mask()
                if _P is not None:
                    _P.record("get_action_mask()", time.perf_counter() - mask_t0)

                # Record select_action() timing if profiling
                if _P is not None:
                    action_t0 = time.perf_counter()
                action = model.select_action(obs, mask, training=True)
                if _P is not None:
                    _P.record("model.select_action()", time.perf_counter() - action_t0)

                # Record env.step() timing if profiling
                if _P is not None:
                    _P._step_bot_s = 0.0
                    _P._step_obs_s = 0.0
                    step_t0 = time.perf_counter()
                next_obs, reward, done, _, _ = env.step(action)
                if _P is not None:
                    step_elapsed = time.perf_counter() - step_t0
                    _P.record("env.step() [total]", step_elapsed)
                    step_bot = _P._step_bot_s
                    step_obs = _P._step_obs_s
                    game_logic = step_elapsed - step_bot - step_obs
                    if game_logic > 0:
                        _P.record("env.step() [game_logic]", game_logic)

                # Next-state legal mask for Double DQN; all-legal at terminal.
                next_mask = env.get_action_mask() if not done else np.ones(7, dtype=np.float32)
                # Terminal-only reward in Q-buffer (exclude mid-hand intrinsic shaping).
                q_reward = reward if done else 0.0

                # Record store_transition() timing if profiling
                if _P is not None:
                    store_t0 = time.perf_counter()
                model.store_transition(obs, action, q_reward, next_obs, done, next_mask=next_mask)
                if _P is not None:
                    _P.record("store_transition()", time.perf_counter() - store_t0)

                result = model.train_step()

                if result["q_loss"] is not None:
                    last_q_loss = result["q_loss"]
                    last_policy_loss = result["policy_loss"]

                obs = next_obs
                hand_step += 1
                if hand_step >= _MAX_STEPS_PER_HAND:
                    break

            if done:
                session_deltas.append(reward)
            else:
                # The hand did not finish, so the sitting cannot carry stacks.
                break

        episode_times.append(time.time() - ep_start)
        if session_deltas:
            delta_text = ",".join(f"{delta:+.5f}" for delta in session_deltas)
            logger.log(f"[Ep {episode}] hand_deltas={delta_text}")

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if episode % args.eval_every == 0 and not args.no_validation:
            print(f"\n  [EVAL] Episode {episode} | {sp_scheduler.status}", flush=True)
            sp_before = sp_scheduler.fraction
            model.q_network.eval()
            model.policy_network.eval()
            print("  [EVAL] Random+Flop", flush=True)
            random_eval = evaluate(
                model,
                num_hands_per_size=args.random_eval_hands,
                min_players=args.min_players,
                max_players=args.max_players,
                seed=_EVAL_SEED,
                max_bet=args.max_bet,
                stall_guard=args.stall_guard,
                stall_counts=stall_counts,
            )
            print(
                f"  [EVAL] SD-CFR  ({args.trust_hands} hands/size, seed {_EVAL_SEED})",
                flush=True,
            )
            sdcfr_eval = evaluate_vs_sdcfr(
                model,
                sdcfr_model,
                num_hands_per_size=args.trust_hands,
                min_players=args.min_players,
                max_players=args.max_players,
                seed=_EVAL_SEED,
                max_bet=args.max_bet,
                stall_guard=args.stall_guard,
                stall_counts=stall_counts,
            )
            stall_line = f"  [STALL] repeated no-progress moves: {format_stall_summary(stall_counts)}"
            print(stall_line, flush=True)
            logger.log(stall_line)
            append_csv_row(
                history_path,
                eval_history_row(
                    episode,
                    generation_count,
                    sp_scheduler.fraction,
                    random_eval,
                    sdcfr_eval,
                ),
            )
            model.q_network.train()
            model.policy_network.train()
            ramp_note = sp_scheduler.apply_scheduled_step(
                sdcfr_eval.preflop_fold_rate, sdcfr_eval.lower_bb100
            )
            print(f"  [RAMP] {ramp_note}", flush=True)
            q_lr_reduced = _on_self_play_ramp(
                model, sp_before, sp_scheduler.fraction, q_lr_reduced
            )
            if sdcfr_eval.lower_bb100 > best_lower:
                best_lower = sdcfr_eval.lower_bb100
                stalls = 0
                best_path = models_dir / "nfsp_best.pt"
                model.save(
                    str(best_path),
                    training_meta=_build_training_meta(
                        sp_scheduler, generation_count, episode, best_lower, stalls
                    ),
                )
                print(
                    f"  [BEST] SD-CFR lower bound {best_lower:+.1f} BB/100 -> {best_path}",
                    flush=True,
                )
            else:
                stalls += 1
                if episode >= _STOP_MIN_EPISODE and stalls >= _STOP_PATIENCE:
                    stop_training = True
                    print(
                        f"  [STOP] SD-CFR lower bound did not improve for {stalls} evals "
                        f"(best {best_lower:+.1f})",
                        flush=True,
                    )
            q_str = f"{last_q_loss:.4f}" if last_q_loss is not None else "n/a"
            p_str = f"{last_policy_loss:.4f}" if last_policy_loss is not None else "n/a"
            elapsed = time.time() - start_time
            avg_ep = (np.mean(episode_times[-200:])
                      if len(episode_times) >= 200 else np.mean(episode_times))
            eta = (args.episodes - episode) * avg_ep / 60.0
            fold_str = (
                "n/a" if sdcfr_eval.preflop_fold_rate is None
                else f"{sdcfr_eval.preflop_fold_rate:.0%}"
            )

            line = (
                f"Ep {episode:7d}/{args.episodes} | "
                f"q_loss={q_str} | p_loss={p_str} | "
                f"eval={_format_bb100(random_eval.mean_bb100, random_eval.lower_bb100, random_eval.upper_bb100, random_eval.n)} win={random_eval.win_rate:.0%} | "
                f"sdcfr={_format_bb100(sdcfr_eval.mean_bb100, sdcfr_eval.lower_bb100, sdcfr_eval.upper_bb100, sdcfr_eval.n)} win={sdcfr_eval.win_rate:.0%} | "
                f"pf_fold={fold_str} | "
                f"sp={sp_scheduler.fraction:.0%} | "
                f"elapsed={elapsed:.0f}s | ETA={eta:.1f}m"
            )
            print(f"  {line}", flush=True)
            logger.log(line)

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if episode > 0 and episode % args.checkpoint_every == 0:
            gen_path = models_dir / f"nfsp_gen_{generation_count}.pt"
            print(f"  [GEN {generation_count}] Saving -> {gen_path}", end="", flush=True)
            meta = _build_training_meta(
                sp_scheduler, generation_count + 1, episode, best_lower, stalls
            )
            model.save(str(gen_path), training_meta=meta)
            print(" done")
            logger.log(f"[Ep {episode}] Gen {generation_count} saved: {gen_path}")

            # Reuse this epoch's paired eval when it just ran. Otherwise
            # score the saved generation with that same eval.
            if not args.no_validation:
                just_scored = episode % args.eval_every == 0
                if just_scored:
                    val_summary = (
                        f"random={_format_bb100(random_eval.mean_bb100, random_eval.lower_bb100, random_eval.upper_bb100, random_eval.n)}"
                        f" win={random_eval.win_rate:.0%} | "
                        f"sdcfr={_format_bb100(sdcfr_eval.mean_bb100, sdcfr_eval.lower_bb100, sdcfr_eval.upper_bb100, sdcfr_eval.n)}"
                        f" win={sdcfr_eval.win_rate:.0%}"
                    )
                    print(f"  [VALIDATE] Gen {generation_count} {val_summary}")
                else:
                    print(f"  [VALIDATE] Gen {generation_count}", flush=True)
                    model.q_network.eval()
                    model.policy_network.eval()
                    val_summary, _, _ = run_checkpoint_validation(
                        model,
                        sdcfr_model,
                        args.min_players,
                        args.max_players,
                        random_hands=args.random_eval_hands,
                        trust_hands=args.trust_hands,
                        max_bet=args.max_bet,
                        stall_guard=args.stall_guard,
                        stall_counts=stall_counts,
                    )
                    model.q_network.train()
                    model.policy_network.train()
                    print(f"  [VALIDATE] {val_summary}")
                logger.log(f"[Ep {episode}] Gen {generation_count} validation: {val_summary}")

                if args.min_players <= 6 <= args.max_players:
                    # log_game derives obs_version/bet_scheme from the model
                    # itself via _env_encoding(model); passing them here would
                    # not reach PokerEnv and is not in log_game's signature.
                    log_game(
                        model,
                        num_players=6,
                        num_hands=args.sample_games,
                        games_dir=games_dir,
                        learning_seat=0,
                        max_bet=args.max_bet,
                        stall_guard=args.stall_guard,
                        stall_counts=stall_counts,
                    )

            generation_count += 1

        if stop_training:
            break

    # Final artifact is the checkpoint with the best SD-CFR lower bound.
    if best_path is not None and best_path.is_file():
        shutil.copy(best_path, save_path)
        print(f"  [BEST] Copied {best_path.name} -> {save_path.name} (lower bound {best_lower:+.1f} BB/100)")
    else:
        meta = _build_training_meta(
            sp_scheduler, generation_count, args.episodes - 1, best_lower, stalls
        )
        model.save(str(save_path), training_meta=meta)
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE  |  {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Final self-play fraction: {sp_scheduler.fraction:.0%}")
    print(f"Generations saved: {generation_count}")
    print(f"Model folder: {models_dir}")
    print(f"Final model: {save_path}")
    print(f"Game logs: {games_dir}")
    print(f"Eval history: {models_dir / 'eval_history.csv'}")
    print(f"Stall summary: {format_stall_summary(stall_counts)}")
    print(f"{'='*70}\n")

    if not args.skip_ladder:
        score_checkpoint_ladder(
            models_dir=models_dir,
            sdcfr_model=sdcfr_model,
            min_players=args.min_players,
            max_players=args.max_players,
            random_hands=args.random_eval_hands,
            trust_hands=args.trust_hands,
            max_bet=args.max_bet,
            stride=args.ladder_stride,
            batch_size=args.batch_size,
            train_every=args.train_every,
        )

    # Print profiling report if enabled
    if _print_report is not None:
        _print_report()


if __name__ == "__main__":
    main()
