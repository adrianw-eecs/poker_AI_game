#!/usr/bin/env python
"""NFSP v2 Multi-Seat Self-Play Training with eta sweep support.

This script trains NFSP v2 (98-wide obs v2, pot-fraction betting) with
support for command-line eta and updates_per_train parameters.

Usage:
  python scripts/train_nfsp_v2.py --checkpoint models/nfsp_v2/nfsp_v2.pt --eta 0.2 --updates-per-train 2
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.bots.sdcfr_bot import SDCFRBot
from poker.ml.action_space import BetScheme
from poker.ml.env import PokerEnv
from poker.ml.metrics import bb100_interval
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.nfsp_networks_v3 import checkpoint_obs_version
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.ml.population import PopulationManager

# Evaluation constants
_EVAL_SEED = 17_001
_HANDS_PER_SIZE = 20
_MIN_PLAYERS = 2
_MAX_PLAYERS = 9

# Stop conditions
_STOP_PATIENCE = 10
_STOP_MIN_EPISODE = 20_000
_TRUST_HANDS_PER_SIZE = 400
_RANDOM_EVAL_HANDS = 100
_HU_HANDS = 40


def build_opponents(
    num_opponents: int,
    self_play_fraction: float,
    pop_manager: PopulationManager,
    generation_count: int,
    sdcfr_model: SDCFRModel | None = None,
) -> list:
    """Build opponent list with SD-CFR seats first, then self-play, then baseline bots."""
    if num_opponents == 0:
        return []

    # Reserve SD-CFR seats first (40% of opponents)
    n_sdcfr = min(num_opponents, max(1, int(round(num_opponents * 0.40))))
    n_self = int(round((num_opponents - n_sdcfr) * self_play_fraction))
    n_baseline = num_opponents - n_sdcfr - n_self

    bots: list = []

    # SD-CFR seats (frozen, shared model)
    if n_sdcfr and sdcfr_model is None:
        raise RuntimeError("SD-CFR seats requested but no frozen SDCFRModel was loaded")

    for i in range(n_sdcfr):
        bots.append(SDCFRBot(name=f"SDCFR_{i}", model=sdcfr_model))

    # Baseline bots: RandomBot, FlopBot, CallBot
    if n_baseline > 0:
        bot_types = [RandomBot, FlopBot, CallBot]
        for i in range(n_baseline):
            bot_cls = bot_types[i % len(bot_types)]
            if bot_cls is RandomBot:
                bots.append(RandomBot(name=f"Random_{i}", seed=int(np.random.randint(0, 2**31 - 1))))
            elif bot_cls is FlopBot:
                bots.append(FlopBot(name=f"Flop_{i}"))
            else:
                bots.append(CallBot())

    # Self-play opponents
    if n_self > 0:
        for i in range(n_self):
            if generation_count > 0:
                try:
                    gen_id = generation_count - 1 - (i % 5)
                    prev_model = pop_manager.load_generation(gen_id)
                    bots.append(NFSPBot(name=f"Self_{gen_id}_{i}", model=prev_model, training=False))
                except FileNotFoundError:
                    bots.append(RandomBot(name=f"SelfFallback_{i}", seed=int(np.random.randint(0, 2**31 - 1))))
            else:
                bots.append(RandomBot(name=f"SelfFallback_{i}", seed=int(np.random.randint(0, 2**31 - 1))))

    return bots


def evaluate(
    model: NFSPModel,
    num_hands_per_size: int = _HANDS_PER_SIZE,
    min_players: int = _MIN_PLAYERS,
    max_players: int = _MAX_PLAYERS,
    seed: int = _EVAL_SEED,
    max_bet: int = 0,
    stall_guard: bool = True,
    stall_counts: Dict[str, int] | None = None,
) -> tuple[float, float, float, int]:
    """Evaluate model performance across multiple player counts.

    Returns: (mean_bb100, lower_bb100, upper_bb100, n)
    """
    rng = np.random.RandomState(seed)
    all_rewards: list[float] = []

    counts = sorted(set(n for n in [_MIN_PLAYERS, 4, 6, 9] if min_players <= n <= max_players))
    
    for n in counts:
        for config in range(2):
            for h in range(num_hands_per_size):
                env_seed = int(rng.randint(0, 2**31 - 1))
                if config == 0:
                    opponents = [RandomBot(seed=int(rng.randint(0, 2**31 - 1))) for _ in range(n - 1)]
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
                    obs_version=int(getattr(model, "obs_version", 2)),
                    bet_scheme=BetScheme.POT_FRACTION,
                )
                env.hand_number = h
                
                obs, _ = env.reset()
                reward = 0.0
                
                while True:
                    mask = env.get_action_mask()
                    action = model.select_action(np.asarray(obs), mask, training=False)
                    obs, reward, done, _, _ = env.step(action)
                    if done:
                        break
                    if env.state is None or env.state.action_on_seat != env.learning_seat:
                        break
                    obs = np.asarray(obs)
                
                all_rewards.append(reward)

    mean, lower, upper, count = bb100_interval(all_rewards)
    return mean, lower, upper, count


def evaluate_vs_sdcfr(
    model: NFSPModel,
    sdcfr_model: SDCFRModel,
    num_hands_per_size: int = _TRUST_HANDS_PER_SIZE,
    min_players: int = _MIN_PLAYERS,
    max_players: int = _MAX_PLAYERS,
    seed: int = _EVAL_SEED,
    max_bet: int = 0,
    stall_guard: bool = True,
    stall_counts: Dict[str, int] | None = None,
) -> tuple[float, float, float, int]:
    """Evaluate against frozen SD-CFR model."""
    rng = np.random.RandomState(seed + 1)
    all_rewards: list[float] = []

    counts = sorted(set(n for n in [_MIN_PLAYERS, 4, 6, 9] if min_players <= n <= max_players))
    
    for n in counts:
        for h in range(num_hands_per_size):
            env_seed = int(rng.randint(0, 2**31 - 1))
            opp_list = [SDCFRBot(name=f"SDCFR_eval_{i}", model=sdcfr_model) for i in range(n - 1)]
            
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
                obs_version=int(getattr(model, "obs_version", 2)),
                bet_scheme=BetScheme.POT_FRACTION,
            )
            env.hand_number = h
            
            obs, _ = env.reset()
            reward = 0.0
            
            while True:
                mask = env.get_action_mask()
                action = model.select_action(np.asarray(obs), mask, training=False)
                obs, reward, done, _, _ = env.step(action)
                if done:
                    break
                if env.state is None or env.state.action_on_seat != env.learning_seat:
                    break
                obs = np.asarray(obs)
            
            all_rewards.append(reward)

    mean, lower, upper, count = bb100_interval(all_rewards)
    return mean, lower, upper, count


def evaluate_sdcfr_only(
    model: NFSPModel,
    sdcfr_model: SDCFRModel,
    num_hands_per_size: int = _TRUST_HANDS_PER_SIZE,
    min_players: int = _MIN_PLAYERS,
    max_players: int = _MAX_PLAYERS,
    seed: int = _EVAL_SEED,
    max_bet: int = 0,
    stall_guard: bool = True,
    stall_counts: Dict[str, int] | None = None,
) -> tuple[float, float, float, int]:
    """Evaluate ONLY against frozen SD-CFR model (no Random+Flop baseline)."""
    rng = np.random.RandomState(seed + 1)
    all_rewards: list[float] = []

    counts = sorted(set(n for n in [_MIN_PLAYERS, 4, 6, 9] if min_players <= n <= max_players))
    
    for n in counts:
        for h in range(num_hands_per_size):
            env_seed = int(rng.randint(0, 2**31 - 1))
            opp_list = [SDCFRBot(name=f"SDCFR_only_eval_{i}", model=sdcfr_model) for i in range(n - 1)]
            
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
                obs_version=int(getattr(model, "obs_version", 2)),
                bet_scheme=BetScheme.POT_FRACTION,
            )
            env.hand_number = h
            
            obs, _ = env.reset()
            reward = 0.0
            
            while True:
                mask = env.get_action_mask()
                action = model.select_action(np.asarray(obs), mask, training=False)
                obs, reward, done, _, _ = env.step(action)
                if done:
                    break
                if env.state is None or env.state.action_on_seat != env.learning_seat:
                    break
                obs = np.asarray(obs)
            
            all_rewards.append(reward)

    mean, lower, upper, count = bb100_interval(all_rewards)
    return mean, lower, upper, count


def _build_training_meta(
    episode: int,
    best_lower: float,
    stalls: int,
    model: NFSPModel,
) -> Dict[str, Any]:
    return {
        "best_lower_bb100": best_lower,
        "sdcfr_stalls": stalls,
        "eta": model.eta,
        "updates_per_train": model.updates_per_train,
    }


def _write_history_row(path: Path, row: dict[str, object]) -> None:
    """Append one evaluation row, writing the header on first use.

    Columns are taken from the first row written, so a later row with a
    different key set (sdcfr-only vs standard eval) would misalign. Keys are
    therefore unioned against the existing header when the file already exists.
    """
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh), [])
        if header:
            row = {k: row.get(k, "") for k in header}
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if fh.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="NFSP v2 Multi-Seat Self-Play Training")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from nfsp_v2 checkpoint")
    parser.add_argument("--sdcfr-opponent", type=str,
                        default="models/sdcfr_prod/sdcfr_ckpt_500.pt",
                        help="Frozen SD-CFR checkpoint")
    parser.add_argument("--episodes", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--log-file", type=str, default="training_v2.txt")
    parser.add_argument("--min-players", type=int, default=2)
    parser.add_argument("--max-players", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size (default: 2048)")
    parser.add_argument("--train-every", type=int, default=64,
                        help="Run a gradient update every N decisions (counts train_step calls, "
                             "not episodes; see docs/PERFORMANCE.md)")
    parser.add_argument("--eta", type=float, default=0.1,
                        help="Best-response mix fraction (0.0-1.0)")
    parser.add_argument("--updates-per-train", type=int, default=1,
                        help="Number of updates per training step")
    parser.add_argument("--random-eval-hands", type=int, default=100,
                        help="Random+Flop evaluation hands per size")
    parser.add_argument("--stop-min-episode", type=int, default=20_000,
                        help="Min episodes before stall check (default: 20,000)")
    parser.add_argument("--stop-patience", type=int, default=10,
                        help="Number of evals before stopping (default: 10)")
    parser.add_argument("--sdcfr-eval-only", action="store_true",
                        help="Evaluate only against SD-CFR (not Random+Flop)")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--policy-batch-threshold", type=int, default=None,
                        help="Policy batch threshold (default: batch_size)")
    parser.add_argument("--model-name", type=str, default="nfsp_v2",
                        help="Model folder name (saved to models/MODEL_NAME/)")
    parser.add_argument("--obs-version", type=int, default=3, choices=(2, 3, 4),
                        help="Observation encoder: 2 (318-wide), 3 (354-wide, adds "
                             "opponent tendencies + attention), or 4 (372-wide, adds "
                             "equity and pot structure). Checkpoints do not load across "
                             "versions.")
    parser.add_argument("--value-distribution", type=str, default="c51",
                        choices=("none", "c51"),
                        help="Q head: scalar ('none') or categorical distribution ('c51')")
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--vmin", type=float, default=-10.0)
    parser.add_argument("--vmax", type=float, default=10.0)
    parser.add_argument("--n-step", type=int, default=3,
                        help="Bootstrap every N decisions instead of every 1")
    parser.add_argument("--reservoir-min-prob", type=float, default=0.25,
                        help="Exponential averaging floor for the average-policy "
                             "reservoir; 0 = uniform (drags the policy backward)")

    args = parser.parse_args()

    if not (2 <= args.min_players <= args.max_players <= 9):
        parser.error("Require 2 <= min-players <= max-players <= 9")

    # Use model-name argument
    models_dir = Path("models") / args.model_name
    models_dir.mkdir(parents=True, exist_ok=True)

    games_dir = Path("games") / "AI_games"
    games_dir.mkdir(parents=True, exist_ok=True)

    save_path = models_dir / "nfsp_v2.pt"
    log_file = models_dir / args.log_file
    history_path = models_dir / "eval_history.csv"

    pop_manager = PopulationManager(models_dir=str(models_dir))
    
    # Fix for Issue #1: Set policy batch threshold if specified
    if args.policy_batch_threshold is not None:
        print(f"[INFO] Setting policy batch threshold to {args.policy_batch_threshold}")

    # The env, eval helpers and _begin all follow model.obs_version, so this
    # flag controls the encoder for the whole run.
    model = NFSPModel(
        obs_version=args.obs_version,
        bet_scheme=BetScheme.POT_FRACTION,
        eta=args.eta,
        updates_per_train=args.updates_per_train,
        batch_size=args.batch_size,
        train_every=args.train_every,
        q_capacity=1_000_000,
        policy_capacity=1_000_000,
        q_lr=2.5e-4,
        policy_lr=1e-4,
        target_sync_every=300,
        # These were previously implicit defaults of NFSPModel. They are passed
        # explicitly so this trainer's behaviour is reproducible and can be
        # ablated from the command line rather than changing under it.
        reservoir_min_prob=args.reservoir_min_prob,
        n_step=args.n_step,
        value_distribution=args.value_distribution,
        num_atoms=args.num_atoms,
        vmin=args.vmin,
        vmax=args.vmax,
    )

    training_meta: Dict = {}
    if args.checkpoint:
        # Check the encoder BEFORE loading: load() rebuilds the network from the
        # checkpoint's own shapes, so a width mismatch surfaces as an opaque
        # "Error(s) in loading state_dict" instead of an actionable message.
        _ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        _q_state = _ck.get("q_network", {})
        _ck_version = checkpoint_obs_version(_q_state)
        del _ck, _q_state
        if _ck_version != args.obs_version:
            raise SystemExit(
                f"Refusing {args.checkpoint}: checkpoint obs_version={_ck_version}, "
                f"but --obs-version is {args.obs_version}. Pass a matching "
                "--obs-version or a matching checkpoint."
            )
        print(f"[LOAD] {args.checkpoint}")
        training_meta = model.load(args.checkpoint)
        print(f"[LOAD] eta={model.eta}, updates_per_train={model.updates_per_train}")
        print(f"[OVERRIDE] Command line: eta={args.eta}, updates_per_train={args.updates_per_train}")
        
        model.eta = args.eta
        model.updates_per_train = args.updates_per_train
    else:
        print("[INIT] NFSP v2 from scratch")

    sdcfr_path = Path(args.sdcfr_opponent)
    if not sdcfr_path.is_file():
        raise SystemExit(f"SD-CFR checkpoint not found: {sdcfr_path}")

    print(f"[{datetime.now()}] [LOAD] {sdcfr_path}")
    sdcfr_model = SDCFRModel()
    sdcfr_model.load(str(sdcfr_path))
    sdcfr_model.advantage_network.eval()

    best_lower = float(training_meta.get("best_lower_bb100", -1e18)) if training_meta else -1e18
    stalls = int(training_meta.get("sdcfr_stalls", 0)) if training_meta else 0
    best_path: Optional[Path] = None

    print(f"\n{'='*70}")
    print(f"NFSP V2 SELF-PLAY TRAINING")
    print(f"{'='*70}")
    print(f"  Model:        nfsp_v2")
    print(f"  Eta:          {model.eta:.0%} eta={model.eta}")
    print(f"  Updates/train:{model.updates_per_train}")
    print(f"  Episodes:     {args.episodes:,}")
    print(f"  Players:      {args.min_players}-{args.max_players}")
    print(f"  Batch:        {args.batch_size:,}")
    print(f"  Train every:  {args.train_every}")
    print(f"  SD-CFR:       {sdcfr_path}")
    print(f"  Save every:   {args.checkpoint_every} episodes")
    print()

    start = time.time()
    episode = 0
    last_q_loss: float | None = None
    last_p_loss: float | None = None

    for episode in range(args.episodes):
        ep_start = time.time()

        num_players = int(np.random.randint(args.min_players, args.max_players + 1))
        learning_seat = int(np.random.randint(0, num_players))

        num_opponents = num_players - 1
        opponents = build_opponents(
            num_opponents,
            model.eta,
            pop_manager,
            0,
            sdcfr_model,
        )

        env = PokerEnv(
            num_players=num_players,
            learning_seat=learning_seat,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
            seed=int(np.random.randint(0, 2**31 - 1)),
            obs_version=int(getattr(model, "obs_version", 2)),
            bet_scheme=BetScheme.POT_FRACTION,
        )

        obs, _ = env.reset()

        # Anticipatory dynamics: sample one mode (best response vs average
        # policy) per episode. Without this the mode is drawn lazily on the very
        # first decision and then held for the entire run, so the reservoir only
        # ever fills if that single draw happened to land on best response.
        model.begin_episode(learning_seat)

        while True:
            mask = env.get_action_mask()
            current_obs = np.asarray(obs)
            action = model.select_action(current_obs, mask, training=True, seat=learning_seat)
            obs, reward, done, _, _ = env.step(action)

            if env.state is not None and env.state.action_on_seat == learning_seat:
                if done:
                    hand_return = reward
                else:
                    hand_return = 0.0
                model.store_transition(current_obs, action, hand_return, np.asarray(obs), done, mask)

            if done:
                break
            if env.state is None:
                break
            if env.state.action_on_seat != learning_seat:
                break

        # Training step: one decision = one step. train_step() decides internally
        # whether this step falls on a train_every boundary.
        train_result = model.train_step()

        # One hand finished. The Q warm-up gate counts hands, not decisions.
        model.end_hand()

        # Track the most recent real loss OUTSIDE the logging branch. train_step
        # returns None on every non-boundary step, so the episode%100 log line
        # almost never coincides with a boundary and would print "None" forever.
        if train_result["q_loss"] is not None:
            last_q_loss = train_result["q_loss"]
        if train_result["policy_loss"] is not None:
            last_p_loss = train_result["policy_loss"]

        # Periodic logging. This MUST NOT call train_step() again: doing so
        # injected an extra gradient update per log line (~1% of steps), which
        # meant logged losses came from a different step than the training loop.
        if episode % 100 == 0:
            q_loss_val = train_result["q_loss"]
            p_loss_val = train_result["policy_loss"]

            # "*" marks a loss measured on this exact step; a bare number is the
            # most recent earlier value.
            q_str = (
                f"{q_loss_val:8.4f}*" if q_loss_val is not None
                else f"{last_q_loss:8.4f} " if last_q_loss is not None
                else "    None "
            )
            p_str = (
                f"{p_loss_val:8.4f}*" if p_loss_val is not None
                else f"{last_p_loss:8.4f} " if last_p_loss is not None
                else "    None "
            )

            print(
                f"[{datetime.now()}]   [TRAIN] Ep {episode:5d}: "
                f"Q_buf={len(model.q_buffer):6d}, P_buf={len(model.policy_buffer):6d}, "
                f"Q_upd={model._q_updates:6d}, P_upd={model._p_updates:6d}, "
                f"Q_loss={q_str}, P_loss={p_str}"
            )

        # Evaluation
        if episode % args.eval_every == 0 and not args.no_validation:
            print(f"[{datetime.now()}]   [EVAL] Episode {episode}")
            model.q_network.eval()
            model.policy_network.eval()

            if args.sdcfr_eval_only:
                # SD-CFR only evaluation (Issue #3 fix)
                print(f"      [SD-CFR ONLY EVAL]")
                sdcfr_mean, sdcfr_lower, sdcfr_upper, sdcfr_n = evaluate_vs_sdcfr(
                    model,
                    sdcfr_model,
                    num_hands_per_size=_TRUST_HANDS_PER_SIZE,
                    min_players=args.min_players,
                    max_players=args.max_players,
                    seed=_EVAL_SEED,
                )
                print(f"      vs SD-CFR:    {sdcfr_lower:+.1f} to {sdcfr_upper:+.1f} BB/100 (n={sdcfr_n})")
            else:
                # Standard evaluation with Random+Flop baseline
                print(f"      [STANDARD EVAL]")
                mean, lower, upper, n = evaluate(
                    model,
                    num_hands_per_size=args.random_eval_hands,
                    min_players=args.min_players,
                    max_players=args.max_players,
                    seed=_EVAL_SEED,
                )
                sdcfr_mean, sdcfr_lower, sdcfr_upper, sdcfr_n = evaluate_vs_sdcfr(
                    model,
                    sdcfr_model,
                    num_hands_per_size=_TRUST_HANDS_PER_SIZE,
                    min_players=args.min_players,
                    max_players=args.max_players,
                    seed=_EVAL_SEED,
                )

                print(f"      Random+Flop:  {lower:+.1f} to {upper:+.1f} BB/100 (n={n})")
                print(f"      vs SD-CFR:    {sdcfr_lower:+.1f} to {sdcfr_upper:+.1f} BB/100 (n={sdcfr_n})")

            # Append this evaluation to the CSV history. history_path was
            # computed but never written, so runs produced no record to compare
            # later (docs/PERFORMANCE.md relies on this file).
            try:
                _row = {
                    "episode": episode,
                    "eta": model.eta,
                    "updates_per_train": model.updates_per_train,
                    "q_buffer": len(model.q_buffer),
                    "policy_buffer": len(model.policy_buffer),
                    "q_updates": model._q_updates,
                    "p_updates": model._p_updates,
                    "q_loss": last_q_loss,
                    "p_loss": last_p_loss,
                    "sdcfr_mean_bb100": sdcfr_mean,
                    "sdcfr_lower_bb100": sdcfr_lower,
                    "sdcfr_upper_bb100": sdcfr_upper,
                    "sdcfr_n": sdcfr_n,
                    "best_lower_bb100": best_lower,
                    "stalls": stalls,
                }
                if not args.sdcfr_eval_only:
                    _row.update(
                        {
                            "random_mean_bb100": mean,
                            "random_lower_bb100": lower,
                            "random_upper_bb100": upper,
                            "random_n": n,
                        }
                    )
                _write_history_row(history_path, _row)
            except OSError as exc:
                print(f"      [WARN] could not write eval history: {exc}")

            # Always track SD-CFR performance for best model selection
            if sdcfr_lower > best_lower:
                best_lower = sdcfr_lower
                stalls = 0
                best_path = models_dir / "nfsp_best.pt"
                model.save(str(best_path), training_meta=_build_training_meta(episode, best_lower, stalls, model))
                print(f"      [BEST] {best_lower:+.1f} BB/100 (vs SD-CFR) -> {best_path}")
            else:
                stalls += 1
                if episode >= args.stop_min_episode and stalls >= args.stop_patience:
                    print(f"      [STOP] No improvement for {stalls} evals (stall patience: {args.stop_patience})")
                    break

            model.q_network.train()
            model.policy_network.train()

        if episode > 0 and episode % args.checkpoint_every == 0:
            gen_path = models_dir / f"nfsp_gen_{episode // args.checkpoint_every}.pt"
            model.save(str(gen_path), training_meta=_build_training_meta(episode, best_lower, stalls, model))
            print(f"[{datetime.now()}]   [GEN {episode // args.checkpoint_every}] {gen_path}")

        if episode >= _STOP_MIN_EPISODE and stalls >= _STOP_PATIENCE:
            break

    elapsed = time.time() - start

    # Write the final artifact. Without this the run only produced nfsp_best.pt
    # and the periodic generations, leaving save_path pointing at whatever old
    # file already occupied that name.
    if best_path is not None and best_path.is_file():
        shutil.copy(best_path, save_path)
        print(f"  [FINAL] Copied {best_path.name} -> {save_path.name}")
    else:
        model.save(
            str(save_path),
            training_meta=_build_training_meta(episode, best_lower, stalls, model),
        )
        print(f"  [FINAL] Saved {save_path.name}")

    # Fix for Issue #2: Calculate actual training time vs expected
    expected_time_minutes = args.episodes / 100  # Estimate: 1 episode ~0.1 min
    actual_time_minutes = elapsed / 60
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Time:               {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Episodes:           {episode + 1:,}")
    print(f"  Final eta:          {model.eta:.0%}")
    print(f"  Final updates:      {model.updates_per_train}")
    print(f"  Best vs SD-CFR:     {best_lower:+.1f} BB/100")
    print(f"  Best checkpoint:    {save_path}")
    print(f"  Q-buffer size:      {len(model.q_buffer):,}")
    print(f"  Policy-buffer size: {len(model.policy_buffer):,}")
    print(f"  Policy updates:     {model._q_updates}")
    print(f"{'='*70}\n")
    
    # Summary for Issue #1 verification
    print(f"[SUMMARY]")
    print(f"  Issue #1 - Policy Updates:")
    print(f"    - Policy buffer final size: {len(model.policy_buffer):,}")
    print(f"    - Total policy updates: {model._q_updates}")
    print(f"    - Batch size: {args.batch_size}")
    if len(model.policy_buffer) > 0:
        print(f"    ✅ Policy buffer populated and updated")
    else:
        print(f"    ❌ Policy buffer empty - see training logs for details")
    
    print(f"  Issue #2 - Stall Detection:")
    print(f"    - Max episodes allowed: {args.episodes}")
    print(f"    - Episodes trained: {episode + 1}")
    print(f"    - Reason stopped: {'Completed all episodes' if episode + 1 == args.episodes else f'Stall detection after {stalls} evals'}")
    
    print(f"  Issue #3 - Evaluation Mode:")
    print(f"    - Evaluation type: {'SD-CFR only' if args.sdcfr_eval_only else 'Standard (Random+Flop + SD-CFR)'}")


if __name__ == "__main__":
    main()
