#!/usr/bin/env python
"""SD-CFR Self-Play Training with adaptive self-play scheduling.

Each traversal samples a player count in [--min-players, --max-players].
Opponent composition starts with only classic bots (RandomBot + FlopBot)
and gradually shifts toward self-play opponents once the model consistently
wins against baseline bots — matching the NFSP self-play ramp-up logic.

Training loop:
  1. Run --traversals-per-iteration full hands as the traverser.
  2. Outcome-sampling stores (obs || advantages) in the regret buffer.
  3. model.train_iteration() reinitialises and re-trains the advantage net.
  4. Repeat.

USAGE:
  python scripts/train_sdcfr_selfplay.py --iterations 2000
  python scripts/train_sdcfr_selfplay.py --checkpoint models/sdcfr_selfplay.pt
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.bots.sdcfr_bot import SDCFRBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.action_space import build_action_mask
from poker.ml.cfr.traversal import ExternalSamplingTraverser
from poker.ml.env import PokerEnv
from poker.ml.models.sdcfr_model import SDCFRModel


# ---------------------------------------------------------------------------
# Self-play scheduler (identical logic to NFSP script)
# ---------------------------------------------------------------------------

class SelfPlayScheduler:
    """Gradually increases self-play fraction as the model becomes profitable."""

    WIN_THRESHOLD = 0.02
    RAMP_STEP = 0.05
    DECAY_STEP = 0.10
    MAX_FRACTION = 0.70
    WINDOW = 3
    RAMP_AFTER = 3

    def __init__(self) -> None:
        self.fraction: float = 0.0
        self._history: list[float] = []
        self._consecutive_wins: int = 0

    def update(self, eval_reward: float) -> None:
        self._history.append(eval_reward)
        recent_avg = float(np.mean(self._history[-self.WINDOW:]))

        if recent_avg > self.WIN_THRESHOLD:
            self._consecutive_wins += 1
            if self._consecutive_wins >= self.RAMP_AFTER:
                self.fraction = min(self.MAX_FRACTION, self.fraction + self.RAMP_STEP)
        else:
            self._consecutive_wins = 0
            if recent_avg <= 0.0:
                self.fraction = max(0.0, self.fraction - self.DECAY_STEP)

    @property
    def status(self) -> str:
        recent = self._history[-self.WINDOW:] if self._history else [0.0]
        avg = float(np.mean(recent))
        return (f"self_play={self.fraction:.0%} | "
                f"consec_wins={self._consecutive_wins} | "
                f"recent_avg={avg:+.4f}")


# ---------------------------------------------------------------------------
# Opponent builder
# ---------------------------------------------------------------------------

def build_opponents(
    num_opponents: int,
    self_play_fraction: float,
    model: SDCFRModel,
) -> list:
    """Build a mixed opponent list for one traversal.

    Baseline bots (RandomBot + FlopBot) always fill at least one slot.
    Self-play opponents are live SDCFRBot instances sharing the current
    model weights (read-only inference, no gradient flow).
    """
    if num_opponents == 0:
        return []

    n_selfplay = int(round(num_opponents * self_play_fraction))
    n_baseline = num_opponents - n_selfplay

    bots: list = []
    for i in range(n_baseline):
        bots.append(RandomBot(name=f"Random_{i}") if i % 2 == 0
                    else FlopBot(name=f"Flop_{i}"))

    for i in range(n_selfplay):
        bots.append(SDCFRBot(name=f"Self_{i}", model=model))

    return bots


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_EVAL_PLAYER_COUNTS = [2, 4, 6, 9]


def evaluate(
    model: SDCFRModel,
    num_hands_per_size: int = 20,
    min_players: int = 2,
    max_players: int = 9,
) -> float:
    """Evaluate across multiple table sizes against fixed baseline bots only."""
    target_counts = sorted(set(
        n for n in _EVAL_PLAYER_COUNTS if min_players <= n <= max_players
    ))
    if not target_counts:
        target_counts = [min_players]

    total_reward = 0.0
    total_hands = 0

    for n in target_counts:
        configs = [
            [RandomBot(seed=i) for i in range(n - 1)],
            [FlopBot() if i % 2 == 0 else RandomBot(seed=i + 10)
             for i in range(n - 1)],
        ]
        size_reward = 0.0
        for opp_list in configs:
            for h in range(num_hands_per_size):
                env = PokerEnv(
                    num_players=n,
                    learning_seat=h % n,
                    opponent_bots=opp_list,
                    small_blind=25,
                    big_blind=50,
                    seed=h * 100 + n,
                )
                obs, _ = env.reset()
                if env.state is None or env.state.action_on_seat != env.learning_seat:
                    continue  # hand ended before learning agent got to act
                done = False
                while not done:
                    mask = build_action_mask(env.state, env.learning_seat).astype(float)
                    strategy = model.get_strategy(obs, mask)
                    action = int(np.random.choice(7, p=strategy))
                    obs, reward, done, _, _ = env.step(action)
                size_reward += reward
                total_hands += 1
        per_hand = size_reward / (num_hands_per_size * len(configs))
        total_reward += size_reward
        print(f"    {n:2d}-player: {per_hand:+.4f} reward/hand", flush=True)

    return total_reward / max(total_hands, 1)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    def __init__(self, log_file: str) -> None:
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"SD-CFR SELF-PLAY TRAINING SESSION\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n")

    def log(self, msg: str) -> None:
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name()} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB)")
    else:
        print("[WARNING] No GPU — training will be slow")

    parser = argparse.ArgumentParser(description="SD-CFR Self-Play Training")
    parser.add_argument("--model-name",               type=str,   default="sdcfr_selfplay",
                        help="Model folder name (e.g. 'sdcfr_selfplay' creates models/sdcfr_selfplay/).")
    parser.add_argument("--min-players",              type=int,   default=2)
    parser.add_argument("--max-players",              type=int,   default=9)
    parser.add_argument("--iterations",               type=int,   default=2_000)
    parser.add_argument("--traversals-per-iteration", type=int,   default=200)
    parser.add_argument("--eval-every",               type=int,   default=200)
    parser.add_argument("--checkpoint-every",         type=int,   default=500)
    parser.add_argument("--log-file",                 type=str,   default="training_sdcfr_selfplay.txt")
    parser.add_argument("--checkpoint",               type=str,   default=None)
    parser.add_argument("--batch-size",               type=int,   default=2_048)
    parser.add_argument("--capacity",                 type=int,   default=500_000)
    parser.add_argument("--sgd-steps",                type=int,   default=1_000)
    parser.add_argument("--lr",                       type=float, default=0.0005)
    args = parser.parse_args()

    if not (2 <= args.min_players <= args.max_players <= 9):
        parser.error("Require 2 <= min-players <= max-players <= 9")

    # Create model folder (like NFSP: models/{model_name}/)
    models_dir = Path("models") / args.model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "sdcfr.pt"
    log_file = models_dir / args.log_file

    logger = TrainingLogger(str(log_file))
    sp_scheduler = SelfPlayScheduler()

    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=25, big=50, ante=0)],
        hands_per_level=1000,
        fixed=True,
    )
    config = GameConfig(
        num_players=args.max_players,
        starting_stack=1000,
        small_blind=25,
        big_blind=50,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )

    model = SDCFRModel(
        lr=args.lr,
        batch_size=args.batch_size,
        sgd_steps_per_iteration=args.sgd_steps,
        capacity=args.capacity,
    )

    start_iteration = 0
    if args.checkpoint:
        print(f"[LOAD] {args.checkpoint}")
        model.load(args.checkpoint)
        start_iteration = model.cfr_iteration
        print(f"       Resuming from iteration {start_iteration}")

    traverser = ExternalSamplingTraverser(
        config=config,
        advantage_network=model.advantage_network,
        device=model.device,
        regret_buffer=model.regret_buffer,
    )

    print(f"\n{'='*70}")
    print(f"SD-CFR SELF-PLAY TRAINING")
    print(f"{'='*70}")
    print(f"  Model folder:       {models_dir}")
    print(f"  Players/traversal:  {args.min_players}-{args.max_players}")
    print(f"  Iterations:         {args.iterations:,}")
    print(f"  Traversals/iter:    {args.traversals_per_iteration:,}")
    print(f"  Self-play ramp:     starts at 0%, grows to 70% as model wins")
    print(f"  Eval player sizes:  {_EVAL_PLAYER_COUNTS}")
    print()

    start_time = time.time()
    iter_times: list[float] = []
    generation_count = 0

    for iteration in range(start_iteration, args.iterations):
        iter_start = time.time()
        traverser.advantage_network = model.advantage_network
        episode_rewards: list[float] = []

        for _ in range(args.traversals_per_iteration):
            num_players = np.random.randint(args.min_players, args.max_players + 1)
            num_opponents = num_players - 1
            traverser_seat = iteration % num_players

            opponents = build_opponents(
                num_opponents=num_opponents,
                self_play_fraction=sp_scheduler.fraction,
                model=model,
            )
            r = traverser.traverse(
                traverser_seat=traverser_seat,
                cfr_iteration=iteration,
                opponent_bots=opponents,
                num_players=num_players,
            )
            episode_rewards.append(r)

        result = model.train_iteration()
        traverser.advantage_network = model.advantage_network
        iter_times.append(time.time() - iter_start)

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if iteration % args.eval_every == 0:
            elapsed = time.time() - start_time
            avg_iter = (np.mean(iter_times[-20:])
                        if len(iter_times) >= 20 else np.mean(iter_times))
            eta_min = (args.iterations - iteration) * avg_iter / 60.0
            loss_str = f"{result['loss']:.4f}" if result["loss"] >= 0 else "warming up"

            print(f"\nIter {iteration:6d}/{args.iterations} | "
                  f"loss={loss_str} | "
                  f"buf={result['buffer_size']:6d} | "
                  f"avg_r={np.mean(episode_rewards):+.4f} | "
                  f"{sp_scheduler.status} | "
                  f"elapsed={elapsed:.0f}s | ETA={eta_min:.1f}m",
                  flush=True)

            print(f"  Evaluation ({len(_EVAL_PLAYER_COUNTS)} sizes x 2 configs x 20 hands):")
            eval_reward = evaluate(
                model, num_hands_per_size=20,
                min_players=args.min_players, max_players=args.max_players,
            )
            sp_scheduler.update(eval_reward)
            print(f"  Mean eval: {eval_reward:+.4f}  |  Updated {sp_scheduler.status}",
                  flush=True)
            logger.log(
                f"[Iter {iteration}] loss={loss_str} buf={result['buffer_size']} "
                f"eval={eval_reward:.4f} sp={sp_scheduler.fraction:.0%}"
            )

        # ------------------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------------------
        if iteration > 0 and iteration % args.checkpoint_every == 0:
            gen_path = str(save_path.parent / f"sdcfr_gen_{generation_count}.pt")
            model.save(gen_path)
            print(f"  [GEN {generation_count}] Checkpoint saved: {gen_path}")
            logger.log(f"[Iter {iteration}] Gen {generation_count} saved: {gen_path}")
            generation_count += 1

    # Final save
    model.save(str(save_path))
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SD-CFR SELF-PLAY TRAINING COMPLETE  |  {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Final self-play fraction: {sp_scheduler.fraction:.0%}")
    print(f"Generations saved: {generation_count}")
    print(f"Model folder: {models_dir}")
    print(f"Final model: {save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
