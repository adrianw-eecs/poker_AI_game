#!/usr/bin/env python
"""NFSP Self-Play Training: 4-PLAYER with Frozen Model Copies.

This script trains an NFSP model by having it play against 3 frozen copies
of previous generations of itself. This is the core principle of NFSP:
generational improvement through population-based self-play.

Configuration:
- 4 players: learning_seat=0 (being trained) + 3 frozen opponents
- Opponent composition (from PopulationManager):
  * 40% RandomBot (baseline)
  * 30% FlopBot (strategy variation)
  * 20% Previous generation self-play (learning from past versions)
  * 10% Historical generations (diversity from older versions)
- Starting stack: 1000, Blinds: 25/50
- Checkpoints saved every N episodes for generational training

USAGE:
  # Continue training from existing model
  python scripts/train_nfsp_selfplay.py --checkpoint models/nfsp_4player.pt

  # Start fresh
  python scripts/train_nfsp_selfplay.py

  # Custom episodes
  python scripts/train_nfsp_selfplay.py --episodes 500000 --eval-every 5000
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.population import PopulationManager
from poker.engine.action_validator import legal_actions, validate
from poker.exceptions import IllegalActionError
from poker.ml.action_space import action_index_to_action


class TrainingLogger:
    """Logs training progress and events."""

    def __init__(self, log_file: str = "training_selfplay.txt"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_start = datetime.now()
        self._write_header()

    def _write_header(self):
        """Write session header."""
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*70}\n")
            f.write(f"NFSP SELF-PLAY TRAINING SESSION\n")
            f.write(f"Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n")

    def log_checkpoint(self, episode: int, gen_id: int, path: str):
        """Log generation checkpoint save."""
        message = f"[Ep {episode}] Generation {gen_id} saved: {path}"
        with open(self.log_file, "a") as f:
            f.write(message + "\n")

    def log_eval(self, episode: int, eval_reward: float, q_loss: float, p_loss: float):
        """Log evaluation results."""
        message = f"[Ep {episode}] Eval reward: {eval_reward:.4f} | Q-loss: {q_loss:.4f} | P-loss: {p_loss:.4f}"
        with open(self.log_file, "a") as f:
            f.write(message + "\n")


def evaluate(model: NFSPModel, pop_manager: PopulationManager, num_hands: int = 30, current_gen: int = 0) -> float:
    """Evaluate model against current population."""
    try:
        opponents = pop_manager.get_opponent_roster(
            current_gen=max(0, current_gen - 1),  # Use previous generation
            num_opponents=3
        )
    except Exception as e:
        print(f"    [WARN] Error getting opponent roster: {e}, using fallback")
        from poker.bots.call_bot import CallBot
        from poker.bots.flop_bot import FlopBot
        from poker.bots.random_bot import RandomBot
        opponents = [RandomBot(), FlopBot(), CallBot()]

    total_reward = 0.0
    total_hands = 0

    for _ in range(num_hands):
        env = PokerEnv(
            num_players=4,
            learning_seat=0,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
        )

        obs, _ = env.reset()
        done = False

        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=False)
            obs, reward, done, _, _ = env.step(action)

        total_reward += reward
        total_hands += 1

    return total_reward / max(total_hands, 1)


def main() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] GPU Detected: {torch.cuda.get_device_name()}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("[WARNING] No GPU detected - training will be slow")

    parser = argparse.ArgumentParser(
        description="NFSP Self-Play Training: Train vs 3 frozen model copies"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to continue training from",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100_000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5_000,
        help="Evaluate every N episodes",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10_000,
        help="Save generation checkpoint every N episodes",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/nfsp_selfplay.pt",
        help="Path to save final model",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="training_selfplay.txt",
        help="Training log file",
    )
    args = parser.parse_args()

    # Initialize components
    logger = TrainingLogger(args.log_file)
    pop_manager = PopulationManager(models_dir=str(Path(args.save_path).parent))

    # Load or create model
    model = NFSPModel()
    if args.checkpoint:
        print(f"[LOAD] Loading checkpoint: {args.checkpoint}")
        model.load(args.checkpoint)
    else:
        print("[INIT] Training new model from scratch")

    print(f"\n{'='*70}")
    print(f"NFSP SELF-PLAY TRAINING")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Game mode: 4-PLAYER")
    print(f"    Learning seat: 0")
    print(f"    Opponent seats: 1, 2, 3 (frozen model copies + classic bots)")
    print(f"  Episodes: {args.episodes:,}")
    print(f"  Blinds: 25/50")
    print(f"  Starting stack: 1000 per player")
    print(f"  Opponent composition:")
    print(f"    - 40% RandomBot")
    print(f"    - 30% FlopBot")
    print(f"    - 20% Previous generation self-play")
    print(f"    - 10% Historical generations")
    print(f"  Save path: {args.save_path}")
    print(f"  Log file: {args.log_file}")
    print()

    last_q_loss: float | None = None
    last_policy_loss: float | None = None
    start_time = time.time()
    episode_times = []
    generation_count = 0

    for episode in range(args.episodes):
        ep_start = time.time()

        # Get fresh opponent roster for this episode
        # Use generation_count-1 since we haven't incremented yet
        # (generations are saved retroactively)
        try:
            gen_id = max(0, generation_count - 1)
            opponents = pop_manager.get_opponent_roster(
                current_gen=gen_id,
                num_opponents=3
            )
        except Exception as e:
            # Fallback if generation doesn't exist yet
            print(f"    [ERROR] Failed to get opponent roster: {e}, using fallback", flush=True)
            from poker.bots.random_bot import RandomBot
            from poker.bots.flop_bot import FlopBot
            from poker.bots.call_bot import CallBot
            opponents = [RandomBot(), FlopBot(), CallBot()]

        seed = np.random.randint(0, 2**31 - 1)

        env = PokerEnv(
            num_players=4,
            learning_seat=0,
            opponent_bots=opponents,
            small_blind=25,
            big_blind=50,
            seed=seed,
        )

        obs, _ = env.reset()
        done = False
        hand_step = 0
        hand_invalid_count = 0

        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=True)

            # Validate action
            try:
                if env.state is not None:
                    legal = legal_actions(env.state, env.learning_seat)
                    concrete_action = action_index_to_action(action, env.state, env.learning_seat)
                    validate(env.state, env.learning_seat, concrete_action)
            except (IllegalActionError, IndexError, ValueError):
                hand_invalid_count += 1

            next_obs, reward, done, truncated, _ = env.step(action)

            model.store_transition(obs, action, reward, next_obs, done)
            result = model.train_step()

            if result["q_loss"] is not None:
                last_q_loss = result["q_loss"]
                last_policy_loss = result["policy_loss"]

            obs = next_obs
            hand_step += 1

            if hand_step > 100:
                break

        episode_times.append(time.time() - ep_start)

        # Periodic evaluation
        if episode % args.eval_every == 0:
            print(f"  [EVAL] Starting evaluation at episode {episode}...", flush=True)
            eval_reward = evaluate(model, pop_manager, num_hands=30, current_gen=generation_count)
            print(f"  [EVAL] Evaluation complete, reward={eval_reward:.4f}", flush=True)
            q_str = f"{last_q_loss:.4f}" if last_q_loss is not None else "n/a"
            p_str = f"{last_policy_loss:.4f}" if last_policy_loss is not None else "n/a"
            elapsed = time.time() - start_time
            avg_ep_time = (
                np.mean(episode_times[-100:])
                if len(episode_times) >= 100
                else np.mean(episode_times)
            )
            eta = (args.episodes - episode) * avg_ep_time / 60

            status = (
                f"Ep {episode:7d}/{args.episodes:7d} | "
                f"q_loss={q_str} | policy_loss={p_str} | "
                f"eval_reward={eval_reward:7.4f} | "
                f"elapsed={elapsed:7.1f}s | ETA={eta:7.1f}m"
            )
            print(status)
            q_loss_val = float(q_str) if q_str != "n/a" else 0.0
            p_loss_val = float(p_str) if p_str != "n/a" else 0.0
            logger.log_eval(episode, eval_reward, q_loss_val, p_loss_val)

        # Periodic checkpointing - save generation
        if episode > 0 and episode % args.checkpoint_every == 0:
            gen_ckpt = str(Path(args.save_path).parent / f"nfsp_gen_{generation_count}.pt")
            print(f"  [SAVING] Generation {generation_count}...", end='', flush=True)
            model.save(gen_ckpt)
            print(f" Done!")
            print(f"  [GEN {generation_count}] Checkpoint saved: {gen_ckpt}")
            logger.log_checkpoint(episode, generation_count, gen_ckpt)
            generation_count += 1
            print(f"  [INFO] Generation count incremented to {generation_count}")

    # Final save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SELF-PLAY TRAINING COMPLETE!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Avg time per episode: {np.mean(episode_times):.4f}s")
    print(f"Final model: {save_path}")
    print(f"Generations trained: {generation_count}")
    print(f"Log file: {args.log_file}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
