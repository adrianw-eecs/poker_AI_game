#!/usr/bin/env python
"""Extended NFSP training: 500K episodes with checkpointing."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.population import PopulationManager


def evaluate(model: NFSPModel, num_players: int, num_hands: int = 30, seed: int = 42) -> float:
    """Play num_hands against RandomBot, return mean reward."""
    env = PokerEnv(
        num_players=num_players,
        learning_seat=0,
        opponent_bots=[RandomBot(seed=seed + i) for i in range(num_players - 1)],
        seed=seed,
    )
    total_reward = 0.0
    for _ in range(num_hands):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=False)
            obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    return total_reward / num_hands


def main() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] GPU Detected: {torch.cuda.get_device_name()}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("[WARNING] No GPU detected - training will be slow")

    parser = argparse.ArgumentParser(description="Extended NFSP training (500K episodes)")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/nfsp_extended.pt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None for no seeding)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"Extended NFSP Training: {args.episodes:,} episodes")
    print(f"{'='*70}")
    print(f"Players: {args.num_players}, Eval every: {args.eval_every:,} episodes")
    print(f"Checkpoints every: {args.checkpoint_every:,} episodes")
    print(f"Expected time: ~20 minutes on RTX 3080\n")

    model = NFSPModel()
    pop_manager = PopulationManager(models_dir=str(Path(args.save_path).parent))

    # Create environment with diverse opponents
    num_opponents = args.num_players - 1
    opponents = pop_manager.get_opponent_roster(args.generation, num_opponents)
    env = PokerEnv(
        num_players=args.num_players,
        learning_seat=0,
        opponent_bots=opponents,
        seed=args.seed,
    )

    last_q_loss: float | None = None
    last_policy_loss: float | None = None
    start_time = time.time()
    episode_times = []

    for episode in range(args.episodes):
        ep_start = time.time()

        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=True)
            next_obs, reward, done, truncated, _ = env.step(action)
            model.store_transition(obs, action, reward, next_obs, done)
            result = model.train_step()
            if result["q_loss"] is not None:
                last_q_loss = result["q_loss"]
                last_policy_loss = result["policy_loss"]
            obs = next_obs

        episode_times.append(time.time() - ep_start)

        # Periodic evaluation
        if episode % args.eval_every == 0:
            eval_reward = evaluate(model, args.num_players, num_hands=30)
            q_str = f"{last_q_loss:.4f}" if last_q_loss is not None else "n/a"
            p_str = f"{last_policy_loss:.4f}" if last_policy_loss is not None else "n/a"
            elapsed = time.time() - start_time
            avg_ep_time = np.mean(episode_times[-100:]) if len(episode_times) >= 100 else np.mean(episode_times)
            eta = (args.episodes - episode) * avg_ep_time / 60
            print(
                f"Ep {episode:7d}/{args.episodes:7d} | q_loss={q_str} | policy_loss={p_str} "
                f"| eval={eval_reward:7.4f} | elapsed={elapsed:7.1f}s | ETA={eta:7.1f}m"
            )

        # Periodic checkpointing
        if episode > 0 and episode % args.checkpoint_every == 0:
            ckpt_path = str(Path(args.save_path).parent / f"nfsp_ckpt_{episode:06d}.pt")
            model.save(ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    # Final save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Extended training complete!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Avg time per episode: {np.mean(episode_times):.4f}s")
    print(f"Saved final model to {save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
