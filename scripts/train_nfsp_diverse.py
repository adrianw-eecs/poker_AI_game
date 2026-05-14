#!/usr/bin/env python
"""Extended NFSP training with diverse opponents to prevent local optima.

Trains NFSP against a mix of:
- RandomBot (unpredictable)
- FlopBot (hand strength aware)
- CallBot (weak calling station)

This diversity prevents the model from converging to degenerate strategies like
"always fold" by forcing it to learn value betting, hand strength evaluation,
and aggression.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.population import PopulationManager


def evaluate(model: NFSPModel, num_players: int, num_hands: int = 30, seed: int = 42) -> float:
    """Evaluate against all three opponent types, return average reward."""
    opponents_list = [
        [RandomBot(seed=seed + 1)],
        [FlopBot(seed=seed + 2)],
        [CallBot(seed=seed + 3)],
    ]

    total_reward = 0.0
    total_hands = 0

    for opponents in opponents_list:
        env = PokerEnv(
            num_players=num_players,
            learning_seat=0,
            opponent_bots=opponents,
            seed=seed,
        )

        for _ in range(num_hands // 3):  # Distribute hands across opponent types
            obs, _ = env.reset()
            done = False
            while not done:
                mask = env.get_action_mask()
                action = model.select_action(obs, mask, training=False)
                obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            total_hands += 1

    return total_reward / max(total_hands, 1)


def get_opponent_for_episode(episode: int, num_players: int, seed: int) -> list:
    """Rotate through opponent types for diversity.

    Episode 0-33333:   RandomBot
    Episode 33334-66666: FlopBot
    Episode 66667-99999: CallBot
    (repeat)
    """
    cycle_length = 33334
    cycle_type = (episode // cycle_length) % 3

    if cycle_type == 0:
        return [RandomBot(seed=seed + episode)]
    elif cycle_type == 1:
        return [FlopBot(seed=seed + episode)]
    else:
        return [CallBot(seed=seed + episode)]


def main() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] GPU Detected: {torch.cuda.get_device_name()}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("[WARNING] No GPU detected - training will be slow")

    parser = argparse.ArgumentParser(
        description="NFSP training with diverse opponents (RandomBot + FlopBot + CallBot)"
    )
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=250_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--save-path", type=str, default="models/nfsp_diverse.pt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"NFSP Training with Diverse Opponents: {args.episodes:,} episodes")
    print(f"{'='*70}")
    print(f"Opponents: RandomBot + FlopBot + CallBot (rotating)")
    print(f"Eval every: {args.eval_every:,} episodes (vs all opponent types)")
    print(f"Checkpoints every: {args.checkpoint_every:,} episodes")
    print(f"Expected time: ~20 minutes on RTX 3080\n")

    model = NFSPModel()
    pop_manager = PopulationManager(models_dir=str(Path(args.save_path).parent))

    last_q_loss: float | None = None
    last_policy_loss: float | None = None
    start_time = time.time()
    episode_times = []

    for episode in range(args.episodes):
        ep_start = time.time()

        # Rotate opponent type for diversity
        opponents = get_opponent_for_episode(episode, args.num_players, args.seed or 42)

        env = PokerEnv(
            num_players=args.num_players,
            learning_seat=0,
            opponent_bots=opponents,
            seed=args.seed,
        )

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
            ckpt_path = str(Path(args.save_path).parent / f"nfsp_diverse_ckpt_{episode:06d}.pt")
            model.save(ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    # Final save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Avg time per episode: {np.mean(episode_times):.4f}s")
    print(f"Saved final model to {save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
