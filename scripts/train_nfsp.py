#!/usr/bin/env python
"""Train an NFSP agent via self-play in PokerEnv."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.population import PopulationManager


def evaluate(model: NFSPModel, num_players: int, num_hands: int = 200, seed: int = 42) -> float:
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
    parser = argparse.ArgumentParser(description="Train NFSP poker agent via self-play")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=500_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--roster-refresh-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--generation", type=int, default=0, help="Generation ID for population-based training")
    parser.add_argument("--save-path", type=str, default="models/nfsp.pt")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

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

    for episode in range(args.episodes):
        # Refresh opponent roster every N episodes (dynamic variation)
        if episode % args.roster_refresh_every == 0 and episode > 0:
            num_opponents = args.num_players - 1
            opponents = pop_manager.refresh_opponent_roster_for_episode(
                args.generation, episode, num_opponents
            )
            env.opponent_bots = opponents

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

        if episode % args.eval_every == 0:
            eval_reward = evaluate(model, args.num_players)
            q_str = f"{last_q_loss:.4f}" if last_q_loss is not None else "n/a"
            p_str = f"{last_policy_loss:.4f}" if last_policy_loss is not None else "n/a"
            elapsed = time.time() - start_time
            gen_str = f"Gen{args.generation}" if args.generation > 0 else "Gen0"
            print(
                f"[{gen_str}] Episode {episode:7d} | q_loss={q_str} | policy_loss={p_str} "
                f"| eval_reward={eval_reward:7.4f} | elapsed={elapsed:.1f}s"
            )

        # Checkpoint every N episodes
        if episode % args.checkpoint_every == 0 and episode > 0:
            ckpt_path = Path(args.save_path).parent / f"nfsp_gen{args.generation}_ckpt_{episode}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt_path))

    # Save as generation checkpoint
    gen_path = pop_manager.save_generation(model, args.generation)

    # Also save to the main save path
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    elapsed = time.time() - start_time
    print(f"Training complete. Total time: {elapsed/3600:.1f}h")
    print(f"Saved generation {args.generation} to {gen_path}")
    print(f"Saved final model to {save_path}")


if __name__ == "__main__":
    main()
