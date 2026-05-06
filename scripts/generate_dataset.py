#!/usr/bin/env python
"""Generate self-play dataset for training poker agents."""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.action_space import build_action_mask
from poker.ml.env import PokerEnv
from poker.training.dataset import Experience, ReplayBuffer


def collect_self_play_data(
    num_hands: int,
    num_players: int = 2,
    starting_stack: int = 1000,
    small_blind: int = 5,
    big_blind: int = 10,
    opponent_type: str = "random",
    seed: int | None = None,
) -> ReplayBuffer:
    """Collect self-play data using random bots as opponents.

    Args:
        num_hands: Number of hands to play.
        num_players: Number of players in the game.
        starting_stack: Starting chip stack per player.
        small_blind: Small blind amount.
        big_blind: Big blind amount.
        opponent_type: "random", "flop", or "mixed".
        seed: Random seed for reproducibility.

    Returns:
        ReplayBuffer with collected experiences.
    """
    # Set up bots
    if opponent_type == "random":
        opponent_bots = [RandomBot(seed=seed) for _ in range(num_players - 1)]
    elif opponent_type == "flop":
        opponent_bots = [FlopBot(seed=seed) for _ in range(num_players - 1)]
    elif opponent_type == "mixed":
        opponent_bots = []
        for i in range(num_players - 1):
            if i % 2 == 0:
                opponent_bots.append(RandomBot(seed=seed))
            else:
                opponent_bots.append(FlopBot(seed=seed))
    else:
        raise ValueError(f"Unknown opponent_type: {opponent_type}")

    # Create environment (learning agent always in seat 0)
    env = PokerEnv(
        num_players=num_players,
        starting_stack=starting_stack,
        small_blind=small_blind,
        big_blind=big_blind,
        learning_seat=0,
        opponent_bots=opponent_bots,
        seed=seed,
    )

    # Collect data
    buffer = ReplayBuffer(max_size=num_hands * 20)  # ~20 decisions per hand
    hand_id = 0

    for hand_idx in range(num_hands):
        obs, info = env.reset()

        done = False
        while not done:
            # Get legal actions
            mask = env.get_action_mask()

            # Record this state-action pair
            # For now, pick a random legal action
            legal_actions = np.where(mask)[0]
            action = np.random.choice(legal_actions)

            # Step the environment
            next_obs, reward, done, _, info = env.step(action)

            # Add to buffer
            experience = Experience(
                observation=obs,
                action=action,
                reward=reward,
                legal_mask=mask,
                seat=0,
                hand_id=hand_id,
            )
            buffer.add(experience)

            # Move to next observation
            obs = next_obs

        hand_id += 1

        # Print progress
        if (hand_idx + 1) % 100 == 0:
            print(f"Completed {hand_idx + 1}/{num_hands} hands, "
                  f"Buffer size: {buffer.size}, Stats: {buffer.stats()}")

    print(f"\nCompleted all {num_hands} hands")
    print(f"Total experiences collected: {buffer.size}")
    print(f"Buffer stats: {buffer.stats()}")

    return buffer


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate self-play dataset for poker agents")
    parser.add_argument("--hands", type=int, default=10000, help="Number of hands to play")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--stack", type=int, default=1000, help="Starting stack")
    parser.add_argument("--sb", type=int, default=5, help="Small blind")
    parser.add_argument("--bb", type=int, default=10, help="Big blind")
    parser.add_argument("--opponents", choices=["random", "flop", "mixed"], default="random",
                        help="Opponent type")
    parser.add_argument("--out", type=str, default="data/selfplay.npz", help="Output file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Create output directory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Collect data
    print(f"Collecting {args.hands} hands of self-play data...")
    print(f"  Players: {args.players}")
    print(f"  Stack: {args.stack}")
    print(f"  Blinds: {args.sb}/{args.bb}")
    print(f"  Opponents: {args.opponents}")
    print(f"  Seed: {args.seed}")
    print()

    buffer = collect_self_play_data(
        num_hands=args.hands,
        num_players=args.players,
        starting_stack=args.stack,
        small_blind=args.sb,
        big_blind=args.bb,
        opponent_type=args.opponents,
        seed=args.seed,
    )

    # Save
    print(f"\nSaving to {args.out}...")
    buffer.save(args.out)
    print("Done!")


if __name__ == "__main__":
    main()
