#!/usr/bin/env python
"""Generate training data using trained models in self-play."""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.deep_bot import DeepBot
from poker.bots.linear_bot import LinearBot
from poker.bots.tree_bot import TreeBot
from poker.bots.random_bot import RandomBot
from poker.ml.env import PokerEnv
from poker.ml.models.deep_q import DeepQModel
from poker.ml.models.linear_q import LinearQModel
from poker.ml.models.tree_q import TreeQModel
from poker.training.dataset import Experience, ReplayBuffer
from poker.engine.action_validator import legal_actions
from poker.ml.action_space import action_to_action_index, build_action_mask
from poker.ml.observation import build_observation


def generate_selfplay_data(
    learning_bot_name: str,
    opponent_bot_names: list[str],
    num_hands: int = 100,
    output_file: str = "data/selfplay_v2.npz",
) -> None:
    """Generate training data using trained models in self-play.

    Args:
        learning_bot_name: Name of bot to learn (will be seat 0).
        opponent_bot_names: Names of opponent bots.
        num_hands: Number of hands to play.
        output_file: Output file path.
    """
    # Load models
    print(f"Loading models...")
    bots = {}

    if Path("models/linear_q.pkl").exists():
        linear_model = LinearQModel()
        linear_model.load("models/linear_q.pkl")
        bots["linear"] = LinearBot(name="LinearBot", model=linear_model)
        print("  Loaded LinearBot")

    if Path("models/tree_q.pkl").exists():
        tree_model = TreeQModel()
        tree_model.load("models/tree_q.pkl")
        bots["tree"] = TreeBot(name="TreeBot", model=tree_model)
        print("  Loaded TreeBot")

    if Path("models/deep_q.pt").exists():
        deep_model = DeepQModel()
        deep_model.load("models/deep_q.pt")
        bots["deep"] = DeepBot(name="DeepBot", model=deep_model)
        print("  Loaded DeepBot")

    bots["random"] = RandomBot(name="RandomBot")

    # Validate requested bots exist
    learning_bot = bots.get(learning_bot_name)
    if learning_bot is None:
        print(f"ERROR: Learning bot '{learning_bot_name}' not found")
        sys.exit(1)

    for opp_name in opponent_bot_names:
        if opp_name not in bots:
            print(f"ERROR: Opponent bot '{opp_name}' not found")
            sys.exit(1)

    print(f"\nGenerating {num_hands} hands of self-play data...")
    print(f"  Learning bot: {learning_bot_name}")
    print(f"  Opponent bots: {opponent_bot_names}")

    # Initialize replay buffer
    buffer = ReplayBuffer()

    # Play hands
    for hand_idx in range(num_hands):
        # Select opponent (round-robin through opponent list)
        opp_name = opponent_bot_names[hand_idx % len(opponent_bot_names)]
        opponent_bot = bots[opp_name]

        # Create environment with learning bot at seat 0
        env = PokerEnv(
            num_players=2,
            starting_stack=1000,
            learning_seat=0,
            opponent_bots=[opponent_bot],
        )

        obs, info = env.reset()
        done = False

        hand_experiences = []

        while not done:
            # Get legal actions for learning agent
            legal_acts = legal_actions(env.state, env.learning_seat)

            # Get learning bot's action
            action = learning_bot.act(env.state.view_for(env.learning_seat), legal_acts)

            # Convert action to index
            action_idx = action_to_action_index(action, env.state, env.learning_seat)

            # Store observation and action for this step
            current_obs = build_observation(env.state, env.learning_seat)

            # Build legal action mask
            mask = build_action_mask(env.state, env.learning_seat)

            # Take step
            obs, reward, done, _, info = env.step(action_idx)

            # If hand is done, record experience with final reward
            if done:
                experience = Experience(
                    observation=current_obs,
                    action=action_idx,
                    reward=reward,
                    legal_mask=mask,
                    seat=env.learning_seat,
                    hand_id=hand_idx,
                )
                buffer.add(experience)

        if (hand_idx + 1) % 20 == 0:
            print(f"  Completed {hand_idx + 1}/{num_hands} hands...")

    print(f"\nGenerated {buffer.size} experiences")
    print(f"Stats: {buffer.stats()}")

    # Save buffer
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    buffer.save(output_file)
    print(f"Saved to {output_file}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate training data using trained models in self-play"
    )
    parser.add_argument(
        "--learning",
        type=str,
        default="linear",
        choices=["linear", "tree", "deep", "random"],
        help="Learning bot to generate data for",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="random",
        help="Comma-separated list of opponent bot types (default: random)",
    )
    parser.add_argument("--hands", type=int, default=100, help="Number of hands to play")
    parser.add_argument("--out", type=str, default="data/selfplay_v2.npz", help="Output file")

    args = parser.parse_args()

    opponent_names = [o.strip() for o in args.opponents.split(",")]

    generate_selfplay_data(
        learning_bot_name=args.learning,
        opponent_bot_names=opponent_names,
        num_hands=args.hands,
        output_file=args.out,
    )


if __name__ == "__main__":
    main()
