#!/usr/bin/env python
"""Compare three trained poker models in a round-robin tournament."""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.deep_bot import DeepBot
from poker.bots.linear_bot import LinearBot
from poker.bots.tree_bot import TreeBot
from poker.bots.random_bot import RandomBot
from poker.evaluation.tournament import TournamentEvaluator
from poker.ml.models.linear_q import LinearQModel
from poker.ml.models.tree_q import TreeQModel
from poker.ml.models.deep_q import DeepQModel


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare trained poker models in a round-robin tournament"
    )
    parser.add_argument(
        "--linear", type=str, default="models/linear_q.pkl", help="Path to linear model"
    )
    parser.add_argument("--tree", type=str, default="models/tree_q.pkl", help="Path to tree model")
    parser.add_argument("--deep", type=str, default="models/deep_q.pt", help="Path to deep model")
    parser.add_argument(
        "--hands", type=int, default=100, help="Number of hands per match (default 100)"
    )
    parser.add_argument(
        "--include-random",
        action="store_true",
        help="Include random bot as baseline",
    )

    args = parser.parse_args()

    # Load models
    bots = []

    print("Loading models...")
    if Path(args.linear).exists():
        linear_model = LinearQModel()
        linear_model.load(args.linear)
        linear_bot = LinearBot(name="LinearBot", model=linear_model)
        bots.append(linear_bot)
        print(f"  Loaded LinearBot from {args.linear}")
    else:
        print(f"  WARNING: Linear model not found at {args.linear}")

    if Path(args.tree).exists():
        tree_model = TreeQModel()
        tree_model.load(args.tree)
        tree_bot = TreeBot(name="TreeBot", model=tree_model)
        bots.append(tree_bot)
        print(f"  Loaded TreeBot from {args.tree}")
    else:
        print(f"  WARNING: Tree model not found at {args.tree}")

    if Path(args.deep).exists():
        deep_model = DeepQModel()
        deep_model.load(args.deep)
        deep_bot = DeepBot(name="DeepBot", model=deep_model)
        bots.append(deep_bot)
        print(f"  Loaded DeepBot from {args.deep}")
    else:
        print(f"  WARNING: Deep model not found at {args.deep}")

    if args.include_random:
        random_bot = RandomBot(name="RandomBot")
        bots.append(random_bot)
        print(f"  Added RandomBot as baseline")

    if len(bots) < 2:
        print("ERROR: Need at least 2 bots to run tournament")
        sys.exit(1)

    # Run tournament
    print(f"\nRunning round-robin tournament ({args.hands} hands per match)...")
    evaluator = TournamentEvaluator(num_hands_per_match=args.hands)
    tournament_stats = evaluator.run_tournament(bots)

    # Print results
    print("\n" + "=" * 70)
    print(tournament_stats)
    print("=" * 70)


if __name__ == "__main__":
    main()
