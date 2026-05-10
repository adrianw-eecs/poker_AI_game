"""Standalone SD-CFR training script."""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the project src is on the path when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.cfr.traversal import ExternalSamplingTraverser
from poker.ml.models.sdcfr_model import SDCFRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD-CFR poker agent")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--cfr-iterations", type=int, default=10_000)
    parser.add_argument("--traversals-per-iteration", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--save-path", type=str, default="models/sdcfr.pt")
    parser.add_argument(
        "--warmstart",
        type=str,
        default=None,
        help="Path to existing NFSP or SD-CFR model to initialise weights from.",
    )
    return parser.parse_args()


def build_config(num_players: int) -> tuple[GameConfig, BlindSchedule]:
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=10, big=20, ante=0)],
        hands_per_level=1000,
        fixed=True,
    )
    config = GameConfig(
        num_players=num_players,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )
    return config, blind_schedule


def main() -> None:
    args = parse_args()

    config, blind_schedule = build_config(args.num_players)
    model = SDCFRModel()

    if args.warmstart is not None:
        model.load(args.warmstart)
        print(f"Loaded warmstart weights from {args.warmstart}")

    traverser = ExternalSamplingTraverser(
        config=config,
        blind_schedule=blind_schedule,
        advantage_network=model.advantage_network,
        device=model.device,
        regret_buffer=model.regret_buffer,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    start_time = time.time()

    for iteration in range(args.cfr_iterations):
        traverser_seat = iteration % args.num_players
        # Keep traverser's network reference current after reinit in train_iteration
        traverser.advantage_network = model.advantage_network

        for _ in range(args.traversals_per_iteration):
            traverser.traverse(traverser_seat=traverser_seat, cfr_iteration=iteration)

        result = model.train_iteration()
        # Sync traverser network after potential reinit
        traverser.advantage_network = model.advantage_network

        if iteration % 200 == 0:
            elapsed = time.time() - start_time
            print(f"Iter {iteration:5d} | {result} | elapsed={elapsed:.1f}s")

        # Checkpoint every N iterations
        if iteration % args.checkpoint_every == 0 and iteration > 0:
            ckpt_path = Path(args.save_path).parent / f"sdcfr_ckpt_{iteration}.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(ckpt_path))

    model.save(args.save_path)
    elapsed = time.time() - start_time
    print(f"Training complete. Total time: {elapsed/3600:.1f}h")
    print(f"Saved final model to {args.save_path}")


if __name__ == "__main__":
    main()
