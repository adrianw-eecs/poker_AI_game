#!/usr/bin/env python
"""Quick SD-CFR training test (15 min): 200 iterations to validate implementation."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.cfr.traversal import ExternalSamplingTraverser
from poker.ml.models.sdcfr_model import SDCFRModel


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
    # Verify GPU availability
    if torch.cuda.is_available():
        print(f"[GPU] GPU Detected: {torch.cuda.get_device_name()}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("[WARNING] No GPU detected - training will be slow")

    parser = argparse.ArgumentParser(description="Quick SD-CFR training test (200 iterations)")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--cfr-iterations", type=int, default=200)
    parser.add_argument("--traversals-per-iteration", type=int, default=500)
    parser.add_argument("--save-path", type=str, default="models/sdcfr_quick_test.pt")
    parser.add_argument(
        "--warmstart",
        type=str,
        default=None,
        help="Path to existing model to initialise weights from.",
    )
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Quick SD-CFR Training Test: {args.cfr_iterations} iterations")
    print(f"{'='*70}")
    print(f"Players: {args.num_players}, Traversals per iter: {args.traversals_per_iteration}")
    print(f"Expected time: ~15 minutes on RTX 3060+\n")

    config, blind_schedule = build_config(args.num_players)
    model = SDCFRModel()

    if args.warmstart is not None:
        model.load(args.warmstart)
        print(f"Loaded warmstart weights from {args.warmstart}\n")

    traverser = ExternalSamplingTraverser(
        config=config,
        blind_schedule=blind_schedule,
        advantage_network=model.advantage_network,
        device=model.device,
        regret_buffer=model.regret_buffer,
    )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    start_time = time.time()
    iteration_times = []

    for iteration in range(args.cfr_iterations):
        iter_start = time.time()

        traverser_seat = iteration % args.num_players
        # Keep traverser's network reference current after reinit in train_iteration
        traverser.advantage_network = model.advantage_network

        for _ in range(args.traversals_per_iteration):
            traverser.traverse(traverser_seat=traverser_seat, cfr_iteration=iteration)

        result = model.train_iteration()
        # Sync traverser network after potential reinit
        traverser.advantage_network = model.advantage_network

        iteration_times.append(time.time() - iter_start)

        if iteration % 10 == 0 or iteration == args.cfr_iterations - 1:
            elapsed = time.time() - start_time
            avg_iter_time = np.mean(iteration_times[-20:]) if len(iteration_times) >= 20 else np.mean(iteration_times)
            eta = (args.cfr_iterations - iteration) * avg_iter_time / 60
            print(
                f"Iter {iteration:4d}/{args.cfr_iterations:4d} | loss={result['loss']:7.4f} | buffer={result['buffer_size']:6d} "
                f"| elapsed={elapsed:6.1f}s | avg_iter={avg_iter_time:.2f}s | ETA={eta:6.1f}m"
            )

    model.save(args.save_path)
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Quick training complete!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"Avg time per iteration: {np.mean(iteration_times):.2f}s")
    print(f"Saved final model to {args.save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
