#!/usr/bin/env python
"""Extended SD-CFR training: 10K iterations with checkpointing."""

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
    if torch.cuda.is_available():
        print(f"[GPU] GPU Detected: {torch.cuda.get_device_name()}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("[WARNING] No GPU detected - training will be slow")

    parser = argparse.ArgumentParser(description="Extended SD-CFR training (10K iterations)")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--cfr-iterations", type=int, default=10_000)
    parser.add_argument("--traversals-per-iteration", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--save-path", type=str, default="models/sdcfr_extended.pt")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (None for no seeding)")
    parser.add_argument(
        "--warmstart",
        type=str,
        default=None,
        help="Path to existing model to initialise weights from.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"\n{'='*70}")
    print(f"Extended SD-CFR Training: {args.cfr_iterations:,} iterations")
    print(f"{'='*70}")
    print(f"Players: {args.num_players}, Traversals per iter: {args.traversals_per_iteration:,}")
    print(f"Checkpoints every: {args.checkpoint_every:,} iterations")
    print(f"Expected time: ~3-4 hours on RTX 3080\n")

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

        # Log every 100 iterations to reduce overhead
        if iteration % 100 == 0:
            elapsed = time.time() - start_time
            avg_iter_time = np.mean(iteration_times[-20:]) if len(iteration_times) >= 20 else np.mean(iteration_times)
            eta = (args.cfr_iterations - iteration) * avg_iter_time / 60
            print(
                f"Iter {iteration:6d}/{args.cfr_iterations:6d} | loss={result['loss']:7.4f} | "
                f"buffer={result['buffer_size']:7d} | elapsed={elapsed:7.1f}s | ETA={eta:7.1f}m"
            )

        # Periodic checkpointing
        if iteration > 0 and iteration % args.checkpoint_every == 0:
            ckpt_path = str(Path(args.save_path).parent / f"sdcfr_ckpt_{iteration:06d}.pt")
            model.save(ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

    model.save(args.save_path)
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Extended training complete!")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m = {elapsed/3600:.1f}h)")
    print(f"Avg time per iteration: {np.mean(iteration_times):.2f}s")
    print(f"Saved final model to {args.save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
