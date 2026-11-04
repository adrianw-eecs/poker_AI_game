"""Standalone SD-CFR training script."""

import argparse
import os
import sys
import time
from pathlib import Path

# Ensure the project src is on the path when run directly.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.cfr.traversal import ExternalSamplingTraverser
from poker.ml.models.sdcfr_model import SDCFRModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD-CFR poker agent")
    parser.add_argument("--model-name", type=str, default="sdcfr_v1",
                        help="Model folder name (e.g. 'sdcfr_v1' creates models/sdcfr_v1/).")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--cfr-iterations", type=int, default=10_000)
    parser.add_argument("--traversals-per-iteration", type=int, default=1_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2_048,
                        help="Minibatch size for advantage network SGD (default: 2048).")
    parser.add_argument("--sgd-steps", type=int, default=1_000,
                        help="Max SGD steps per CFR iteration (default: 1000).")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint in the model folder (restores iteration + buffer).",
    )
    parser.add_argument(
        "--warmstart",
        type=str,
        default=None,
        help="Warm-start from an existing model (loads weights only, resets iteration/buffer).",
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

    # Create model folder (like NFSP: models/{model_name}/)
    models_dir = Path("models") / args.model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / "sdcfr.pt"

    config, _ = build_config(args.num_players)
    model = SDCFRModel(batch_size=args.batch_size, sgd_steps_per_iteration=args.sgd_steps)

    start_iteration = 0
    if args.resume is not None:
        resume_path = args.resume if Path(args.resume).is_absolute() else models_dir / args.resume
        model.load(str(resume_path))
        start_iteration = model.cfr_iteration
        buf_size = len(model.regret_buffer)
        print(f"Resuming from {resume_path} (iteration {start_iteration}, buffer={buf_size:,})")
    elif args.warmstart is not None:
        model.load(args.warmstart)
        print(f"Warm-start weights loaded from {args.warmstart} (iteration/buffer reset)")

    traverser = ExternalSamplingTraverser(
        config=config,
        advantage_network=model.advantage_network,
        device=model.device,
        regret_buffer=model.regret_buffer,
    )

    print(f"\n{'='*70}")
    print(f"SD-CFR Training")
    print(f"{'='*70}")
    print(f"  Model folder:       {models_dir}")
    print(f"  Total iterations:   {start_iteration + args.cfr_iterations}")
    print(f"  Starting from iter: {start_iteration}")
    print(f"  Players:            {args.num_players}")
    print(f"  Traversals/iter:    {args.traversals_per_iteration:,}")
    print(f"  Batch size:         {args.batch_size:,}")
    print(f"  SGD steps/iter:     {args.sgd_steps:,}\n")

    start_time = time.time()
    total_iterations = start_iteration + args.cfr_iterations

    # Log ~20 times over the run regardless of total iteration count.
    log_every = max(1, args.cfr_iterations // 20)

    for iteration in range(start_iteration, total_iterations):
        traverser_seat = iteration % args.num_players
        # Keep traverser's network reference current after reinit in train_iteration
        traverser.advantage_network = model.advantage_network

        for _ in range(args.traversals_per_iteration):
            traverser.traverse(traverser_seat=traverser_seat, cfr_iteration=iteration)

        result = model.train_iteration()
        # Sync traverser network after potential reinit
        traverser.advantage_network = model.advantage_network

        if (iteration - start_iteration) % log_every == 0:
            elapsed = time.time() - start_time
            loss_str = f"{result['loss']:.4f}" if result["loss"] >= 0 else "warming up"
            print(
                f"Iter {iteration:5d}/{total_iterations} | "
                f"loss={loss_str} | "
                f"buffer={result['buffer_size']:6d} | "
                f"elapsed={elapsed:.1f}s"
            )

        # Checkpoint every N iterations
        if iteration % args.checkpoint_every == 0 and iteration > 0:
            ckpt_path = models_dir / f"sdcfr_ckpt_{iteration}.pt"
            model.save(str(ckpt_path))

    model.save(str(save_path))
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training complete. Total time: {elapsed/3600:.1f}h")
    print(f"Model folder: {models_dir}")
    print(f"Final model:  {save_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
