#!/usr/bin/env python
"""NFSP parallel training with 4 worker processes collecting data simultaneously.

Architecture:
- Main process: Trains on buffers, manages model weights
- 4 worker processes: Each runs an independent poker environment, sends transitions via queue
- Shared queue: Workers put transitions, main process reads and buffers them
- Weight sync: Every 1000 episodes, workers fetch latest weights

Expected speedup: 3-4× wall-clock time reduction (4 envs in parallel)
"""

import argparse
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.buffers import CircularBuffer, ReservoirBuffer
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.population import PopulationManager


def get_opponent_for_episode(episode: int, seed: int):
    """Rotate through opponent types for diversity."""
    cycle_length = 83334  # 250K / 3 opponents
    cycle_type = (episode // cycle_length) % 3

    if cycle_type == 0:
        return [RandomBot(seed=seed + episode)]
    elif cycle_type == 1:
        return [FlopBot(seed=seed + episode)]
    else:
        return [CallBot(seed=seed + episode)]


def worker_collect_episodes(
    worker_id: int,
    episode_start: int,
    episode_end: int,
    transition_queue: mp.Queue,
    weight_queue: mp.Queue,
    result_queue: mp.Queue,
    seed: int,
) -> None:
    """Worker process: collect episodes in an independent environment.

    Sends transitions to main process via queue instead of writing to shared buffer.

    Args:
        worker_id: ID of this worker (0-3)
        episode_start: Starting episode number
        episode_end: Ending episode number
        transition_queue: Queue to send transitions to main process
        weight_queue: Queue to receive model weights
        result_queue: Queue to send stats back to main process
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    # Create local model
    model = NFSPModel()

    episode_times = []
    total_transitions = 0
    last_weight_sync = 0

    print(f"[Worker {worker_id}] Starting episodes {episode_start}-{episode_end}", flush=True)

    for episode in range(episode_start, episode_end):
        ep_start = time.time()

        # Periodically fetch updated weights from main process
        if episode - last_weight_sync >= 1000:
            try:
                weights = weight_queue.get_nowait()
                model.q_network.load_state_dict(weights["q_network"])
                model.policy_network.load_state_dict(weights["policy_network"])
                model.target_network.load_state_dict(weights["target_network"])
                last_weight_sync = episode
                if episode % 5000 == 0:
                    print(f"[Worker {worker_id}] Synced weights at episode {episode}", flush=True)
            except queue.Empty:
                pass  # No new weights yet

        # Get opponent for this episode
        opponents = get_opponent_for_episode(episode, seed or 42)
        env = PokerEnv(
            num_players=2,
            learning_seat=0,
            opponent_bots=opponents,
            seed=seed + episode if seed else None,
        )

        # Collect transitions
        obs, _ = env.reset()
        done = False
        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=True)
            next_obs, reward, done, _, _ = env.step(action)

            # Send transition to main process via queue
            transition_queue.put({
                "obs": obs.copy(),
                "action": action,
                "reward": reward,
                "next_obs": next_obs.copy(),
                "done": done,
            })
            total_transitions += 1

            obs = next_obs

        episode_times.append(time.time() - ep_start)

        if (episode + 1) % 10000 == 0:
            avg_time = np.mean(episode_times[-100:]) if len(episode_times) >= 100 else np.mean(episode_times)
            print(
                f"[Worker {worker_id}] Episode {episode+1} | "
                f"Avg time: {avg_time:.4f}s | Transitions sent: {total_transitions}",
                flush=True,
            )

    # Report final stats
    result_queue.put(
        {
            "worker_id": worker_id,
            "episodes": episode_end - episode_start,
            "transitions": total_transitions,
            "avg_episode_time": np.mean(episode_times),
        }
    )

    print(f"[Worker {worker_id}] Completed {episode_end - episode_start} episodes", flush=True)


def main() -> None:
    if torch.cuda.is_available():
        print(f"[GPU] GPU Detected: {torch.cuda.get_device_name()}")
        print(f"       Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\n")
    else:
        print("[WARNING] No GPU detected - training will be slow\n")

    parser = argparse.ArgumentParser(description="NFSP Parallel Training (4 Workers)")
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=250_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--save-path", type=str, default="models/nfsp_parallel.pt")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    print(f"{'='*70}")
    print(f"NFSP Parallel Training: {args.num_workers} Workers")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Total episodes: {args.episodes:,}")
    print(f"  Episodes per worker: {args.episodes // args.num_workers:,}")
    print(f"  Eval frequency: every {args.eval_every:,} episodes")
    print(f"  Checkpoint frequency: every {args.checkpoint_every:,} episodes\n")

    # Create buffers (NOT thread-safe, will be filled by main process only)
    q_buffer = CircularBuffer(capacity=1_000_000, obs_dim=155)
    policy_buffer = ReservoirBuffer(capacity=5_000_000, obs_dim=155)

    # Create main training model
    model = NFSPModel()

    # Create queues for communication (thread-safe by design)
    transition_queue = mp.Queue(maxsize=10000)  # Buffer transitions from workers
    weight_queue = mp.Queue()  # Send weights to workers
    result_queue = mp.Queue()  # Receive stats from workers

    # Spawn worker processes
    workers = []
    episodes_per_worker = args.episodes // args.num_workers

    print(f"Spawning {args.num_workers} worker processes...")
    for worker_id in range(args.num_workers):
        episode_start = worker_id * episodes_per_worker
        episode_end = (worker_id + 1) * episodes_per_worker if worker_id < args.num_workers - 1 else args.episodes

        p = mp.Process(
            target=worker_collect_episodes,
            args=(
                worker_id,
                episode_start,
                episode_end,
                transition_queue,
                weight_queue,
                result_queue,
                args.seed,
            ),
            daemon=False,
        )
        p.start()
        workers.append(p)

    print(f"[OK] All workers started\n")

    # Main training loop
    last_q_loss = None
    last_policy_loss = None
    start_time = time.time()
    last_log_time = start_time
    train_step_count = 0
    transitions_received = 0
    last_checkpoint_time = start_time

    try:
        while any(p.is_alive() for p in workers):
            # Continuously drain transition queue and add to buffers
            while True:
                try:
                    transition = transition_queue.get_nowait()
                    q_buffer.add(
                        transition["obs"],
                        transition["action"],
                        transition["reward"],
                        transition["next_obs"],
                        transition["done"],
                    )
                    policy_buffer.add(transition["obs"], transition["action"])
                    transitions_received += 1
                except queue.Empty:
                    break

            # Train on accumulated transitions (more data available now)
            num_training_steps = 0
            while len(q_buffer) >= 512 and len(policy_buffer) >= 64 and num_training_steps < 10:
                result = model.train_step()
                if result["q_loss"] is not None:
                    last_q_loss = result["q_loss"]
                    last_policy_loss = result["policy_loss"]
                train_step_count += 1
                num_training_steps += 1

            # Broadcast updated weights to workers periodically
            for _ in range(args.num_workers):
                try:
                    weight_queue.put_nowait(
                        {
                            "q_network": model.q_network.state_dict(),
                            "policy_network": model.policy_network.state_dict(),
                            "target_network": model.target_network.state_dict(),
                        }
                    )
                except queue.Full:
                    pass  # Worker not ready, skip

            # Log every 2 seconds
            elapsed = time.time() - start_time
            if elapsed - (last_log_time - start_time) >= 2.0:
                q_str = f"{last_q_loss:.4f}" if last_q_loss is not None else "n/a"
                p_str = f"{last_policy_loss:.4f}" if last_policy_loss is not None else "n/a"
                print(
                    f"Ep {transitions_received:7d} | q_loss={q_str} | policy_loss={p_str} | "
                    f"q_buf={len(q_buffer):7d} | p_buf={len(policy_buffer):7d} | "
                    f"transitions={transitions_received:10d} | elapsed={elapsed:7.1f}s"
                )
                last_log_time = time.time()

            # Checkpointing every checkpoint_every seconds
            if transitions_received > 0 and (time.time() - last_checkpoint_time) >= 60 and transitions_received % args.checkpoint_every == 0:
                ckpt_path = str(Path(args.save_path).parent / f"nfsp_parallel_ckpt_{transitions_received:06d}.pt")
                model.save(ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")
                last_checkpoint_time = time.time()

            # Prevent busy-waiting
            time.sleep(0.01)

        # Drain any remaining transitions after workers finish
        print("\nDraining remaining transitions from queue...")
        while True:
            try:
                transition = transition_queue.get_nowait()
                q_buffer.add(
                    transition["obs"],
                    transition["action"],
                    transition["reward"],
                    transition["next_obs"],
                    transition["done"],
                )
                policy_buffer.add(transition["obs"], transition["action"])
                transitions_received += 1
            except queue.Empty:
                break

        # Final training pass on all accumulated data
        print(f"Running final training pass on {len(q_buffer)} buffered transitions...")
        final_train_steps = 0
        while len(q_buffer) >= 512 and len(policy_buffer) >= 64 and final_train_steps < 100:
            result = model.train_step()
            if result["q_loss"] is not None:
                last_q_loss = result["q_loss"]
                last_policy_loss = result["policy_loss"]
            train_step_count += 1
            final_train_steps += 1
        print(f"Final training complete: {final_train_steps} additional steps")

    except KeyboardInterrupt:
        print("\n[INFO] Received interrupt, shutting down workers...")
    finally:
        # Wait for all workers to finish
        print("\nWaiting for workers to complete...")
        for i, p in enumerate(workers):
            p.join(timeout=30)
            if p.is_alive():
                print(f"[WARNING] Worker {i} did not exit cleanly, terminating...")
                p.terminate()

        # Collect results
        print("\nWorker Statistics:")
        while not result_queue.empty():
            try:
                result = result_queue.get_nowait()
                print(
                    f"  Worker {result['worker_id']}: "
                    f"{result['episodes']} episodes, "
                    f"{result['transitions']:,} transitions, "
                    f"avg {result['avg_episode_time']:.4f}s/episode"
                )
            except queue.Empty:
                break

        # Final save
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))

        elapsed = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}m)")
        print(f"Total training steps: {train_step_count}")
        print(f"Total transitions collected: {transitions_received}")
        print(f"Saved final model to {save_path}")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    mp.set_start_method("spawn", force=True)
    main()
