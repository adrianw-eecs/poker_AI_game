#!/usr/bin/env python
"""Collect 10,000 hands of training data using parallel processes.

This script runs multiple data collection jobs in parallel using multiprocessing,
then combines the results into a single dataset. Each worker collects 1,000 hands
against FlopBot (which provides varied reward signals).

Run this to generate all 10,000 hands of training data:
    python scripts/collect_data_parallel.py
"""

import argparse
import sys
from pathlib import Path
from multiprocessing import Pool, get_context
import numpy as np
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import here so multiprocessing workers have access
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.domain.deck import Deck
from poker.engine.action_handler import ActionHandler
from poker.engine.session import Session, SessionConfig
from poker.logging.logger import NullLogger
from poker.ml.action_space import action_to_action_index, build_action_mask
from poker.ml.observation import build_observation
from poker.rng import RNG
from poker.training.dataset import Experience, ReplayBuffer

from poker.training.dataset import ReplayBuffer


def collect_batch(args_tuple):
    """Collect a batch of hands. Must be at module level for multiprocessing.

    Args:
        args_tuple: (batch_num, num_hands_per_batch, num_players, opponent_config)
                   opponent_config can be string like "flop" or "random",
                   or a dict with mixed opponent types

    Returns:
        ReplayBuffer with collected experiences from this batch.
    """
    batch_num, num_hands, num_players, opponent_config = args_tuple

    # Parse opponent config
    if isinstance(opponent_config, str):
        desc = f"{opponent_config} bots"
    else:
        flop_count = opponent_config.get("flop", 0)
        random_count = opponent_config.get("random", 0)
        desc = f"{flop_count} FlopBot + {random_count} RandomBot"

    print(f"\n[Batch {batch_num}] Starting {num_hands} hands against {desc}...")

    # Create game configuration
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10, ante=0)],
        hands_per_level=1000,
        fixed=True,
    )
    config = GameConfig(
        num_players=num_players,
        starting_stack=1000,
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )

    session_config = SessionConfig(duration_hands=num_hands, rebuy_enabled=True)
    session = Session(
        config=config,
        blind_schedule=blind_schedule,
        session_config=session_config,
        logger=NullLogger(),
        action_handler=ActionHandler(),
    )

    # Create bots based on opponent config
    bots = {}
    if isinstance(opponent_config, str):
        # Simple single-type config
        if opponent_config == "random":
            opponent_class = RandomBot
        elif opponent_config == "flop":
            opponent_class = FlopBot
        else:
            raise ValueError(f"Unknown opponent type: {opponent_config}")
        bots = {i: opponent_class(name=f"Opponent{i}") for i in range(num_players)}
    else:
        # Mixed config: {"flop": 2, "random": 1, ...}
        seat_num = 0
        for _ in range(opponent_config.get("flop", 0)):
            if seat_num < num_players:
                bots[seat_num] = FlopBot(name=f"FlopBot{seat_num}")
                seat_num += 1
        for _ in range(opponent_config.get("random", 0)):
            if seat_num < num_players:
                bots[seat_num] = RandomBot(name=f"RandomBot{seat_num}")
                seat_num += 1
        while seat_num < num_players:
            bots[seat_num] = RandomBot(name=f"RandomBot{seat_num}")
            seat_num += 1

    # Collect data
    buffer = ReplayBuffer(max_size=num_hands * 50)
    hand_data: dict[int, list[Experience]] = {}
    learning_seat = 0

    original_act = bots[learning_seat].act

    def tracking_act(state, legal):
        """Track observation before returning action."""
        seat = learning_seat
        obs = build_observation(state, seat)
        mask = build_action_mask(state, seat)
        action = original_act(state, legal)
        action_idx = action_to_action_index(action, state, seat)

        hand_id = state.hand_number
        if hand_id not in hand_data:
            hand_data[hand_id] = []

        hand_data[hand_id].append(
            Experience(
                observation=obs,
                action=action_idx,
                reward=0.0,
                legal_mask=mask.astype(np.int32),
                seat=seat,
                hand_id=hand_id,
            )
        )
        return action

    bots[learning_seat].act = tracking_act

    # Play games (suppress output)
    initial_state = session.create_initial_state(num_players)
    rng = RNG()

    def deck_factory() -> Deck:
        return Deck(rng=rng)

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        final_state = session.run(initial_state, bots, deck_factory)
    finally:
        sys.stdout = old_stdout

    # Process data and assign rewards
    starting_stack = config.starting_stack
    for hand_id in sorted(hand_data.keys()):
        final_stack = final_state.players[learning_seat].stack
        initial_stack = starting_stack
        reward = (final_stack - initial_stack) / starting_stack

        for exp in hand_data[hand_id]:
            updated_exp = Experience(
                observation=exp.observation,
                action=exp.action,
                reward=reward,
                legal_mask=exp.legal_mask,
                seat=exp.seat,
                hand_id=exp.hand_id,
            )
            buffer.add(updated_exp)

    print(f"[Batch {batch_num}] Completed: {buffer.size} experiences collected")
    return buffer


def parse_opponent_config(opponent_str: str, num_players: int) -> dict | str:
    """Parse opponent configuration string.

    Args:
        opponent_str: String like "flop", "random", or "2flop+1random"
        num_players: Total number of players (seats)

    Returns:
        Either a string ("flop" or "random") or dict {"flop": N, "random": M}
    """
    opponent_str = opponent_str.lower().strip()

    # Simple cases
    if opponent_str in ("flop", "random"):
        return opponent_str

    # Mixed format: "2flop+1random" or "2flop+2random"
    if "flop" in opponent_str or "random" in opponent_str:
        config = {}

        # Parse flop count
        if "flop" in opponent_str:
            parts = opponent_str.split("flop")
            if parts[0]:
                config["flop"] = int(parts[0])

        # Parse random count
        if "random" in opponent_str:
            parts = opponent_str.split("random")
            if parts[0]:
                # Extract number before "random"
                num_str = ""
                for i in range(len(parts[0]) - 1, -1, -1):
                    if parts[0][i].isdigit():
                        num_str = parts[0][i] + num_str
                    elif num_str:
                        break
                if num_str:
                    config["random"] = int(num_str)

        # Validate
        total_opponents = sum(config.values())
        if total_opponents >= num_players:
            raise ValueError(
                f"Opponent count ({total_opponents}) must be less than num_players ({num_players})"
            )

        if config:
            return config

    raise ValueError(
        f"Invalid opponent format: {opponent_str}. "
        'Use "random", "flop", or "2flop+1random"'
    )


def combine_buffers(buffers: list) -> ReplayBuffer:
    """Combine multiple ReplayBuffers into one.

    Args:
        buffers: List of ReplayBuffer objects.

    Returns:
        Combined ReplayBuffer with all experiences.
    """
    total_size = sum(buf.size for buf in buffers)
    combined = ReplayBuffer(max_size=total_size + 1000)

    for buf in buffers:
        # Copy all experiences from this buffer
        for i in range(buf.size):
            combined.add(buf.buffer[i])

    return combined


def main():
    """Main entry point for parallel data collection."""
    parser = argparse.ArgumentParser(
        description="Collect 10,000 hands of training data using parallel processes"
    )
    parser.add_argument(
        "--total-hands",
        type=int,
        default=10000,
        help="Total hands to collect (default: 10000)",
    )
    parser.add_argument(
        "--hands-per-batch",
        type=int,
        default=1000,
        help="Hands per batch/worker (default: 1000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, use CPU count for max)",
    )
    parser.add_argument(
        "--players",
        type=int,
        default=4,
        help="Number of players per game (default: 4)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="flop",
        help='Opponent type: "random", "flop", or mixed like "2flop+1random" (default: flop)',
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/selfplay_10k.npz",
        help="Output file for combined data (default: data/selfplay_10k.npz)",
    )

    args = parser.parse_args()

    # Parse opponent configuration
    opponent_config = parse_opponent_config(args.opponents, args.players)

    # Calculate number of batches needed
    num_batches = args.total_hands // args.hands_per_batch
    if args.total_hands % args.hands_per_batch != 0:
        num_batches += 1

    # Adjust num_workers to not exceed batches
    num_workers = min(args.num_workers, num_batches)

    # Format opponent description
    if isinstance(opponent_config, str):
        opponent_desc = f"{args.players - 1} {opponent_config} bots"
    else:
        flop_count = opponent_config.get("flop", 0)
        random_count = opponent_config.get("random", 0)
        opponent_desc = f"{flop_count} FlopBot + {random_count} RandomBot"

    print("=" * 70)
    print("PARALLEL DATA COLLECTION FOR 10,000 POKER HANDS")
    print("=" * 70)
    print(f"Total hands:        {args.total_hands:,}")
    print(f"Hands per batch:    {args.hands_per_batch:,}")
    print(f"Number of batches:  {num_batches}")
    print(f"Parallel workers:   {num_workers}")
    print(f"Players per game:   {args.players}")
    print(f"Opponents:          {opponent_desc}")
    print(f"Output file:        {args.out}")
    print("=" * 70)
    print()

    # Create output directory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Prepare batch arguments
    batch_args = []
    for batch_num in range(1, num_batches + 1):
        # Last batch might have fewer hands
        hands_in_batch = args.hands_per_batch
        if batch_num == num_batches:
            hands_in_batch = args.total_hands - (batch_num - 1) * args.hands_per_batch

        batch_args.append((batch_num, hands_in_batch, args.players, opponent_config))

    # Run parallel data collection
    print(f"Starting {num_workers} parallel workers...")
    print("(Each worker collects hands independently, then results are combined)\n")

    # Use spawn context for Windows compatibility
    try:
        ctx = get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            buffers = pool.map(collect_batch, batch_args)
    except ValueError:
        # Fallback to default context if spawn is not available
        with Pool(processes=num_workers) as pool:
            buffers = pool.map(collect_batch, batch_args)

    print("\n" + "=" * 70)
    print("COMBINING RESULTS")
    print("=" * 70)

    # Combine all buffers
    combined_buffer = combine_buffers(buffers)

    print(f"\nTotal experiences collected: {combined_buffer.size:,}")
    print(f"Average experiences per batch: {combined_buffer.size // num_batches:,}")

    # Analyze reward distribution
    if combined_buffer.size > 0:
        rewards = np.array([exp.reward for exp in combined_buffer.buffer[:combined_buffer.size]])
        print(f"\nReward statistics:")
        print(f"  Mean:   {rewards.mean():+.4f}")
        print(f"  Std:    {rewards.std():.4f}")
        print(f"  Min:    {rewards.min():+.4f}")
        print(f"  Max:    {rewards.max():+.4f}")
        print(f"  Count(-1.0): {(rewards == -1.0).sum():,}")
        print(f"  Count(0.0):  {(rewards == 0.0).sum():,}")
        print(f"  Count(+1.0): {(rewards == 1.0).sum():,}")

    # Save combined data
    print(f"\nSaving combined data to {args.out}...")
    combined_buffer.save(args.out)

    print("✓ Done! Ready for training with improved reward signals.")
    print("\nNext steps:")
    print(f"  1. Train linear model:  python scripts/train_linear.py --data {args.out}")
    print(f"  2. Train tree model:    python scripts/train_tree.py --data {args.out}")
    print(f"  3. Train deep model:    python scripts/train_deep.py --data {args.out}")
    print()


if __name__ == "__main__":
    # Windows multiprocessing requires this guard
    main()
    sys.exit(0)
