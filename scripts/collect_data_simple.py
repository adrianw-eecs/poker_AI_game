#!/usr/bin/env python
"""Simple sequential data collection - more reliable than multiprocessing on Windows.

This script collects training data by playing poker hands sequentially.
While not parallel, it's more reliable on Windows and shows progress in real-time.

Usage:
    python scripts/collect_data_simple.py --total-hands 10000 --opponents 2flop+1random
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import io

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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


def collect_hands(
    num_hands: int,
    num_players: int = 4,
    opponent_config: dict | str = "random",
    learning_seat: int = 0,
    batch_size: int = 50,
) -> ReplayBuffer:
    """Collect training data by playing hands.

    Args:
        num_hands: Number of hands to play.
        num_players: Number of players per game.
        opponent_config: Opponent configuration (string or dict).
        learning_seat: Seat to collect data from.
        batch_size: Print progress after every N hands (default: 50).

    Returns:
        ReplayBuffer with collected experiences.
    """
    # Create game configuration
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=5, big=10, ante=0)],
        hands_per_level=1000,
        fixed=True,
    )
    config = GameConfig(
        num_players=num_players,
        starting_stack=10000,  # Increased from 1000 to prevent early eliminations
        small_blind=5,
        big_blind=10,
        ante=0,
        rake_percent=0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )

    session_config = SessionConfig(
        duration_hands=num_hands,
        rebuy_enabled=True,
        rebuy_stack=10000  # Rebuy to full starting stack
    )
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

    # Progress tracking
    last_printed_hand = 0
    batch_stacks = {}  # Track initial stack at start of each batch

    original_act = bots[learning_seat].act

    def tracking_act(state, legal):
        """Track observation before returning action."""
        nonlocal last_printed_hand

        seat = learning_seat
        obs = build_observation(state, seat)
        mask = build_action_mask(state, seat)
        action = original_act(state, legal)
        action_idx = action_to_action_index(action, state, seat)

        hand_id = state.hand_number
        if hand_id not in hand_data:
            hand_data[hand_id] = []

            # Track stack at start of each batch
            batch_num = hand_id // batch_size
            if batch_num not in batch_stacks:
                batch_stacks[batch_num] = state.players[seat].stack

            # Print progress after every batch_size hands
            if hand_id % batch_size == 0 and hand_id > last_printed_hand:
                completed_batches = hand_id // batch_size
                current_stack = state.players[seat].stack
                batch_start_stack = batch_stacks.get(completed_batches, current_stack)
                profit_loss = current_stack - batch_start_stack

                # Format profit/loss with sign
                if profit_loss >= 0:
                    pl_str = f"+{profit_loss}"
                else:
                    pl_str = f"{profit_loss}"

                print(f"  [OK] Batch {completed_batches} ({hand_id} hands) | Stack: {current_stack} | P&L: {pl_str}")
                last_printed_hand = hand_id

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

    print(f"\nPlaying {num_hands} hands in batches of {batch_size}...")
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        final_state = session.run(initial_state, bots, deck_factory)
    finally:
        sys.stdout = old_stdout

    # Process data and assign rewards
    print(f"\nProcessing {len(hand_data)} hands of data...")
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

    # Summary statistics
    print(f"\nCollection Summary:")
    print(f"  Requested hands:   {num_hands:,}")
    print(f"  Actual hands:      {len(hand_data):,}")
    print(f"  Experiences:       {buffer.size:,}")
    print(f"  Final stack:       {final_state.players[learning_seat].stack}")
    print(f"  Starting stack:    {starting_stack}")
    total_pl = final_state.players[learning_seat].stack - starting_stack
    print(f"  Total P&L:         {total_pl:+.0f} chips")

    return buffer


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect poker training data (simple sequential version)"
    )
    parser.add_argument(
        "--total-hands",
        type=int,
        default=10000,
        help="Total hands to collect (default: 10000)",
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
        default="1flop+2random",
        help='Opponent type: "random", "flop", or mixed like "1flop+2random" (default: 1flop+2random)',
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/selfplay_10k.npz",
        help="Output file (default: data/selfplay_10k.npz)",
    )

    args = parser.parse_args()

    opponent_config = parse_opponent_config(args.opponents, args.players)

    # Format opponent description
    if isinstance(opponent_config, str):
        opponent_desc = f"{args.players - 1} {opponent_config} bots"
    else:
        flop_count = opponent_config.get("flop", 0)
        random_count = opponent_config.get("random", 0)
        opponent_desc = f"{flop_count} FlopBot + {random_count} RandomBot"

    print("=" * 70)
    print("DATA COLLECTION FOR POKER TRAINING")
    print("=" * 70)
    print(f"Target hands:  {args.total_hands:,}")
    print(f"Players:       {args.players}")
    print(f"Opponents:     {opponent_desc}")
    print(f"Output file:   {args.out}")
    print("=" * 70)

    # Create output directory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Collect data in multiple sessions until we reach target
    print(f"\nStarting data collection (running multiple sessions as needed)...")
    combined_buffer = ReplayBuffer(max_size=args.total_hands * 50)
    total_hands_collected = 0
    session_num = 0

    while total_hands_collected < args.total_hands:
        session_num += 1
        hands_remaining = args.total_hands - total_hands_collected

        # Request more hands than needed to account for session termination
        hands_to_request = int(hands_remaining * 1.7)

        print(f"\n[Session {session_num}] Requesting {hands_to_request:,} hands (need {hands_remaining:,} more)...")
        buffer = collect_hands(
            num_hands=hands_to_request,
            num_players=args.players,
            opponent_config=opponent_config,
            learning_seat=0,
        )

        # Add experiences to combined buffer
        for i in range(buffer.size):
            combined_buffer.observations[combined_buffer.size] = buffer.observations[i]
            combined_buffer.actions[combined_buffer.size] = buffer.actions[i]
            combined_buffer.rewards[combined_buffer.size] = buffer.rewards[i]
            combined_buffer.legal_masks[combined_buffer.size] = buffer.legal_masks[i]
            combined_buffer.seats[combined_buffer.size] = buffer.seats[i]
            combined_buffer.hand_ids[combined_buffer.size] = buffer.hand_ids[i]
            combined_buffer.size += 1

        total_hands_collected += buffer.size
        print(f"[Session {session_num}] Collected {buffer.size:,} more experiences (total: {combined_buffer.size:,})")

        if combined_buffer.size >= args.total_hands * 5:  # Rough estimate: 5-6 exp per hand
            break

    buffer = combined_buffer

    # Analyze rewards
    if buffer.size > 0:
        rewards = buffer.rewards[: buffer.size]
        print(f"\nReward statistics:")
        print(f"  Mean:      {rewards.mean():+.4f}")
        print(f"  Std:       {rewards.std():.4f}")
        print(f"  Min:       {rewards.min():+.4f}")
        print(f"  Max:       {rewards.max():+.4f}")
        print(f"  Losses:    {(rewards < -0.3).sum():,}")
        print(f"  Neutral:   {((rewards >= -0.3) & (rewards <= 0.3)).sum():,}")
        print(f"  Wins:      {(rewards > 0.3).sum():,}")

    # Save data
    print(f"\nSaving data to {args.out}...")
    buffer.save(args.out)

    print("[OK] Done!")


if __name__ == "__main__":
    main()
