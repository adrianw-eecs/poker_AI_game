#!/usr/bin/env python
"""Collect training data by playing games with specified bots."""

import argparse
import io
import sys
from pathlib import Path

import numpy as np

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


def collect_data_from_games_mixed(
    num_hands: int,
    num_players: int = 4,
    opponent_config: dict | str = "random",
    learning_seat: int = 0,
    output_file: str = "data/selfplay.npz",
) -> ReplayBuffer:
    """Collect training data with mixed opponent types.

    Args:
        num_hands: Number of hands to play.
        num_players: Number of players per game.
        opponent_config: Either a string ("random"/"flop") or a dict with mixed config
                        e.g., {"flop": 2, "random": 1} for 2 flop + 1 random
        learning_seat: Seat number to collect data from.
        output_file: Where to save the collected data.

    Returns:
        ReplayBuffer with collected experiences.
    """
    # Create config with rebuy enabled to keep games going
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

    # Create session with hands limit and rebuy enabled
    session_config = SessionConfig(duration_hands=num_hands, rebuy_enabled=True)
    session = Session(
        config=config,
        blind_schedule=blind_schedule,
        session_config=session_config,
        logger=NullLogger(),
        action_handler=ActionHandler(),
    )

    # Parse opponent config
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
        bots = {}
        seat_num = 0

        # Add FlopBots
        for _ in range(opponent_config.get("flop", 0)):
            if seat_num < num_players:
                bots[seat_num] = FlopBot(name=f"FlopBot{seat_num}")
                seat_num += 1

        # Add RandomBots
        for _ in range(opponent_config.get("random", 0)):
            if seat_num < num_players:
                bots[seat_num] = RandomBot(name=f"RandomBot{seat_num}")
                seat_num += 1

        # Fill remaining seats with random if needed
        while seat_num < num_players:
            bots[seat_num] = RandomBot(name=f"RandomBot{seat_num}")
            seat_num += 1

    # Play games and collect data
    buffer = ReplayBuffer(max_size=num_hands * 50)  # ~50 decisions per hand
    hand_data: dict[int, list[Experience]] = {}  # hand_id -> list of experiences

    # Override bot.act() for learning seat to track observations
    original_act = bots[learning_seat].act

    def tracking_act(state, legal):
        """Track observation before returning action."""
        seat = learning_seat
        obs = build_observation(state, seat)
        mask = build_action_mask(state, seat)

        # Call original bot to get action
        action = original_act(state, legal)

        # Convert to action index
        action_idx = action_to_action_index(action, state, seat)

        # Store in hand_data (reward will be assigned later)
        hand_id = state.hand_number
        if hand_id not in hand_data:
            hand_data[hand_id] = []

        hand_data[hand_id].append(
            Experience(
                observation=obs,
                action=action_idx,
                reward=0.0,  # Will be updated at end of hand
                legal_mask=mask.astype(np.int32),
                seat=seat,
                hand_id=hand_id,
            )
        )

        return action

    bots[learning_seat].act = tracking_act

    # Play games
    print(f"Playing {num_hands} hands with {num_players} players...")
    print(f"Collecting data from seat {learning_seat}")

    # Describe opponents
    if isinstance(opponent_config, str):
        print(f"Opponents: {num_players - 1} {opponent_config} bots\n")
    else:
        flop_count = opponent_config.get("flop", 0)
        random_count = opponent_config.get("random", 0)
        print(f"Opponents: {flop_count} FlopBot + {random_count} RandomBot\n")

    initial_state = session.create_initial_state(num_players)
    rng = RNG()

    def deck_factory() -> Deck:
        deck = Deck(rng=rng)
        return deck

    # Suppress stdout during game play to avoid encoding issues with Unicode card symbols
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        final_state = session.run(initial_state, bots, deck_factory)
    finally:
        sys.stdout = old_stdout

    # Process hand_data: assign rewards and add to buffer
    print(f"\nProcessing {len(hand_data)} hands of data...")

    starting_stack = config.starting_stack
    for hand_id in sorted(hand_data.keys()):
        # Get final stack for learning agent from final_state
        final_stack = final_state.players[learning_seat].stack
        initial_stack = starting_stack  # Simplified; in full solution would track per-hand
        reward = (final_stack - initial_stack) / starting_stack

        # Assign reward to all experiences in this hand
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

    print(f"Collected {buffer.size} experiences from {len(hand_data)} hands")
    return buffer


def collect_data_from_games(
    num_hands: int,
    num_players: int = 4,
    opponent_type: str = "random",
    learning_seat: int = 0,
    output_file: str = "data/selfplay.npz",
) -> ReplayBuffer:
    """Collect training data by playing games.

    Args:
        num_hands: Number of hands to play.
        num_players: Number of players per game.
        opponent_type: "random" or "flop" for opponents.
        learning_seat: Seat number to collect data from.
        output_file: Where to save the collected data.

    Returns:
        ReplayBuffer with collected experiences.
    """
    # Create config with rebuy enabled to keep games going
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

    # Create session with hands limit and rebuy enabled
    session_config = SessionConfig(duration_hands=num_hands, rebuy_enabled=True)
    session = Session(
        config=config,
        blind_schedule=blind_schedule,
        session_config=session_config,
        logger=NullLogger(),
        action_handler=ActionHandler(),
    )

    # Create bots - all opponents
    if opponent_type == "random":
        opponent_class = RandomBot
    elif opponent_type == "flop":
        opponent_class = FlopBot
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    bots = {i: opponent_class(name=f"Opponent{i}") for i in range(num_players)}

    # Play games and collect data
    buffer = ReplayBuffer(max_size=num_hands * 50)  # ~50 decisions per hand
    hand_data: dict[int, list[Experience]] = {}  # hand_id -> list of experiences

    # Override bot.act() for learning seat to track observations
    original_act = bots[learning_seat].act

    def tracking_act(state, legal):
        """Track observation before returning action."""
        seat = learning_seat
        obs = build_observation(state, seat)
        mask = build_action_mask(state, seat)

        # Call original bot to get action
        action = original_act(state, legal)

        # Convert to action index
        action_idx = action_to_action_index(action, state, seat)

        # Store in hand_data (reward will be assigned later)
        hand_id = state.hand_number
        if hand_id not in hand_data:
            hand_data[hand_id] = []

        hand_data[hand_id].append(
            Experience(
                observation=obs,
                action=action_idx,
                reward=0.0,  # Will be updated at end of hand
                legal_mask=mask.astype(np.int32),
                seat=seat,
                hand_id=hand_id,
            )
        )

        return action

    bots[learning_seat].act = tracking_act

    # Play games
    print(f"Playing {num_hands} hands with {num_players} players...")
    print(f"Collecting data from seat {learning_seat}")
    print(f"Opponents: {num_players - 1} {opponent_type} bots\n")

    initial_state = session.create_initial_state(num_players)
    rng = RNG()

    def deck_factory() -> Deck:
        deck = Deck(rng=rng)
        return deck

    # Suppress stdout during game play to avoid encoding issues with Unicode card symbols
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        final_state = session.run(initial_state, bots, deck_factory)
    finally:
        sys.stdout = old_stdout

    # Process hand_data: assign rewards and add to buffer
    print(f"\nProcessing {len(hand_data)} hands of data...")

    starting_stack = config.starting_stack
    for hand_id in sorted(hand_data.keys()):
        # Get final stack for learning agent from final_state
        final_stack = final_state.players[learning_seat].stack
        initial_stack = starting_stack  # Simplified; in full solution would track per-hand
        reward = (final_stack - initial_stack) / starting_stack

        # Assign reward to all experiences in this hand
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

    print(f"Collected {buffer.size} experiences from {len(hand_data)} hands")
    return buffer


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Collect training data from poker games")
    parser.add_argument("--hands", type=int, default=100, help="Number of hands to play")
    parser.add_argument("--players", type=int, default=4, help="Number of players")
    parser.add_argument("--opponents", choices=["random", "flop"], default="random",
                        help="Opponent type")
    parser.add_argument("--out", type=str, default="data/selfplay.npz", help="Output file")

    args = parser.parse_args()

    # Create output directory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # Collect data
    buffer = collect_data_from_games(
        num_hands=args.hands,
        num_players=args.players,
        opponent_type=args.opponents,
        output_file=args.out,
    )

    # Save
    print(f"\nSaving to {args.out}...")
    buffer.save(args.out)
    print("Done!")


if __name__ == "__main__":
    main()
