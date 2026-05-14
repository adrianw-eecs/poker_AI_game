#!/usr/bin/env python
"""Debug test script - logs every action taken by the model."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.random_bot import RandomBot
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.config.game_config import GameConfig
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel

# Action names for clarity
ACTION_NAMES = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "RAISE_2x",
    4: "RAISE_3x",
    5: "RAISE_4x",
    6: "ALL_IN",
}


def build_game_config(num_players: int) -> tuple[GameConfig, BlindSchedule]:
    """Build standard 2/4 blind game config."""
    blind_schedule = BlindSchedule(
        levels=[BlindLevel(small=2, big=4, ante=0)],
        hands_per_level=1000,
        fixed=True,
    )
    config = GameConfig(
        num_players=num_players,
        starting_stack=1000,
        small_blind=2,
        big_blind=4,
        ante=0,
        rake_percent=0.0,
        rake_cap=None,
        blind_schedule=blind_schedule,
        run_it_twice=False,
    )
    return config, blind_schedule


def main():
    print("\n" + "=" * 80)
    print("NFSP MODEL DEBUG - ACTION LOGGING")
    print("=" * 80 + "\n")

    # Load model
    model = NFSPModel()
    model.load("models/nfsp_diverse_v2.pt")
    print("[OK] NFSP model loaded from models/nfsp_diverse_v2.pt\n")

    # Setup game
    config, blind_schedule = build_game_config(2)
    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[RandomBot(seed=42)],
        seed=42,
    )

    # Run 5 hands with detailed logging
    num_hands = 5
    action_counts = {i: 0 for i in range(7)}
    total_profit = 0

    for hand_idx in range(num_hands):
        print(f"\n{'='*80}")
        print(f"HAND {hand_idx + 1}/5")
        print(f"{'='*80}")

        obs, _ = env.reset()
        done = False
        action_sequence = []
        hand_profit = 0

        step_count = 0
        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=False)
            action_counts[action] += 1
            action_sequence.append(action)

            # Log the action
            legal_actions = np.where(mask == 1)[0]
            action_name = ACTION_NAMES.get(action, f"UNKNOWN({action})")
            legal_action_names = [ACTION_NAMES.get(a, f"UNKNOWN({a})") for a in legal_actions]

            print(
                f"  Step {step_count+1}: Action={action_name:12s} "
                f"(legal: {', '.join(legal_action_names)})"
            )

            obs, reward, done, _, _ = env.step(action)
            step_count += 1

        # End of hand
        hand_profit = reward
        total_profit += hand_profit
        status = "WON" if hand_profit > 0 else ("LOST" if hand_profit < 0 else "TIED")
        print(f"\n  Hand Result: {status} (${hand_profit:+.0f})")
        print(f"  Actions taken: {[ACTION_NAMES.get(a, str(a)) for a in action_sequence]}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print("Action Frequency:")
    for action_id, count in action_counts.items():
        action_name = ACTION_NAMES.get(action_id, f"UNKNOWN({action_id})")
        pct = 100 * count / sum(action_counts.values()) if sum(action_counts.values()) > 0 else 0
        print(f"  {action_name:12s}: {count:3d} times ({pct:5.1f}%)")

    print(f"\nTotal Profit: ${total_profit:+.0f}")
    print(f"Avg per Hand: ${total_profit / num_hands:+.2f}\n")


if __name__ == "__main__":
    main()
