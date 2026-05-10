#!/usr/bin/env python
"""Test the v3 model to verify it learned proper strategy."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.random_bot import RandomBot
from poker.config.game_config import GameConfig
from poker.config.blind_schedule import BlindLevel, BlindSchedule
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel

ACTION_NAMES = {
    0: "FOLD",
    1: "CHECK",
    2: "CALL",
    3: "RAISE_2x",
    4: "RAISE_3x",
    5: "RAISE_4x",
    6: "ALL_IN",
}


def main():
    print("\n" + "=" * 80)
    print("NFSP v3 MODEL TEST - Verify Strategy Learning")
    print("=" * 80 + "\n")

    model = NFSPModel()
    try:
        model.load("models/nfsp_v3.pt")
        print("[OK] Loaded v3 model from models/nfsp_v3.pt\n")
    except FileNotFoundError:
        print("[ERROR] Model not found. Run train_nfsp_v3.py first.\n")
        return

    env = PokerEnv(
        num_players=2,
        learning_seat=0,
        opponent_bots=[RandomBot(seed=42)],
        seed=42,
    )

    num_hands = 10
    action_counts = {i: 0 for i in range(7)}
    total_profit = 0

    for hand_idx in range(num_hands):
        print(f"Hand {hand_idx + 1}/{num_hands}")

        obs, _ = env.reset()
        done = False
        action_sequence = []

        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=False)
            action_counts[action] += 1
            action_sequence.append(action)

            obs, reward, done, _, _ = env.step(action)

        total_profit += reward
        status = "WON" if reward > 0 else ("LOST" if reward < 0 else "TIED")
        print(
            f"  Result: {status} (${reward:+.0f}) | "
            f"Actions: {[ACTION_NAMES[a] for a in action_sequence]}"
        )

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    print("Action Distribution:")
    total = sum(action_counts.values())
    for action_id in range(7):
        count = action_counts[action_id]
        pct = 100 * count / total if total > 0 else 0
        print(f"  {ACTION_NAMES[action_id]:10s}: {count:3d} ({pct:5.1f}%)")

    print(f"\nTotal Profit: ${total_profit:+.0f}")
    print(f"Avg per Hand: ${total_profit / num_hands:+.2f}")
    print()

    # Compare against expectations
    fold_pct = 100 * action_counts[0] / total if total > 0 else 0
    print("Analysis:")
    if fold_pct > 70:
        print(f"  Status: FOLDING TOO MUCH ({fold_pct:.1f}%)")
        print("  Issue: v3 fix didn't resolve the problem")
    elif fold_pct > 50:
        print(f"  Status: IMPROVING ({fold_pct:.1f}% fold)")
        print("  Progress: Moving away from 100% fold, but not optimal yet")
    elif fold_pct > 30:
        print(f"  Status: GOOD ({fold_pct:.1f}% fold)")
        print("  Progress: Learning diverse strategy")
    else:
        print(f"  Status: EXCELLENT ({fold_pct:.1f}% fold)")
        print("  Progress: Learned proper fold/play balance")


if __name__ == "__main__":
    main()
