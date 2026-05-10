#!/usr/bin/env python
"""Test the parallel-trained NFSP model's action distribution.

Validates that Phase 1 parallel training produces a model with diverse action patterns
(not 100% folding like the broken versions).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.random_bot import RandomBot
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel


def test_model_action_distribution(model_path: str, num_hands: int = 100):
    """Test a trained model's action distribution across multiple hands."""
    print(f"\n{'='*70}")
    print(f"Testing Model: {model_path}")
    print(f"{'='*70}\n")

    # Load model
    model = NFSPModel()
    model.load(model_path)

    opponents = [RandomBot(seed=42), FlopBot(seed=43), CallBot(seed=44)]
    action_counts = {"fold": 0, "call": 0, "raise": 0}
    action_opportunities = 0
    win_count = 0
    total_reward = 0

    for hand_idx in range(num_hands):
        # Rotate opponent for diversity
        opponent_idx = hand_idx % 3
        env = PokerEnv(
            num_players=2,
            learning_seat=0,
            opponent_bots=[opponents[opponent_idx]],
            seed=42 + hand_idx,
        )

        obs, _ = env.reset()
        done = False
        hand_reward = 0

        while not done:
            mask = env.get_action_mask()
            action = model.select_action(obs, mask, training=False)
            next_obs, reward, done, _, _ = env.step(action)

            # Track actions only when it's our turn
            if mask.sum() > 0:  # Only when we have legal actions
                action_names = ["fold", "call", "raise"]
                if 0 <= action < len(action_names):
                    action_counts[action_names[action]] += 1
                    action_opportunities += 1

            hand_reward += reward
            obs = next_obs

        total_reward += hand_reward
        if hand_reward > 0:
            win_count += 1

    # Calculate percentages
    fold_pct = 100 * action_counts["fold"] / action_opportunities if action_opportunities > 0 else 0
    call_pct = 100 * action_counts["call"] / action_opportunities if action_opportunities > 0 else 0
    raise_pct = 100 * action_counts["raise"] / action_opportunities if action_opportunities > 0 else 0
    win_pct = 100 * win_count / num_hands

    print(f"Results over {num_hands} hands ({action_opportunities} decision points):")
    print(f"  Fold:  {action_counts['fold']:4d} ({fold_pct:5.1f}%)")
    print(f"  Call:  {action_counts['call']:4d} ({call_pct:5.1f}%)")
    print(f"  Raise: {action_counts['raise']:4d} ({raise_pct:5.1f}%)")
    print(f"\nPerformance:")
    print(f"  Win rate: {win_pct:.1f}% ({win_count}/{num_hands})")
    print(f"  Total reward: {total_reward:.1f}")
    print(f"  Avg reward per hand: {total_reward/num_hands:.3f}")

    # Validation checks
    print(f"\n{'='*70}")
    print("Validation Checks:")
    print(f"{'='*70}")

    checks = [
        ("Not 100% folding", fold_pct < 100),
        ("Fold percentage < 95%", fold_pct < 95),
        ("Has raise actions", action_counts["raise"] > 0),
        ("Has call actions", action_counts["call"] > 0),
        ("Positive win rate", win_pct > 20),  # Should beat random (50%) sometimes
    ]

    for check_name, result in checks:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {check_name}")

    return all(result for _, result in checks)


if __name__ == "__main__":
    # Test parallel-trained model
    parallel_model = "models/nfsp_parallel.pt"
    if Path(parallel_model).exists():
        success = test_model_action_distribution(parallel_model, num_hands=50)
        if success:
            print(f"\n[PASS] Model validation PASSED")
        else:
            print(f"\n[FAIL] Model validation FAILED")
    else:
        print(f"Model not found: {parallel_model}")
