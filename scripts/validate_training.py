#!/usr/bin/env python
"""Validate NFSP/SDCFR training progress.

Usage:
  # Test a checkpoint
  python scripts/validate_training.py models/nfsp_v3_ckpt_050000.pt

  # Test the latest model
  python scripts/validate_training.py models/nfsp_v3.pt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from poker.ml.env import PokerEnv
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel
from poker.bots.random_bot import RandomBot
from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.engine.action_validator import legal_actions


def test_nfsp(model_path: str, num_hands: int = 100, num_players: int = 2, rake_percent: float = 0.0) -> dict:
    """Test NFSP model against multiple opponent types.

    Args:
        model_path: Path to trained model
        num_hands: Number of hands per opponent
        num_players: Number of players (2 for heads-up, 4 for multi-way)
        rake_percent: Rake percentage (0.0 to 5.0)
    """
    model = NFSPModel()
    model.load(model_path)

    results = {}
    all_actions = [0] * 7
    all_hands = []  # Track all hands for winning hands analysis
    winning_hands = []  # Track winning hands
    pot_wins = 0  # Count pots won

    # Adjust opponents based on player count
    if num_players == 2:
        opponent_configs = {
            "RandomBot": [RandomBot()],
            "CallBot": [CallBot()],
            "FlopBot": [FlopBot()],
        }
    elif num_players == 4:
        opponent_configs = {
            "RandomBot x3": [RandomBot(), RandomBot(), RandomBot()],
            "RandomBot+FlopBot+CallBot": [RandomBot(), FlopBot(), CallBot()],
            "FlopBot x3": [FlopBot(), FlopBot(), FlopBot()],
        }
    else:
        raise ValueError(f"Unsupported num_players: {num_players}")

    for opp_config_name, opponent_list in opponent_configs.items():
        total_reward = 0.0
        hand_count = 0

        for hand_idx in range(num_hands):
            # Use random seed for each game (not fixed seed)
            seed = np.random.randint(0, 2**31 - 1)

            env = PokerEnv(
                num_players=num_players,
                learning_seat=0,
                opponent_bots=opponent_list,
                small_blind=25,
                big_blind=50,
                ante=0,
                seed=seed,
            )

            obs, _ = env.reset()

            # Extract hole cards directly from game state
            try:
                if hasattr(env, 'state') and env.state and env.state.players:
                    player = env.state.players[env.learning_seat]
                    if player.hole_cards and len(player.hole_cards) == 2:
                        # Get ASCII representation of cards
                        import re
                        def card_to_ascii(card):
                            card_str = str(card)
                            # Remove ANSI color codes
                            card_str = re.sub(r'\x1b\[[0-9;]*m', '', card_str)
                            # Replace suit symbols with ASCII
                            card_str = card_str.replace('♣', 'c').replace('♦', 'd').replace('♥', 'h').replace('♠', 's')
                            return card_str

                        card1 = card_to_ascii(player.hole_cards[0])
                        card2 = card_to_ascii(player.hole_cards[1])
                        hole_cards = f"{card1}{card2}"
                    else:
                        hole_cards = "Unknown"
                else:
                    hole_cards = "Unknown"
            except (AttributeError, IndexError, ValueError):
                hole_cards = "Unknown"

            done = False

            while not done:
                mask = env.get_action_mask()
                action = model.select_action(obs, mask, training=False)
                all_actions[action] += 1

                obs, reward, done, _, _ = env.step(action)

            total_reward += reward
            hand_count += 1

            # Track hand results
            hand_result = {
                "hole_cards": hole_cards,
                "opponent": opp_config_name,
                "reward": reward,
                "profit": reward > 0,  # Only count actual profits, not break-even
            }
            all_hands.append(hand_result)

            # Consider a hand "won" if reward > 0 (actual profit, not break-even)
            if reward > 0:
                pot_wins += 1
                winning_hands.append(hand_result)

        results[opp_config_name] = total_reward / hand_count

    # Calculate action distribution
    total_actions = sum(all_actions)
    action_names = ["FOLD", "CHECK", "CALL", "RAISE-0.5", "RAISE-POT", "RAISE-2x", "ALL_IN"]
    action_dist = {
        action_names[i]: 100.0 * all_actions[i] / total_actions
        for i in range(7)
    }

    # Sort winning hands by reward (highest first)
    winning_hands.sort(key=lambda x: x["reward"], reverse=True)
    top_5_wins = winning_hands[:5]

    # Calculate breakdown of hand results
    break_even_hands = sum(1 for h in all_hands if h["reward"] == 0)
    losing_hands = sum(1 for h in all_hands if h["reward"] < 0)
    actual_winning_hands = sum(1 for h in all_hands if h["reward"] > 0)

    return {
        "opponent_rewards": results,
        "action_distribution": action_dist,
        "total_hands": len(all_hands),  # Already includes all hands
        "pot_wins": pot_wins,
        "total_hands_played": len(all_hands),
        "win_rate": 100.0 * pot_wins / len(all_hands) if all_hands else 0.0,
        "top_5_wins": top_5_wins,
        "break_even_hands": break_even_hands,
        "losing_hands": losing_hands,
        "actual_winning_hands": actual_winning_hands,
        "num_players": num_players,
    }


def test_sdcfr(model_path: str, num_hands: int = 100) -> dict:
    """Test SDCFR model against RandomBot."""
    model = SDCFRModel()
    model.load(model_path)

    from poker.config.game_config import GameConfig
    from poker.config.blind_schedule import BlindLevel, BlindSchedule

    config = GameConfig(
        num_players=2,
        starting_stack=1000,
        small_blind=10,
        big_blind=20,
        ante=0,
        rake_percent=0,
        blind_schedule=BlindSchedule(
            levels=[BlindLevel(10, 20, 0)],
            hands_per_level=1000,
            fixed=True,
        ),
    )

    total_reward = 0.0
    hand_count = 0
    all_actions = [0] * 7

    from poker.engine.dealer import deal_hole_cards
    from poker.state.game_state import GameState, Street
    from poker.state.player_state import PlayerState
    from poker.state.pot import Pot
    from poker.domain.deck import Deck
    from poker.rng import RNG

    # Simplified SDCFR testing (uses game state directly, not PokerEnv)
    print("  (SDCFR testing requires game state traversal - skipping for now)")

    return {
        "status": "SDCFR testing not yet implemented in this script",
    }


def print_report(model_path: str, model_type: str, results: dict) -> None:
    """Print a formatted validation report."""
    print(f"\n{'='*70}")
    print(f"Validation Report: {Path(model_path).name}")
    print(f"Model Type: {model_type}")
    print(f"{'='*70}")

    if model_type == "NFSP":
        # Win/Loss Summary
        hands_per_opp = results['total_hands_played'] // 3
        print("\nHand Results Breakdown:")
        print("-" * 70)
        print(f"  Hands per Opponent:  {hands_per_opp}")
        print(f"  Opponents Tested:    3 (RandomBot, CallBot, FlopBot)")
        print(f"  Total Hands Played:  {results['total_hands_played']}")
        print()
        print(f"  Actual Wins (profit > 0):    {results['actual_winning_hands']:5d} ({100.0*results['actual_winning_hands']/results['total_hands_played']:5.2f}%)")
        print(f"  Break-Even (profit = 0):    {results['break_even_hands']:5d} ({100.0*results['break_even_hands']/results['total_hands_played']:5.2f}%)")
        print(f"  Losses (profit < 0):        {results['losing_hands']:5d} ({100.0*results['losing_hands']/results['total_hands_played']:5.2f}%)")

        # Opponent rewards
        print("\nPerformance vs Opponents:")
        print("-" * 70)
        for opp, reward in results["opponent_rewards"].items():
            status = "[+] POSITIVE" if reward > 0 else "[-] NEGATIVE" if reward < -0.5 else "[~] NEUTRAL"
            print(f"  {opp:12s}: {reward:7.4f} reward/hand  {status}")

        avg_reward = np.mean(list(results["opponent_rewards"].values()))
        print(f"\n  Average:     {avg_reward:7.4f} reward/hand")

        # Top 5 Winning Hands
        if results.get("top_5_wins"):
            print("\nTop 5 Winning Hands:")
            print("-" * 70)
            for idx, hand in enumerate(results["top_5_wins"], 1):
                print(f"  {idx}. {hand['hole_cards']:6s} vs {hand['opponent']:10s} - Profit: {hand['reward']:+.4f}")
        else:
            print("\nTop 5 Winning Hands:")
            print("-" * 70)
            print("  (No winning hands found)")

        # Action distribution
        print("\nAction Distribution:")
        print("-" * 70)
        actions = results["action_distribution"]

        # Group raise actions
        raise_total = actions["RAISE-0.5"] + actions["RAISE-POT"] + actions["RAISE-2x"]

        print(f"  FOLD:        {actions['FOLD']:6.2f}%")
        print(f"  CHECK/CALL:  {actions['CHECK'] + actions['CALL']:6.2f}%")
        print(f"  RAISE:       {raise_total:6.2f}%")
        print(f"  ALL_IN:      {actions['ALL_IN']:6.2f}%")

        # Interpretation
        print("\nInterpretation:")
        print("-" * 70)
        fold_pct = actions["FOLD"]

        if fold_pct > 80:
            print("  [WARNING] FOLD% too high (>80%) - Model may be learning only folding")
            print("           Issue: Reward signal may still be broken")
        elif fold_pct > 60:
            print("  [WARNING] FOLD% high (>60%) - Still early in training")
            print("           Expect: Should decrease toward 30-40% by ep 100K")
        elif fold_pct < 30:
            print("  [OK] FOLD% healthy (<30%) - Learning balanced strategy")
        else:
            print("  [OK] FOLD% good (30-40%) - Expected for mid-training")

        if avg_reward > 0:
            print("  [OK] Profitable against opponents - Learning is working")
        elif avg_reward > -0.5:
            print("  [OK] Breaking even - Good progress, need more training")
        else:
            print("  [WARNING] Large losses - Model needs more training or fix")

        if results['win_rate'] < 30:
            print("  [WARNING] Win rate very low (<30%) - Model needs more training")
        elif results['win_rate'] < 40:
            print("  [OK] Win rate acceptable (30-40%) - Model showing promise")
        else:
            print("  [OK] Win rate good (>40%) - Model playing well")

    print(f"\n{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate training progress of NFSP/SDCFR models"
    )
    parser.add_argument("model_path", help="Path to model checkpoint (.pt file)")
    parser.add_argument("--hands", type=int, default=100, help="Number of test hands per opponent")
    parser.add_argument("--type", default=None, help="Model type (nfsp/sdcfr), auto-detect if not specified")
    parser.add_argument("--players", type=int, default=2, choices=[2, 4], help="Number of players (2=heads-up, 4=multi-way)")
    parser.add_argument("--rake", type=float, default=0.0, help="Rake percentage (0.0 to 5.0)")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    # Auto-detect model type
    model_type = args.type
    if model_type is None:
        if "sdcfr" in model_path.name.lower():
            model_type = "SDCFR"
        else:
            model_type = "NFSP"

    print(f"\nLoading model from {model_path}...")
    print(f"Configuration: {args.players}-player game, {args.rake}% rake")
    print()

    try:
        if model_type.upper() == "NFSP":
            results = test_nfsp(str(model_path), num_hands=args.hands, num_players=args.players, rake_percent=args.rake)
            print_report(str(model_path), "NFSP", results)
        elif model_type.upper() == "SDCFR":
            results = test_sdcfr(str(model_path), num_hands=args.hands)
            print_report(str(model_path), "SDCFR", results)
        else:
            print(f"Error: Unknown model type {model_type}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading or testing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
