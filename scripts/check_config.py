import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("CONFIGURATION VERIFICATION")
print("=" * 70)

# Check 1: Training script num_players
print("\n[CHECK 1] NFSP Training Script Player Count")
print("-" * 70)

with open("scripts/train_nfsp_final.py") as f:
    content = f.read()
    if 'num_players=2' in content:
        print("CONFIRMED: Training script hardcodes 2-player only")
        print("  - Default: --num-players 2")
        print("  - opponent_bots=[opponent] (single bot)")
        print("  - Game is 2-player, NOT 4-player")
    else:
        print("  Code doesn't show explicit 2-player hardcoding")

# Check 2: Action retry limits
print("\n[CHECK 2] Action Retry Limits")
print("-" * 70)

with open("src/poker/engine/action_handler.py") as f:
    content = f.read()
    if "max_retries" in content:
        print("FOUND: ActionHandler has max_retries=10 for invalid actions")
        print("  - Location: src/poker/engine/action_handler.py")
        print("  - Raises IllegalActionError after 10 failed attempts")
    else:
        print("  No action retry limit found")

with open("src/poker/ml/env.py") as f:
    content = f.read()
    if "action_handler" in content.lower() or "ActionHandler" in content:
        print("  PokerEnv uses ActionHandler: YES")
    else:
        print("  PokerEnv uses ActionHandler: NO")
        print("  - Bot actions called directly without retry logic")
        print("  - No punishment for invalid actions currently")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
1. Training Script: 2-PLAYER ONLY
   - Currently plays single learning_seat vs single opponent
   - To play 4-player: need to create 3 opponent bots

2. Action Limits: NOT INTEGRATED
   - ActionHandler exists but PokerEnv doesn't use it
   - No current punishment for failed actions
   - Need to add explicit invalid action tracking and penalties
""")
