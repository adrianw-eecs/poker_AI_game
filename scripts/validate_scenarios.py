#!/usr/bin/env python
"""Validate benchmark scenario results by checking game logs for hand counts."""

import json
import sys
from pathlib import Path
from collections import defaultdict

def count_hands_in_log(log_file: Path) -> int:
    """Count number of HandStarted events in a game log."""
    count = 0
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get('event_type') == 'HandStarted':
                        count += 1
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return 0
    return count

def main():
    games_dir = Path("games/AI_games")
    if not games_dir.exists():
        print("No game logs directory found")
        return 1

    # Get all game session files
    session_files = sorted(games_dir.glob("session_*.txt"))
    if not session_files:
        print("No session files found")
        return 1

    # Get corresponding JSONL files
    jsonl_files = sorted(games_dir.glob("game_tests_*.jsonl"))

    print("=" * 70)
    print("GAME LOG VALIDATION")
    print("=" * 70)

    if jsonl_files:
        print(f"\nFound {len(jsonl_files)} game log files")
        for jsonl_file in jsonl_files:
            hand_count = count_hands_in_log(jsonl_file)
            print(f"  {jsonl_file.name}: {hand_count} hands")

    if session_files:
        print(f"\nFound {len(session_files)} session transcript files")
        for session_file in session_files[-6:]:  # Show last 6
            print(f"  {session_file.name}")

    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
