#!/usr/bin/env python3
"""Validate a poker game log for consistency issues."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def validate_log(log_file: str) -> None:
    """Parse and validate a poker game log.

    Args:
        log_file: Path to the JSONL log file.
    """
    issues = []
    events = []

    try:
        with open(log_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    issues.append(f"JSON parse error: {e}")
                    continue
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        return

    if not events:
        print("No events found in log")
        return

    # Validate event sequence
    hand_starts = 0
    hand_ends = 0
    actions_per_hand = defaultdict(int)
    current_hand = None

    for event in events:
        event_type = event.get("type")

        if event_type == "HandStarted":
            hand_starts += 1
            current_hand = event.get("hand_number")
        elif event_type == "HandEnded":
            hand_ends += 1
            if current_hand is not None:
                if actions_per_hand[current_hand] == 0:
                    issues.append(f"Hand {current_hand}: No actions taken")
                current_hand = None
        elif event_type == "ActionTaken":
            if current_hand is not None:
                actions_per_hand[current_hand] += 1

    # Validate counts
    if hand_starts != hand_ends:
        issues.append(f"Hand mismatch: {hand_starts} starts, {hand_ends} ends")

    if hand_starts == 0:
        issues.append("No hands recorded in log")

    # Check for timing events
    timing_events = [e for e in events if e.get("type") == "TimingEvent"]
    if timing_events:
        print(f"Timing events found: {len(timing_events)}")
    else:
        print("Warning: No timing events in log")

    # Print summary
    print("\n" + "=" * 70)
    print(f"GAME LOG VALIDATION: {log_file}")
    print("=" * 70)
    print(f"Total events: {len(events)}")
    print(f"Hands recorded: {hand_starts}")

    if issues:
        print(f"\nValidation Issues ({len(issues)}):")
        for issue in issues:
            print(f"  [FAIL] {issue}")
    else:
        print("\n[PASS] All validations passed!")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_game.py <log_file>")
        sys.exit(1)

    validate_log(sys.argv[1])
