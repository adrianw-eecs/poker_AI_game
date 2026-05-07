#!/usr/bin/env python3
"""Analyze timing events from a poker session log."""

import json
import sys
from collections import defaultdict
from pathlib import Path


def analyze_log(log_file: str) -> None:
    """Parse JSONL log and aggregate timing events.

    Args:
        log_file: Path to the JSONL log file.
    """
    timing_by_phase = defaultdict(list)
    total_session_time = 0
    total_hands = 0

    try:
        with open(log_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "TimingEvent":
                    phase = event["phase"]
                    elapsed = event["elapsed_us"]
                    notes = event.get("notes", "")

                    # Create phase key with notes if present
                    phase_key = f"{phase}:{notes}" if notes else phase
                    timing_by_phase[phase_key].append(elapsed)

                    # Track total session time
                    if phase == "session_total":
                        total_session_time = elapsed

                    # Count hands
                    if phase == "hand_complete":
                        total_hands += 1
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        return

    if not timing_by_phase:
        print("No timing events found in log")
        return

    # Print header
    print("\n" + "=" * 80)
    print(f"TIMING ANALYSIS: {log_file}")
    print("=" * 80)
    print(f"Total hands: {total_hands}")
    print(f"Total session time: {total_session_time / 1_000_000:.3f} seconds")
    if total_hands > 0:
        print(f"Average hand time: {total_session_time / total_hands / 1_000:.1f} ms")
    print("\n" + "-" * 80)
    print(f"{'Phase':<30} {'Count':>6} {'Total (us)':>12} {'Avg (us)':>12} {'% Total':>10}")
    print("-" * 80)

    # Sort by total time (descending)
    sorted_phases = sorted(
        timing_by_phase.items(),
        key=lambda x: sum(x[1]),
        reverse=True
    )

    for phase, times in sorted_phases:
        count = len(times)
        total = sum(times)
        avg = total / count if count > 0 else 0
        pct = (total / total_session_time * 100) if total_session_time > 0 else 0

        print(f"{phase:<30} {count:>6} {total:>12} {avg:>12.1f} {pct:>9.1f}%")

    print("-" * 80)
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_timing.py <log_file>")
        sys.exit(1)

    analyze_log(sys.argv[1])
