"""Analyze game logs to compute bot statistics."""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from collections import defaultdict


class GameStatsAnalyzer:
    """Analyze game JSONL logs to compute bot statistics."""

    def __init__(self, jsonl_path: Path, seat_to_bot: Optional[Dict[int, str]] = None, final_stacks: Optional[Dict[int, int]] = None):
        """
        Initialize analyzer.

        Args:
            jsonl_path: Path to game log JSONL file
            seat_to_bot: Mapping of seat numbers to bot names {0: "NFSPBot", ...}
            final_stacks: Final stacks for each seat {0: 1234, ...}
        """
        self.jsonl_path = jsonl_path
        self.seat_to_bot = seat_to_bot or {}
        self.final_stacks = final_stacks or {}

    def analyze(self) -> Dict[str, Dict[str, Any]]:
        """Read JSONL log and compute statistics."""
        pot_winners = defaultdict(int)  # Track pot wins per seat
        hand_count = 0

        with open(self.jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                # Count hands and track pot winners
                if event_type == "HandEnded":
                    hand_count += 1
                    chip_dist = event.get("chip_distribution", {})

                    # Find who won the pot (non-zero entries in chip_distribution)
                    for seat_str, chips_awarded in chip_dist.items():
                        seat = int(seat_str)
                        if chips_awarded > 0:
                            pot_winners[seat] += 1

        # Build final statistics
        stats_final = {}
        for seat in range(4):
            bot_name = self.seat_to_bot.get(seat, f"Player{seat+1}")
            final_stack = self.final_stacks.get(seat, 1000)
            starting_stack = 1000
            net_change = final_stack - starting_stack

            stats_final[bot_name] = {
                "pot_wins": pot_winners.get(seat, 0),
                "stack_gains": max(0, net_change),  # Positive change
                "stack_losses": abs(min(0, net_change)),  # Absolute value of negative change
                "net_change": net_change,
                "hands_played": hand_count,
            }

        return stats_final

    def get_summary(self) -> str:
        """Return formatted summary of game statistics."""
        stats = self.analyze()

        if not stats:
            return "No statistics available"

        lines = []
        lines.append("\n" + "=" * 90)
        lines.append("DETAILED GAME STATISTICS")
        lines.append("=" * 90)
        lines.append(
            f"{'Player':<20} {'Pots Won':>10} {'Stack Wins':>12} "
            f"{'Stack Loss':>12} {'Net Change':>12}"
        )
        lines.append("-" * 90)

        total_pots = 0
        total_gains = 0
        total_losses = 0
        total_net = 0

        for player_name in sorted(stats.keys()):
            stat = stats[player_name]
            pot_wins = stat["pot_wins"]
            gains = stat["stack_gains"]
            losses = stat["stack_losses"]
            net = stat["net_change"]

            total_pots += pot_wins
            total_gains += gains
            total_losses += losses
            total_net += net

            net_str = f"+{net}" if net > 0 else str(net)
            lines.append(
                f"{player_name:<20} {pot_wins:>10} {gains:>12} "
                f"{losses:>12} {net_str:>12}"
            )

        lines.append("-" * 90)
        lines.append(
            f"{'TOTAL':<20} {total_pots:>10} {total_gains:>12} "
            f"{total_losses:>12} {total_net:>12}"
        )
        lines.append("=" * 90)

        return "\n".join(lines)
