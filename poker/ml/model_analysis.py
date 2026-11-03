"""Shared model analysis utilities for extracting game statistics.

Provides reusable functions for analyzing model behavior across:
- Position-based folding
- Street-based action distribution
- Card strength-based action distribution
- Diagnostic checks
"""

import numpy as np
import numpy.typing as npt
from typing import Dict, List, Tuple


CARD_STRENGTH_NAMES = [
    "HighCard",
    "OnePair",
    "PairDraw",
    "TwoPair",
    "StraightDraw",
    "FlushDraw",
    "MadeStraightFlush",
    "Trips+",
]

PREFLOP_TIERS = ("premium", "playable", "trash")
# Hand-strength warnings on fewer decisions than this are sampling noise.
MIN_STRENGTH_DECISIONS = 50
# A push/fold chart at this stack depth folds much more often than this.
MIN_PREFLOP_FOLD = 0.15


def preflop_tier(hole: tuple) -> str:
    """Bucket two hole cards as premium, playable, or trash."""
    if len(hole) != 2:
        return "trash"
    high, low = sorted((int(hole[0].rank), int(hole[1].rank)), reverse=True)
    suited = hole[0].suit == hole[1].suit
    if (high == low and high >= 10) or (high == 14 and low >= 12):
        return "premium"
    if high == low or (high == 14 and suited):
        return "playable"
    if suited and high - low == 1 and low >= 5:
        return "playable"
    return "trash"


def extract_card_strength_from_obs(obs: npt.NDArray[np.float32]) -> int:
    """Extract card strength bucket (0-7) from observation one-hot encoding.

    Card strength is a one-hot in the fixed observation (indices 81-88).
    Legacy 155-d vectors place it just before the zero pad, so callers that
    still pass those vectors get bucket 0 unless the one-hot is in range.
    """
    if len(obs) >= 89:
        return int(np.argmax(obs[81:89]))
    return 0


class ModelStatistics:
    """Collect and analyze model behavior statistics from games."""

    def __init__(self):
        """Initialize tracking dictionaries."""
        # Position tracking - support all positions across different player counts
        position_names = [
            "Button", "SB", "BB", "UTG", "UTG+1", "UTG+2", "HJ", "CO", "MP"
        ]
        self.position_stats = {
            pos: {"folds": 0, "total": 0, "rewards": []}
            for pos in position_names
        }

        # Street tracking
        self.street_stats = {
            "preflop": {"folds": 0, "calls": 0, "raises": 0},
            "flop": {"folds": 0, "calls": 0, "raises": 0},
            "turn": {"folds": 0, "calls": 0, "raises": 0},
            "river": {"folds": 0, "calls": 0, "raises": 0},
        }

        # Card strength tracking
        self.card_stats = {
            name: {"folds": 0, "calls": 0, "raises": 0, "total": 0}
            for name in CARD_STRENGTH_NAMES
        }
        self.preflop_class_stats = {
            tier: {"folds": 0, "calls": 0, "raises": 0}
            for tier in PREFLOP_TIERS
        }
        self.preflop_position_stats = {
            pos: {"folds": 0, "calls": 0, "raises": 0}
            for pos in position_names
        }

    def record_action(
        self,
        action: int,
        street: str,
        position: str,
        card_strength_idx: int,
    ) -> None:
        """Record a single action.

        Args:
            action: 0=FOLD, 1=CHECK/CALL, 2-6=RAISE
            street: "preflop", "flop", "turn", "river"
            position: "Button", "SB", "BB", "UTG", "HJ", "CO"
            card_strength_idx: 0-7 from observation
        """
        # Track by position
        self.position_stats[position]["total"] += 1

        # Track by street
        street_key = street.lower()
        if action == 0:  # FOLD
            self.position_stats[position]["folds"] += 1
            self.street_stats[street_key]["folds"] += 1
            self.card_stats[CARD_STRENGTH_NAMES[min(card_strength_idx, 7)]]["folds"] += 1
        elif action == 1:  # CHECK/CALL
            self.street_stats[street_key]["calls"] += 1
            self.card_stats[CARD_STRENGTH_NAMES[min(card_strength_idx, 7)]]["calls"] += 1
        else:  # RAISE (2-6)
            self.street_stats[street_key]["raises"] += 1
            self.card_stats[CARD_STRENGTH_NAMES[min(card_strength_idx, 7)]]["raises"] += 1

        # Track card strength
        self.card_stats[CARD_STRENGTH_NAMES[min(card_strength_idx, 7)]]["total"] += 1

    def record_preflop(self, position: str, tier: str, action: int) -> None:
        """Record one preflop decision by seat position and hand class."""
        if position not in self.preflop_position_stats:
            self.preflop_position_stats[position] = {"folds": 0, "calls": 0, "raises": 0}
        if tier not in self.preflop_class_stats:
            self.preflop_class_stats[tier] = {"folds": 0, "calls": 0, "raises": 0}
        key = "folds" if action == 0 else "calls" if action == 1 else "raises"
        self.preflop_position_stats[position][key] += 1
        self.preflop_class_stats[tier][key] += 1

    def preflop_fold_rate(self) -> float | None:
        """Share of preflop decisions that are folds, or None if none were seen."""
        stats = self.street_stats["preflop"]
        total = stats["folds"] + stats["calls"] + stats["raises"]
        if total == 0:
            return None
        return stats["folds"] / total

    def record_reward(self, position: str, reward: float) -> None:
        """Record hand outcome reward for position analysis."""
        self.position_stats[position]["rewards"].append(reward)

    def print_position_stats(self) -> None:
        """Print fold rates and average rewards by position."""
        print("\nFOLD RATES BY POSITION:")
        print("-" * 70)
        # Print positions in order, skip if no data
        position_order = ["Button", "SB", "BB", "UTG", "UTG+1", "UTG+2", "HJ", "CO", "MP"]
        has_data = False
        for pos in position_order:
            if pos not in self.position_stats:
                continue
            stats = self.position_stats[pos]
            if stats["total"] > 0:
                has_data = True
                fold_pct = 100.0 * stats["folds"] / stats["total"]
                avg_reward = float(np.mean(stats["rewards"]))
                print(
                    f"  {pos:8s}: {fold_pct:5.1f}% folds ({stats['folds']:3d}/{stats['total']:4d}) "
                    f"| avg reward: {avg_reward:+.4f}"
                )
        if not has_data:
            print("  (No position data collected)")

    def print_street_stats(self) -> None:
        """Print action distribution by street."""
        print("\nACTION DISTRIBUTION BY STREET:")
        print("-" * 70)
        for street in ["preflop", "flop", "turn", "river"]:
            stats = self.street_stats[street]
            total = stats["folds"] + stats["calls"] + stats["raises"]
            if total > 0:
                fold_pct = 100.0 * stats["folds"] / total
                call_pct = 100.0 * stats["calls"] / total
                raise_pct = 100.0 * stats["raises"] / total
                print(
                    f"  {street.upper():8s}: "
                    f"Fold={fold_pct:5.1f}% Call={call_pct:5.1f}% Raise={raise_pct:5.1f}%"
                )

    def print_preflop_breakdown(self) -> None:
        """Print preflop fold and raise rates by position and hand class."""
        print("\nPREFLOP BY POSITION:")
        print("-" * 70)
        position_order = ["Button", "SB", "BB", "UTG", "UTG+1", "UTG+2", "HJ", "CO", "MP"]
        any_position = False
        for pos in position_order:
            stats = self.preflop_position_stats.get(pos)
            if stats is None:
                continue
            total = stats["folds"] + stats["calls"] + stats["raises"]
            if total == 0:
                continue
            any_position = True
            print(
                f"  {pos:8s}: Fold={100.0 * stats['folds'] / total:5.1f}% "
                f"Raise={100.0 * stats['raises'] / total:5.1f}% (n={total:3d})"
            )
        if not any_position:
            print("  (No preflop position data)")

        print("\nPREFLOP BY HAND CLASS:")
        print("-" * 70)
        for tier in PREFLOP_TIERS:
            stats = self.preflop_class_stats[tier]
            total = stats["folds"] + stats["calls"] + stats["raises"]
            if total == 0:
                continue
            print(
                f"  {tier:10s}: Fold={100.0 * stats['folds'] / total:5.1f}% "
                f"Raise={100.0 * stats['raises'] / total:5.1f}% (n={total:3d})"
            )

    def print_card_stats(self) -> None:
        """Print action distribution by card strength."""
        print("\nACTION DISTRIBUTION BY CARD STRENGTH:")
        print("-" * 70)
        for card_name in CARD_STRENGTH_NAMES:
            stats = self.card_stats[card_name]
            if stats["total"] > 0:
                fold_pct = 100.0 * stats["folds"] / stats["total"]
                call_pct = 100.0 * stats["calls"] / stats["total"]
                raise_pct = 100.0 * stats["raises"] / stats["total"]
                print(
                    f"  {card_name:20s}: "
                    f"Fold={fold_pct:5.1f}% Call={call_pct:5.1f}% Raise={raise_pct:5.1f}% "
                    f"(n={stats['total']:3d})"
                )

    def print_diagnosis(self) -> None:
        """Print diagnostic checks."""
        print("\nDIAGNOSIS:")
        print("-" * 70)

        # Check fold rates
        high_fold_positions = [
            (pos, stats["folds"] / max(1, stats["total"]))
            for pos, stats in self.position_stats.items()
            if stats["total"] > 0
        ]
        high_fold_positions.sort(key=lambda x: x[1], reverse=True)

        if high_fold_positions and high_fold_positions[0][1] > 0.5:
            print(
                f"  [WARN] {high_fold_positions[0][0]} folds {high_fold_positions[0][1]*100:.0f}% "
                f"— too tight"
            )
        else:
            max_fold = high_fold_positions[0][1] * 100 if high_fold_positions else 0
            print(f"  [OK] Fold rates reasonable (max: {max_fold:.0f}%)")

        # Check preflop
        preflop_total = (
            self.street_stats["preflop"]["folds"]
            + self.street_stats["preflop"]["calls"]
            + self.street_stats["preflop"]["raises"]
        )
        if preflop_total > 0:
            preflop_fold_pct = 100.0 * self.street_stats["preflop"]["folds"] / preflop_total
            if preflop_fold_pct < MIN_PREFLOP_FOLD * 100:
                print(
                    f"  [FAIL] Preflop fold % = {preflop_fold_pct:.0f}% "
                    f"— under {MIN_PREFLOP_FOLD * 100:.0f}%"
                )
            elif preflop_fold_pct > 40:
                print(f"  [WARN] Preflop fold % = {preflop_fold_pct:.0f}% — too tight preflop")
            else:
                print(f"  [OK] Preflop fold % = {preflop_fold_pct:.0f}%")

        # Check position rewards
        position_rewards = [
            (pos, np.mean(stats["rewards"]))
            for pos, stats in self.position_stats.items()
            if stats["total"] > 0 and len(stats["rewards"]) > 0
        ]
        if position_rewards:
            position_rewards.sort(key=lambda x: x[1])
            if position_rewards[0][1] < -2.0:
                print(
                    f"  [WARN] {position_rewards[0][0]} loses {abs(position_rewards[0][1]):.2f} "
                    f"BB/hand — position leak"
                )
            else:
                print(f"  [OK] Reward variance by position reasonable")

        # Check card strength strategy
        weak_hands = ["HighCard", "OnePair"]
        strong_hands = ["MadeStraightFlush", "Trips+"]

        weak_raise_pcts = [
            100.0 * (self.card_stats[h]["raises"] / max(1, self.card_stats[h]["total"]))
            for h in weak_hands
            if self.card_stats[h]["total"] > 0
        ]
        strong_raise_pcts = [
            100.0 * (self.card_stats[h]["raises"] / max(1, self.card_stats[h]["total"]))
            for h in strong_hands
            if self.card_stats[h]["total"] > 0
        ]

        weak_n = sum(self.card_stats[name]["total"] for name in weak_hands)
        strong_n = sum(self.card_stats[name]["total"] for name in strong_hands)
        if weak_raise_pcts and strong_raise_pcts and (
            weak_n < MIN_STRENGTH_DECISIONS or strong_n < MIN_STRENGTH_DECISIONS
        ):
            print(
                f"  [WARN] Hand strength sample too small "
                f"(weak n={weak_n}, strong n={strong_n}, need {MIN_STRENGTH_DECISIONS})"
            )
        elif weak_raise_pcts and strong_raise_pcts:
            weak_agg = np.mean(weak_raise_pcts)
            strong_agg = np.mean(strong_raise_pcts)
            if weak_agg > strong_agg:
                print(
                    f"  [WARN] Weak hands ({weak_agg:.0f}%) more aggressive than strong "
                    f"({strong_agg:.0f}%) — backwards"
                )
            elif strong_agg - weak_agg < 10:
                print(
                    f"  [WARN] Weak ({weak_agg:.0f}%) vs Strong ({strong_agg:.0f}%) "
                    f"— not differentiated enough"
                )
            else:
                print(
                    f"  [OK] Hand strength scaling good (Weak={weak_agg:.0f}% Raise, "
                    f"Strong={strong_agg:.0f}% Raise)"
                )

    def print_all(self) -> None:
        """Print all statistics and diagnosis."""
        self.print_position_stats()
        self.print_street_stats()
        self.print_preflop_breakdown()
        self.print_card_stats()
        self.print_diagnosis()
