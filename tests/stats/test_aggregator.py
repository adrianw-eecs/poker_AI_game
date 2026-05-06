"""Tests for the stats aggregator.

Uses hand-crafted JSONL logs with known events to verify that the computed
per-seat statistics match expected values exactly.
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from poker.stats.aggregator import BotStats, aggregate_from_log

# ---------------------------------------------------------------------------
# Helpers for building test JSONL logs
# ---------------------------------------------------------------------------


def _write_log(lines: list[dict[str, object]], path: Path) -> None:
    """Write a list of event dicts as a JSONL file."""
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def _hand_started(hand: int, dealer: int = 0, sb: int = 5, bb: int = 10) -> dict[str, object]:
    return {"type": "HandStarted", "hand_number": hand, "dealer_seat": dealer,
            "small_blind": sb, "big_blind": bb}


def _blind_posted(hand: int, seat: int, amount: int, is_big: bool) -> dict[str, object]:
    return {"type": "BlindPosted", "hand_number": hand, "seat": seat,
            "amount": amount, "is_big_blind": is_big}


def _action(hand: int, street: str, seat: int, action_type: str, amount: int = 0) -> dict[str, object]:
    return {"type": "ActionTaken", "hand_number": hand, "street": street,
            "seat": seat, "action_type": action_type, "amount": amount}


def _hand_ended(hand: int, distribution: dict[int, int]) -> dict[str, object]:
    return {"type": "HandEnded", "hand_number": hand,
            "chip_distribution": {str(k): v for k, v in distribution.items()}}


# ---------------------------------------------------------------------------
# Two-hand scenario with known statistics
#
# Hand 0:
#   Seat 0 (SB) raises preflop to 30.  (VPIP + PFR for seat 0)
#   Seat 1 (BB) calls 30.              (VPIP for seat 1)
#   Flop: seat 1 checks; seat 0 raises 50 (aggression).
#   Seat 1 folds.
#   Seat 0 wins: chip_distribution = {0: +30, 1: -30}
#
# Hand 1:
#   Seat 1 (SB) raises preflop to 30.  (VPIP + PFR for seat 1)
#   Seat 0 (BB) folds.                  (no VPIP for seat 0)
#   Seat 1 wins: chip_distribution = {0: -10, 1: +10}
#
# Expected stats:
#   Seat 0: played=2, won=1, vpip=1/2, pfr=1/2, agg=2, pas=0, af=inf,
#           chips_won=30, delta=+20, avg_won=30.0
#   Seat 1: played=2, won=1, vpip=2/2, pfr=1/2, agg=1, pas=1, af=1.0,
#           chips_won=10, delta=-20, avg_won=10.0
# ---------------------------------------------------------------------------

TWO_HAND_LOG: list[dict[str, object]] = [
    # --- Hand 0 ---
    _hand_started(0, dealer=5, sb=5, bb=10),
    _blind_posted(0, 0, 5, is_big=False),
    _blind_posted(0, 1, 10, is_big=True),
    _action(0, "PREFLOP", 0, "RAISE", 30),   # seat 0: VPIP + PFR, aggressive
    _action(0, "PREFLOP", 1, "CALL", 30),    # seat 1: VPIP, passive
    _action(0, "FLOP", 1, "CHECK", 0),
    _action(0, "FLOP", 0, "RAISE", 50),     # seat 0: aggressive (post-flop)
    _action(0, "FLOP", 1, "FOLD", 0),
    _hand_ended(0, {0: 30, 1: -30}),

    # --- Hand 1 ---
    _hand_started(1, dealer=0, sb=5, bb=10),
    _blind_posted(1, 1, 5, is_big=False),
    _blind_posted(1, 0, 10, is_big=True),
    _action(1, "PREFLOP", 1, "RAISE", 30),   # seat 1: VPIP + PFR, aggressive
    _action(1, "PREFLOP", 0, "FOLD", 0),     # seat 0: no VPIP
    _hand_ended(1, {0: -10, 1: 10}),
]


@pytest.mark.smoke
def test_aggregator_two_hand_scenario() -> None:
    """Verify stats computation on two-hand scenario."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "game.jsonl"
        _write_log(TWO_HAND_LOG, path)
        stats = aggregate_from_log(path)

    assert 0 in stats and 1 in stats
    s0, s1 = stats[0], stats[1]

    assert s0.hands_played == 2 and s1.hands_played == 2
    assert s0.hands_won == 1 and s1.hands_won == 1
    assert s0.vpip == pytest.approx(0.5) and s1.vpip == pytest.approx(1.0)
    assert s0.pfr == pytest.approx(0.5) and s1.pfr == pytest.approx(0.5)
    assert math.isinf(s0.af) and s1.af == pytest.approx(1.0)
    assert s0.total_chips_delta == 20 and s1.total_chips_delta == -20


class TestEdgeCases:
    """Test edge cases for the aggregator."""

    def test_empty_log(self) -> None:
        """Verify empty log returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            path.write_text("")
            result = aggregate_from_log(path)
        assert result == {}

    def test_no_wins(self) -> None:
        """Verify avg_pot_won is 0.0 when no hands were won."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _action(0, "PREFLOP", 0, "FOLD"),
            _hand_ended(0, {0: -10, 1: 10}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "no_wins.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert 0 in result
        assert result[0].avg_pot_won == 0.0
        assert result[0].hands_won == 0
        assert result[0].win_rate == 0.0

    def test_always_aggressive_af_is_infinite(self) -> None:
        """Verify AF is inf when player never called."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _action(0, "PREFLOP", 0, "RAISE", 30),
            _action(0, "PREFLOP", 1, "FOLD"),
            _hand_ended(0, {0: 10, 1: -10}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "aggressive.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert 0 in result
        assert math.isinf(result[0].af)

    def test_never_aggressive_af_is_zero(self) -> None:
        """Verify AF is 0.0 when player never raised or called."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _action(0, "PREFLOP", 0, "FOLD"),
            _hand_ended(0, {0: -10, 1: 10}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "passive.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert 0 in result
        assert result[0].af == 0.0

    def test_vpip_does_not_count_folds(self) -> None:
        """Verify folding preflop does not count as VPIP."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _action(0, "PREFLOP", 0, "FOLD"),
            _hand_ended(0, {0: -10, 1: 10}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fold.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert result[0].vpip_count == 0
        assert result[0].vpip == 0.0

    def test_vpip_does_not_count_postflop_actions(self) -> None:
        """Verify post-flop actions do not count toward VPIP."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _action(0, "PREFLOP", 0, "FOLD"),
            _action(0, "FLOP", 0, "CALL", 20),   # shouldn't count for VPIP
            _hand_ended(0, {0: -10, 1: 10}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "postflop.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert result[0].vpip_count == 0

    def test_all_in_counts_as_vpip_and_pfr(self) -> None:
        """Verify ALL_IN preflop counts for both VPIP and PFR."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _action(0, "PREFLOP", 0, "ALL_IN", 1000),
            _hand_ended(0, {0: 500, 1: -500}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "allin.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert result[0].vpip_count == 1
        assert result[0].pfr_count == 1

    def test_multiple_hands_chips_delta(self) -> None:
        """Verify total_chips_delta sums correctly across hands."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _hand_ended(0, {0: 100, 1: -100}),
            _hand_started(1),
            _hand_ended(1, {0: -50, 1: 50}),
            _hand_started(2),
            _hand_ended(2, {0: 25, 1: -25}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert result[0].total_chips_delta == 75   # 100 - 50 + 25
        assert result[1].total_chips_delta == -75  # -100 + 50 - 25
        assert result[0].hands_played == 3
        assert result[0].hands_won == 2  # hands 0 and 2

    def test_seat_number_stored_in_stats(self) -> None:
        """Verify the seat field is correctly stored."""
        events: list[dict[str, object]] = [
            _hand_started(0),
            _hand_ended(0, {3: 50, 5: -50}),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seats.jsonl"
            _write_log(events, path)
            result = aggregate_from_log(path)
        assert result[3].seat == 3
        assert result[5].seat == 5
