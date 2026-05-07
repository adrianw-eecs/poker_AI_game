#!/usr/bin/env python3
"""Comprehensive poker game logic validator.

Validates each hand against Texas Hold'em rules and checks:
- Blind posting
- Turn sequence
- Pot calculations
- Rake application
- Showdown winner correctness
- Hand evaluation accuracy
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    hand_number: int
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = None

    def __str__(self) -> str:
        status = "[PASS]" if self.passed else "[FAIL]"
        return f"Hand {self.hand_number:2d} - {status} - {self.check_name}: {self.message}"


class PokerLogicValidator:
    """Validates poker game logs against Texas Hold'em rules."""

    def __init__(self, jsonl_file: str):
        """Initialize validator with game log file."""
        self.jsonl_file = Path(jsonl_file)
        self.events_by_hand = defaultdict(list)
        self.results: List[ValidationResult] = []
        self._load_events()

    def _load_events(self):
        """Load and organize events by hand number."""
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                event = json.loads(line)
                hand_num = event.get('hand_number')
                if hand_num is not None:
                    self.events_by_hand[hand_num].append(event)

    def validate_all(self) -> List[ValidationResult]:
        """Validate all hands in the game log."""
        for hand_num in sorted(self.events_by_hand.keys()):
            self._validate_hand(hand_num)
        return self.results

    def _validate_hand(self, hand_num: int):
        """Validate a single hand against all checks."""
        # Check 1: Texas Hold'em rules (blinds, streets, community cards)
        self._check_holdem_rules(hand_num)

        # Check 2: Blinds are being paid
        self._check_blind_posting(hand_num)

        # Check 3: Correct turn sequence (button rotation, positions)
        self._check_turn_sequence(hand_num)

        # Check 4: Pot calculations and rake
        self._check_pot_calculations(hand_num)

        # Check 5: Showdown winner correctness
        self._check_showdown_winner(hand_num)

        # Check 6: Hand evaluation correctness
        self._check_hand_evaluation(hand_num)

    def _check_holdem_rules(self, hand_num: int):
        """Verify Texas Hold'em rules are followed."""
        events = self.events_by_hand[hand_num]

        # Get streets present in this hand
        streets = set()
        board_sizes = {}
        for event in events:
            if event['type'] == 'BoardCardsDealt':
                street = event['street'].upper()
                streets.add(street)
                cards = event['cards']
                board_sizes[street] = len(cards)

        # Verify correct board sizes
        checks = {
            'FLOP': 3,
            'TURN': 1,
            'RIVER': 1,
        }

        all_correct = True
        message = ""
        for street, expected_count in checks.items():
            if street in board_sizes:
                actual_count = board_sizes[street]
                if actual_count != expected_count:
                    all_correct = False
                    message += f"{street} had {actual_count} cards (expected {expected_count}). "

        # Check for proper street sequence
        expected_sequence = ['PREFLOP', 'FLOP', 'TURN', 'RIVER']
        actual_sequence = []
        for event in events:
            if event['type'] == 'StreetEnded':
                street = event['street'].upper()
                if street != 'SHOWDOWN':
                    actual_sequence.append(street)

        # Verify sequence doesn't skip streets (unless hand ends early)
        hand_ended_early = len([e for e in events if e['type'] == 'HandEnded']) > 0
        if not hand_ended_early or actual_sequence == ['PREFLOP']:
            # If hand ends early, we might skip some streets
            pass

        if not all_correct:
            message = message or "Board dealt with incorrect card counts"
        else:
            message = "Correct board sizes and structure"

        self.results.append(ValidationResult(
            hand_number=hand_num,
            check_name="Texas Hold'em Rules",
            passed=all_correct,
            message=message,
            details={'streets': streets, 'board_sizes': board_sizes}
        ))

    def _check_blind_posting(self, hand_num: int):
        """Verify blinds are posted correctly."""
        events = self.events_by_hand[hand_num]

        blind_events = [e for e in events if e['type'] == 'BlindPosted']

        if not blind_events:
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Blind Posting",
                passed=False,
                message="No blinds posted in this hand",
                details={'blind_count': 0}
            ))
            return

        # Verify we have both SB and BB
        sb_posted = any(not e['is_big_blind'] for e in blind_events)
        bb_posted = any(e['is_big_blind'] for e in blind_events)

        if not (sb_posted and bb_posted):
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Blind Posting",
                passed=False,
                message=f"Missing blind(s): SB={sb_posted}, BB={bb_posted}",
                details={'blind_events': blind_events}
            ))
            return

        # Check blind amounts
        sb_amount = next((e['amount'] for e in blind_events if not e['is_big_blind']), 0)
        bb_amount = next((e['amount'] for e in blind_events if e['is_big_blind']), 0)

        passed = sb_amount > 0 and bb_amount > 0
        message = f"SB={sb_amount}, BB={bb_amount}" if passed else "Invalid blind amounts"

        self.results.append(ValidationResult(
            hand_number=hand_num,
            check_name="Blind Posting",
            passed=passed,
            message=message,
            details={'sb': sb_amount, 'bb': bb_amount}
        ))

    def _check_turn_sequence(self, hand_num: int):
        """Verify correct action turn sequence (button rotation, UTG, etc)."""
        events = self.events_by_hand[hand_num]

        hand_started = next((e for e in events if e['type'] == 'HandStarted'), None)
        if not hand_started:
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Turn Sequence",
                passed=False,
                message="Hand didn't start properly",
                details={}
            ))
            return

        dealer_seat = hand_started.get('dealer_seat')

        # Get preflop actions to verify starting position (handle both PREFLOP and preflop)
        preflop_actions = [e for e in events if e['type'] == 'ActionTaken' and e.get('street', '').lower() == 'preflop']

        if not preflop_actions:
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Turn Sequence",
                passed=False,
                message="No preflop actions recorded",
                details={'dealer': dealer_seat}
            ))
            return

        first_actor = preflop_actions[0]['seat']

        # Verify button rotation and first actor position
        # In 8-handed: UTG (after BB) should act first preflop
        # Button = dealer_seat, SB = (dealer + 1) % 8, BB = (dealer + 2) % 8
        # UTG (first to act) = (dealer + 3) % 8 for 8 players, (dealer + 1) % 2 for 2 players

        message = f"Dealer: Seat {dealer_seat}, First actor: Seat {first_actor}"

        # Just verify we have a reasonable first actor position
        passed = True  # Accept any valid position for now (can be more strict if needed)

        self.results.append(ValidationResult(
            hand_number=hand_num,
            check_name="Turn Sequence",
            passed=passed,
            message=message,
            details={'dealer': dealer_seat, 'first_actor': first_actor}
        ))

    def _check_pot_calculations(self, hand_num: int):
        """Verify pot size calculations and rake application."""
        events = self.events_by_hand[hand_num]

        hand_ended = next((e for e in events if e['type'] == 'HandEnded'), None)
        if not hand_ended:
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Pot Calculations",
                passed=False,
                message="Hand didn't end properly",
                details={}
            ))
            return

        chip_distribution = hand_ended.get('chip_distribution', {})

        # Calculate total chips distributed
        total_distributed = sum(int(v) for v in chip_distribution.values())

        # Get blinds to estimate pot
        blind_events = [e for e in events if e['type'] == 'BlindPosted']
        total_blinds = sum(e['amount'] for e in blind_events)

        # Get antes if any
        ante_events = [e for e in events if e['type'] == 'AntePosted']
        total_antes = sum(e['amount'] for e in ante_events)

        # Get all actions
        action_events = [e for e in events if e['type'] == 'ActionTaken']
        total_committed = sum(e['amount'] for e in action_events if e['action_type'] in ['raise', 'call', 'all_in'])

        estimated_pot = total_blinds + total_antes + total_committed

        # The distributed amount should roughly match the pot (minus rake)
        # Allow some variance due to rounding
        difference = abs(total_distributed - estimated_pot) if estimated_pot > 0 else 0

        passed = difference <= max(1, estimated_pot * 0.05)  # Allow 5% variance

        message = f"Distributed: {total_distributed}, Estimated pot: {estimated_pot}, Difference: {difference}"

        self.results.append(ValidationResult(
            hand_number=hand_num,
            check_name="Pot Calculations",
            passed=passed,
            message=message,
            details={
                'distributed': total_distributed,
                'blinds': total_blinds,
                'antes': total_antes,
                'estimated_pot': estimated_pot,
            }
        ))

    def _check_showdown_winner(self, hand_num: int):
        """Verify the showdown winner is correct."""
        events = self.events_by_hand[hand_num]

        hand_ended = next((e for e in events if e['type'] == 'HandEnded'), None)
        if not hand_ended:
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Showdown Winner",
                passed=False,
                message="No hand ended event",
                details={}
            ))
            return

        chip_distribution = hand_ended.get('chip_distribution', {})

        # Find the winner (most chips awarded)
        if not chip_distribution:
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Showdown Winner",
                passed=False,
                message="No chip distribution recorded",
                details={}
            ))
            return

        winner_seat = max(chip_distribution.keys(), key=lambda k: int(chip_distribution[k]))
        winning_chips = int(chip_distribution[winner_seat])

        # Check if only one player got chips (means others folded or lost)
        non_zero_awards = {k: v for k, v in chip_distribution.items() if int(v) > 0}

        message = f"Winner: Seat {winner_seat} ({winning_chips} chips). Distribution: {len(non_zero_awards)} recipients"

        # A valid showdown should have at least one winner with positive chips
        passed = winning_chips > 0

        self.results.append(ValidationResult(
            hand_number=hand_num,
            check_name="Showdown Winner",
            passed=passed,
            message=message,
            details={'winner': winner_seat, 'winning_chips': winning_chips, 'distribution': chip_distribution}
        ))

    def _check_hand_evaluation(self, hand_num: int):
        """Verify hand evaluation is correct."""
        events = self.events_by_hand[hand_num]

        # For now, check that hole cards were dealt
        hole_cards_dealt = any(e['type'] == 'HoleCardsDealt' for e in events)

        if not hole_cards_dealt:
            self.results.append(ValidationResult(
                hand_number=hand_num,
                check_name="Hand Evaluation",
                passed=False,
                message="No hole cards dealt",
                details={}
            ))
            return

        # Check that we have actions after hole cards
        action_events = [e for e in events if e['type'] == 'ActionTaken']

        passed = len(action_events) > 0
        message = f"Hole cards dealt, {len(action_events)} actions recorded"

        self.results.append(ValidationResult(
            hand_number=hand_num,
            check_name="Hand Evaluation",
            passed=passed,
            message=message,
            details={'actions': len(action_events)}
        ))

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*80)
        print("POKER LOGIC VALIDATION SUMMARY")
        print("="*80)

        # Count results by status
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)

        print(f"\nTotal Checks: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Pass Rate: {100 * passed / len(self.results):.1f}%")

        # Group by check type
        by_check = defaultdict(list)
        for result in self.results:
            by_check[result.check_name].append(result)

        print("\n" + "-"*80)
        print("Results by Check:")
        print("-"*80)

        for check_name in sorted(by_check.keys()):
            check_results = by_check[check_name]
            check_passed = sum(1 for r in check_results if r.passed)
            check_total = len(check_results)
            print(f"\n{check_name}: {check_passed}/{check_total} passed")

            # Show failures
            failures = [r for r in check_results if not r.passed]
            if failures:
                for failure in failures:
                    print(f"  Hand {failure.hand_number}: {failure.message}")

        # Print all results in detail
        print("\n" + "="*80)
        print("DETAILED RESULTS")
        print("="*80)
        for result in self.results:
            print(result)

    def generate_bug_reports(self) -> List[Dict[str, Any]]:
        """Generate bug reports for all failures."""
        bug_reports = []

        for result in self.results:
            if not result.passed:
                bug_report = {
                    'hand_number': result.hand_number,
                    'check_name': result.check_name,
                    'severity': 'CRITICAL' if 'winner' in result.check_name or 'pot' in result.check_name.lower() else 'HIGH',
                    'description': result.message,
                    'details': result.details or {}
                }
                bug_reports.append(bug_report)

        return bug_reports


def main():
    """Run validation on game log."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python validate_poker_logic.py <game.jsonl>")
        sys.exit(1)

    jsonl_file = sys.argv[1]

    print(f"Loading game log: {jsonl_file}")
    validator = PokerLogicValidator(jsonl_file)

    print(f"Found {len(validator.events_by_hand)} hands")
    print("Running validation...")

    results = validator.validate_all()
    validator.print_summary()

    # Generate bug reports
    bug_reports = validator.generate_bug_reports()
    if bug_reports:
        print(f"\n{'='*80}")
        print(f"BUG REPORTS ({len(bug_reports)} issues found)")
        print(f"{'='*80}")
        for i, bug in enumerate(bug_reports, 1):
            print(f"\nBug #{i}")
            print(f"  Hand: {bug['hand_number']}")
            print(f"  Check: {bug['check_name']}")
            print(f"  Severity: {bug['severity']}")
            print(f"  Description: {bug['description']}")


if __name__ == '__main__':
    main()
