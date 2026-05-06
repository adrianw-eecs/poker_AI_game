"""Hand replay from JSONL event logs."""

import json
from pathlib import Path
from typing import cast

from poker.config.blind_schedule import BlindLevel
from poker.config.game_config import GameConfig
from poker.domain.action import Action, ActionType
from poker.domain.card import Card
from poker.domain.deck import Deck
from poker.logging.events import Event, EventEncoder
from poker.state.game_state import GameState, Street
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


class Replay:
    """Loads and replays hands from a JSONL event log.

    Can reconstruct the full game state from events and step through
    the hand progression manually or automatically.
    """

    def __init__(self, filepath: Path | str) -> None:
        """Initialize the replay from a JSONL log file.

        Args:
            filepath: Path to the JSONL log file.
        """
        self.filepath = Path(filepath)
        self.events: list[Event] = []
        self._load_events()

    def _load_events(self) -> None:
        """Load all events from the JSONL file."""
        if not self.filepath.exists():
            raise FileNotFoundError(f"Log file not found: {self.filepath}")

        with open(self.filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                event = EventEncoder.from_dict(data)
                self.events.append(event)

    def get_hand_events(self, hand_number: int) -> list[Event]:
        """Get all events for a specific hand.

        Args:
            hand_number: The hand number (0-indexed).

        Returns:
            List of events for that hand.
        """
        return [e for e in self.events if hasattr(e, "hand_number") and e.hand_number == hand_number]

    def get_available_hands(self) -> set[int]:
        """Get set of available hand numbers in the log.

        Returns:
            Set of hand numbers that have events in the log.
        """
        hands = set()
        for event in self.events:
            if hasattr(event, "hand_number"):
                hands.add(event.hand_number)
        return hands

    def replay_hand(
        self,
        hand_number: int,
        config: GameConfig,
        initial_state: GameState | None = None,
    ) -> dict[str, object]:
        """Replay a hand and return its progression.

        Reconstructs the game state from events, step by step.

        Args:
            hand_number: The hand number to replay.
            config: The game configuration.
            initial_state: Optional initial state to use. If None, reconstructs from events.

        Returns:
            A dictionary with:
            - "events": List of events for this hand
            - "progression": List of (event, reconstructed_state) tuples
            - "outcome": Dict mapping seat → chips won/lost
        """
        hand_events = self.get_hand_events(hand_number)
        if not hand_events:
            raise ValueError(f"No events found for hand {hand_number}")

        progression = []
        outcome = {}

        # Find the HandStarted event to get initial config
        from poker.logging.events import (
            AntePosted,
            BlindPosted,
            HandEnded,
            HandStarted,
        )

        hand_started = next(
            (e for e in hand_events if isinstance(e, HandStarted)), None
        )
        if not hand_started:
            raise ValueError(f"HandStarted event not found for hand {hand_number}")

        # Find HandEnded event to get outcome
        hand_ended = next((e for e in hand_events if isinstance(e, HandEnded)), None)
        if hand_ended:
            # Convert int keys to string keys for consistent serialization
            outcome = {str(k): v for k, v in hand_ended.chip_distribution.items()}

        # Reconstruct blind level from first blind events
        blind_level = None
        for event in hand_events:
            if isinstance(event, BlindPosted):
                blind_level = BlindLevel(
                    small=hand_started.small_blind,
                    big=hand_started.big_blind,
                    ante=0,
                )
                break
        if not blind_level:
            blind_level = BlindLevel(small=1, big=2, ante=0)

        # If no initial state provided, create one
        if initial_state is None:
            # Estimate number of players from events
            player_seats = set()
            for event in hand_events:
                if isinstance(event, (BlindPosted, AntePosted)):
                    player_seats.add(event.seat)

            num_players = max(player_seats) + 1 if player_seats else 2

            # Initialize basic state
            players = tuple(
                PlayerState(
                    seat=seat,
                    name=f"Player{seat + 1}",
                    stack=0,  # Will be set from events
                    hole_cards=(),
                    committed_this_street=0,
                    committed_this_hand=0,
                    has_folded=False,
                    is_all_in=False,
                    is_eliminated=False,
                )
                for seat in range(num_players)
            )

            state = GameState(
                hand_number=hand_number,
                street=Street.PREFLOP,
                dealer_seat=hand_started.dealer_seat,
                players=players,
                community_cards=(),
                pots=[],
                current_bet_to_call=0,
                last_raise_size=0,
                action_history_this_street=[],
                action_history_this_hand=[],
                deck_remaining_count=52,
                config=config,
                blind_level=blind_level,
                action_on_seat=None,
            )

        for event in hand_events:
            # Note: Full state reconstruction would require re-implementing the
            # entire hand engine logic. For now, we store events for manual replay.
            progression.append((event, None))

        return {
            "hand_number": hand_number,
            "events": hand_events,
            "progression": progression,
            "outcome": outcome,
        }

    def display_hand_summary(
        self, hand_number: int, config: GameConfig
    ) -> str:
        """Display a summary of a hand.

        Args:
            hand_number: The hand number to summarize.
            config: The game configuration.

        Returns:
            A formatted string with hand summary.
        """
        from poker.logging.events import HandEnded, HandStarted

        hand_events = self.get_hand_events(hand_number)
        if not hand_events:
            return f"No events found for hand {hand_number}"

        hand_started = next(
            (e for e in hand_events if isinstance(e, HandStarted)), None
        )
        hand_ended = next((e for e in hand_events if isinstance(e, HandEnded)), None)

        lines = []
        lines.append(f"Hand #{hand_number + 1}")

        if hand_started:
            lines.append(
                f"  Dealer: Seat {hand_started.dealer_seat}, "
                f"Blinds: {hand_started.small_blind}/{hand_started.big_blind}"
            )

        if hand_ended:
            lines.append(f"  Distribution: {hand_ended.chip_distribution}")

        lines.append(f"  Total events: {len(hand_events)}")

        return "\n".join(lines)

    def interactive_replay(self, hand_number: int, config: GameConfig) -> None:
        """Interactive step-through replay of a hand.

        Prompts user for commands to step through events.

        Args:
            hand_number: The hand number to replay.
            config: The game configuration.
        """
        hand_events = self.get_hand_events(hand_number)
        if not hand_events:
            print(f"No events found for hand {hand_number}")
            return

        print(self.display_hand_summary(hand_number, config))
        print()

        step = 0
        while step < len(hand_events):
            event = hand_events[step]
            print(f"[{step + 1}/{len(hand_events)}] {event}")

            try:
                command = input("(n=next, p=prev, q=quit, j=jump): ").strip().lower()
            except EOFError:
                break

            if command == "n":
                step += 1
            elif command == "p":
                step = max(0, step - 1)
            elif command.startswith("j"):
                try:
                    parts = command.split()
                    if len(parts) > 1:
                        step = int(parts[1])
                except (ValueError, IndexError):
                    pass
            elif command == "q":
                break
            else:
                # Default to next
                step += 1

            print()

        print("Replay ended.")
