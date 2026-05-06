"""Game state snapshot."""

from dataclasses import dataclass
from enum import Enum
from typing import cast

from poker.config.blind_schedule import BlindLevel
from poker.config.game_config import GameConfig
from poker.domain.action import Action, ActionType
from poker.domain.card import Card
from poker.state.player_state import PlayerState
from poker.state.pot import Pot


class Street(Enum):
    """Poker street enumeration."""

    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"
    SHOWDOWN = "showdown"


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of the entire game state.

    Attributes:
        hand_number: The current hand number (0-indexed).
        street: The current betting street.
        dealer_seat: The seat of the dealer/button.
        players: Tuple of all players' states.
        community_cards: The community cards revealed so far.
        pots: List of current pots (main + side pots).
        current_bet_to_call: The amount a player must match to stay in.
        last_raise_size: The size of the last raise (for min-raise calculation).
        action_history_this_street: List of actions taken this street.
        action_history_this_hand: List of all actions this hand (across streets).
        deck_remaining_count: Number of cards remaining in the deck.
        config: The game configuration.
        blind_level: The current blind level.
        action_on_seat: The seat number of the player whose turn it is, or None if hand is over.
    """

    hand_number: int
    street: Street
    dealer_seat: int
    players: tuple[PlayerState, ...]
    community_cards: tuple[Card, ...]
    pots: list[Pot]
    current_bet_to_call: int
    last_raise_size: int
    action_history_this_street: list[tuple[int, Action]]  # (seat, action) pairs
    action_history_this_hand: list[tuple[int, Action]]  # (seat, action) pairs
    deck_remaining_count: int
    config: GameConfig
    blind_level: BlindLevel
    action_on_seat: int | None

    def view_for(self, seat: int) -> "GameState":
        """Return a state with hole cards hidden from other seats.

        Used to provide bots with a view that doesn't reveal opponent cards.

        Args:
            seat: The seat number viewing the state.

        Returns:
            A new GameState with other players' hole cards stripped.
        """
        # Strip hole cards from all seats except the one viewing
        masked_players = tuple(
            player.with_hole_cards(()) if player.seat != seat else player
            for player in self.players
        )

        return GameState(
            hand_number=self.hand_number,
            street=self.street,
            dealer_seat=self.dealer_seat,
            players=masked_players,
            community_cards=self.community_cards,
            pots=self.pots,
            current_bet_to_call=self.current_bet_to_call,
            last_raise_size=self.last_raise_size,
            action_history_this_street=self.action_history_this_street,
            action_history_this_hand=self.action_history_this_hand,
            deck_remaining_count=self.deck_remaining_count,
            config=self.config,
            blind_level=self.blind_level,
            action_on_seat=self.action_on_seat,
        )

    def to_dict(self) -> dict[str, object]:
        """Convert state to a dictionary (JSON-safe).

        Returns:
            A dictionary representation of the state.
        """
        return {
            "hand_number": self.hand_number,
            "street": self.street.value,
            "dealer_seat": self.dealer_seat,
            "players": [
                {
                    "seat": p.seat,
                    "name": p.name,
                    "stack": p.stack,
                    "hole_cards": [str(c) for c in p.hole_cards],
                    "committed_this_street": p.committed_this_street,
                    "committed_this_hand": p.committed_this_hand,
                    "has_folded": p.has_folded,
                    "is_all_in": p.is_all_in,
                    "is_eliminated": p.is_eliminated,
                }
                for p in self.players
            ],
            "community_cards": [str(c) for c in self.community_cards],
            "pots": [
                {
                    "amount": p.amount,
                    "eligible_seats": sorted(p.eligible_seats),
                }
                for p in self.pots
            ],
            "current_bet_to_call": self.current_bet_to_call,
            "last_raise_size": self.last_raise_size,
            "action_history_this_street": [
                (seat, action.type.value, action.amount)
                for seat, action in self.action_history_this_street
            ],
            "action_history_this_hand": [
                (seat, action.type.value, action.amount)
                for seat, action in self.action_history_this_hand
            ],
            "deck_remaining_count": self.deck_remaining_count,
            "action_on_seat": self.action_on_seat,
            # config and blind_level handled separately by caller
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        config: GameConfig,
        blind_level: BlindLevel,
    ) -> "GameState":
        """Create state from a dictionary.

        Args:
            data: The dictionary representation.
            config: The game configuration.
            blind_level: The current blind level.

        Returns:
            A GameState instance.
        """
        # Import here to avoid circular dependency
        from poker.domain.card import Card

        # Reconstruct players
        players_data = cast(list[dict[str, object]], data["players"])
        players = tuple(
            PlayerState(
                seat=cast(int, p["seat"]),
                name=cast(str, p["name"]),
                stack=cast(int, p["stack"]),
                hole_cards=tuple(
                    Card.from_string(card_str) for card_str in cast(list[str], p["hole_cards"])
                ),
                committed_this_street=cast(int, p["committed_this_street"]),
                committed_this_hand=cast(int, p["committed_this_hand"]),
                has_folded=cast(bool, p["has_folded"]),
                is_all_in=cast(bool, p["is_all_in"]),
                is_eliminated=cast(bool, p["is_eliminated"]),
            )
            for p in players_data
        )

        # Reconstruct community cards
        community_cards = tuple(
            Card.from_string(card_str) for card_str in cast(list[str], data["community_cards"])
        )

        # Reconstruct pots
        pots_data = cast(list[dict[str, object]], data["pots"])
        pots = [
            Pot(
                amount=cast(int, p["amount"]),
                eligible_seats=frozenset(cast(list[int], p["eligible_seats"])),
            )
            for p in pots_data
        ]

        # Reconstruct actions
        action_history_this_street = [
            (seat, Action(type=_action_type_from_str(action_type), amount=amount))
            for seat, action_type, amount in cast(list[tuple[int, str, int]], data["action_history_this_street"])
        ]

        action_history_this_hand = [
            (seat, Action(type=_action_type_from_str(action_type), amount=amount))
            for seat, action_type, amount in cast(list[tuple[int, str, int]], data["action_history_this_hand"])
        ]

        return cls(
            hand_number=cast(int, data["hand_number"]),
            street=Street(cast(str, data["street"])),
            dealer_seat=cast(int, data["dealer_seat"]),
            players=players,
            community_cards=community_cards,
            pots=pots,
            current_bet_to_call=cast(int, data["current_bet_to_call"]),
            last_raise_size=cast(int, data["last_raise_size"]),
            action_history_this_street=action_history_this_street,
            action_history_this_hand=action_history_this_hand,
            deck_remaining_count=cast(int, data["deck_remaining_count"]),
            config=config,
            blind_level=blind_level,
            action_on_seat=cast(int | None, data["action_on_seat"]),
        )


def _action_type_from_str(s: str) -> ActionType:
    """Convert string representation back to ActionType."""
    return ActionType(s)
