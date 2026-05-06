"""Event types for game logging and replay."""

from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class HandStarted:
    """Emitted when a new hand begins."""

    hand_number: int
    dealer_seat: int
    small_blind: int
    big_blind: int


@dataclass(frozen=True)
class BlindPosted:
    """Emitted when a blind is posted."""

    hand_number: int
    seat: int
    amount: int
    is_big_blind: bool


@dataclass(frozen=True)
class AntePosted:
    """Emitted when an ante is posted."""

    hand_number: int
    seat: int
    amount: int


@dataclass(frozen=True)
class HoleCardsDealt:
    """Emitted when hole cards are dealt."""

    hand_number: int
    num_players: int


@dataclass(frozen=True)
class BoardCardsDealt:
    """Emitted when community cards are dealt."""

    hand_number: int
    street: str  # "FLOP", "TURN", or "RIVER"
    cards: tuple[str, ...]  # Card strings like "As", "Kh"


@dataclass(frozen=True)
class ActionTaken:
    """Emitted when a player takes an action."""

    hand_number: int
    street: str
    seat: int
    action_type: str  # "FOLD", "CHECK", "CALL", "RAISE", "ALL_IN"
    amount: int


@dataclass(frozen=True)
class PotsBuilt:
    """Emitted when pots are computed."""

    hand_number: int
    street: str
    pots: tuple[tuple[int, tuple[int, ...]], ...]  # (amount, eligible_seats)


@dataclass(frozen=True)
class Showdown:
    """Emitted at showdown with hand results."""

    hand_number: int
    hands: dict[int, tuple[int, tuple[int, ...]]]  # seat -> (type_value, kickers)


@dataclass(frozen=True)
class HandEnded:
    """Emitted when a hand finishes."""

    hand_number: int
    chip_distribution: dict[int, int]  # seat -> chips won/lost


@dataclass(frozen=True)
class StreetEnded:
    """Emitted when a street ends with a full-state snapshot.

    This event is intended for "human-readable" session transcripts and
    for ML/offline analysis where periodic full-state snapshots are useful.
    """

    hand_number: int
    street: str  # "PREFLOP", "FLOP", "TURN", "RIVER", "SHOWDOWN"
    snapshot: str  # human-readable formatted GameState snapshot


# Discriminated union of all event types
Event = (
    HandStarted
    | BlindPosted
    | AntePosted
    | HoleCardsDealt
    | BoardCardsDealt
    | ActionTaken
    | PotsBuilt
    | Showdown
    | HandEnded
    | StreetEnded
)


class EventEncoder:
    """Encodes and decodes events to/from JSON-compatible dicts."""

    @staticmethod
    def to_dict(event: Event) -> dict[str, object]:
        """Convert an event to a JSON-compatible dict."""
        if isinstance(event, HandStarted):
            return {
                "type": "HandStarted",
                "hand_number": event.hand_number,
                "dealer_seat": event.dealer_seat,
                "small_blind": event.small_blind,
                "big_blind": event.big_blind,
            }
        elif isinstance(event, BlindPosted):
            return {
                "type": "BlindPosted",
                "hand_number": event.hand_number,
                "seat": event.seat,
                "amount": event.amount,
                "is_big_blind": event.is_big_blind,
            }
        elif isinstance(event, AntePosted):
            return {
                "type": "AntePosted",
                "hand_number": event.hand_number,
                "seat": event.seat,
                "amount": event.amount,
            }
        elif isinstance(event, HoleCardsDealt):
            return {
                "type": "HoleCardsDealt",
                "hand_number": event.hand_number,
                "num_players": event.num_players,
            }
        elif isinstance(event, BoardCardsDealt):
            return {
                "type": "BoardCardsDealt",
                "hand_number": event.hand_number,
                "street": event.street,
                "cards": list(event.cards),
            }
        elif isinstance(event, ActionTaken):
            return {
                "type": "ActionTaken",
                "hand_number": event.hand_number,
                "street": event.street,
                "seat": event.seat,
                "action_type": event.action_type,
                "amount": event.amount,
            }
        elif isinstance(event, PotsBuilt):
            pots_list = [
                {"amount": amount, "eligible_seats": list(seats)}
                for amount, seats in event.pots
            ]
            return {
                "type": "PotsBuilt",
                "hand_number": event.hand_number,
                "street": event.street,
                "pots": pots_list,
            }
        elif isinstance(event, Showdown):
            hands_dict = {
                str(seat): {"type": hand_type_val, "kickers": list(kickers)}
                for seat, (hand_type_val, kickers) in event.hands.items()
            }
            return {
                "type": "Showdown",
                "hand_number": event.hand_number,
                "hands": hands_dict,
            }
        elif isinstance(event, HandEnded):
            chip_dist = {str(seat): chips for seat, chips in event.chip_distribution.items()}
            return {
                "type": "HandEnded",
                "hand_number": event.hand_number,
                "chip_distribution": chip_dist,
            }
        elif isinstance(event, StreetEnded):
            return {
                "type": "StreetEnded",
                "hand_number": event.hand_number,
                "street": event.street,
                "snapshot": event.snapshot,
            }
        else:
            raise ValueError(f"Unknown event type: {type(event)}")

    @staticmethod
    def from_dict(data: dict[str, object]) -> Event:
        """Reconstruct an event from a JSON-compatible dict."""
        event_type = data.get("type")

        if event_type == "HandStarted":
            return HandStarted(
                hand_number=cast(int, data["hand_number"]),
                dealer_seat=cast(int, data["dealer_seat"]),
                small_blind=cast(int, data["small_blind"]),
                big_blind=cast(int, data["big_blind"]),
            )
        elif event_type == "BlindPosted":
            return BlindPosted(
                hand_number=cast(int, data["hand_number"]),
                seat=cast(int, data["seat"]),
                amount=cast(int, data["amount"]),
                is_big_blind=cast(bool, data["is_big_blind"]),
            )
        elif event_type == "AntePosted":
            return AntePosted(
                hand_number=cast(int, data["hand_number"]),
                seat=cast(int, data["seat"]),
                amount=cast(int, data["amount"]),
            )
        elif event_type == "HoleCardsDealt":
            return HoleCardsDealt(
                hand_number=cast(int, data["hand_number"]),
                num_players=cast(int, data["num_players"]),
            )
        elif event_type == "BoardCardsDealt":
            return BoardCardsDealt(
                hand_number=cast(int, data["hand_number"]),
                street=cast(str, data["street"]),
                cards=tuple(cast(list[str], data["cards"])),
            )
        elif event_type == "ActionTaken":
            return ActionTaken(
                hand_number=cast(int, data["hand_number"]),
                street=cast(str, data["street"]),
                seat=cast(int, data["seat"]),
                action_type=cast(str, data["action_type"]),
                amount=cast(int, data["amount"]),
            )
        elif event_type == "PotsBuilt":
            pots_data = cast(list[object], data["pots"])
            pots = tuple(
                (
                    cast(int, cast(dict[str, object], pot)["amount"]),
                    tuple(cast(list[int], cast(dict[str, object], pot)["eligible_seats"])),
                )
                for pot in pots_data
            )
            return PotsBuilt(
                hand_number=cast(int, data["hand_number"]),
                street=cast(str, data["street"]),
                pots=pots,
            )
        elif event_type == "Showdown":
            hands_data = cast(dict[str, object], data["hands"])
            hands = {
                int(seat): (
                    cast(int, cast(dict[str, object], hand_info)["type"]),
                    tuple(cast(list[int], cast(dict[str, object], hand_info)["kickers"])),
                )
                for seat, hand_info in hands_data.items()
            }
            return Showdown(
                hand_number=cast(int, data["hand_number"]),
                hands=hands,
            )
        elif event_type == "HandEnded":
            chip_dist_data = cast(dict[str, object], data["chip_distribution"])
            chip_distribution = {
                int(seat): cast(int, chips) for seat, chips in chip_dist_data.items()
            }
            return HandEnded(
                hand_number=cast(int, data["hand_number"]),
                chip_distribution=chip_distribution,
            )
        elif event_type == "StreetEnded":
            return StreetEnded(
                hand_number=cast(int, data["hand_number"]),
                street=cast(str, data["street"]),
                snapshot=cast(str, data["snapshot"]),
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")
