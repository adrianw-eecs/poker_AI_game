"""Event types for game logging and replay.

JSONL is the canonical record. Each event is one fact. New fields have
defaults so logs written before this schema still decode. ``seq`` is added
by GameLogger and ignored here.
"""

from dataclasses import dataclass, field
from typing import cast


@dataclass(frozen=True)
class SessionStarted:
    """Emitted once when a session begins."""

    seats: tuple[int, ...]
    names: dict[int, str]
    stacks: dict[int, int]
    small_blind: int
    big_blind: int
    ante: int
    rake_percent: float
    run_it_twice: bool
    seed: int | None = None


@dataclass(frozen=True)
class SessionEnded:
    """Emitted once when a session stops."""

    reason: str  # "one_player", "hand_limit", "time_limit"
    stacks: dict[int, int]
    net: dict[int, int]
    names: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HandStarted:
    """Emitted when a new hand begins."""

    hand_number: int
    dealer_seat: int
    small_blind: int
    big_blind: int
    ante: int = 0
    sb_seat: int | None = None
    bb_seat: int | None = None
    stacks: dict[int, int] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)
    active_seats: tuple[int, ...] = ()


@dataclass(frozen=True)
class BlindPosted:
    """Emitted when a blind is posted."""

    hand_number: int
    seat: int
    amount: int
    is_big_blind: bool
    required: int = 0
    stack_before: int = 0
    stack_after: int = 0
    all_in: bool = False
    name: str = ""
    position: str = ""


@dataclass(frozen=True)
class AntePosted:
    """Emitted when an ante is posted."""

    hand_number: int
    seat: int
    amount: int
    stack_before: int = 0
    stack_after: int = 0
    all_in: bool = False
    name: str = ""
    position: str = ""


@dataclass(frozen=True)
class HoleCardsDealt:
    """Emitted when hole cards are dealt."""

    hand_number: int
    num_players: int
    cards: dict[int, tuple[str, ...]] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BoardCardsDealt:
    """Emitted when community cards are dealt."""

    hand_number: int
    street: str  # "flop", "turn", or "river"
    cards: tuple[str, ...]  # Newly dealt card codes, e.g. "As"
    board: tuple[str, ...] = ()  # Full board after this deal


@dataclass(frozen=True)
class ActionTaken:
    """Emitted when a player takes a voluntary action."""

    hand_number: int
    street: str
    seat: int
    action_type: str  # engine type: "fold", "check", "call", "raise", "all_in"
    amount: int
    chips_added: int = 0
    pot_after: int = 0
    action_num: int = 0
    log_line: str = ""
    label: str = ""  # "fold", "check", "call", "bet", "raise", "all_in"
    to_call_before: int = 0
    stack_before: int = 0
    stack_after: int = 0
    all_in: bool = False
    reopens: bool = False
    name: str = ""
    position: str = ""


@dataclass(frozen=True)
class PlayerOut:
    """Emitted when a player leaves the hand or the game."""

    hand_number: int
    seat: int
    reason: str  # "fold", "sit_out", "busted"
    street: str = ""
    stack: int = 0
    name: str = ""
    position: str = ""


@dataclass(frozen=True)
class PlayerRebought:
    """Emitted when an eliminated player rebuys."""

    hand_number: int
    seat: int
    amount: int
    stack_after: int
    name: str = ""


@dataclass(frozen=True)
class ButtonMoved:
    """Emitted when the button moves to the next hand."""

    hand_number: int
    dealer_seat: int


@dataclass(frozen=True)
class BlindLevelChanged:
    """Emitted when the scheduled blind level changes."""

    hand_number: int
    small_blind: int
    big_blind: int
    ante: int = 0


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
    ended_early: bool = False
    hole_cards: dict[int, tuple[str, ...]] = field(default_factory=dict)
    rank_names: dict[int, str] = field(default_factory=dict)
    best_five: dict[int, tuple[str, ...]] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RunItTwice:
    """Emitted when the pot is run twice."""

    hand_number: int
    board1: tuple[str, ...]
    board2: tuple[str, ...]
    awards1: dict[int, int]
    awards2: dict[int, int]
    ranks1: dict[int, str] = field(default_factory=dict)
    ranks2: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PotAwarded:
    """Emitted when a pot is given to one or more winners."""

    hand_number: int
    pot_index: int
    amount: int
    winning_seats: tuple[int, ...]
    split: bool
    run: int = 0  # 0 for a single run; 1 or 2 when run-it-twice


@dataclass(frozen=True)
class RakeTaken:
    """Emitted when the house takes rake. Omitted when the amount is 0."""

    hand_number: int
    amount: int


@dataclass(frozen=True)
class HandEnded:
    """Emitted when a hand finishes."""

    hand_number: int
    chip_distribution: dict[int, int]  # seat -> gross chips collected from pots
    net: dict[int, int] = field(default_factory=dict)  # award minus committed
    stacks_after: dict[int, int] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class StreetEnded:
    """Emitted when a betting street closes.

    ``snapshot`` is kept so older logs still decode. New records leave it
    empty and fill the structured fields instead.
    """

    hand_number: int
    street: str
    snapshot: str = ""
    pot: int = 0
    board: tuple[str, ...] = ()
    active_seats: tuple[int, ...] = ()
    folded_seats: tuple[int, ...] = ()
    all_in_seats: tuple[int, ...] = ()
    reason: str = ""  # "matched", "uncontested", "all_in"


@dataclass(frozen=True)
class TimingEvent:
    """Emitted when tracking timing of game actions. Diagnostic only."""

    hand_number: int
    phase: str
    seat: int
    elapsed_us: int


@dataclass(frozen=True)
class InvalidAction:
    """Emitted when a bot offers an illegal action. Diagnostic only."""

    hand_number: int
    seat: int
    street: str
    attempt: int
    bot: str
    action: str
    error: str


Event = (
    SessionStarted
    | SessionEnded
    | HandStarted
    | BlindPosted
    | AntePosted
    | HoleCardsDealt
    | BoardCardsDealt
    | ActionTaken
    | PlayerOut
    | PlayerRebought
    | ButtonMoved
    | BlindLevelChanged
    | PotsBuilt
    | Showdown
    | RunItTwice
    | PotAwarded
    | RakeTaken
    | HandEnded
    | StreetEnded
    | TimingEvent
    | InvalidAction
)


def is_diagnostic(event: Event) -> bool:
    """Return True for events that are not part of the hand history."""
    return isinstance(event, (TimingEvent, InvalidAction))


def _int_map(mapping: dict[int, object]) -> dict[str, object]:
    return {str(key): value for key, value in mapping.items()}


def _load_int_int(raw: object) -> dict[int, int]:
    if not isinstance(raw, dict):
        return {}
    return {int(key): int(cast(int, value)) for key, value in raw.items()}


def _load_int_str(raw: object) -> dict[int, str]:
    if not isinstance(raw, dict):
        return {}
    return {int(key): str(value) for key, value in raw.items()}


def _load_int_cards(raw: object) -> dict[int, tuple[str, ...]]:
    if not isinstance(raw, dict):
        return {}
    loaded: dict[int, tuple[str, ...]] = {}
    for key, value in raw.items():
        loaded[int(key)] = tuple(str(card) for card in cast(list[object], value))
    return loaded


def _load_str_tuple(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(str(item) for item in raw)


def _load_int_tuple(raw: object) -> tuple[int, ...]:
    if not isinstance(raw, list):
        return ()
    return tuple(int(cast(int, item)) for item in raw)


def _optional_int(data: dict[str, object], key: str) -> int | None:
    if key not in data or data[key] is None:
        return None
    return int(cast(int, data[key]))


class EventEncoder:
    """Encodes and decodes events to/from JSON-compatible dicts."""

    @staticmethod
    def to_dict(event: Event) -> dict[str, object]:
        """Convert an event to a JSON-compatible dict."""
        if isinstance(event, SessionStarted):
            return {
                "type": "SessionStarted",
                "seats": list(event.seats),
                "names": _int_map(cast(dict[int, object], event.names)),
                "stacks": _int_map(cast(dict[int, object], event.stacks)),
                "small_blind": event.small_blind,
                "big_blind": event.big_blind,
                "ante": event.ante,
                "rake_percent": event.rake_percent,
                "run_it_twice": event.run_it_twice,
                "seed": event.seed,
            }
        elif isinstance(event, SessionEnded):
            return {
                "type": "SessionEnded",
                "reason": event.reason,
                "stacks": _int_map(cast(dict[int, object], event.stacks)),
                "net": _int_map(cast(dict[int, object], event.net)),
                "names": _int_map(cast(dict[int, object], event.names)),
            }
        elif isinstance(event, HandStarted):
            return {
                "type": "HandStarted",
                "hand_number": event.hand_number,
                "dealer_seat": event.dealer_seat,
                "small_blind": event.small_blind,
                "big_blind": event.big_blind,
                "ante": event.ante,
                "sb_seat": event.sb_seat,
                "bb_seat": event.bb_seat,
                "stacks": _int_map(cast(dict[int, object], event.stacks)),
                "names": _int_map(cast(dict[int, object], event.names)),
                "active_seats": list(event.active_seats),
            }
        elif isinstance(event, BlindPosted):
            return {
                "type": "BlindPosted",
                "hand_number": event.hand_number,
                "seat": event.seat,
                "amount": event.amount,
                "is_big_blind": event.is_big_blind,
                "required": event.required,
                "stack_before": event.stack_before,
                "stack_after": event.stack_after,
                "all_in": event.all_in,
                "name": event.name,
                "position": event.position,
            }
        elif isinstance(event, AntePosted):
            return {
                "type": "AntePosted",
                "hand_number": event.hand_number,
                "seat": event.seat,
                "amount": event.amount,
                "stack_before": event.stack_before,
                "stack_after": event.stack_after,
                "all_in": event.all_in,
                "name": event.name,
                "position": event.position,
            }
        elif isinstance(event, HoleCardsDealt):
            cards = {
                str(seat): list(codes) for seat, codes in event.cards.items()
            }
            return {
                "type": "HoleCardsDealt",
                "hand_number": event.hand_number,
                "num_players": event.num_players,
                "cards": cards,
                "names": _int_map(cast(dict[int, object], event.names)),
            }
        elif isinstance(event, BoardCardsDealt):
            return {
                "type": "BoardCardsDealt",
                "hand_number": event.hand_number,
                "street": event.street,
                "cards": list(event.cards),
                "board": list(event.board),
            }
        elif isinstance(event, ActionTaken):
            return {
                "type": "ActionTaken",
                "hand_number": event.hand_number,
                "street": event.street,
                "seat": event.seat,
                "action_type": event.action_type,
                "amount": event.amount,
                "chips_added": event.chips_added,
                "pot_after": event.pot_after,
                "action_num": event.action_num,
                "log_line": event.log_line,
                "label": event.label,
                "to_call_before": event.to_call_before,
                "stack_before": event.stack_before,
                "stack_after": event.stack_after,
                "all_in": event.all_in,
                "reopens": event.reopens,
                "name": event.name,
                "position": event.position,
            }
        elif isinstance(event, PlayerOut):
            return {
                "type": "PlayerOut",
                "hand_number": event.hand_number,
                "seat": event.seat,
                "reason": event.reason,
                "street": event.street,
                "stack": event.stack,
                "name": event.name,
                "position": event.position,
            }
        elif isinstance(event, PlayerRebought):
            return {
                "type": "PlayerRebought",
                "hand_number": event.hand_number,
                "seat": event.seat,
                "amount": event.amount,
                "stack_after": event.stack_after,
                "name": event.name,
            }
        elif isinstance(event, ButtonMoved):
            return {
                "type": "ButtonMoved",
                "hand_number": event.hand_number,
                "dealer_seat": event.dealer_seat,
            }
        elif isinstance(event, BlindLevelChanged):
            return {
                "type": "BlindLevelChanged",
                "hand_number": event.hand_number,
                "small_blind": event.small_blind,
                "big_blind": event.big_blind,
                "ante": event.ante,
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
            hole = {str(seat): list(codes) for seat, codes in event.hole_cards.items()}
            best = {str(seat): list(codes) for seat, codes in event.best_five.items()}
            return {
                "type": "Showdown",
                "hand_number": event.hand_number,
                "hands": hands_dict,
                "ended_early": event.ended_early,
                "hole_cards": hole,
                "rank_names": _int_map(cast(dict[int, object], event.rank_names)),
                "best_five": best,
                "names": _int_map(cast(dict[int, object], event.names)),
            }
        elif isinstance(event, RunItTwice):
            return {
                "type": "RunItTwice",
                "hand_number": event.hand_number,
                "board1": list(event.board1),
                "board2": list(event.board2),
                "awards1": _int_map(cast(dict[int, object], event.awards1)),
                "awards2": _int_map(cast(dict[int, object], event.awards2)),
                "ranks1": _int_map(cast(dict[int, object], event.ranks1)),
                "ranks2": _int_map(cast(dict[int, object], event.ranks2)),
            }
        elif isinstance(event, PotAwarded):
            return {
                "type": "PotAwarded",
                "hand_number": event.hand_number,
                "pot_index": event.pot_index,
                "amount": event.amount,
                "winning_seats": list(event.winning_seats),
                "split": event.split,
                "run": event.run,
            }
        elif isinstance(event, RakeTaken):
            return {
                "type": "RakeTaken",
                "hand_number": event.hand_number,
                "amount": event.amount,
            }
        elif isinstance(event, HandEnded):
            return {
                "type": "HandEnded",
                "hand_number": event.hand_number,
                "chip_distribution": _int_map(cast(dict[int, object], event.chip_distribution)),
                "net": _int_map(cast(dict[int, object], event.net)),
                "stacks_after": _int_map(cast(dict[int, object], event.stacks_after)),
                "names": _int_map(cast(dict[int, object], event.names)),
            }
        elif isinstance(event, StreetEnded):
            return {
                "type": "StreetEnded",
                "hand_number": event.hand_number,
                "street": event.street,
                "snapshot": event.snapshot,
                "pot": event.pot,
                "board": list(event.board),
                "active_seats": list(event.active_seats),
                "folded_seats": list(event.folded_seats),
                "all_in_seats": list(event.all_in_seats),
                "reason": event.reason,
            }
        elif isinstance(event, TimingEvent):
            return {
                "type": "TimingEvent",
                "hand_number": event.hand_number,
                "phase": event.phase,
                "seat": event.seat,
                "elapsed_us": event.elapsed_us,
            }
        elif isinstance(event, InvalidAction):
            return {
                "type": "InvalidAction",
                "hand_number": event.hand_number,
                "seat": event.seat,
                "street": event.street,
                "attempt": event.attempt,
                "bot": event.bot,
                "action": event.action,
                "error": event.error,
            }
        else:
            raise ValueError(f"Unknown event type: {type(event)}")

    @staticmethod
    def from_dict(data: dict[str, object]) -> Event:
        """Reconstruct an event from a JSON-compatible dict."""
        event_type = data.get("type")

        if event_type == "SessionStarted":
            return SessionStarted(
                seats=_load_int_tuple(data.get("seats")),
                names=_load_int_str(data.get("names")),
                stacks=_load_int_int(data.get("stacks")),
                small_blind=cast(int, data["small_blind"]),
                big_blind=cast(int, data["big_blind"]),
                ante=cast(int, data.get("ante", 0)),
                rake_percent=float(cast(float, data.get("rake_percent", 0.0))),
                run_it_twice=bool(data.get("run_it_twice", False)),
                seed=cast(int, data["seed"]) if data.get("seed") is not None else None,
            )
        elif event_type == "SessionEnded":
            return SessionEnded(
                reason=cast(str, data["reason"]),
                stacks=_load_int_int(data.get("stacks")),
                net=_load_int_int(data.get("net")),
                names=_load_int_str(data.get("names")),
            )
        elif event_type == "HandStarted":
            return HandStarted(
                hand_number=cast(int, data["hand_number"]),
                dealer_seat=cast(int, data["dealer_seat"]),
                small_blind=cast(int, data["small_blind"]),
                big_blind=cast(int, data["big_blind"]),
                ante=cast(int, data.get("ante", 0)),
                sb_seat=_optional_int(data, "sb_seat"),
                bb_seat=_optional_int(data, "bb_seat"),
                stacks=_load_int_int(data.get("stacks")),
                names=_load_int_str(data.get("names")),
                active_seats=_load_int_tuple(data.get("active_seats")),
            )
        elif event_type == "BlindPosted":
            return BlindPosted(
                hand_number=cast(int, data["hand_number"]),
                seat=cast(int, data["seat"]),
                amount=cast(int, data["amount"]),
                is_big_blind=cast(bool, data["is_big_blind"]),
                required=cast(int, data.get("required", 0)),
                stack_before=cast(int, data.get("stack_before", 0)),
                stack_after=cast(int, data.get("stack_after", 0)),
                all_in=bool(data.get("all_in", False)),
                name=cast(str, data.get("name", "")),
                position=cast(str, data.get("position", "")),
            )
        elif event_type == "AntePosted":
            return AntePosted(
                hand_number=cast(int, data["hand_number"]),
                seat=cast(int, data["seat"]),
                amount=cast(int, data["amount"]),
                stack_before=cast(int, data.get("stack_before", 0)),
                stack_after=cast(int, data.get("stack_after", 0)),
                all_in=bool(data.get("all_in", False)),
                name=cast(str, data.get("name", "")),
                position=cast(str, data.get("position", "")),
            )
        elif event_type == "HoleCardsDealt":
            return HoleCardsDealt(
                hand_number=cast(int, data["hand_number"]),
                num_players=cast(int, data["num_players"]),
                cards=_load_int_cards(data.get("cards")),
                names=_load_int_str(data.get("names")),
            )
        elif event_type == "BoardCardsDealt":
            return BoardCardsDealt(
                hand_number=cast(int, data["hand_number"]),
                street=cast(str, data["street"]),
                cards=_load_str_tuple(data.get("cards")),
                board=_load_str_tuple(data.get("board")),
            )
        elif event_type == "ActionTaken":
            return ActionTaken(
                hand_number=cast(int, data["hand_number"]),
                street=cast(str, data["street"]),
                seat=cast(int, data["seat"]),
                action_type=cast(str, data["action_type"]),
                amount=cast(int, data["amount"]),
                chips_added=cast(int, data.get("chips_added", 0)),
                pot_after=cast(int, data.get("pot_after", 0)),
                action_num=cast(int, data.get("action_num", 0)),
                log_line=cast(str, data.get("log_line", "")),
                label=cast(str, data.get("label", "")),
                to_call_before=cast(int, data.get("to_call_before", 0)),
                stack_before=cast(int, data.get("stack_before", 0)),
                stack_after=cast(int, data.get("stack_after", 0)),
                all_in=bool(data.get("all_in", False)),
                reopens=bool(data.get("reopens", False)),
                name=cast(str, data.get("name", "")),
                position=cast(str, data.get("position", "")),
            )
        elif event_type == "PlayerOut":
            return PlayerOut(
                hand_number=cast(int, data["hand_number"]),
                seat=cast(int, data["seat"]),
                reason=cast(str, data["reason"]),
                street=cast(str, data.get("street", "")),
                stack=cast(int, data.get("stack", 0)),
                name=cast(str, data.get("name", "")),
                position=cast(str, data.get("position", "")),
            )
        elif event_type == "PlayerRebought":
            return PlayerRebought(
                hand_number=cast(int, data["hand_number"]),
                seat=cast(int, data["seat"]),
                amount=cast(int, data["amount"]),
                stack_after=cast(int, data["stack_after"]),
                name=cast(str, data.get("name", "")),
            )
        elif event_type == "ButtonMoved":
            return ButtonMoved(
                hand_number=cast(int, data["hand_number"]),
                dealer_seat=cast(int, data["dealer_seat"]),
            )
        elif event_type == "BlindLevelChanged":
            return BlindLevelChanged(
                hand_number=cast(int, data["hand_number"]),
                small_blind=cast(int, data["small_blind"]),
                big_blind=cast(int, data["big_blind"]),
                ante=cast(int, data.get("ante", 0)),
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
                ended_early=bool(data.get("ended_early", False)),
                hole_cards=_load_int_cards(data.get("hole_cards")),
                rank_names=_load_int_str(data.get("rank_names")),
                best_five=_load_int_cards(data.get("best_five")),
                names=_load_int_str(data.get("names")),
            )
        elif event_type == "RunItTwice":
            return RunItTwice(
                hand_number=cast(int, data["hand_number"]),
                board1=_load_str_tuple(data.get("board1")),
                board2=_load_str_tuple(data.get("board2")),
                awards1=_load_int_int(data.get("awards1")),
                awards2=_load_int_int(data.get("awards2")),
                ranks1=_load_int_str(data.get("ranks1")),
                ranks2=_load_int_str(data.get("ranks2")),
            )
        elif event_type == "PotAwarded":
            return PotAwarded(
                hand_number=cast(int, data["hand_number"]),
                pot_index=cast(int, data["pot_index"]),
                amount=cast(int, data["amount"]),
                winning_seats=_load_int_tuple(data.get("winning_seats")),
                split=bool(data["split"]),
                run=cast(int, data.get("run", 0)),
            )
        elif event_type == "RakeTaken":
            return RakeTaken(
                hand_number=cast(int, data["hand_number"]),
                amount=cast(int, data["amount"]),
            )
        elif event_type == "HandEnded":
            return HandEnded(
                hand_number=cast(int, data["hand_number"]),
                chip_distribution=_load_int_int(data.get("chip_distribution")),
                net=_load_int_int(data.get("net")),
                stacks_after=_load_int_int(data.get("stacks_after")),
                names=_load_int_str(data.get("names")),
            )
        elif event_type == "StreetEnded":
            return StreetEnded(
                hand_number=cast(int, data["hand_number"]),
                street=cast(str, data["street"]),
                snapshot=cast(str, data.get("snapshot", "")),
                pot=cast(int, data.get("pot", 0)),
                board=_load_str_tuple(data.get("board")),
                active_seats=_load_int_tuple(data.get("active_seats")),
                folded_seats=_load_int_tuple(data.get("folded_seats")),
                all_in_seats=_load_int_tuple(data.get("all_in_seats")),
                reason=cast(str, data.get("reason", "")),
            )
        elif event_type == "TimingEvent":
            return TimingEvent(
                hand_number=cast(int, data["hand_number"]),
                phase=cast(str, data["phase"]),
                seat=cast(int, data["seat"]),
                elapsed_us=cast(int, data["elapsed_us"]),
            )
        elif event_type == "InvalidAction":
            return InvalidAction(
                hand_number=cast(int, data["hand_number"]),
                seat=cast(int, data["seat"]),
                street=cast(str, data["street"]),
                attempt=cast(int, data["attempt"]),
                bot=cast(str, data["bot"]),
                action=cast(str, data["action"]),
                error=cast(str, data["error"]),
            )
        else:
            raise ValueError(f"Unknown event type: {event_type}")
