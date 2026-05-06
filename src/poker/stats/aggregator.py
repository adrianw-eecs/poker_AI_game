"""Stats aggregator: compute per-seat poker statistics from JSONL game logs."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BotStats:
    """Per-seat statistics computed from a JSONL game log.

    Attributes:
        seat: The player seat number.
        hands_played: Total number of hands played.
        hands_won: Hands where the player had a positive chip delta.
        win_rate: hands_won / hands_played (0.0 if no hands played).
        vpip_count: Hands with a voluntary preflop action (CALL/RAISE/ALL_IN).
        vpip: vpip_count / hands_played — Voluntarily Put money In Pot %.
        pfr_count: Hands with a preflop raise (RAISE or ALL_IN).
        pfr: pfr_count / hands_played — Pre-Flop Raise %.
        aggressive_actions: Total RAISE + ALL_IN actions across all streets.
        passive_actions: Total CALL actions across all streets.
        af: Aggression Factor = aggressive_actions / passive_actions.
              float('inf') when passive_actions == 0 and aggressive_actions > 0;
              0.0 when both are zero.
        total_chips_won: Sum of positive chip deltas (chips collected from pots).
        total_chips_delta: Net chip change across all hands.
        avg_pot_won: total_chips_won / hands_won (0.0 if no wins).
    """

    seat: int
    hands_played: int
    hands_won: int
    win_rate: float
    vpip_count: int
    vpip: float
    pfr_count: int
    pfr: float
    aggressive_actions: int
    passive_actions: int
    af: float
    total_chips_won: int
    total_chips_delta: int
    avg_pot_won: float


def aggregate_from_log(path: "Path | str") -> dict[int, BotStats]:
    """Compute per-seat statistics from a JSONL game log.

    Reads all events in the file, processes ActionTaken and HandEnded events,
    and returns a mapping from seat number to BotStats.

    VPIP counts hands where the player voluntarily entered the pot preflop
    (i.e., made a CALL, RAISE, or ALL_IN action on the PREFLOP street). Forced
    blind/ante postings are not counted because they appear as BlindPosted /
    AntePosted events, not ActionTaken.

    AF (Aggression Factor) = (RAISE + ALL_IN) / CALL across all streets.
    If a player never called, AF is float('inf') when they have aggressive
    actions, or 0.0 when they have neither.

    Args:
        path: Path to the JSONL log file.

    Returns:
        A dict mapping seat number to BotStats for every seat that appeared in
        at least one HandEnded or ActionTaken event in the log.
    """
    filepath = Path(path)

    # Per-seat raw accumulators
    vpip_hands: dict[int, set[int]] = {}  # seat -> set of hand_numbers
    pfr_hands: dict[int, set[int]] = {}
    aggressive: dict[int, int] = {}  # RAISE + ALL_IN count
    passive: dict[int, int] = {}  # CALL count

    # Chip tracking (populated from HandEnded events)
    chip_delta: dict[int, int] = {}
    chips_won: dict[int, int] = {}
    hands_played: dict[int, int] = {}
    hands_won_count: dict[int, int] = {}

    # All seats seen across any event
    all_seats: set[int] = set()

    def _ensure_seat(seat: int) -> None:
        """Initialize per-seat accumulators on first encounter."""
        if seat not in vpip_hands:
            vpip_hands[seat] = set()
            pfr_hands[seat] = set()
            aggressive[seat] = 0
            passive[seat] = 0
        all_seats.add(seat)

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data: dict[str, object] = json.loads(line)
            event_type = data.get("type")

            if event_type == "ActionTaken":
                seat = int(str(data["seat"]))
                street = str(data["street"]).upper()
                action_type = str(data["action_type"]).upper()
                hand_number = int(str(data["hand_number"]))

                _ensure_seat(seat)

                # VPIP: voluntary preflop action
                if street == "PREFLOP" and action_type in ("CALL", "RAISE", "ALL_IN"):
                    vpip_hands[seat].add(hand_number)

                # PFR: preflop raise or all-in
                if street == "PREFLOP" and action_type in ("RAISE", "ALL_IN"):
                    pfr_hands[seat].add(hand_number)

                # Aggression factor components (all streets)
                if action_type in ("RAISE", "ALL_IN"):
                    aggressive[seat] += 1
                elif action_type == "CALL":
                    passive[seat] += 1

            elif event_type == "HandEnded":
                raw_dist = data["chip_distribution"]
                if not isinstance(raw_dist, dict):
                    continue
                chip_distribution = {int(k): int(str(v)) for k, v in raw_dist.items()}

                for seat, delta in chip_distribution.items():
                    _ensure_seat(seat)
                    chip_delta[seat] = chip_delta.get(seat, 0) + delta
                    hands_played[seat] = hands_played.get(seat, 0) + 1
                    if delta > 0:
                        chips_won[seat] = chips_won.get(seat, 0) + delta
                        hands_won_count[seat] = hands_won_count.get(seat, 0) + 1

    # Build final BotStats for each seat
    result: dict[int, BotStats] = {}
    for seat in all_seats:
        played = hands_played.get(seat, 0)
        won = hands_won_count.get(seat, 0)
        win_rate = won / played if played > 0 else 0.0

        vpip_count = len(vpip_hands.get(seat, set()))
        pfr_count = len(pfr_hands.get(seat, set()))
        vpip = vpip_count / played if played > 0 else 0.0
        pfr = pfr_count / played if played > 0 else 0.0

        agg = aggressive.get(seat, 0)
        pas = passive.get(seat, 0)
        if pas > 0:
            af = agg / pas
        elif agg > 0:
            af = float("inf")
        else:
            af = 0.0

        total_won = chips_won.get(seat, 0)
        avg_won = total_won / won if won > 0 else 0.0

        result[seat] = BotStats(
            seat=seat,
            hands_played=played,
            hands_won=won,
            win_rate=win_rate,
            vpip_count=vpip_count,
            vpip=vpip,
            pfr_count=pfr_count,
            pfr=pfr,
            aggressive_actions=agg,
            passive_actions=pas,
            af=af,
            total_chips_won=total_won,
            total_chips_delta=chip_delta.get(seat, 0),
            avg_pot_won=avg_won,
        )

    return result
