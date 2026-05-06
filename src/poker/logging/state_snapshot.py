"""Human-readable GameState snapshot formatting."""

from __future__ import annotations

from poker.state.game_state import GameState


_POSITION_LABELS_BY_COUNT: dict[int, list[str]] = {
    2: ["BTN/SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "UTG"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
    7: ["BTN", "SB", "BB", "UTG", "MP", "HJ", "CO"],
    8: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"],
    9: ["BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "HJ", "CO"],
    10: ["BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "MP+1", "HJ", "CO"],
}


def _position_label(seat: int, dealer_seat: int, num_players: int) -> str:
    labels = _POSITION_LABELS_BY_COUNT.get(num_players)
    if labels is None:
        return f"Seat{seat}"
    offset = (seat - dealer_seat) % num_players
    return labels[offset]


def format_state_snapshot(state: GameState) -> str:
    """Format a full, non-viewer-relative snapshot of the state.

    This is intended to read similarly to the terminal output used for humans,
    but is stable for logging and does not hide information based on viewer seat.
    """
    lines: list[str] = []
    street = state.street.value.upper()

    lines.append(f"=== Hand #{state.hand_number + 1} | Street: {street} ===")

    # Board
    if state.community_cards:
        # IMPORTANT: Card.__str__ includes ANSI color codes for terminal play.
        # For logs, we emit a plain, non-colored representation.
        board = " ".join(f"{c.rank}{c.suit}" for c in state.community_cards)
        lines.append(f"Board: [ {board} ]")
    else:
        lines.append("Board: [ - ]")

    # Pots
    total_pot = sum(p.amount for p in state.pots)
    if len(state.pots) <= 1:
        lines.append(f"Pot: {total_pot}")
    else:
        parts = [f"{p.amount}@{sorted(p.eligible_seats)}" for p in state.pots]
        lines.append(f"Pot: {total_pot}  ({' | '.join(parts)})")

    if state.current_bet_to_call > 0:
        lines.append(f"To call: {state.current_bet_to_call}")

    lines.append("")
    lines.append("Players:")

    num_players = len(state.players)
    for p in state.players:
        pos = _position_label(p.seat, state.dealer_seat, num_players)
        status: list[str] = []
        if p.is_eliminated:
            status.append("ELIM")
        if p.has_folded:
            status.append("FOLDED")
        if p.is_all_in:
            status.append("ALL-IN")
        if state.action_on_seat == p.seat:
            status.append("ACTION")
        status_str = f" ({', '.join(status)})" if status else ""

        # Hole cards: show only at showdown (or when the hand ended early and engine moved to SHOWDOWN)
        if street == "SHOWDOWN" and p.hole_cards:
            hole = " ".join(f"{c.rank}{c.suit}" for c in p.hole_cards)
            hole_str = f"[{hole}]"
        else:
            hole_str = "[?? ??]" if p.hole_cards and len(p.hole_cards) == 2 else "[     ]"

        lines.append(
            f"  [{pos:7s}] {p.name:12s} stack={p.stack:5d} "
            f"committed(hand={p.committed_this_hand}, street={p.committed_this_street}) "
            f"{hole_str}{status_str}"
        )

    # Action history (street)
    if state.action_history_this_street:
        lines.append("")
        lines.append("Action this street:")
        for seat, action in state.action_history_this_street:
            actor = state.players[seat].name
            amt = f" {action.amount}" if action.amount else ""
            lines.append(f"  - {actor} (seat {seat}): {action.type.value.upper()}{amt}")

    lines.append("")
    return "\n".join(lines)

