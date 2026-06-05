"""Human-readable GameState snapshot formatting."""

from __future__ import annotations

from poker.domain.action import Action, ActionType
from poker.state.game_state import GameState
from poker.state.pot_manager import rebuild_pots


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


def position_label(seat: int, dealer_seat: int, num_players: int) -> str:
    """Return the position abbreviation for a seat."""
    labels = _POSITION_LABELS_BY_COUNT.get(num_players)
    if labels is None:
        return f"Seat{seat}"
    offset = (seat - dealer_seat) % num_players
    return labels[offset]


def _format_pot_line(state: GameState) -> str:
    """Format pot total with optional side-pot breakdown."""
    total_pot = sum(p.amount for p in state.pots)
    if len(state.pots) <= 1:
        return f"Pot: {total_pot}"
    parts = [f"{p.amount}@{sorted(p.eligible_seats)}" for p in state.pots]
    return f"Pot: {total_pot}  ({' | '.join(parts)})"


def _chips_added_for_action(action: Action, committed_before: int) -> int:
    """Compute incremental chips put in for an action."""
    if action.type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
        return action.amount - committed_before
    return 0


def format_action_description(
    action: Action,
    *,
    chips_added: int | None = None,
) -> str:
    """Return a human-readable action string with optional chip delta."""
    if action.type == ActionType.FOLD:
        base = "FOLD"
    elif action.type == ActionType.CHECK:
        base = "CHECK"
    elif action.type == ActionType.CALL:
        base = f"CALL to {action.amount}"
    elif action.type == ActionType.RAISE:
        base = f"RAISE to {action.amount}"
    else:
        base = f"ALL-IN to {action.amount}"

    if chips_added is not None and chips_added > 0:
        return f"{base} (+{chips_added})"
    return base


def format_action_history(
    state: GameState,
    actions: list[tuple[int, Action]],
    *,
    numbered: bool = True,
    indent: str = "  ",
) -> list[str]:
    """Format a list of (seat, action) pairs with incremental chip deltas."""
    lines: list[str] = []
    street_commits: dict[int, int] = {p.seat: 0 for p in state.players}

    for i, (seat, action) in enumerate(actions, start=1):
        player = state.players[seat]
        pos = position_label(seat, state.dealer_seat, len(state.players))
        before = street_commits[seat]
        chips_added = _chips_added_for_action(action, before)
        if action.type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
            street_commits[seat] = action.amount

        desc = format_action_description(action, chips_added=chips_added)
        prefix = f"{i}. " if numbered else "- "
        lines.append(
            f"{indent}{prefix}{player.name} ({pos}, seat {seat}): {desc}"
        )
    return lines


def format_blinds_line(state: GameState) -> str | None:
    """Return a blinds-posted summary for preflop, or None if not applicable."""
    if state.street.value != "preflop":
        return None
    num_players = len(state.players)
    sb_seat = (state.dealer_seat + 1) % num_players
    bb_seat = (state.dealer_seat + 2) % num_players
    sb = state.players[sb_seat]
    bb = state.players[bb_seat]
    sb_amt = state.config.small_blind
    bb_amt = state.config.big_blind
    if sb.committed_this_street < sb_amt and bb.committed_this_street < bb_amt:
        return None
    return (
        f"Blinds posted: {sb.name} (SB) +{sb_amt}, "
        f"{bb.name} (BB) +{bb_amt}"
    )


def format_hand_replay(
    hand_log: list[tuple[str, int, Action]],
    state: GameState,
) -> str:
    """Format a full hand action replay grouped by street."""
    if not hand_log:
        return ""

    lines: list[str] = ["=== Hand Action Replay ==="]
    current_street: str | None = None
    street_commits: dict[int, int] = {p.seat: 0 for p in state.players}
    street_parts: list[str] = []

    for street, seat, action in hand_log:
        if street != current_street:
            if street_parts and current_street is not None:
                lines.append(f"{current_street}:  {' | '.join(street_parts)}")
            current_street = street
            street_parts = []
            street_commits = {p.seat: 0 for p in state.players}

        player = state.players[seat]
        pos = position_label(seat, state.dealer_seat, len(state.players))
        before = street_commits[seat]
        chips_added = _chips_added_for_action(action, before)
        if action.type in (ActionType.CALL, ActionType.RAISE, ActionType.ALL_IN):
            street_commits[seat] = action.amount

        short = format_action_description(action, chips_added=chips_added)
        # Compact: "P2(SB) RAISE to 100 (+75)"
        short = short.replace("RAISE to", "RAISE").replace("CALL to", "CALL").replace(
            "ALL-IN to", "ALL-IN"
        )
        street_parts.append(f"{player.name}({pos}) {short}")

    if street_parts and current_street is not None:
        lines.append(f"{current_street}:  {' | '.join(street_parts)}")

    lines.append("")
    return "\n".join(lines)


def _format_hole_cards(
    state: GameState,
    seat: int,
    *,
    viewer_seat: int | None,
    use_color: bool,
) -> str:
    """Format hole cards respecting viewer visibility."""
    player = state.players[seat]
    street = state.street.value.upper()
    show_all = viewer_seat is None and street == "SHOWDOWN"
    is_viewer = viewer_seat is not None and seat == viewer_seat

    if not player.hole_cards:
        return "[     ]"

    if show_all or is_viewer:
        if use_color:
            return f"[{' '.join(str(c) for c in player.hole_cards)}]"
        return f"[{' '.join(f'{c.rank}{c.suit}' for c in player.hole_cards)}]"

    if len(player.hole_cards) == 2:
        return "[?? ??]"
    return "[?]"


def _format_player_status(state: GameState, seat: int) -> str:
    """Build status suffix for a player line."""
    player = state.players[seat]
    parts: list[str] = []
    if player.is_eliminated:
        parts.append("OUT")
    if player.has_folded:
        parts.append("FOLDED")
    if player.is_all_in:
        parts.append("ALL-IN")
    if state.action_on_seat == seat:
        parts.append("<<< action")
    return f" ({', '.join(parts)})" if parts else ""


def _format_last_action(action: Action) -> str:
    """Format the most recent action for inline player display."""
    if action.type == ActionType.FOLD:
        return "folded"
    if action.type == ActionType.CHECK:
        return "checked"
    if action.type == ActionType.CALL:
        return f"called to {action.amount}"
    if action.type == ActionType.RAISE:
        return f"raised to {action.amount}"
    return f"all-in to {action.amount}"


def format_state_snapshot(
    state: GameState,
    *,
    viewer_seat: int | None = None,
    indent: str = "",
    use_color_cards: bool = False,
) -> str:
    """Format a GameState snapshot for terminal or session logging.

    Args:
        state: Game state (pots are rebuilt for display accuracy).
        viewer_seat: If set, only this seat's hole cards are revealed.
        indent: Prefix for every line (terminal uses two spaces).
        use_color_cards: Use ANSI-colored card strings (terminal).
    """
    state = rebuild_pots(state)
    lines: list[str] = []
    street = state.street.value.upper()
    num_players = len(state.players)

    lines.append(
        f"{indent}=== Hand #{state.hand_number + 1}  |  Street: {street} ==="
    )
    lines.append("")

    if state.community_cards:
        if use_color_cards:
            board = " ".join(str(c) for c in state.community_cards)
        else:
            board = " ".join(f"{c.rank}{c.suit}" for c in state.community_cards)
        lines.append(f"{indent}  Board: [ {board} ]")
    else:
        lines.append(f"{indent}  Board: [ - ]")

    lines.append(f"{indent}  {_format_pot_line(state)}")
    if state.current_bet_to_call > 0:
        lines.append(f"{indent}  To call: {state.current_bet_to_call}")

    blinds = format_blinds_line(state)
    if blinds:
        lines.append(f"{indent}  {blinds}")

    lines.append("")
    lines.append(f"{indent}  Players:")

    last_action: dict[int, Action] = {}
    for seat, action in state.action_history_this_street:
        last_action[seat] = action

    for p in state.players:
        pos = position_label(p.seat, state.dealer_seat, num_players)
        hole_str = _format_hole_cards(
            state, p.seat, viewer_seat=viewer_seat, use_color=use_color_cards
        )
        viewer_marker = " (YOU)" if viewer_seat is not None and p.seat == viewer_seat else ""
        action_part = ""
        if p.seat in last_action:
            action_part = f" — {_format_last_action(last_action[p.seat])}"

        lines.append(
            f"{indent}    [{pos:8s}] {p.name}{viewer_marker}  "
            f"stack={p.stack}  "
            f"(in: {p.committed_this_street} street / {p.committed_this_hand} hand)  "
            f"{hole_str}{action_part}{_format_player_status(state, p.seat)}"
        )

    if state.action_history_this_street:
        lines.append("")
        lines.append(f"{indent}  Action this street:")
        lines.extend(
            format_action_history(
                state,
                state.action_history_this_street,
                indent=f"{indent}  ",
            )
        )

    lines.append("")
    return "\n".join(lines)


def format_action_log_line(
    state: GameState,
    seat: int,
    action: Action,
    action_num: int,
    chips_added: int,
) -> str:
    """Format a single per-action log line for live session transcripts."""
    player = state.players[seat]
    pos = position_label(seat, state.dealer_seat, len(state.players))
    street = state.street.value.upper()
    pot_total = sum(p.amount for p in state.pots)
    desc = format_action_description(action, chips_added=chips_added)
    next_actor = ""
    if state.action_on_seat is not None:
        next_p = state.players[state.action_on_seat]
        next_pos = position_label(
            state.action_on_seat, state.dealer_seat, len(state.players)
        )
        to_call = state.current_bet_to_call
        next_actor = (
            f"  Next: {next_p.name} ({next_pos})"
            f" | To call: {to_call} | Pot: {pot_total}"
        )
    else:
        next_actor = f"  Street complete | Pot: {pot_total}"

    return (
        f"[Hand #{state.hand_number + 1} | {street} | #{action_num}] "
        f"{player.name} ({pos}): {desc}{next_actor}"
    )
