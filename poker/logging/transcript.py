"""Render game events as a readable hand history.

The JSONL stream is the source of truth. This module only formats it.
Diagnostic events (timing, rejected actions) return None and stay out of
the default transcript.
"""

from poker.logging.events import (
    ActionTaken,
    AntePosted,
    BlindLevelChanged,
    BlindPosted,
    BoardCardsDealt,
    ButtonMoved,
    Event,
    HandEnded,
    HandStarted,
    HoleCardsDealt,
    PlayerOut,
    PlayerRebought,
    PotAwarded,
    PotsBuilt,
    RakeTaken,
    RunItTwice,
    SessionEnded,
    SessionStarted,
    Showdown,
    StreetEnded,
    is_diagnostic,
)


def render_event(event: Event) -> str | None:
    """Return one transcript block for an event, or None to skip it."""
    if is_diagnostic(event):
        return None
    if isinstance(event, SessionStarted):
        return _session_started(event)
    if isinstance(event, SessionEnded):
        return _session_ended(event)
    if isinstance(event, HandStarted):
        return _hand_started(event)
    if isinstance(event, BlindPosted):
        return _blind(event)
    if isinstance(event, AntePosted):
        return _ante(event)
    if isinstance(event, HoleCardsDealt):
        return _hole_cards(event)
    if isinstance(event, BoardCardsDealt):
        return _board(event)
    if isinstance(event, ActionTaken):
        return _action(event)
    if isinstance(event, PlayerOut):
        return _player_out(event)
    if isinstance(event, PlayerRebought):
        return _rebuy(event)
    if isinstance(event, ButtonMoved):
        return f"Button moves to seat {event.dealer_seat}\n"
    if isinstance(event, BlindLevelChanged):
        ante = f" ante {event.ante}" if event.ante else ""
        return f"Blinds now {event.small_blind}/{event.big_blind}{ante}\n"
    if isinstance(event, StreetEnded):
        return _street_ended(event)
    if isinstance(event, PotsBuilt):
        return _pots(event)
    if isinstance(event, Showdown):
        return _showdown(event)
    if isinstance(event, RunItTwice):
        return _run_it_twice(event)
    if isinstance(event, PotAwarded):
        return _pot_awarded(event)
    if isinstance(event, RakeTaken):
        return f"  Rake: {event.amount}\n"
    if isinstance(event, HandEnded):
        return _hand_ended(event)
    return None


def _who(name: str, position: str, seat: int) -> str:
    label = name or f"seat {seat}"
    if position:
        return f"{label} ({position} seat {seat})"
    return f"{label} (seat {seat})"


def _signed(amount: int) -> str:
    if amount > 0:
        return f"+{amount}"
    return str(amount)


def _session_started(event: SessionStarted) -> str:
    players = []
    for seat in event.seats:
        name = event.names.get(seat, f"seat {seat}")
        stack = event.stacks.get(seat, 0)
        players.append(f"seat {seat} {name} stack {stack}")
    seed = "none" if event.seed is None else str(event.seed)
    ante = f" ante {event.ante}" if event.ante else ""
    return (
        "=== POKER SESSION ===\n"
        f"Players: {'; '.join(players)}\n"
        f"Blinds {event.small_blind}/{event.big_blind}{ante} "
        f"rake {event.rake_percent:g}% run_it_twice={event.run_it_twice} seed={seed}\n\n"
    )


def _session_ended(event: SessionEnded) -> str:
    lines = [f"=== SESSION ENDED ({event.reason}) ===\n"]
    for seat in sorted(event.stacks):
        name = event.names.get(seat, f"seat {seat}")
        stack = event.stacks[seat]
        if event.net:
            lines.append(f"  {name} {stack} ({_signed(event.net.get(seat, 0))})\n")
        else:
            lines.append(f"  {name} {stack}\n")
    lines.append("\n")
    return "".join(lines)


def _hand_started(event: HandStarted) -> str:
    stacks = []
    seats = sorted(event.stacks) if event.stacks else list(event.active_seats)
    for seat in seats:
        name = event.names.get(seat, f"seat{seat}")
        if event.stacks:
            stacks.append(f"{name}={event.stacks[seat]}")
        else:
            stacks.append(name)
    stack_part = f" | stacks {' '.join(stacks)}" if stacks else ""
    ante = f" ante {event.ante}" if event.ante else ""
    return (
        f"{'=' * 60}\n"
        f"Hand #{event.hand_number + 1} | dealer seat {event.dealer_seat} "
        f"| blinds {event.small_blind}/{event.big_blind}{ante}{stack_part}\n"
        f"{'=' * 60}\n"
    )


def _blind(event: BlindPosted) -> str:
    kind = "BB" if event.is_big_blind else "SB"
    note = ""
    if event.all_in and event.required and event.amount < event.required:
        note = f" (all-in, short {event.required - event.amount})"
    elif event.all_in:
        note = " (all-in)"
    stack = f" | stack {event.stack_after}" if event.stack_before or event.stack_after else ""
    return f"  {_who(event.name, event.position, event.seat)} posts {kind} {event.amount}{note}{stack}\n"


def _ante(event: AntePosted) -> str:
    note = " (all-in)" if event.all_in else ""
    stack = f" | stack {event.stack_after}" if event.stack_before or event.stack_after else ""
    return f"  {_who(event.name, event.position, event.seat)} posts ante {event.amount}{note}{stack}\n"


def _hole_cards(event: HoleCardsDealt) -> str:
    if not event.cards:
        return f"  Hole cards dealt to {event.num_players} players\n"
    lines: list[str] = []
    for seat in sorted(event.cards):
        name = event.names.get(seat, f"seat {seat}")
        cards = " ".join(event.cards[seat])
        lines.append(f"  {name} is dealt {cards}\n")
    return "".join(lines)


def _board(event: BoardCardsDealt) -> str:
    cards = event.board or event.cards
    shown = " ".join(cards)
    return f"{event.street.upper()} [{shown}]\n"


def _action(event: ActionTaken) -> str:
    if event.log_line and not event.label:
        text = event.log_line
        return text if text.endswith("\n") else text + "\n"

    who = _who(event.name, event.position, event.seat)
    label = (event.label or event.action_type).lower()
    all_in = " (all-in)" if event.all_in and label != "all_in" else ""
    added = f" (+{event.chips_added})" if event.chips_added else ""
    tail = f" | pot {event.pot_after} | stack {event.stack_after}"

    if label == "fold":
        return f"  {who} folds\n"
    if label == "check":
        return f"  {who} checks | pot {event.pot_after}\n"
    if label == "call":
        return f"  {who} calls {event.amount}{added}{all_in}{tail}\n"
    if label == "bet":
        return f"  {who} bets {event.amount}{added}{all_in}{tail}\n"
    if label == "raise":
        return f"  {who} raises to {event.amount}{added}{all_in}{tail}\n"
    if label == "all_in":
        return f"  {who} all-in to {event.amount}{added}{tail}\n"
    return f"  {who} {label} {event.amount}{added}{all_in}{tail}\n"


def _player_out(event: PlayerOut) -> str:
    who = _who(event.name, event.position, event.seat)
    if event.reason == "fold":
        return f"  {who} is out (folded)\n"
    if event.reason == "sit_out":
        return f"  {who} sits out — out\n"
    if event.reason == "busted":
        return f"  {who} is eliminated (stack {event.stack})\n"
    return f"  {who} is out ({event.reason})\n"


def _rebuy(event: PlayerRebought) -> str:
    name = event.name or f"seat {event.seat}"
    return f"  {name} (seat {event.seat}) rebuys to {event.stack_after}\n"


def _street_ended(event: StreetEnded) -> str:
    if event.reason or event.board or event.active_seats or event.pot:
        board = f" | board {' '.join(event.board)}" if event.board else ""
        active = " ".join(str(seat) for seat in event.active_seats) or "-"
        folded = " ".join(str(seat) for seat in event.folded_seats) or "-"
        reason = event.reason or "closed"
        return (
            f"  Street {event.street} closed ({reason}) | pot {event.pot}{board}"
            f" | still in [{active}] | folded [{folded}]\n"
        )
    if event.snapshot:
        text = event.snapshot
        return text if text.endswith("\n") else text + "\n"
    return f"  Street {event.street} closed\n"


def _pots(event: PotsBuilt) -> str:
    if not event.pots:
        return "  Pots: (empty)\n"
    parts: list[str] = []
    for index, (amount, seats) in enumerate(event.pots):
        kind = "main" if index == 0 else f"side {index}"
        eligible = ", ".join(str(seat) for seat in seats)
        parts.append(f"{kind} {amount} eligible [{eligible}]")
    return "  Pots: " + "; ".join(parts) + "\n"


def _showdown(event: Showdown) -> str:
    header = "SHOWDOWN (uncontested)\n" if event.ended_early else "SHOWDOWN\n"
    lines = [header]
    seats = sorted(set(event.hole_cards) | set(event.rank_names) | set(event.hands))
    if event.ended_early and not event.rank_names:
        if not seats:
            lines.append("  hand ended early\n")
            return "".join(lines)
        for seat in seats:
            name = event.names.get(seat, f"seat {seat}")
            cards = " ".join(event.hole_cards.get(seat, ()))
            card_part = f" {cards}" if cards else ""
            lines.append(f"  {name} wins without showdown{card_part}\n")
        return "".join(lines)

    for seat in seats:
        name = event.names.get(seat, f"seat {seat}")
        cards = " ".join(event.hole_cards.get(seat, ()))
        rank = event.rank_names.get(seat, "")
        if cards and rank:
            lines.append(f"  {name} shows {cards} — {rank}\n")
        elif cards:
            lines.append(f"  {name} shows {cards}\n")
        elif rank:
            lines.append(f"  {name} — {rank}\n")
    return "".join(lines)


def _run_it_twice(event: RunItTwice) -> str:
    lines = [
        "  Run it twice\n",
        f"  Board 1: [{' '.join(event.board1)}]\n",
        f"  Board 2: [{' '.join(event.board2)}]\n",
    ]
    lines.append("  Board 1 awards: " + _award_map(event.awards1) + "\n")
    if event.ranks1:
        lines.append("  Board 1 hands: " + _rank_map(event.ranks1) + "\n")
    lines.append("  Board 2 awards: " + _award_map(event.awards2) + "\n")
    if event.ranks2:
        lines.append("  Board 2 hands: " + _rank_map(event.ranks2) + "\n")
    return "".join(lines)


def _award_map(awards: dict[int, int]) -> str:
    return " ".join(f"seat{seat}={amount}" for seat, amount in sorted(awards.items()))


def _rank_map(ranks: dict[int, str]) -> str:
    return "; ".join(f"seat{seat} {rank}" for seat, rank in sorted(ranks.items()))


def _pot_awarded(event: PotAwarded) -> str:
    winners = ", ".join(str(seat) for seat in event.winning_seats)
    split = " (split)" if event.split else ""
    run = f"Run {event.run} " if event.run else ""
    return f"  {run}Pot {event.pot_index}: {event.amount} to seats [{winners}]{split}\n"


def _hand_ended(event: HandEnded) -> str:
    source = event.net if event.net else event.chip_distribution
    parts: list[str] = []
    for seat in sorted(source):
        name = event.names.get(seat, f"seat{seat}")
        if event.stacks_after:
            stack = event.stacks_after.get(seat, 0)
            parts.append(f"{name} {_signed(source[seat])} (stack {stack})")
        else:
            parts.append(f"{name}={source[seat]}")
    return "Hand result: " + ", ".join(parts) + "\n\n"
