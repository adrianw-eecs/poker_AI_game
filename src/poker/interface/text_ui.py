"""Text-based user interface for rendering game state and prompting humans."""

from poker.domain.action import Action, ActionType
from poker.domain.hand import HandRank, HandType
from poker.evaluation.evaluator import evaluate, evaluate_with_best_cards
from poker.evaluation.kicker_display import should_show_kicker_and_card
from poker.state.game_state import GameState

# Position labels in seat order relative to the dealer button
_POSITION_LABELS_BY_COUNT: dict[int, list[str]] = {
    2: ["BTN/SB", "BB"],
    3: ["BTN", "SB", "BB"],
    4: ["BTN", "SB", "BB", "UTG"],
    5: ["BTN", "SB", "BB", "UTG", "CO"],
    6: ["BTN", "SB", "BB", "UTG", "MP", "CO"],
}


def _position_label(seat: int, dealer_seat: int, num_players: int) -> str:
    """Return the position label for a seat.

    Args:
        seat: The seat number to label.
        dealer_seat: The current dealer/button seat.
        num_players: Total number of players.

    Returns:
        A position abbreviation like "BTN", "SB", "BB", "UTG", etc.
    """
    labels = _POSITION_LABELS_BY_COUNT.get(num_players)
    if labels is None:
        return f"Seat{seat}"
    # Offset from dealer: dealer is position 0 (BTN)
    offset = (seat - dealer_seat) % num_players
    return labels[offset]


def _action_type_to_str(action_type: ActionType) -> str:
    """Return a short uppercase label for an action type."""
    return {
        ActionType.FOLD: "FOLD",
        ActionType.CHECK: "CHECK",
        ActionType.CALL: "CALL",
        ActionType.RAISE: "RAISE",
        ActionType.ALL_IN: "ALL-IN",
    }[action_type]


def render(state: GameState, viewer_seat: int) -> str:
    """Render the game state as a human-readable string.

    Displays community cards, pot totals, each player's stack and last action
    this street, a position marker, and the action history for this street.

    Args:
        state: The current game state (typically state.view_for(viewer_seat)).
        viewer_seat: The seat number of the human viewer.

    Returns:
        A formatted multi-line string representing the game state.
    """
    lines: list[str] = []

    # Header: hand number and street
    street_name = state.street.value.upper()
    lines.append(f"=== Hand #{state.hand_number + 1}  |  Street: {street_name} ===")
    lines.append("")

    # Community cards
    if state.community_cards:
        card_str = " ".join(str(c) for c in state.community_cards)
        lines.append(f"  Board: [ {card_str} ]")
    else:
        lines.append("  Board: [ - ]")

    # Pot(s)
    total_pot = sum(p.amount for p in state.pots)
    if len(state.pots) == 1:
        lines.append(f"  Pot:   {total_pot}")
    else:
        # Show each player's contribution this street
        player_commits = [
            str(p.committed_this_street)
            for p in state.players
            if not p.is_eliminated
        ]
        pot_detail = " + ".join(player_commits)
        lines.append(f"  Pot:   {total_pot}  ({pot_detail})")

    if state.current_bet_to_call > 0:
        lines.append(f"  To call: {state.current_bet_to_call}")

    lines.append("")
    lines.append("  Players:")

    num_active = sum(1 for p in state.players if not p.is_eliminated)

    # Build action-this-street map (last action per seat)
    last_action: dict[int, Action] = {}
    for seat, action in state.action_history_this_street:
        last_action[seat] = action

    for player in state.players:
        if player.is_eliminated:
            continue

        pos = _position_label(player.seat, state.dealer_seat, num_active)
        is_viewer = player.seat == viewer_seat
        is_acting = player.seat == state.action_on_seat

        # Hole cards: show viewer's own cards, hide others
        if is_viewer and player.hole_cards:
            hole_str = " ".join(str(c) for c in player.hole_cards)
            card_part = f"[{hole_str}]"
        elif player.hole_cards and len(player.hole_cards) > 0:
            card_part = "[?? ??]" if len(player.hole_cards) == 2 else "[?]"
        else:
            card_part = "[     ]"

        # Status flags
        status_parts: list[str] = []
        if player.has_folded:
            status_parts.append("FOLDED")
        elif player.is_all_in:
            status_parts.append("ALL-IN")
        if is_acting:
            status_parts.append("<<< action")
        status = f"  ({', '.join(status_parts)})" if status_parts else ""

        # Last action this street
        if player.seat in last_action:
            act = last_action[player.seat]
            if act.type == ActionType.FOLD:
                action_str = "folded"
            elif act.type == ActionType.CHECK:
                action_str = "checked"
            elif act.type == ActionType.CALL:
                action_str = f"called {act.amount}"
            elif act.type == ActionType.RAISE:
                action_str = f"raised to {act.amount}"
            else:  # ALL_IN
                action_str = f"all-in {act.amount}"
            action_part = f" — {action_str}"
        else:
            action_part = ""

        viewer_marker = " (YOU)" if is_viewer else ""
        available_stack = state.config.starting_stack - player.committed_this_hand
        lines.append(
            f"    [{pos:8s}] {player.name}{viewer_marker}  "
            f"stack={available_stack}  {card_part}{action_part}{status}"
        )

    # Action history this street
    if state.action_history_this_street:
        lines.append("")
        lines.append("  Action this street:")
        for seat, action in state.action_history_this_street:
            player = state.players[seat]
            act_label = _action_type_to_str(action.type)
            if action.amount > 0:
                lines.append(f"    {player.name} (seat {seat}): {act_label} {action.amount}")
            else:
                lines.append(f"    {player.name} (seat {seat}): {act_label}")

    lines.append("")
    return "\n".join(lines)


def _parse_input(raw: str, legal: list[Action]) -> Action | None:
    """Parse a raw input string into a legal Action, or return None on failure.

    Accepted input formats (case-insensitive):
        "f"        → FOLD
        "k"        → CHECK
        "c"        → CALL
        "r <N>"    → RAISE to N chips
        "a"        → ALL_IN

    Args:
        raw: The raw user input string.
        legal: The list of legal actions.

    Returns:
        The matching legal Action, or None if the input is invalid.
    """
    token = raw.strip().lower()
    legal_types = {a.type for a in legal}

    if token == "f":
        if ActionType.FOLD in legal_types:
            for a in legal:
                if a.type == ActionType.FOLD:
                    return a
        return None

    if token == "k":
        if ActionType.CHECK in legal_types:
            for a in legal:
                if a.type == ActionType.CHECK:
                    return a
        return None

    if token == "c":
        if ActionType.CALL in legal_types:
            for a in legal:
                if a.type == ActionType.CALL:
                    return a
        return None

    if token == "a":
        if ActionType.ALL_IN in legal_types:
            for a in legal:
                if a.type == ActionType.ALL_IN:
                    return a
        return None

    if token.startswith("r "):
        parts = token.split()
        if len(parts) == 2:
            try:
                amount = int(parts[1])
            except ValueError:
                return None
            if ActionType.RAISE not in legal_types:
                return None
            # Find the min and max valid raise amounts
            min_raise = next((a.amount for a in legal if a.type == ActionType.RAISE), None)
            all_in_amount = next((a.amount for a in legal if a.type == ActionType.ALL_IN), None)
            max_raise = all_in_amount if all_in_amount is not None else min_raise
            if min_raise is None:
                return None
            if amount < min_raise:
                return None
            if max_raise is not None and amount > max_raise:
                return None
            return Action.raise_to(amount)

    return None


def _legal_actions_summary(legal: list[Action]) -> str:
    """Return a human-readable summary of the available actions.

    Args:
        legal: The list of legal actions.

    Returns:
        A one-line summary string for displaying to the user.
    """
    parts: list[str] = []
    for action in legal:
        if action.type == ActionType.FOLD:
            parts.append("f=fold")
        elif action.type == ActionType.CHECK:
            parts.append("k=check")
        elif action.type == ActionType.CALL:
            parts.append(f"c=call({action.amount})")
        elif action.type == ActionType.RAISE:
            # Find all-in to show max
            all_in = next((a.amount for a in legal if a.type == ActionType.ALL_IN), None)
            if all_in is not None and all_in > action.amount:
                parts.append(f"r N=raise({action.amount}-{all_in})")
            else:
                parts.append(f"r N=raise({action.amount})")
        elif action.type == ActionType.ALL_IN:
            parts.append(f"a=all-in({action.amount})")
    return "  Actions: " + "  |  ".join(parts)


def prompt(state: GameState, legal: list[Action]) -> Action:
    """Read user input and return a valid legal action.

    Displays the available actions, reads from stdin, validates the input
    against the legal set, and retries on invalid input.

    Args:
        state: The current game state (used for context display).
        legal: The list of legal actions.

    Returns:
        A legal Action chosen by the user.
    """
    summary = _legal_actions_summary(legal)
    while True:
        print(summary)
        try:
            raw = input("  Your action: ")
        except EOFError:
            # Non-interactive mode: default to fold
            return Action.fold()

        action = _parse_input(raw, legal)
        if action is not None:
            return action

        # Invalid input: show an error and retry
        valid_cmds: list[str] = []
        for a in legal:
            if a.type == ActionType.FOLD:
                valid_cmds.append('"f"')
            elif a.type == ActionType.CHECK:
                valid_cmds.append('"k"')
            elif a.type == ActionType.CALL:
                valid_cmds.append('"c"')
            elif a.type == ActionType.RAISE:
                valid_cmds.append('"r <amount>"')
            elif a.type == ActionType.ALL_IN:
                valid_cmds.append('"a"')
        print(f"  Invalid input '{raw}'. Valid commands: {', '.join(valid_cmds)}")


def display_showdown(final_state: GameState, awards: dict[int, int]) -> None:
    """Display the showdown information including cards, hands, and results.

    Shows all players' hole cards, the final board, best hand evaluations,
    and who won what chips. Kickers are shown only when they are the deciding
    factor between hands of the same rank.

    Args:
        final_state: The final game state after showdown.
        awards: Dict mapping seat → chips won (from the pot).
    """
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 50)
    lines.append("SHOWDOWN")
    lines.append("=" * 50)
    lines.append("")

    # Display board
    if final_state.community_cards:
        card_str = " ".join(str(c) for c in final_state.community_cards)
        lines.append(f"Board: [ {card_str} ]")
    else:
        lines.append("Board: (hand ended early - no board)")

    lines.append("")

    # First pass: evaluate all active players' hands and store the results
    player_hands: dict[int, tuple[str, tuple, tuple]] = {}  # seat -> (hand_eval, hand_rank, best_5_cards)
    for seat, player in enumerate(final_state.players):
        if player.is_eliminated:
            continue

        if player.has_folded:
            player_hands[seat] = ("folded", None, None)
        else:
            # Evaluate hand for active players (only if we have enough community cards)
            all_cards = list(player.hole_cards) + list(final_state.community_cards)
            if len(all_cards) >= 5:
                hand_rank, best_5_cards = evaluate_with_best_cards(all_cards)
                player_hands[seat] = (hand_rank, best_5_cards, hand_rank)
            else:
                # Not enough cards (hand ended early)
                player_hands[seat] = ("no showdown", None, None)

    # Second pass: determine which hands get kicker display
    hand_displays: dict[int, str] = {}  # seat -> display string
    for seat, (hand_rank, best_5_cards, _) in player_hands.items():
        if isinstance(hand_rank, str):  # "folded" or "no showdown"
            hand_displays[seat] = hand_rank
        else:
            # Get all other hands for comparison (excluding folded/no-showdown)
            other_hands = [
                (rank, cards)
                for s, (rank, cards, _) in player_hands.items()
                if s != seat and isinstance(rank, HandRank)
            ]

            # Check if kicker should be shown
            show_kicker, kicker_rank = should_show_kicker_and_card(
                hand_rank, best_5_cards, other_hands
            )

            # Build the hand display string
            if show_kicker and kicker_rank is not None:
                hand_eval = _hand_rank_with_kicker(hand_rank, kicker_rank)
            else:
                hand_eval = _hand_rank_without_extra_kicker(hand_rank)

            hand_displays[seat] = hand_eval

    # Third pass: display each player's cards and hand evaluation
    for seat, player in enumerate(final_state.players):
        if player.is_eliminated:
            continue

        pos = _position_label(seat, final_state.dealer_seat, len(final_state.players))

        # Format hole cards
        if player.hole_cards:
            card_str = " ".join(str(c) for c in player.hole_cards)
            card_part = f"[{card_str}]"
        else:
            card_part = "[??]"

        # Get hand evaluation
        hand_eval = hand_displays.get(seat, "error")

        # Check if net winner (profit > 0)
        committed = player.committed_this_hand
        award = awards.get(seat, 0)
        net_change = award - committed
        if net_change > 0:
            winner_marker = f"  (WINNER: +{net_change})"
        else:
            winner_marker = ""

        lines.append(f"  {player.name:20s} ({pos:8s}): {card_part}  -> {hand_eval}{winner_marker}")

    lines.append("")

    # Display chip distribution relative to what they committed
    for seat, player in enumerate(final_state.players):
        if player.is_eliminated:
            continue

        committed = player.committed_this_hand
        award = awards.get(seat, 0)
        net_change = award - committed

        if net_change > 0:
            lines.append(f"  Hand result: {player.name} wins {net_change} chips")
        elif net_change < 0:
            lines.append(f"  Hand result: {player.name} loses {-net_change} chips")
        else:
            lines.append(f"  Hand result: {player.name} breaks even")

    lines.append("")

    print("\n".join(lines))


def _hand_rank_without_extra_kicker(hand_rank: "HandRank") -> str:
    """Return hand description without showing kicker information.

    This suppresses the default kicker display from HandRank.__str__().
    """
    from poker.domain.hand import _RANK_NAMES

    if hand_rank.type == HandType.HIGH_CARD:
        kicker_strs = [_RANK_NAMES[k] for k in hand_rank.kickers]
        return f"High card, {', '.join(kicker_strs[:3])}"

    if hand_rank.type == HandType.PAIR:
        pair_rank = hand_rank.kickers[0]
        return f"Pair of {_RANK_NAMES[pair_rank]}s"

    if hand_rank.type == HandType.TWO_PAIR:
        high = hand_rank.kickers[0]
        low = hand_rank.kickers[1]
        return f"Two pair, {_RANK_NAMES[high]}s and {_RANK_NAMES[low]}s"

    if hand_rank.type == HandType.THREE_OF_A_KIND:
        trips = hand_rank.kickers[0]
        return f"Three of a kind, {_RANK_NAMES[trips]}s"

    if hand_rank.type == HandType.STRAIGHT:
        high = hand_rank.kickers[0]
        return f"Straight, {_RANK_NAMES[high]}-high"

    if hand_rank.type == HandType.FLUSH:
        high = hand_rank.kickers[0]
        return f"Flush, {_RANK_NAMES[high]}-high"

    if hand_rank.type == HandType.FULL_HOUSE:
        trips = hand_rank.kickers[0]
        pair = hand_rank.kickers[1]
        return f"Full house, {_RANK_NAMES[trips]}s full of {_RANK_NAMES[pair]}s"

    if hand_rank.type == HandType.FOUR_OF_A_KIND:
        quads = hand_rank.kickers[0]
        return f"Four of a kind, {_RANK_NAMES[quads]}s"

    # STRAIGHT_FLUSH
    high = hand_rank.kickers[0]
    return f"Straight flush, {_RANK_NAMES[high]}-high"


def _hand_rank_with_kicker(hand_rank: "HandRank", kicker_rank: int) -> str:
    """Return hand description with a specific kicker highlighted.

    Args:
        hand_rank: The HandRank object.
        kicker_rank: The rank value of the kicker to display (2-14).

    Returns:
        Hand description string with kicker appended.
    """
    from poker.domain.hand import _RANK_NAMES

    base_desc = _hand_rank_without_extra_kicker(hand_rank)
    kicker_name = _RANK_NAMES[kicker_rank]
    return f"{base_desc}, {kicker_name} kicker"
