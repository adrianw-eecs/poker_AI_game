"""Text-based user interface for rendering game state and prompting humans."""

from poker.domain.action import Action, ActionType
from poker.domain.hand import HandRank, HandType
from poker.evaluation.kicker_display import should_show_kicker_and_card
from poker.logging.state_snapshot import format_state_snapshot, position_label
from poker.state.game_state import GameState


def render(state: GameState, viewer_seat: int) -> str:
    """Render the game state as a human-readable string.

    Delegates to the shared snapshot formatter with viewer-relative card masking
    and live pot rebuild for accurate mid-street display.

    Args:
        state: The current game state (typically state.view_for(viewer_seat)).
        viewer_seat: The seat number of the human viewer.

    Returns:
        A formatted multi-line string representing the game state.
    """
    return format_state_snapshot(
        state,
        viewer_seat=viewer_seat,
        indent="",
        use_color_cards=True,
    )


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
    """Return a human-readable summary of the available actions."""
    parts: list[str] = []
    for action in legal:
        if action.type == ActionType.FOLD:
            parts.append("f=fold")
        elif action.type == ActionType.CHECK:
            parts.append("k=check")
        elif action.type == ActionType.CALL:
            parts.append(f"c=call({action.amount})")
        elif action.type == ActionType.RAISE:
            all_in = next((a.amount for a in legal if a.type == ActionType.ALL_IN), None)
            if all_in is not None and all_in > action.amount:
                parts.append(f"r N=raise({action.amount}-{all_in})")
            else:
                parts.append(f"r N=raise({action.amount})")
        elif action.type == ActionType.ALL_IN:
            parts.append(f"a=all-in({action.amount})")
    return "  Actions: " + "  |  ".join(parts)


def prompt(state: GameState, legal: list[Action]) -> Action:
    """Read user input and return a valid legal action."""
    summary = _legal_actions_summary(legal)
    while True:
        print(summary)
        try:
            raw = input("  Your action: ")
        except EOFError:
            return Action.fold()

        action = _parse_input(raw, legal)
        if action is not None:
            return action

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
    """Display showdown information including cards, hands, and results."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 50)
    lines.append("SHOWDOWN")
    lines.append("=" * 50)
    lines.append("")

    if final_state.community_cards:
        card_str = " ".join(str(c) for c in final_state.community_cards)
        lines.append(f"Board: [ {card_str} ]")
    else:
        lines.append("Board: (hand ended early - no board)")

    lines.append("")

    player_hands: dict[int, tuple[str, tuple, tuple]] = {}
    for seat, player in enumerate(final_state.players):
        if player.is_eliminated:
            continue

        if player.has_folded:
            player_hands[seat] = ("folded", None, None)
        else:
            all_cards = list(player.hole_cards) + list(final_state.community_cards)
            if len(all_cards) >= 5:
                from poker.evaluation.evaluator import evaluate_with_best_cards
                hand_rank, best_5_cards = evaluate_with_best_cards(all_cards)
                player_hands[seat] = (hand_rank, best_5_cards, hand_rank)
            else:
                player_hands[seat] = ("no showdown", None, None)

    hand_displays: dict[int, str] = {}
    for seat, (hand_rank, best_5_cards, _) in player_hands.items():
        if isinstance(hand_rank, str):
            hand_displays[seat] = hand_rank
        else:
            other_hands = [
                (rank, cards)
                for s, (rank, cards, _) in player_hands.items()
                if s != seat and isinstance(rank, HandRank)
            ]
            show_kicker, kicker_rank = should_show_kicker_and_card(
                hand_rank, best_5_cards, other_hands
            )
            if show_kicker and kicker_rank is not None:
                hand_eval = _hand_rank_with_kicker(hand_rank, kicker_rank)
            else:
                hand_eval = _hand_rank_without_extra_kicker(hand_rank)
            hand_displays[seat] = hand_eval

    num_players = len(final_state.players)
    for seat, player in enumerate(final_state.players):
        if player.is_eliminated:
            continue

        pos = position_label(seat, final_state.dealer_seat, num_players)

        if player.hole_cards:
            card_str = " ".join(str(c) for c in player.hole_cards)
            card_part = f"[{card_str}]"
        else:
            card_part = "[??]"

        hand_eval = hand_displays.get(seat, "error")

        committed = player.committed_this_hand
        award = awards.get(seat, 0)
        net_change = award - committed
        if net_change > 0:
            winner_marker = f"  (WINNER: +{net_change})"
        else:
            winner_marker = ""

        status = ""
        if player.has_folded:
            status = "  (FOLDED)"
        elif player.is_eliminated:
            status = "  (OUT)"

        lines.append(
            f"  {player.name:20s} ({pos:8s}): {card_part}  -> "
            f"{hand_eval}{winner_marker}{status}"
        )

    lines.append("")

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
    """Return hand description without showing kicker information."""
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

    high = hand_rank.kickers[0]
    return f"Straight flush, {_RANK_NAMES[high]}-high"


def _hand_rank_with_kicker(hand_rank: "HandRank", kicker_rank: int) -> str:
    """Return hand description with a specific kicker highlighted."""
    from poker.domain.hand import _RANK_NAMES

    base_desc = _hand_rank_without_extra_kicker(hand_rank)
    kicker_name = _RANK_NAMES[kicker_rank]
    return f"{base_desc}, {kicker_name} kicker"
