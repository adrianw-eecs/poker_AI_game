"""Showdown resolution and pot distribution."""

from poker.domain.deck import Deck
from poker.domain.hand import HandRank
from poker.evaluation.evaluator import evaluate
from poker.state.game_state import GameState, Street
from poker.state.pot_manager import apply_rake, build_pots, distribute


def resolve(state: GameState, deck: Deck) -> tuple[dict[int, int], int]:
    """Determine chip awards at showdown.

    Evaluates hand outcomes, applies rake, and distributes pots.
    Handles standard showdown and run-it-twice scenarios.

    Args:
        state: The final GameState (assumed to have completed all betting).
        deck: The remaining deck (needed for run-it-twice).

    Returns:
        Tuple of (awards dict mapping seat → chips won/lost, total rake taken).
    """
    # Identify non-folded players
    non_folded = [
        i for i, p in enumerate(state.players)
        if not p.has_folded and not p.is_eliminated
    ]

    # Fast path: single winner (everyone else folded)
    if len(non_folded) == 1:
        winner_seat = non_folded[0]
        total_pot = sum(p.committed_this_hand for p in state.players)
        # No rake on uncalled portion (hand ended early)
        awards: dict[int, int] = {}
        for seat in range(len(state.players)):
            awards[seat] = total_pot if seat == winner_seat else 0
        return awards, 0  # No rake for uncontested pots

    # Multi-way showdown: evaluate all remaining hands
    hand_ranks = {}
    for seat in non_folded:
        player = state.players[seat]
        # Combine hole cards with community cards
        all_cards = list(player.hole_cards) + list(state.community_cards)
        hand_ranks[seat] = evaluate(all_cards)

    # Check if run-it-twice should be triggered
    if _should_run_it_twice(state, non_folded):
        return _resolve_run_it_twice(state, non_folded, hand_ranks, deck)

    # Standard showdown: build pots, apply rake, distribute
    folded_seats = {
        i for i, p in enumerate(state.players)
        if p.has_folded or p.is_eliminated
    }
    committed = {i: state.players[i].committed_this_hand for i in range(len(state.players))}

    pots = build_pots(committed, folded_seats)
    pots_after_rake, rake_taken = apply_rake(
        pots, state.config.rake_percent, state.config.rake_cap
    )
    awards = distribute(pots_after_rake, hand_ranks, state.dealer_seat)

    return awards, rake_taken


def _should_run_it_twice(state: GameState, non_folded: list[int]) -> bool:
    """Check if run-it-twice should be triggered.

    Run-it-twice is triggered when:
    - Exactly 2 players remain
    - Both are all-in
    - At least 1 street remains to deal (not on river)
    - config.run_it_twice is True

    Args:
        state: Current game state.
        non_folded: List of non-folded player seats.

    Returns:
        True if run-it-twice should be triggered.
    """
    if not state.config.run_it_twice:
        return False

    if len(non_folded) != 2:
        return False

    # Both must be all-in
    for seat in non_folded:
        if not state.players[seat].is_all_in:
            return False

    # Check if we're before the river (so there are streets left to deal)
    remaining_streets = 0
    if state.street == Street.PREFLOP:
        remaining_streets = 3  # flop, turn, river
    elif state.street == Street.FLOP:
        remaining_streets = 2  # turn, river
    elif state.street == Street.TURN:
        remaining_streets = 1  # river
    else:  # RIVER or SHOWDOWN
        remaining_streets = 0

    return remaining_streets >= 1


def _resolve_run_it_twice(
    state: GameState,
    non_folded: list[int],
    hand_ranks: dict[int, HandRank],
    deck: Deck,
) -> dict[int, int]:
    """Resolve a hand using run-it-twice rules.

    Deals two independent boards and evaluates both. Awards are averaged
    across the two boards.

    Args:
        state: Current game state.
        non_folded: List of non-folded player seats.
        hand_ranks: Hand ranks from current board.
        deck: The remaining deck for dealing new boards.

    Returns:
        Dict mapping seat → averaged chips won across both run-outs.
    """
    # Calculate how many streets remain
    if state.street == Street.PREFLOP:
        remaining_streets = 3
    elif state.street == Street.FLOP:
        remaining_streets = 2
    elif state.street == Street.TURN:
        remaining_streets = 1
    else:
        remaining_streets = 0

    if remaining_streets < 1:
        # Fallback to standard showdown if no streets remain
        folded_seats = {
            i for i, p in enumerate(state.players)
            if p.has_folded or p.is_eliminated
        }
        committed = {i: state.players[i].committed_this_hand for i in range(len(state.players))}
        pots = build_pots(committed, folded_seats)
        pots_after_rake, rake_taken = apply_rake(
            pots, state.config.rake_percent, state.config.rake_cap
        )
        awards = distribute(pots_after_rake, hand_ranks, state.dealer_seat)
        return awards, rake_taken

    # Deal two independent boards
    from poker.engine.dealer import deal_run_it_twice

    boards = deal_run_it_twice(deck, state.community_cards, remaining_streets)
    board1, board2 = boards

    # Evaluate both boards for each player
    awards_board1: dict[int, int] = {}
    awards_board2: dict[int, int] = {}
    total_rake = 0

    for board in [board1, board2]:
        # Evaluate hands on this board
        hand_ranks_this_board: dict[int, HandRank] = {}
        for seat in non_folded:
            player = state.players[seat]
            all_cards = list(player.hole_cards) + list(board)
            hand_ranks_this_board[seat] = evaluate(all_cards)

        # Distribute pots for this board
        folded_seats = {
            i for i, p in enumerate(state.players)
            if p.has_folded or p.is_eliminated
        }
        committed = {i: state.players[i].committed_this_hand for i in range(len(state.players))}
        pots = build_pots(committed, folded_seats)
        pots_after_rake, rake_from_board = apply_rake(
            pots, state.config.rake_percent, state.config.rake_cap
        )
        board_awards = distribute(pots_after_rake, hand_ranks_this_board, state.dealer_seat)
        total_rake += rake_from_board

        # Accumulate awards
        if board == board1:
            awards_board1 = board_awards
        else:
            awards_board2 = board_awards

    # Average the awards across both boards
    final_awards: dict[int, int] = {}
    for seat in range(len(state.players)):
        board1_award = awards_board1.get(seat, 0)
        board2_award = awards_board2.get(seat, 0)
        # Integer division for chips (any remainder is already handled by rake)
        final_awards[seat] = (board1_award + board2_award) // 2

    # Note: run-it-twice rake is taken twice (once per board), so we return total
    return final_awards, total_rake
