"""Showdown resolution and pot distribution."""

from dataclasses import dataclass, field

from poker.domain.card import Card
from poker.domain.deck import Deck
from poker.domain.hand import HandRank
from poker.evaluation.evaluator import evaluate_with_best_cards
from poker.state.game_state import GameState, Street
from poker.state.pot import Pot
from poker.state.pot_manager import apply_rake, build_pots, distribute_detailed


@dataclass(frozen=True)
class HandResolution:
    """Chip awards plus the facts needed to log showdown."""

    awards: dict[int, int]
    rake: int
    ended_early: bool
    pots: list[Pot]
    ranks: dict[int, HandRank] = field(default_factory=dict)
    best_five: dict[int, tuple[Card, ...]] = field(default_factory=dict)
    # (run, amount, winning_seats, split). run is 0, or 1 and 2 for run-it-twice.
    pot_awards: list[tuple[int, int, tuple[int, ...], bool]] = field(default_factory=list)
    board1: tuple[Card, ...] | None = None
    board2: tuple[Card, ...] | None = None
    awards1: dict[int, int] | None = None
    awards2: dict[int, int] | None = None
    ranks1: dict[int, HandRank] | None = None
    ranks2: dict[int, HandRank] | None = None
    best1: dict[int, tuple[Card, ...]] | None = None
    best2: dict[int, tuple[Card, ...]] | None = None


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
    details = resolve_hand(state, deck)
    return details.awards, details.rake


def resolve_hand(state: GameState, deck: Deck) -> HandResolution:
    """Resolve the hand and return awards plus showdown facts."""
    non_folded = [
        i for i, p in enumerate(state.players)
        if not p.has_folded and not p.is_eliminated
    ]

    if len(non_folded) == 1:
        winner_seat = non_folded[0]
        total_pot = sum(p.committed_this_hand for p in state.players)
        awards: dict[int, int] = {}
        for seat in range(len(state.players)):
            awards[seat] = total_pot if seat == winner_seat else 0
        pot = Pot(amount=total_pot, eligible_seats=frozenset({winner_seat}))
        return HandResolution(
            awards=awards,
            rake=0,
            ended_early=True,
            pots=[pot],
            pot_awards=[(0, total_pot, (winner_seat,), False)],
        )

    ranks, best_five = _evaluate_board(state, non_folded, state.community_cards)

    if _should_run_it_twice(state, non_folded):
        return _resolve_run_it_twice(state, non_folded, deck)

    folded_seats = {
        i for i, p in enumerate(state.players)
        if p.has_folded or p.is_eliminated
    }
    committed = {i: state.players[i].committed_this_hand for i in range(len(state.players))}

    pots = build_pots(committed, folded_seats)
    pots_after_rake, rake_taken = apply_rake(
        pots, state.config.rake_percent, state.config.rake_cap
    )
    awards, details = distribute_detailed(pots_after_rake, ranks, state.dealer_seat)
    pot_awards = [(0, amount, winners, split) for amount, winners, split in details]
    return HandResolution(
        awards=awards,
        rake=rake_taken,
        ended_early=False,
        pots=pots_after_rake,
        ranks=ranks,
        best_five=best_five,
        pot_awards=pot_awards,
    )


def _evaluate_board(
    state: GameState,
    seats: list[int],
    board: tuple[Card, ...],
) -> tuple[dict[int, HandRank], dict[int, tuple[Card, ...]]]:
    ranks: dict[int, HandRank] = {}
    best: dict[int, tuple[Card, ...]] = {}
    for seat in seats:
        player = state.players[seat]
        all_cards = list(player.hole_cards) + list(board)
        if len(all_cards) not in (5, 6, 7):
            continue
        rank, cards = evaluate_with_best_cards(all_cards)
        ranks[seat] = rank
        best[seat] = cards
    return ranks, best


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

    for seat in non_folded:
        if not state.players[seat].is_all_in:
            return False

    remaining_streets = 0
    if state.street == Street.PREFLOP:
        remaining_streets = 3
    elif state.street == Street.FLOP:
        remaining_streets = 2
    elif state.street == Street.TURN:
        remaining_streets = 1
    else:
        remaining_streets = 0

    return remaining_streets >= 1


def _resolve_run_it_twice(
    state: GameState,
    non_folded: list[int],
    deck: Deck,
) -> HandResolution:
    """Resolve a hand using run-it-twice rules.

    Deals two independent boards and evaluates both. Awards are averaged
    across the two boards.
    """
    if state.street == Street.PREFLOP:
        remaining_streets = 3
    elif state.street == Street.FLOP:
        remaining_streets = 2
    elif state.street == Street.TURN:
        remaining_streets = 1
    else:
        remaining_streets = 0

    folded_seats = {
        i for i, p in enumerate(state.players)
        if p.has_folded or p.is_eliminated
    }
    committed = {i: state.players[i].committed_this_hand for i in range(len(state.players))}

    if remaining_streets < 1:
        ranks, best_five = _evaluate_board(state, non_folded, state.community_cards)
        pots = build_pots(committed, folded_seats)
        pots_after_rake, rake_taken = apply_rake(
            pots, state.config.rake_percent, state.config.rake_cap
        )
        awards, details = distribute_detailed(pots_after_rake, ranks, state.dealer_seat)
        return HandResolution(
            awards=awards,
            rake=rake_taken,
            ended_early=False,
            pots=pots_after_rake,
            ranks=ranks,
            best_five=best_five,
            pot_awards=[(0, amount, winners, split) for amount, winners, split in details],
        )

    from poker.engine.dealer import deal_run_it_twice

    boards = deal_run_it_twice(deck, state.community_cards, remaining_streets)
    board1, board2 = boards

    awards_board1: dict[int, int] = {}
    awards_board2: dict[int, int] = {}
    ranks1: dict[int, HandRank] = {}
    ranks2: dict[int, HandRank] = {}
    best1: dict[int, tuple[Card, ...]] = {}
    best2: dict[int, tuple[Card, ...]] = {}
    pot_awards: list[tuple[int, int, tuple[int, ...], bool]] = []
    pots_after: list[Pot] = []
    total_rake = 0

    for run_index, board in enumerate((board1, board2), start=1):
        hand_ranks, best_cards = _evaluate_board(state, non_folded, board)
        pots = build_pots(committed, folded_seats)
        pots_after_rake, rake_from_board = apply_rake(
            pots, state.config.rake_percent, state.config.rake_cap
        )
        board_awards, details = distribute_detailed(
            pots_after_rake, hand_ranks, state.dealer_seat
        )
        total_rake = rake_from_board
        if run_index == 1:
            awards_board1 = board_awards
            ranks1 = hand_ranks
            best1 = best_cards
            pots_after = pots_after_rake
        else:
            awards_board2 = board_awards
            ranks2 = hand_ranks
            best2 = best_cards
        for amount, winners, split in details:
            pot_awards.append((run_index, amount, winners, split))

    final_awards: dict[int, int] = {}
    # Both boards award the same pot, so the hand's rake is what one run takes,
    # not the sum over both runs. Accumulating reported double the real rake.
    hand_rake = total_rake
    for seat in range(len(state.players)):
        board1_award = awards_board1.get(seat, 0)
        board2_award = awards_board2.get(seat, 0)
        final_awards[seat] = (board1_award + board2_award) // 2

    # Averaging an odd total with floor division silently destroys a chip.
    # Give the remainder to the first eligible seat clockwise from the dealer,
    # matching the odd-chip rule used for a normal chopped pot.
    pot_total = sum(pot.amount for pot in pots_after)
    leftover = pot_total - sum(final_awards.values())
    if leftover > 0:
        num_players = len(state.players)
        for offset in range(1, num_players + 1):
            seat = (state.dealer_seat + offset) % num_players
            if seat in awards_board1 or seat in awards_board2:
                final_awards[seat] += leftover
                break
        else:
            final_awards[next(iter(final_awards))] += leftover

    return HandResolution(
        awards=final_awards,
        rake=hand_rake,
        ended_early=False,
        pots=pots_after,
        ranks=ranks1,
        best_five=best1,
        pot_awards=pot_awards,
        board1=board1,
        board2=board2,
        awards1=awards_board1,
        awards2=awards_board2,
        ranks1=ranks1,
        ranks2=ranks2,
        best1=best1,
        best2=best2,
    )
