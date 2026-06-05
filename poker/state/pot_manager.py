"""Pot calculation and distribution manager."""

from dataclasses import replace

from poker.domain.hand import HandRank
from poker.state.game_state import GameState
from poker.state.pot import Pot


def build_pots(committed_by_seat: dict[int, int], folded: set[int]) -> list[Pot]:
    """Build main pot and side pots from player commitments.

    Creates pots based on stack levels, with separate eligibility sets.
    A folded player is not eligible for any pot, but their chips contribute
    to pot amounts.

    Args:
        committed_by_seat: Dict mapping seat to total chips committed this hand.
        folded: Set of seat numbers that have folded.

    Returns:
        List of Pot objects, ordered from main pot to side pots.
    """
    if not committed_by_seat:
        return []

    # Get unique commitment levels, sorted ascending
    all_seats = set(committed_by_seat.keys())
    active_seats = all_seats - folded
    levels = sorted(set(committed_by_seat.values()))

    pots: list[Pot] = []
    previous_level = 0

    for level in levels:
        # Seats eligible for this pot: those still active at this commitment level
        # (had enough to match this pot round)
        eligible = frozenset(
            seat for seat in active_seats if committed_by_seat.get(seat, 0) >= level
        )

        if not eligible:
            # No one is eligible for this pot (shouldn't happen with good data)
            continue

        # Amount in this pot: difference from previous level, times number of contributors
        contributors = frozenset(
            seat for seat in all_seats if committed_by_seat.get(seat, 0) >= level
        )
        amount = (level - previous_level) * len(contributors)

        pots.append(Pot(amount=amount, eligible_seats=eligible))
        previous_level = level

    return pots


def rebuild_pots(state: GameState) -> GameState:
    """Rebuild pots from current player commitments."""
    committed = {
        i: state.players[i].committed_this_hand for i in range(len(state.players))
    }
    folded = {
        i for i, p in enumerate(state.players) if p.has_folded or p.is_eliminated
    }
    return replace(state, pots=build_pots(committed, folded))


def apply_rake(
    pots: list[Pot], rake_percent: float, rake_cap: int | None
) -> tuple[list[Pot], int]:
    """Apply rake to pots and return rake-adjusted pots.

    Rake is taken independently from each pot, then capped at rake_cap if specified.

    Args:
        pots: List of pots to apply rake to.
        rake_percent: Rake percentage (0-100).
        rake_cap: Maximum rake to take, or None for no cap.

    Returns:
        Tuple of (rake-adjusted pots, total rake taken).
    """
    if rake_percent == 0 or not pots:
        return pots, 0

    adjusted_pots: list[Pot] = []
    total_rake = 0

    for pot in pots:
        # Calculate rake for this pot
        rake_from_pot = int(pot.amount * rake_percent / 100)
        # Cap the rake if specified
        if rake_cap is not None:
            remaining_cap = max(0, rake_cap - total_rake)
            rake_from_pot = min(rake_from_pot, remaining_cap)

        total_rake += rake_from_pot
        adjusted_amount = pot.amount - rake_from_pot
        adjusted_pots.append(Pot(amount=adjusted_amount, eligible_seats=pot.eligible_seats))

    return adjusted_pots, total_rake


def distribute(
    pots: list[Pot],
    hand_ranks: dict[int, HandRank],
    dealer_seat: int,
) -> dict[int, int]:
    """Distribute pots to winning seat(s).

    Awards pots to the best hand. In case of a tie (chop), odd chip(s) go
    clockwise from the dealer.

    Args:
        pots: List of pots to distribute.
        hand_ranks: Dict mapping seat to HandRank (for all eligible seats).
        dealer_seat: Seat of the dealer (for chop tiebreaker).

    Returns:
        Dict mapping seat to chips awarded from this distribution.
    """
    awards: dict[int, int] = {}

    for pot in pots:
        # Find winners in this pot (best hand among eligible seats)
        eligible_seats = list(pot.eligible_seats)
        if not eligible_seats:
            continue

        # Get hand ranks for eligible seats only
        eligible_ranks = {
            seat: hand_ranks[seat] for seat in eligible_seats if seat in hand_ranks
        }

        if not eligible_ranks:
            continue

        # Find the best hand
        best_rank = max(eligible_ranks.values())
        winners = [seat for seat, rank in eligible_ranks.items() if rank == best_rank]

        if len(winners) == 1:
            # Single winner gets the whole pot
            awards[winners[0]] = awards.get(winners[0], 0) + pot.amount
        else:
            # Chop: split pot, odd chips go clockwise from dealer
            chips_per_winner = pot.amount // len(winners)
            odd_chips = pot.amount % len(winners)

            for winner in winners:
                awards[winner] = awards.get(winner, 0) + chips_per_winner

            # Distribute odd chips clockwise from dealer (starting after dealer)
            if odd_chips > 0:
                # Create a list of all seats in clockwise order starting after dealer
                max_seat = max(eligible_seats) if eligible_seats else max(winners)
                next_seat = dealer_seat + 1
                clockwise_order = list(range(next_seat, max_seat + 1)) + list(
                    range(0, next_seat)
                )
                # Distribute odd chips to winners in clockwise order
                winner_list = set(winners)
                for _ in range(odd_chips):
                    for seat in clockwise_order:
                        if seat in winner_list:
                            awards[seat] = awards.get(seat, 0) + 1
                            winner_list.remove(seat)
                            break

    return awards
