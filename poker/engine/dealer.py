"""Card dealing primitives for poker games."""

from poker.domain.card import Card
from poker.domain.deck import Deck
from poker.state.player_state import PlayerState


def deal_hole_cards(
    deck: Deck,
    players: tuple[PlayerState, ...],
    button_seat: int,
) -> tuple[PlayerState, ...]:
    """Deal 2 hole cards to each player, starting left of button.

    In poker, the small blind position is to the left of the button (dealer).
    Cards are dealt starting from the small blind and continuing clockwise.

    Args:
        deck: The deck to deal from.
        players: All players in the game.
        button_seat: The seat number of the dealer/button.

    Returns:
        A new tuple of PlayerState objects with hole cards dealt.
    """
    num_players = len(players)
    eligible = [seat for seat, player in enumerate(players) if not player.is_eliminated]
    if not eligible:
        return players

    # Deal clockwise starting at the first live seat left of the button.
    start_seat = (button_seat + 1) % num_players
    for _ in range(num_players):
        if not players[start_seat].is_eliminated:
            break
        start_seat = (start_seat + 1) % num_players

    dealt_players = list(players)
    seat = start_seat
    for _ in range(len(eligible)):
        while players[seat].is_eliminated:
            seat = (seat + 1) % num_players
        dealt_players[seat] = dealt_players[seat].with_hole_cards(tuple(deck.deal(2)))
        seat = (seat + 1) % num_players

    return tuple(dealt_players)


def deal_flop(deck: Deck) -> tuple[Card, ...]:
    """Deal the flop: burn 1 card, return 3 cards.

    In poker, a burn card is discarded before each community card street
    as a security measure against marked cards or physical manipulation.

    Args:
        deck: The deck to deal from.

    Returns:
        A tuple of 3 flop cards.
    """
    # Burn one card
    deck.deal(1)

    # Deal three cards for the flop
    cards = deck.deal(3)
    return tuple(cards)


def deal_turn(deck: Deck) -> Card:
    """Deal the turn: burn 1 card, return 1 card.

    Args:
        deck: The deck to deal from.

    Returns:
        The turn card.
    """
    # Burn one card
    deck.deal(1)

    # Deal one card for the turn
    cards = deck.deal(1)
    return cards[0]


def deal_river(deck: Deck) -> Card:
    """Deal the river: burn 1 card, return 1 card.

    Args:
        deck: The deck to deal from.

    Returns:
        The river card.
    """
    # Burn one card
    deck.deal(1)

    # Deal one card for the river
    cards = deck.deal(1)
    return cards[0]


def deal_run_it_twice(
    deck: Deck,
    current_board: tuple[Card, ...],
    remaining_streets: int,
) -> tuple[tuple[Card, ...], tuple[Card, ...]]:
    """Deal two parallel run-outs from the current board state.

    Run-it-twice is used when players are all-in. The remaining community
    cards are dealt twice, and each player's equity is based on averaging
    their win rate across both boards.

    Args:
        deck: The deck to deal from (must have enough cards for 2 x remaining cards).
        current_board: The community cards already revealed.
        remaining_streets: Number of streets left to deal (1, 2, or 3). A flop
            run-out deals three cards, so this is street count, not card count.

    Returns:
        A tuple of two boards, each as a tuple of Cards.
    """
    if remaining_streets < 1:
        raise ValueError("remaining_streets must be at least 1")

    # Three remaining streets means a preflop all-in: flop, turn and river.
    # The old guard rejected >2, so every preflop all-in crashed.
    if remaining_streets > 3:
        raise ValueError("remaining_streets must be at most 3")

    # Street sizes still to come. remaining_streets counts streets, and a flop
    # is three cards, so a preflop run-out deals 5 cards, not 3.
    if remaining_streets == 3:
        street_sizes = (3, 1, 1)
    elif remaining_streets == 2:
        street_sizes = (1, 1)
    else:
        street_sizes = (1,)

    def _run_out() -> tuple[Card, ...]:
        drawn: list[Card] = []
        for street_size in street_sizes:
            deck.deal(1)  # Burn before each street's cards.
            drawn.extend(deck.deal(street_size))
        return tuple(drawn)

    board1 = current_board + _run_out()
    board2 = current_board + _run_out()
    return (board1, board2)
