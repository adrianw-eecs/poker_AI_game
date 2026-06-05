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

    # Find the starting seat (small blind, left of button)
    start_seat = (button_seat + 1) % num_players

    # Deal 2 cards to each player in order
    dealt_players = list(players)
    for i in range(num_players):
        seat = (start_seat + i) % num_players
        cards = deck.deal(2)
        dealt_players[seat] = dealt_players[seat].with_hole_cards(tuple(cards))

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
        deck: The deck to deal from (must have enough cards for 2 x remaining_streets).
        current_board: The community cards already revealed.
        remaining_streets: Number of streets left to deal (1 for turn, 2 for turn+river).

    Returns:
        A tuple of two boards, each as a tuple of Cards.
    """
    if remaining_streets < 1:
        raise ValueError("remaining_streets must be at least 1")

    if remaining_streets > 2:
        raise ValueError("remaining_streets must be at most 2")

    # Deal cards for the first run-out
    run1_cards: list[Card] = []
    for _ in range(remaining_streets):
        deck.deal(1)  # Burn
        cards = deck.deal(1)
        run1_cards.extend(cards)

    board1 = current_board + tuple(run1_cards)

    # Deal cards for the second run-out from remaining deck
    run2_cards: list[Card] = []
    for _ in range(remaining_streets):
        deck.deal(1)  # Burn
        cards = deck.deal(1)
        run2_cards.extend(cards)

    board2 = current_board + tuple(run2_cards)

    return (board1, board2)
