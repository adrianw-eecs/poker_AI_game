"""Tests for card dealing functions."""

from poker.domain.deck import Deck
from poker.engine.dealer import deal_flop, deal_hole_cards, deal_river, deal_turn
from poker.rng import RNG
from poker.state.player_state import PlayerState


def _players(n: int) -> tuple[PlayerState, ...]:
    return tuple(PlayerState(
        seat=i, name=f"P{i}", stack=1000, hole_cards=(),
        committed_this_street=0, committed_this_hand=0,
        has_folded=False, is_all_in=False, is_eliminated=False,
    ) for i in range(n))


def test_hole_cards_dealt() -> None:
    deck = Deck(RNG(seed=42))
    dealt = deal_hole_cards(deck, _players(6), button_seat=0)
    assert all(len(p.hole_cards) == 2 for p in dealt)
    assert deck.remaining() == 40


def test_community_cards_dealt() -> None:
    deck = Deck(RNG(seed=42))
    deal_hole_cards(deck, _players(6), button_seat=0)
    flop = deal_flop(deck)
    assert len(flop) == 3
    turn = deal_turn(deck)
    river = deal_river(deck)
    assert deck.remaining() == 32  # 52 - 12 - 4 - 2 - 2


def test_full_deal_no_duplicates() -> None:
    deck = Deck(RNG(seed=42))
    dealt = deal_hole_cards(deck, _players(6), button_seat=0)
    flop = deal_flop(deck)
    turn = deal_turn(deck)
    river = deal_river(deck)
    all_cards = [c for p in dealt for c in p.hole_cards] + list(flop) + [turn, river]
    assert len(set(all_cards)) == len(all_cards)
