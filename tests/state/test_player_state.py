"""Tests for PlayerState."""

import pytest

from poker.domain.card import Card, Rank, Suit
from poker.state.player_state import PlayerState


def _make_player(seat: int = 0, stack: int = 1000, **kwargs) -> PlayerState:
    defaults = dict(
        seat=seat, name=f"Player{seat}", stack=stack, hole_cards=(),
        committed_this_street=0, committed_this_hand=0,
        has_folded=False, is_all_in=False, is_eliminated=False,
    )
    defaults.update(kwargs)
    return PlayerState(**defaults)


def test_player_state_construction() -> None:
    p = _make_player(seat=0, stack=1000)
    assert p.seat == 0
    assert p.stack == 1000
    assert not p.has_folded


def test_player_state_frozen() -> None:
    with pytest.raises(AttributeError):
        _make_player().stack = 500  # type: ignore


def test_player_state_is_active() -> None:
    cards = (Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.HEARTS))
    assert _make_player(hole_cards=cards).is_active
    assert not _make_player(hole_cards=cards, has_folded=True).is_active
    assert not _make_player(hole_cards=cards, is_eliminated=True).is_active
    assert not _make_player(hole_cards=()).is_active


def test_player_state_with_methods() -> None:
    original = _make_player(stack=1000)
    updated = original.with_stack(500)
    assert updated.stack == 500
    assert original.stack == 1000

    folded = original.with_folded(True)
    assert folded.has_folded
    assert not original.has_folded

    all_in = original.with_all_in(True)
    assert all_in.is_all_in
