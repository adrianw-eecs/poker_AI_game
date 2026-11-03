"""Tests for event types and serialization."""

import json
import pytest

from poker.logging.events import (
    ActionTaken,
    AntePosted,
    BlindPosted,
    BoardCardsDealt,
    Event,
    EventEncoder,
    HandEnded,
    HandStarted,
    HoleCardsDealt,
    PlayerOut,
    PotAwarded,
    PotsBuilt,
    SessionStarted,
    Showdown,
    StreetEnded,
)


def test_events_round_trip() -> None:
    events: list[Event] = [
        HandStarted(1, 0, 5, 10),
        BlindPosted(1, 1, 5, False),
        AntePosted(1, 0, 1),
        HoleCardsDealt(1, 6),
        BoardCardsDealt(1, "FLOP", ("As", "Kh", "Qd")),
        ActionTaken(1, "PREFLOP", 2, "CALL", 10),
        PotsBuilt(1, "PREFLOP", ((30, (0, 1, 2, 3, 4, 5)),)),
        Showdown(1, {0: (1, (14, 13, 12)), 1: (0, (14, 13, 12, 11, 10))}),
        StreetEnded(1, "FLOP", "snapshot"),
        HandEnded(1, {0: 100, 1: -100, 2: 0, 3: 0, 4: 0, 5: 0}),
        PlayerOut(1, 2, "busted", street="showdown", stack=0, name="P3", position="BB"),
        PotAwarded(1, 0, 300, (0, 1), True, run=0),
        SessionStarted(
            seats=(0, 1),
            names={0: "P1", 1: "P2"},
            stacks={0: 1000, 1: 1000},
            small_blind=5,
            big_blind=10,
            ante=0,
            rake_percent=0.0,
            run_it_twice=False,
            seed=None,
        ),
    ]
    for event in events:
        data = EventEncoder.to_dict(event)
        reconstructed = EventEncoder.from_dict(data)
        assert reconstructed == event


def test_events_are_json_serializable() -> None:
    event = HandStarted(1, 0, 5, 10)
    data = EventEncoder.to_dict(event)
    json_str = json.dumps(data)
    loaded = json.loads(json_str)
    reconstructed = EventEncoder.from_dict(loaded)
    assert reconstructed == event


def test_old_records_decode_with_defaults() -> None:
    action = EventEncoder.from_dict(
        {
            "type": "ActionTaken",
            "hand_number": 1,
            "street": "preflop",
            "seat": 0,
            "action_type": "fold",
            "amount": 0,
        }
    )
    assert isinstance(action, ActionTaken)
    assert action.label == ""
    assert action.all_in is False
    assert action.log_line == ""

    street = EventEncoder.from_dict(
        {"type": "StreetEnded", "hand_number": 1, "street": "FLOP"}
    )
    assert isinstance(street, StreetEnded)
    assert street.snapshot == ""
    assert street.reason == ""

    showdown = EventEncoder.from_dict(
        {"type": "Showdown", "hand_number": 1, "hands": {"0": {"type": 1, "kickers": [14]}}}
    )
    assert isinstance(showdown, Showdown)
    assert showdown.ended_early is False
    assert showdown.hole_cards == {}

    ended = EventEncoder.from_dict(
        {"type": "HandEnded", "hand_number": 1, "chip_distribution": {"0": 10}}
    )
    assert isinstance(ended, HandEnded)
    assert ended.net == {}
    assert ended.chip_distribution == {0: 10}


def test_encoder_ignores_seq_field() -> None:
    event = EventEncoder.from_dict(
        {
            "type": "HandStarted",
            "seq": 4,
            "hand_number": 0,
            "dealer_seat": 0,
            "small_blind": 5,
            "big_blind": 10,
        }
    )
    assert isinstance(event, HandStarted)
    assert event.hand_number == 0


def test_event_has_required_type_field() -> None:
    event = HandStarted(1, 0, 5, 10)
    data = EventEncoder.to_dict(event)
    assert "type" in data
    assert data["type"] == "HandStarted"
