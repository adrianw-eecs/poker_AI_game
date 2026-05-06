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
    PotsBuilt,
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


def test_event_has_required_type_field() -> None:
    event = HandStarted(1, 0, 5, 10)
    data = EventEncoder.to_dict(event)
    assert "type" in data
    assert data["type"] == "HandStarted"
