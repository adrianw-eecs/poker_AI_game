"""Rolling per-seat opponent tendencies for observation v3.

Observation v2 is hero-relative but carries no memory of *how* opponents play.
The adaptive-NFSP literature attributes most of its gain to exactly this, so v3
appends four smoothed per-seat statistics:

    VPIP          voluntarily put chips in preflop (call or raise, not a check)
    PFR           preflop raise frequency
    AF            aggression factor: (bets + raises) / calls
    fold_to_bet   folds when facing a bet

All four are smoothed toward a neutral prior so a seat with two observed actions
does not read as "always folds". The prior uses a Beta(alpha, beta) mean, which
keeps features in [0, 1] and stable early in a session.

Tendencies are computed from *completed hands* only, so a feature never depends
on the hand currently in progress. That avoids leaking information the acting
player cannot have.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from poker.domain.action import ActionType

MAX_SEATS = 9
# Four statistics per seat.
TENDENCY_PER_SEAT = 4
TENDENCY_DIM = MAX_SEATS * TENDENCY_PER_SEAT

# Smoothing prior: alpha successes out of (alpha + beta) pseudo-observations.
# 0.33 / 0.67 gives a neutral 1/3 mean, which sits between a nit and a station
# without favouring either.
_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 2.0


def _smoothed(hits: float, trials: float, alpha: float = _PRIOR_ALPHA,
              beta: float = _PRIOR_BETA) -> float:
    """Beta-smoothed rate in [0, 1]."""
    return float((hits + alpha) / (trials + alpha + beta))


@dataclass
class _SeatCounts:
    """Raw counters behind one seat's tendencies.

    Floats so an exponential half-life can decay them without rounding the
    early hands away.
    """

    hands: float = 0.0
    vpip_hands: float = 0.0
    pfr_hands: float = 0.0
    aggressive_actions: float = 0.0
    calls: float = 0.0
    folds_to_bet: float = 0.0
    fold_to_bet_opportunities: float = 0.0


@dataclass
class TendencyTracker:
    """Accumulates per-seat tendencies across hands.

    ``record_action`` is called for every action during play; ``end_hand``
    collapses the hand's per-seat flags into the rolling counters. Features are
    read with ``features``, which returns a fixed-width vector regardless of how
    many hands have been seen.
    """

    num_players: int
    half_life: float | None = None
    counts: list[_SeatCounts] = field(default_factory=list)
    _hand_vpip: list[bool] = field(default_factory=list)
    _hand_pfr: list[bool] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.counts:
            self.counts = [_SeatCounts() for _ in range(MAX_SEATS)]
        if not self._hand_vpip:
            self._hand_vpip = [False] * MAX_SEATS
        if not self._hand_pfr:
            self._hand_pfr = [False] * MAX_SEATS

    def start_hand(self) -> None:
        """Clear per-hand flags. Call once per dealt hand."""
        self._hand_vpip = [False] * MAX_SEATS
        self._hand_pfr = [False] * MAX_SEATS

    def record_action(
        self,
        seat: int,
        action_type: ActionType,
        *,
        street_is_preflop: bool,
        facing_bet: bool,
    ) -> None:
        """Record one action for tendency purposes.

        Args:
            seat: Seat that acted.
            action_type: The action taken.
            street_is_preflop: True on the preflop street.
            facing_bet: True when the actor had chips to call (a bet was live).
        """
        if not 0 <= seat < MAX_SEATS:
            return
        c = self.counts[seat]
        decisive = action_type in (
            ActionType.RAISE,
            ActionType.ALL_IN,
            ActionType.CALL,
        )
        if street_is_preflop and decisive:
            # A raise or call preflop is voluntary; the blinds are not counted
            # because they are posted, not chosen.
            self._hand_vpip[seat] = True
        if street_is_preflop and action_type in (ActionType.RAISE, ActionType.ALL_IN):
            self._hand_pfr[seat] = True

        if action_type in (ActionType.RAISE, ActionType.ALL_IN):
            c.aggressive_actions += 1
        elif action_type == ActionType.CALL:
            c.calls += 1

        if facing_bet:
            c.fold_to_bet_opportunities += 1
            if action_type == ActionType.FOLD:
                c.folds_to_bet += 1

    def _decay(self) -> None:
        """Halve every counter once per ``half_life`` hands. No-op when unset."""
        if self.half_life is None or self.half_life <= 0:
            return
        factor = 0.5 ** (1.0 / self.half_life)
        for counts in self.counts:
            counts.hands *= factor
            counts.vpip_hands *= factor
            counts.pfr_hands *= factor
            counts.aggressive_actions *= factor
            counts.calls *= factor
            counts.folds_to_bet *= factor
            counts.fold_to_bet_opportunities *= factor

    def end_hand(self) -> None:
        """Decay older hands, then fold this hand's flags into the counters."""
        self._decay()
        for seat in range(MAX_SEATS):
            counts = self.counts[seat]
            counts.hands += 1.0
            if self._hand_vpip[seat]:
                counts.vpip_hands += 1.0
            if self._hand_pfr[seat]:
                counts.pfr_hands += 1.0

    def features(self) -> npt.NDArray[np.float32]:
        """``(TENDENCY_DIM,)`` float32 vector, all values in [0, 1]."""
        out = np.zeros(TENDENCY_DIM, dtype=np.float32)
        for seat in range(MAX_SEATS):
            counts = self.counts[seat]
            base = seat * TENDENCY_PER_SEAT
            out[base + 0] = _smoothed(counts.vpip_hands, counts.hands)
            out[base + 1] = _smoothed(counts.pfr_hands, counts.hands)
            # Aggression factor is unbounded; map to [0, 1) with x/(x+1).
            af = (
                counts.aggressive_actions / max(counts.calls, 1.0)
                if counts.calls or counts.aggressive_actions
                else 0.0
            )
            out[base + 2] = af / (af + 1.0) if af > 0 else 0.0
            out[base + 3] = _smoothed(counts.folds_to_bet, counts.fold_to_bet_opportunities)
        return out

    def confidence(self, seat: int) -> float:
        """How much the smoothed rate has left the prior, in ``[0, 1]``.

        A seat with no hands is 0. As decayed hands accumulate, this rises
        toward 1 and the prior stops dominating the feature.
        """
        seen = self.counts[seat].hands
        return float(seen / (seen + _PRIOR_ALPHA + _PRIOR_BETA))

    def hands_seen(self, seat: int) -> float:
        """Decayed hand count for a seat. This is the confidence sample size."""
        return float(self.counts[seat].hands)

    def reset(self) -> None:
        self.counts = [_SeatCounts() for _ in range(MAX_SEATS)]
        self.start_hand()
