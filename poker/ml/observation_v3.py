"""Hero-relative observation v3: v2 plus observed opponent tendencies.

The vector is v2's 318 floats followed by ``TENDENCY_DIM`` (36) floats of
smoothed per-seat statistics from :mod:`poker.ml.tendencies`. Layout:

    0    .. 318   identical to observation v2 (see observation_v2.OBS_SPEC_V2)
    318  .. 354   per-seat tendencies, 4 per seat, seat-major

Keeping v2's prefix byte-identical means a v3 encoder differs from v2 only by
appended information, so an ablation can compare them with the same code path
and the same network trunk width apart from the input layer.

Tendencies describe *past hands*, so the encoder takes them as an argument
rather than reading them from ``GameState`` (which is a single hand and has no
memory). Callers that have no tracker get the neutral prior, which is exactly
what a brand-new opponent should look like.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from poker.ml.observation_v2 import OBS_SPEC_V2, OBS_V2_DIM, build_observation_v2
from poker.ml.tendencies import TENDENCY_DIM, TendencyTracker
from poker.state.game_state import GameState

OBS_V3_DIM = OBS_V2_DIM + TENDENCY_DIM

# Number of seats the tendency block covers.
MAX_SEATS = 9
TENDENCY_PER_SEAT = 4


@dataclass(frozen=True)
class ObsSpecV3:
    """Byte offsets of each block inside the v3 vector."""

    dim: int = OBS_V3_DIM
    # v2 blocks, unchanged.
    v2_prefix: int = OBS_V2_DIM
    tendencies: tuple[int, int] = (OBS_V2_DIM, OBS_V3_DIM)
    tendency_per_seat: int = TENDENCY_PER_SEAT
    seat_dim: int = OBS_SPEC_V2.seat_dim
    seat_block: tuple[int, int] = OBS_SPEC_V2.seats


OBS_SPEC_V3 = ObsSpecV3()


def neutral_tendencies() -> npt.NDArray[np.float32]:
    """All-prior tendency block, for a tracker that has seen nothing.

    An untrained tracker must not read as "never folds, never bets": the
    smoothed prior already encodes a neutral rate, so a bare tracker is used
    here rather than zeros.
    """
    tracker = TendencyTracker(num_players=MAX_SEATS)
    return tracker.features()


def build_observation_v3(
    state: GameState,
    seat: int,
    tendencies: TendencyTracker | None = None,
) -> npt.NDArray[np.float32]:
    """Build the v3 vector for ``seat``.

    Args:
        state: Current hand state.
        seat: Hero seat.
        tendencies: Optional rolling tracker. ``None`` yields the neutral prior.

    Returns:
        ``(OBS_V3_DIM,)`` float32. The first ``OBS_V2_DIM`` entries are exactly
        what ``build_observation_v2`` returns for the same state and seat.
    """
    base = build_observation_v2(state, seat)
    block = (
        tendencies.features()
        if tendencies is not None
        else neutral_tendencies()
    )
    if block.shape[0] != TENDENCY_DIM:
        raise ValueError(
            f"tendency block is {block.shape[0]} wide, expected {TENDENCY_DIM}"
        )
    out = np.empty(OBS_V3_DIM, dtype=np.float32)
    out[:OBS_V2_DIM] = base
    out[OBS_V2_DIM:] = block
    return out


def seat_occupancy(state: GameState, hero: int) -> npt.NDArray[np.float32]:
    """``(MAX_SEATS,)`` 1.0 for real seats, 0.0 for padding.

    Attention over seats needs to know which slots exist; v2 encodes a padded
    seat as an all-zero block, which is indistinguishable from a seat whose
    stack happens to be zero.
    """
    num_players = len(state.players)
    out = np.zeros(MAX_SEATS, dtype=np.float32)
    for offset in range(min(num_players, MAX_SEATS)):
        out[offset] = 1.0
    return out
