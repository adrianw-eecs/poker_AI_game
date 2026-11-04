"""Who sits across from the learner in a training sitting.

About half the seats are frozen past checkpoints (the recent generations and
the current best) when those files exist and match the learner's observation
version. One seat is the SD-CFR teacher when it speaks the same observation.
The remaining seats are Random, Flop, and Call. The list is shuffled so a
strategy is not glued to a seat index. Missing checkpoints fall back to the
baseline bots rather than leaving a seat empty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from poker.bots.call_bot import CallBot
from poker.bots.flop_bot import FlopBot
from poker.bots.nfsp_bot import NFSPBot
from poker.bots.random_bot import RandomBot
from poker.bots.sdcfr_bot import SDCFRBot
from poker.ml.action_space import BetScheme
from poker.ml.models.nfsp_model import NFSPModel
from poker.ml.models.sdcfr_model import SDCFRModel


def _generation_number(path: Path) -> int:
    digits = "".join(character for character in path.stem if character.isdigit())
    return int(digits) if digits else -1


class CheckpointRoster:
    """Frozen opponents loaded from ``hybrid_gen_*.pt`` and ``hybrid_best.pt``."""

    def __init__(
        self,
        directory: Path,
        obs_version: int,
        teacher: SDCFRModel | None = None,
    ) -> None:
        self.directory = directory
        self.obs_version = int(obs_version)
        self.teacher = teacher
        self._loaded: list[tuple[str, NFSPModel]] = []
        self._cache: dict[str, tuple[int, NFSPModel]] = {}

    def refresh(self) -> int:
        """Load the current best and the last five generations that match."""
        candidates: list[Path] = []
        best = self.directory / "hybrid_best.pt"
        if best.is_file():
            candidates.append(best)
        generations = sorted(
            self.directory.glob("hybrid_gen_*.pt"),
            key=_generation_number,
        )
        candidates.extend(generations[-5:])
        wanted: list[Path] = []
        seen_names: set[str] = set()
        for path in candidates:
            if path.name in seen_names:
                continue
            seen_names.add(path.name)
            wanted.append(path)
        loaded: list[tuple[str, NFSPModel]] = []
        for path in wanted:
            stamp = path.stat().st_mtime_ns
            cached = self._cache.get(path.name)
            if cached is not None and cached[0] == stamp:
                loaded.append((path.name, cached[1]))
                continue
            model = NFSPModel(
                obs_version=self.obs_version,
                bet_scheme=BetScheme.POT_FRACTION,
            )
            try:
                model.load(str(path))
            except (OSError, ValueError, RuntimeError, KeyError):
                continue
            if int(model.obs_version) != self.obs_version:
                continue
            model.q_network.eval()
            model.policy_network.eval()
            self._cache[path.name] = (stamp, model)
            loaded.append((path.name, model))
        self._loaded = loaded
        return len(self._loaded)

    def seat(self, num_opponents: int, rng: np.random.Generator) -> list:
        """Build ``num_opponents`` bots and shuffle them across seats."""
        if num_opponents <= 0:
            return []
        teacher_ok = (
            self.teacher is not None
            and int(getattr(self.teacher, "obs_version", -1)) == self.obs_version
        )
        n_teacher = 1 if teacher_ok else 0
        n_past = min(len(self._loaded), max(0, num_opponents // 2))
        if n_teacher + n_past > num_opponents:
            n_past = max(0, num_opponents - n_teacher)
        n_base = num_opponents - n_teacher - n_past
        bots: list = []
        if n_teacher and self.teacher is not None:
            bots.append(SDCFRBot(name="SDCFR", model=self.teacher))
        if n_past and self._loaded:
            replace = len(self._loaded) < n_past
            picks = rng.choice(len(self._loaded), size=n_past, replace=replace)
            for index in np.atleast_1d(picks):
                name, model = self._loaded[int(index)]
                bots.append(NFSPBot(name=f"past_{name}", model=model, training=False))
        for index in range(n_base):
            kind = index % 3
            if kind == 0:
                bots.append(
                    RandomBot(name=f"Random_{index}", seed=int(rng.integers(0, 2**31 - 1)))
                )
            elif kind == 1:
                bots.append(FlopBot(name=f"Flop_{index}"))
            else:
                bots.append(CallBot(name=f"Call_{index}"))
        rng.shuffle(bots)
        return bots
