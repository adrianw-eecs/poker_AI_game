"""Self-play steps on a fixed schedule only when the behavior gates pass."""

from scripts.train_nfsp_selfplay import SelfPlayScheduler
from scripts.validate_training import _paired_seed


def test_schedule_holds_until_fold_rate_and_lower_bound_pass() -> None:
    scheduler = SelfPlayScheduler()
    assert scheduler.apply_scheduled_step(0.04, 20.0).startswith("hold")
    assert scheduler.fraction == 0.0
    assert scheduler.apply_scheduled_step(0.20, -80.0).startswith("hold")
    assert scheduler.fraction == 0.0
    assert scheduler.apply_scheduled_step(0.20, -40.0).startswith("ramp")
    assert scheduler.fraction == 0.05


def test_schedule_stops_at_the_cap() -> None:
    scheduler = SelfPlayScheduler()
    scheduler.fraction = SelfPlayScheduler.MAX_FRACTION
    assert scheduler.apply_scheduled_step(0.30, 10.0).startswith("cap")
    assert scheduler.fraction == SelfPlayScheduler.MAX_FRACTION


def test_paired_seed_is_stable() -> None:
    assert _paired_seed(7, 1000, 3, 0) == _paired_seed(7, 1000, 3, 0)
    assert _paired_seed(7, 1000, 3, 0) != _paired_seed(7, 1000, 4, 0)
