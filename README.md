# Poker ML Environment

A No-Limit Texas Hold'em engine in Python, built as a simulation environment for
training machine learning agents via self-play.

**Full documentation is in [`docs/`](docs/README.md).** Start with
[`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

---

## Quick Start

```bash
pip install -e ".[dev]"

python -m poker.main -n 6 --hands 20 --seed 1 \
  --bots human nfsp:models/nfsp_v2/nfsp_v2.pt \
         sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random random flop
```

## Current checkpoints

| Role | Path | Observation | Sizing |
|---|---|---|---|
| NFSP | `models/nfsp_v2/nfsp_v2.pt` | v2, 318-wide | pot-fraction |
| SD-CFR | `models/sdcfr_prod/sdcfr_ckpt_500.pt` | v1, 98-wide | legacy |

These are the only two checkpoints in the repo. Older runs were removed on
2026-11-03. There is **no v3 checkpoint** — the retired `nfsp_v3*.pt` files used a
155-wide observation and predated the current architecture.

## Evaluate a table

```bash
python scripts/validate_training.py --hands 200 --seed 1 \
  --bots nfsp:models/nfsp_v2/nfsp_v2.pt \
         sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt random flop
```

## Tests

```bash
pytest -m smoke   # 36 tests, ~4 s
pytest            # 235 tests, ~10 s
```

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | Current state, how to run, known limitations |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Project layout, refactor and optimisation proposals |
| [`docs/ENGINE.md`](docs/ENGINE.md) | Engine internals: data structures, betting, showdown, testing |
| [`docs/ML.md`](docs/ML.md) | Environment, observation encoders, action space, models |
| [`docs/NFSP_IMPROVEMENTS.md`](docs/NFSP_IMPROVEMENTS.md) | Observation gaps and training-speed proposals for NFSP |
| [`docs/TRAINING.md`](docs/TRAINING.md) | Training commands, long-form runs, troubleshooting |
| [`docs/HYBRID_PLAN.md`](docs/HYBRID_PLAN.md) | Combined NFSP + SD-CFR design |
| [`docs/BOTS.md`](docs/BOTS.md) | Bots and how to add one |
| [`docs/USER_NOTES.md`](docs/USER_NOTES.md) | Stable design decisions |

Rules and game configuration are documented in
[`docs/ENGINE.md`](docs/ENGINE.md) and [`docs/USER_NOTES.md`](docs/USER_NOTES.md).

## License

MIT
