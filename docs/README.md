# Docs

Concise documentation for the current state of the project.

| File | Contents |
|------|---------|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Current models, what works, how to run it, known limitations |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Project layout, refactor and optimisation proposals |
| [`PERFORMANCE.md`](PERFORMANCE.md) | Measured engine↔model costs and the ranked optimisation plan |
| [`ENGINE.md`](ENGINE.md) | Engine internals: data structures, betting, showdown, testing |
| [`ML.md`](ML.md) | Environment, observation encoders, action space, models |
| [`NFSP_IMPROVEMENTS.md`](NFSP_IMPROVEMENTS.md) | Observation gaps and training improvements for NFSP |
| [`TRAINING.md`](TRAINING.md) | Training commands including long-form runs, reward design, troubleshooting |
| [`HYBRID_PLAN.md`](HYBRID_PLAN.md) | Design plan for combining NFSP + SD-CFR into one model |
| [`NFSP_Notes.md`](NFSP_Notes.md) | NFSP paper and reference-implementation notes |
| [`SD-CFR_Notes.md`](SD-CFR_Notes.md) | SD-CFR notes |
| [`BOTS.md`](BOTS.md) | Available bots and how to add a new one |
| [`USER_NOTES.md`](USER_NOTES.md) | Stable decisions and assumptions (kept short) |

`README.md` in the repository root is a short pointer for packaging metadata
(`pyproject.toml` requires it). All substantive documentation lives here.

## Current checkpoints

| Role | Path | Observation | Sizing |
|------|------|-------------|--------|
| NFSP | `models/nfsp_v2/nfsp_v2.pt` | v2, 318-wide | pot-fraction |
| SD-CFR | `models/sdcfr_prod/sdcfr_ckpt_500.pt` | v1, 98-wide | legacy |
