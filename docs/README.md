# Docs

Concise documentation for the current state of the project.

| File | Contents |
|------|---------|
| [`PROJECT_STATUS.md`](PROJECT_STATUS.md) | Current models, what works, how to run it, known limitations |
| [`ENGINE.md`](ENGINE.md) | Deep technical dive: data structures, betting logic, showdown, session management |
| [`ML.md`](ML.md) | ML environment surface — PokerEnv, observation encoders, action space, models |
| [`TRAINING.md`](TRAINING.md) | Training and validation: NFSP/SD-CFR commands, reward design, troubleshooting |
| [`HYBRID_PLAN.md`](HYBRID_PLAN.md) | Design plan for combining NFSP + SD-CFR into one model |
| [`NFSP_Notes.md`](NFSP_Notes.md) | NFSP paper and reference-implementation notes |
| [`SD-CFR_Notes.md`](SD-CFR_Notes.md) | SD-CFR notes |
| [`BOTS.md`](BOTS.md) | Available bots and how to add a new one |
| [`USER_NOTES.md`](USER_NOTES.md) | Stable decisions and assumptions (kept short) |

## Current checkpoints

| Role | Path |
|------|------|
| NFSP | `models/nfsp_v2/nfsp_v2.pt` |
| SD-CFR | `models/sdcfr_prod/sdcfr_ckpt_500.pt` |
