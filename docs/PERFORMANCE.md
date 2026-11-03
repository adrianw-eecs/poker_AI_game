# Performance: Engine ↔ Model Interaction

All figures measured on an RTX 3080, 6-player tables, 2026-11-03.
Method and scripts are described per section so numbers can be re-derived.

---

## 1. Where production wall-time actually goes

The stock profiler (`scripts/profile_training.py`) runs with `train_every=4`,
which makes weight updates look like 89% of the loop. **Production uses
`train_every=64`**, a completely different ratio. Measured end-to-end:

| Config | ms/decision | dec/s | inference | training | engine |
|---|---|---|---|---|---|
| profiler default (te=4, bs=2048, u=1) | 15.68 | 64 | 27% | 66% | 7% |
| **production (te=64, bs=2048, u=1)** | **5.28** | **189** | **68%** | **11%** | **21%** |
| production with `updates_per_train=2` | 6.61 | 151 | 60% | 20% | 20% |
| alt (te=16, bs=1024, u=2) | 9.27 | 108 | 38% | 50% | 12% |

**The production bottleneck is inference, not training.** Two thirds of every
decision is spent on a single forward pass. The profiler's own recommendation
("increase batch_size to better utilise GPU") is misleading here — batch size is
already 2048 and the GPU is not the constraint.

---

## 2. The dominant problem: batch-1 forward costs 2.9 ms

A 695,992-parameter MLP should do batch-1 inference in well under 100 µs. Measured
(with and without per-call synchronisation, and with pure GPU event timing to
exclude launch gaps):

```
batch-1 forward, synced per call     =   2964.6 us
batch-1 forward, amortised launches  =   2895.3 us
batch-1 forward, GPU event time      =   2804.3 us   <-- real GPU time
cpu-side op calls                    =    154 per forward
```

Since GPU event time is ~2.8 ms, this is **not** CPU launch overhead being
mistaken for compute — the GPU really is taking 2.8 ms for one sample. The cause
is that each forward launches ~154 tiny kernels (three embedding lookups, concat,
two MLPs with LayerNorm, three residual blocks, merge, heads), and at batch 1 each
kernel has almost no work to hide its own latency.

### Batching is the fix, and it scales

GPU event time for the same total sample count:

| Batch size | µs per sample |
|---|---|
| 1 | 3,140 |
| 8 | 358 |
| 32 | 88 |
| 128 | 24 |
| 512 | 5.4 |

Inference cost is essentially **flat until batch 32**, so every additional
concurrent hand is nearly free. This is the single largest available win.

### What works today and what does not

| Approach | Result |
|---|---|
| CPU inference for batch 1 | **947 µs vs 2,882 µs — 3.0× faster than GPU** |
| `torch.compile(mode="reduce-overhead")` | Not available: Triton missing on this install |
| Batching across concurrent hands | Only approach that scales with N |

The 3× CPU win is worth noting: for single-sample inference this network is so
latency-bound that the CPU genuinely beats the GPU. That is unusual and is itself
a signal that the architecture is kernel-count-dominated.

---

## 3. Recommended change: batched multi-hand inference

This is the highest-value refactor in the codebase. Currently
`poker/ml/nfsp_loop.py::consume_hand` plays **one hand at a time**, so every
decision is a batch-1 forward.

**Design:** run N independent hands concurrently and batch their pending
observations into one forward pass. With N = 32, per-sample inference drops from
3,140 µs to 88 µs — a ~36× reduction in the dominant cost.

Expected effect on the measured 5.28 ms/decision budget: inference falls from
~3.6 ms to roughly 0.1 ms. Even allowing for the engine's 21%, throughput should
move from ~189 to well over 1,000 decisions/s. Training would then become the
bottleneck, which is the correct place for it.

**Sketch:**

```python
class BatchedEnvPool:
    """N hands in flight; one forward pass per decision round."""

    def __init__(self, num_hands, make_env, model):
        self.hands = [HandState(make_env()) for _ in range(num_hands)]
        self.model = model

    def step_round(self):
        pending = [h for h in self.hands if h.awaiting_decision]
        if not pending:
            return
        obs = np.stack([h.obs for h in pending])
        mask = np.stack([h.mask for h in pending])
        actions = self.model.select_action_batch(obs, mask)   # new method
        for h, a in zip(pending, actions):
            h.apply(a)
```

**Prerequisites:**
1. `NFSPModel.select_action` needs a batched variant that preserves per-hand
   anticipatory mode (`_episode_br_by_seat` is keyed by seat, but mode must now be
   per *hand*, so a hand id is needed alongside the seat id).
2. ε-greedy exploration and policy sampling must respect per-row legal masks —
   both already work elementwise, so this is mostly plumbing.
3. `consume_hand` stays as the single-hand API for evaluation and validation.

**Do this before any architecture work.** It is a larger change than it sounds,
but every alternative optimises a cost that this change makes irrelevant.

---

## 4. Observation building: 233 µs, dominated by scalar `np.clip`

`build_observation_v2` costs 233 µs per call — 4.4% of a production decision.
Breakdown at 6 players, preflop:

| Block | Cost |
|---|---|
| `_seat_block` × 6 | 100.5 µs |
| `_street_summary` × 4 | 86.6 µs |
| `_scalars` | 23.3 µs |
| `_hero_block` | 10.8 µs |
| `_derived_cards` | 0.9 µs |

The cause is **216 scalar helper calls per observation**, made up of 93 `_ratio`,
123 `_clip01`/`_sat`. Each `_clip01` is `float(np.clip(value, 0.0, 1.0))` on a
Python float, which costs **2,913 ns** versus **51 ns** for a plain comparison:

```python
# current: poker/ml/observation_v2.py
def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))

# proposed
def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)
```

Measured on a representative mix of 216 calls: **415 µs → 13 µs (32×)**.

**Equivalence verified** across 77 boundary cases (`±inf`, `nan`, `±1e-9`,
exactly 0 and 1, large values, zero/negative denominators): **0 mismatches**,
including `nan` propagating and `inf` clamping identically.

This is a pure win with no behaviour change. Apply it to `_clip01`, `_ratio`, and
`_sat` together — the saving is spread across all three.

**Minor, same file:** `blind_seats(state)` is recomputed inside every
`_seat_block`, so 6× per observation. At 0.78 µs per call that is only 3.9 µs
saved by hoisting — real but small. Do it while editing the file, not as its own
task.

---

## 5. Training loop

### 5.1 Replay ratio is very low

At 189 dec/s with `train_every=64` and `updates_per_train=1`, the trainer performs
~3 update events/second. Measured per-event cost is 29.0 ms (Q) + 16.9 ms (policy)
= 45.9 ms, so training occupies only ~11% of wall time. Each transition in a
2048-batch is effectively replayed once per ~90 decisions added to the buffer.

Raising `updates_per_train` to 2 costs 6.61 ms/decision instead of 5.28 but
doubles the gradient signal per decision. **Do this after §3** — once inference is
cheap, the ratio changes completely and the optimal setting must be re-measured.

### 5.2 Eight host→device copies per Q update

`_update_q` calls `torch.as_tensor(..., device=self.device)` for obs, actions,
rewards, next_obs, next_masks, current_masks, dones — 8 separate transfers,
measured at ~1.3 ms total, on top of the 3.7 ms to sample the buffer.

**Proposal.** Keep a single pinned staging buffer and issue one
`copy_(non_blocking=True)` per update. Modest (~1 ms of 46 ms) but free.

### 5.3 float16 buffer costs 60% more sampling time

Measured directly:

| Buffer dtype | `sample(2048)` |
|---|---|
| float16 | 3.67 ms |
| float32 | 2.30 ms |

The float16→float32 upcast costs **1.38 ms per sample** (+60%), which is 12% of a
Q update. It halves memory (a 1M-row 318-wide buffer is 636 MB as float16 vs
1.27 GB as float32), so the trade is defensible — but it is not free, and the
docstring currently presents it as an unambiguous win.

Given that `q_capacity=1_000_000` is rarely filled in shorter runs, **float32 is
probably the better default once memory allows**. Measure at your actual buffer
occupancy before changing it.

### 5.4 Warm-up gate counts decisions, not hands

Documented in `NFSP_PERFORMANCE`-adjacent notes and `NFSP_IMPROVEMENTS.md` §3.1:
`_q_min` is evaluated against `_step`, which increments per decision. With
`q_capacity=1_000_000` the gate opens after 10,000 decisions — potentially within
the first handful of hands, when the buffer holds a highly correlated sample.
Training on that sample is a plausible contributor to the oscillating
`br_proxy_bb100`.

**Fix:** gate on completed hands, and keep `_step` for the epsilon schedule only.

### 5.5 One `.item()` per update forces a pipeline stall

`_update_q` returns `float(total_loss.item())`, which synchronises the stream.
With one update per 64 decisions this is negligible; after §3 makes updates
frequent it will matter. Accumulate losses on-device and read them once per
logging interval.

---

## 6. Game engine

### 6.1 Correction to an earlier claim

`docs/ARCHITECTURE.md` §2.1 previously stated that `legal_actions()` is called
"up to four times per decision". **That was wrong.** Instrumenting the call
counters shows:

```
get_action_mask(): 1 legal_actions call
step():            1 legal_actions call
```

So it is two calls per full decision cycle, and each costs only **2.7 µs** —
0.05% of a decision. Caching it would save nothing meaningful. The earlier
proposal is withdrawn.

Likewise `rebuild_pots` (11.1 µs) and `apply_action` (5.7 µs) are each ~0.2% of a
decision. The frozen-dataclass churn is real but **not** worth optimising at
current scale — it is 10× smaller than the observation builder and 300× smaller
than inference.

### 6.2 Where engine time actually goes (21% of a decision)

From `cProfile` over 400 `env.step` calls, the meaningful entries are:

| Function | Share of step time |
|---|---|
| `_play_opponent_turn` (229 calls) | ~47% |
| `_observe` → `build_observation_v2` | ~40% |
| `_resolve_hand` → showdown + evaluator | ~32% of reset |

So within the engine's 21%, **almost half is scripted opponent bots**, not rule
logic. With SD-CFR opponents in the mix this gets considerably worse, because
each SD-CFR decision is a network forward.

**Proposal.** Opponent decisions are independent of the learner's own — they can
be computed while the learner's forward pass is in flight. Combined with §3, fold
opponent inference into the same batched forward instead of running it
synchronously inside `_run_until_decision`.

### 6.3 Showdown evaluator

`evaluate_with_best_cards` enumerates 21 five-card combinations and appears in the
profile at ~0.020 s per 20 hands (~1 ms/hand). This is 0.3% of a decision, so a
7-card fast path would save very little in the training loop. Keep the current
implementation for clarity; revisit only if hand throughput becomes the goal
(e.g. bulk simulation), where ~21× on this component would matter.

---

## 7. Model architecture

### 7.1 Two separate 700K networks, no shared trunk

`NFSPModel` instantiates `q_network` and `policy_network` independently — 1.39M
parameters total, and two separate forward passes at inference when only one is
needed per decision (mode is chosen per episode, so at runtime only one is
evaluated). The duplication has two costs:

1. **Training:** a Q update and a policy update each pay a full forward+backward
   through a 700K network on *different* data. A shared trunk trained on the union
   of both losses would amortise representation learning.
2. **Hybrid plan:** `HYBRID_PLAN.md` requires one trunk with two heads anyway.

**Proposal.** Factor the trunk out of `PokerNetV2` and attach `value_head`,
`adv_head`, and `policy_head` to one trunk. Keep separate optimizers and separate
losses (they have different scales — Q is Huber on returns, policy is CE on
actions), but share the representation.

**Caveat, stated honestly:** the NFSP paper uses *separate* networks deliberately,
and sharing can cause the two objectives to interfere — a large policy gradient
can perturb the Q trunk and vice versa. Treat this as an experiment with a
control: measure `br_proxy_bb100` stability, not just mean reward, because the
shared-trunk failure mode is exactly the instability that column detects.

### 7.2 154 kernels per forward

The kernel count explains the batch-1 latency. Reducing it helps batch-1 only,
which §3 makes irrelevant — so **do not** spend effort here. If batch-1 inference
matters for interactive play specifically, use CPU (3× faster, measured) rather
than restructuring the network.

### 7.3 Feature gaps

Covered in `NFSP_IMPROVEMENTS.md`. Ranked by expected value: opponent-tendency
features (largest ceiling), then players-to-act-behind, then action-history
truncation. Feature work is orthogonal to everything in this document and can
proceed in parallel.

---

## 8. Priority

| # | Change | Effort | Measured/expected gain |
|---|---|---|---|
| 1 | Replace scalar `np.clip` in observation helpers | XS | 233 µs → ~50 µs per obs (32× on the helper block) |
| 2 | Batched multi-hand inference | L | ~36× on the 68% dominant cost |
| 3 | Raise replay ratio after (2), re-measure | XS | More gradient signal per decision |
| 4 | Single pinned transfer in `_update_q` | S | ~1 ms of 46 ms per update |
| 5 | Warm-up gate on hands, not decisions | S | Correctness of early training |
| 6 | Shared trunk (experiment, with control) | M | Amortised representation; gated on `br_proxy` stability |
| 7 | Reconsider float16 → float32 buffers | XS | +1.38 ms/update, 2× memory |
| 8 | Fold opponent inference into the batch | M | Up to half of engine time |

Item 1 is a ten-line change with verified equivalence — do it now. Item 2 is the
real project; items 4, 5, 7 are small follow-ups to measure after it.

---

## 9. Reproducing these numbers

The measurement scripts were temporary. To re-derive:

- **Production ratio:** drive `PokerEnv` + `NFSPModel` directly with
  `train_every=64`, timing `select_action`, `env.step`, and `train_step`
  separately, with `torch.cuda.synchronize()` around the model calls.
- **Kernel count / GPU event time:** `torch.profiler` with `ProfilerActivity.CUDA`,
  and `torch.cuda.Event` pairs around the forward loop.
- **Observation breakdown:** call the private helpers (`_seat_block`,
  `_street_summary`, `_scalars`) individually.
- **Helper-call count:** monkey-patch `_ratio` / `_clip01` with counters.

Run everything on the target GPU. The float16 and batching findings are
hardware-specific in magnitude, though the directions have held on every device
tested.
