# Hybrid NFSP + SD-CFR: Design Plan

Goal: one model that beats both `nfsp_v2` and `sdcfr_prod` independently.

Chosen direction: **SD-CFR as teacher, NFSP as function approximator.** SD-CFR's
regret estimates become a supervised target for NFSP's average-strategy head.
Gating (SD-CFR proposes, NFSP decides whether to obey) is documented as a later
phase in §7, not the starting point.

> **Implementation status (2026-11-03).** Phases A–C are implemented and tested;
> Phase D has a working driver but the long training run has not been done.
>
> | Component | State |
> |---|---|
> | `SDCFRModel(obs_version=2)` | Done, v1 checkpoints still load |
> | `ExternalSamplingTraverser(obs_version=, bet_scheme=)` | Done |
> | `SDCFRBot` follows the model's encoder | Done |
> | `HybridNetV2` shared trunk + Q/policy heads | Done |
> | `NFSPModel(shared_trunk=, distill_kappa=)` | Done |
> | `scripts/train_sdcfr_v2.py` (v2 teacher) | Done, trains |
> | `scripts/train_hybrid.py` (distillation driver) | Done, runs |
> | `tests/ml/test_hybrid.py` | 16 tests passing |
> | κ=0 reproduces plain NFSP exactly | Verified to 1e-9 with dropout held fixed |
> | Distillation moves policy to CFR target | Verified: 0.0003 → 0.9999 |
> | Long hybrid run vs both parents | **Not done — Phase D remaining** |
>
> A 600-episode smoke comparison gave +443 BB/100 (κ=0.25) vs +259 (κ=0), but the
> 95% intervals overlap almost entirely at n=40, so that is not evidence of a win.
> Phase D needs a long run and the paired evaluation in §5.


---

## 1. Why the two models are complementary

Measured from `models/nfsp_v2/eval_history.csv` (kept here as the design evidence):

| Signal | What it shows |
|---|---|
| vs SD-CFR, 50K episodes | −209 → −122 BB/100, interval straddling zero |
| BR-proxy column | swung `+369`, then `−463`, then `−314` |
| vs Random | consistently positive (+264 → +499) |

Reading: NFSP is roughly break-even against SD-CFR and has **no stable best
response** — the BR proxy oscillates wildly, which is the signature of a best
response that is not actually converging.

The two methods fail in opposite ways:

| | NFSP | SD-CFR |
|---|---|---|
| Function approximation | Strong — 318-d hero-relative encoder, card embeddings, ~700K params | Weak — 98-d flat vector, 7 fixed buckets |
| Convergence guarantee | None. Reservoir of historical BR actions is not regret-weighted | Has one (regret matching) |
| Data efficiency | Reuses one network over the whole history | **Re-initialises the advantage net every iteration** — throws away learning |
| Best response quality | Erratic (measured above) | Sound by construction |

So SD-CFR has the correct *objective* but a poor *representation*; NFSP has the
better representation but a weaker objective. Distillation moves the good
objective into the good representation.

---

## 2. Core idea

Do **not** blend two action distributions at inference. That is a heuristic with
no guarantee — if SD-CFR and NFSP disagree, no averaging rule tells you which is
right, and a mixture of the two is not a Nash equilibrium of either.

Instead, make SD-CFR's output a **training signal** rather than a runtime
component:

```
                    ┌─────────────────────────────┐
  obs v2 (318) ───► │   shared residual trunk     │
                    └──────────┬──────────────────┘
                               │
                 ┌─────────────┴──────────────┐
                 ▼                            ▼
        ┌─────────────────┐          ┌─────────────────┐
        │  value head V+A │          │  policy head π  │
        └────────┬────────┘          └────────┬────────┘
                 │                            │
      Double-DQN on M_RL            cross-entropy on M_SL
      (NFSP best response)        ┌───────────┴───────────┐
                                  ▼                       ▼
                        reservoir BR actions   regret-matching targets
                        (existing NFSP signal)   from SD-CFR (new signal)
```

The average-strategy head is trained on **both** sources at once:

\[
L_\pi = (1-\kappa)\,\underbrace{\mathbb{E}_{(s,a)\sim M_{SL}}\left[-\log \pi_\theta(a\mid s)\right]}_{\text{NFSP reservoir}} \;+\; \kappa\,\underbrace{\mathbb{E}_{s}\left[\mathrm{CE}\big(\pi_\theta(\cdot\mid s),\;\sigma_{RM}(s)\big)\right]}_{\text{SD-CFR distillation}}
\]

where \(\sigma_{RM}(s)\) is regret matching over SD-CFR's advantage predictions.
\(\kappa\) is the **only** new hyperparameter that matters, and §6 explains how to
choose it. At \(\kappa = 0\) this exactly reproduces today's NFSP; at \(\kappa = 1\)
it is pure advantage-net distillation. The interesting regime is in between,
because the two terms correct each other's failure modes:

- The reservoir term keeps the policy grounded in *actually observed* behaviour —
  it prevents the distilled policy from confidently extrapolating into states
  SD-CFR's traversal never visited.
- The regret term supplies the convergence pressure the reservoir lacks — a
  reservoir is uniform over history, so it drifts toward whatever the best
  response used to be, not toward the average strategy.

---

## 3. Prerequisite: unify the observation

This is the largest piece of work and must come first, because a shared trunk
cannot consume two different encodings.

Current split:

| | NFSP v2 | SD-CFR |
|---|---|---|
| Observation | `build_observation_v2` → 318 | `build_observation` → 98 |
| Bet scheme | `POT_FRACTION` | `LEGACY` |
| Network | `PokerNetV2` | `PokerNetwork` |

Steps:

1. Add `obs_version` / `bet_scheme` parameters to `SDCFRModel` mirroring the ones
   `NFSPModel` already accepts (it validates them in `__init__`).
2. Make `ExternalSamplingTraverser` take an encoder callable instead of calling
   `build_observation` directly, so it can emit v2 observations.
3. Make `SDCFRBot` read `model.obs_version` and `model.bet_scheme` instead of
   hard-coding `LEGACY` and `build_observation`.
4. Retrain SD-CFR on v2. **The existing `sdcfr_ckpt_500.pt` cannot be reused** —
   its weights are tied to the 98-d encoder. Budget for a fresh SD-CFR run.

Validate the migration by checking that the retrained SD-CFR is not worse than the
legacy one on `validate_training.py --bots` before wiring distillation.

---

## 4. Prerequisite: stop discarding SD-CFR's learning

`SDCFRModel.train_iteration()` currently does this:

```python
# Reinitialise network weights (SD-CFR resets each iteration)
self.advantage_network = PokerNetwork(input_dim=_OBS_DIM, num_actions=_NUM_ACTIONS)
self.optimizer = optim.Adam(self.advantage_network.parameters(), lr=self.lr)
```

For a single-iteration SD-CFR run that is defensible. For a *shared trunk* it is
fatal: the trunk would be reset every CFR iteration, so it could never accumulate
anything, and distillation targets would come from a network that had just been
wiped.

Replace with warm-starting: keep the network (and ideally the trunk) across
iterations and reset only the *targets*, since the regret buffer already carries
iteration weighting via `WeightedCircularBuffer` with \(\sqrt{t}\) weights.

---

## 5. Implementation phases

### Phase A — plumbing (no learning change)
1. Unify observation and bet scheme per §3.
2. Warm-start SD-CFR per §4.
3. Retrain SD-CFR on v2; confirm it matches or beats legacy on a scored table.
4. Verify: `pytest` green, SD-CFR BB/100 not regressed.

### Phase B — emit regret targets
`ExternalSamplingTraverser` currently stores only the action it sampled:

```python
# traversal.py — current behaviour
advantages = np.zeros(7, dtype=np.float32)
advantages[action_i] = terminal_reward * is_weight
```

This is a single-action outcome sample, not a counterfactual value per action, so
it cannot produce a regret-matching distribution. Extend it to record, for each
traversal, the terminal utility of *each* legal action at each traverser decision
(the standard outcome-sampling estimate), so the buffer holds genuine
per-action advantages.

Store a full `(obs, advantages, legal_mask)` row. `WeightedCircularBuffer` already
has `obs_dim=105` packing for SD-CFR (`98 + 7`); widen to `OBS_V2_DIM + 7`.

5. Verify: for a fixed simple state, regret matching over emitted advantages
   produces a sane distribution (sums to 1 over legal actions, uniform when all
   advantages are ≤ 0).

### Phase C — shared trunk + distillation
1. Add a `HybridModel` (or extend `NFSPModel`) with one trunk and two heads. The
   cheapest correct route is to reuse `PokerNetV2` and expose both `value_head`/
   `adv_head` and `policy_head` from a single trunk — `nfsp_networks.py` already
   builds both variants, so factor the trunk out of the two constructors.
2. Implement \(L_\pi\) from §2 with weight \(\kappa\).
3. Keep the NFSP best-response machinery untouched: Double DQN, target net, η
   anticipatory mode, reservoir. Only the policy loss changes.
4. Verify: with \(\kappa = 0\) the model must reproduce current NFSP behaviour
   bit-for-bit on a fixed seed. This is the regression guard — do it first.

### Phase D — evaluation
Use the existing harness; do not invent new metrics.

There is currently no ``hybrid:`` bot token in ``validate_training.py`` /
``poker.main``. Until one is added, a hybrid checkpoint loads under the existing
``nfsp:`` token, because the hybrid **is** an ``NFSPModel`` — its policy head is
what answers ``select_action``:

```bash
python scripts/validate_training.py --hands 400 --seed 1 \
  --bots nfsp:models/hybrid/hybrid.pt \
         nfsp:models/nfsp_v2/nfsp_v2.pt \
         sdcfr:models/sdcfr_prod/sdcfr_ckpt_500.pt \
         random flop
```

Adding a distinct ``hybrid:`` token is a small follow-up if you want the two
labelled separately in one table.

Success criterion, stated before running: hybrid must beat **the better of the two
parents** on BB/100 with a non-overlapping 95% interval, and must not collapse
Fold% below ~20% or Sense% below ~90%.

Also keep the BR-proxy check from `train_nfsp_v2.py::exploitability_proxy`. The
hybrid should show a *stable* proxy across evaluations; the oscillation in §1 is
the symptom it is meant to cure, so a hybrid that still swings ±400 BB/100 has not
achieved the goal even if its mean is higher.

**Running it (implemented):**

```bash
# 1. v2 teacher (the shipped sdcfr_prod checkpoint is obs v1 and cannot be used)
python scripts/train_sdcfr_v2.py --iterations 400 --traversals 200

# 2. hybrid learner
python scripts/train_hybrid.py \
  --teacher models/sdcfr_v2/sdcfr_v2.pt \
  --kappa 0.25 --shared-trunk \
  --episodes 60000 --eval-every 5000 --checkpoint-every 5000
```


---

## 6. Choosing κ

Do not sweep blindly. Start at \(\kappa = 0.25\) and justify it:

- Below ~0.1 the regret term is too weak to fix the drifting reservoir, so you
  have paid the engineering cost for nothing.
- Above ~0.5 the policy becomes a distillation of SD-CFR and inherits its
  representation error — at which point why keep the reservoir at all.

Treat κ as the tuning knob only after Phase C is proven at κ=0 to equal baseline.
If κ needs to move outside [0.1, 0.5], that is evidence a prerequisite is broken,
not evidence you need a different κ.

A principled refinement, if a fixed κ underperforms: anneal it,
\(\kappa_t = \min(0.5, \kappa_0 \sqrt{t})\), mirroring the existing
`epsilon_schedule="inv_sqrt"` convention in `NFSPModel`.

---

## 7. Phase E (later): gating on Nash-gap

The gating idea — SD-CFR proposes, NFSP decides whether to obey — is worth
keeping, but only *after* distillation works. Two reasons it is not the starting point:

1. A gate needs a signal to gate on. "Player performance" is not observable at
   decision time; you need an estimate of how far the current state is from
   equilibrium, which distillation is what gives you.
2. Gating is an *exploitation* layer. Exploiting weak opponents (which is what the
   adaptive-NFSP preprint in `NFSP_Notes.md` §7 does) is a different objective from
   converging to Nash. Conflating them is how agents end up with a policy that is
   neither.

If pursued, the natural signal is already computable: **disagreement between the
two heads.** When \(\pi_\theta(\cdot\mid s)\) and \(\sigma_{RM}(s)\) agree, the
agent is near its own equilibrium and either choice is safe. When they disagree
sharply, that is measurable Nash-gap and the moment to prefer the more
conservative of the two. That gives a gate with a real input, rather than a
hand-tuned threshold on win rate.

---

## 8. Open questions

1. **Multi-seat.** NFSPv2 trains several seats on one shared model with per-seat
   anticipatory mode (`_episode_br_by_seat`). Should the hybrid keep one trunk
   across seats, or one per seat? One trunk is far cheaper and already works for
   NFSP; start there.
2. **Does the reservoir still earn its place?** If distillation dominates at every
   viable κ, delete the reservoir rather than carry dead weight. Decide this from
   the κ=0.25 ablation, not by intuition.
3. **SD-CFR retraining cost.** The legacy run reached iteration 499. Budget a
   comparable run on v2 before Phase C can start at all.
