# NFSP Implementation Notes

Sources:
- **[NFSP 1.pdf]** Heinrich & Silver (2016), "Deep Reinforcement Learning from Self-Play in Imperfect-Information Games" (arXiv:1603.01121) — the original NFSP paper
- **[NFSP 2.pdf]** Lin (2025), "Adaptive NFSP Agent with Opponent Modeling for No-Limit Texas Hold'em Poker" (TechRxiv preprint) — RLCard-based extension
- **[Repo]** Eric Steinberger, github.com/EricSteinberger/Neural-Fictitous-Self-Play — reference implementation in the PokerRL framework (verified against source, Sept 2026)
- **[Article]** Ziad Salloum, "Neural Fictitious Self-Play in Practice" — code walkthrough of the repo

---

## 1. Core Concept

NFSP is the first end-to-end deep RL method that converges to approximate Nash equilibria
in imperfect-information games via self-play, without hand-crafted abstractions.

Theory chain:
- Fictitious play converges to Nash in two-player zero-sum games
- XFP (Heinrich et al. 2015) makes it efficient in behavioural form; the average strategy is
  a realization-weighted convex combination of past best responses:
  `σ(s,a) ∝ Σ λ_k · x^π_k(s) · π_k(s,a)`
- NFSP replaces exact best-response computation with **deep RL** and exact averaging with
  **supervised learning**

## 2. Architecture: One Agent (per seat) = 2 Networks + 2 Memories

| Component          | Memory | Memory type        | Loss                    | Role                          |
|--------------------|--------|--------------------|-------------------------|-------------------------------|
| Q-network `Q(s,a)` | M_RL   | **circular buffer**| MSE Q-learning (DQN/DDQN)| Best response `β = ε-greedy(Q)`|
| Avg-policy net `Π` | M_SL   | **reservoir**      | Cross-entropy on actions| Average strategy `π = Π`      |

**Critical memory rules:**
- Every transition `(s, a, r, s')` → M_RL (always)
- Own behaviour `(s, a)` → M_SL **only when acting via best response policy**.
  - Repo nuance: with `add_random_actions_to_buffer=True` (default, matching the paper),
    **ε-random exploration actions are also added to the averaging buffer** — they count as
    best-response behaviour.

## 3. Algorithm (per iteration)

Each episode (per env), the agent picks a training policy — **anticipatory dynamics**:

```
σ = ε-greedy(Q)   with probability η      (η = 0.1; "BR mode")
σ = Π             with probability 1 − η   ("AVG mode")
```

Per iteration (repo: `HighLevelAlgo.run_one_iter`):
1. **Play** `n_steps` steps across `n_envs` parallel envs, storing to memories per the rules above
2. **n_br_updates Q-updates** (default 2): grad from M_RL batch → apply → sync nets to actors;
   every `target_net_update_freq` NN-updates, sync target net
3. **n_avg_updates Π-updates** (default 2): grad from M_SL batch → apply → sync
4. Increment average-strategy iteration counter; refresh exploration ε

Losses (paper):
- `L(θ_Π) = E_{(s,a)~M_SL} [ −log Π(s,a) ]`
- `L(θ_Q)  = E_{(s,a,r,s')~M_RL} [ (r + max_a' Q(s',a'; θ_Q') − Q(s,a; θ_Q))² ]`

## 4. Hyperparameters

### Paper — Leduc Hold'em
| Parameter | Value |
|---|---|
| Network | 1 hidden layer, 64 ReLU (tested 8–128; monotonic gain with size) |
| M_RL / M_SL | 200k circular / 2M reservoir |
| LR RL / SL | 0.1 / 0.005, vanilla SGD, no momentum |
| Batch / updates | 128; 2 SGD updates per network per 128 game steps |
| Target net refit | every 300 updates |
| η | 0.1 (0.5 → plateau; violates DFP stability conditions) |
| ε exploration | 0.06 → 0, decay ∝ 1/√iterations |
| State encoding | Leduc = flat vector of **length 30** (6 cards, 2 rounds, 0–2 raises, 3 actions); LHE = 288. k-of-n card encoding + betting-history tensor |

### Paper — Limit Texas Hold'em
- 4 hidden layers 1024/512/1024/512 ReLU; memories 600k / 30M
- M_SL = **exponentially-averaged reservoir** (min probability 0.25)
- LR 0.1 / 0.01; batch 256; 2 updates per 256 steps; target refit every 1000 updates
- η = 0.1; ε from 0.08, slower decay

### Repo — `example_paper.py` (Leduc, "same hyperparameters but 3-layer networks")
| Parameter | Value |
|---|---|
| Networks | **Dueling Q-Net** for BR; AvrgStrategyNet (logits → softmax) for AVG; `use_pre_layers=False` → shallow 3-layer net, **128 units** per layer |
| Memories (per seat per LA) | `cir_buf_size_each_la=2e5`, `res_buf_size_each_la=2e6` |
| Updates | 2 BR + 2 AVG updates per iter; batch 128 each |
| LR | 0.1 (BR, SGD, MSE) / 0.005 (AVG, SGD, CE); **grad-norm clipping 10.0** both |
| ε | start 0.06, exponent 0.5, min 0, plus small constant floor (`eps_const=0.01`) |
| Parallelism | **128 parallel envs**, 128 steps/env/iter; 1 LearnerActor local (Ray workers in distributed mode) |
| Target net | every 300 **NN updates** (not iterations) |
| Iter-0 pretraining | `training_multiplier_iter_0`: extra BR updates in iteration 0 (BR net is clueless but already fills reservoir) |
| Eval | best-response exploitability every 2000 iterations; default eval mode = average strategy |

**Result:** 0.1 mba/g exploitability crossed after 230k iterations (~4.5 h, single m5.xlarge);
0.06 mba/g after 850k iterations (~19 h). Paper's NFSP reached exploitability 0.06 in Leduc
(≈ full-width XFP after ~1000 iterations).

## 5. What Breaks NFSP (paper ablations — do not skip)

- **Sliding-window M_SL** → divergence
- **η = 0.5** → plateau
- **Exponentially-averaged reservoir in Leduc** → noisy (OK at LHE scale with min-prob 0.25)
- **Plain DQN** (greedy or averaged) → highly exploitable; imperfect-info games need
  **stochastic policies**; ε-greedy self-play data is too correlated/narrow. NFSP's slowly
  changing average policy gives a smooth data distribution → stable learning
- Appendix: constant averaging stepsizes plateau (don't diverge); reservoir ≈ stepsize 1/T;
  noise in best response degrades monotonically but stays stable

## 6. How the Repo Actually Implements It (verified from source)

### Structure / actor map (matches the article)
```
Driver.py                 → outer loop: evaluate() → run_one_iter() → checkpoint/log
  _HighLevelAlgo.py       → the paper's Algorithm 1 (play / Q updates / Π updates / syncs)
  workers/la/local.py     → LearnerActor: holds ParallelEnvs + per-seat components:
      SeatActor.py        → one per seat: picks policy per env, samples actions, writes memories
      action_buffer/      → M_SL reservoir (ActionBufferFLAT / RNN)
      (PokerRL provides)  → M_RL circular buffer, DDQN learner, env builders
  workers/ps/             → one ParameterServer per seat: owns authoritative weights, applies grads
  workers/chief/          → orchestration + evaluation
```

### Key implementation details that differ from a naive reading of the paper
1. **Everything is per-seat.** Each LearnerActor holds a separate DDQN, AvgWrapper, circular
   buffer, and reservoir **for each seat id** — the same network plays one seat position
   (data is keyed by which player you are, since ranges/positions matter).
2. **Input format:** networks take `(public_obs, range_idx, legal_action_mask)` — the private
   hand is encoded as a *range index* (private-info embedding), not concatenated into the flat
   vector. Public observation features are processed by a multi-branch "MainPokerModule"
   (card block units + merge/table units) when `use_pre_layers=True`.
3. **AVG net outputs logits → softmax**; actions in AVG mode are **sampled** from the
   distribution (`np.random.choice(p=a_probs)`), not argmax. CE loss applies a legal-action mask.
4. **Reservoir formula** (`ActionBufferFLAT.add_step_with_sampling`):
   - while not full: append
   - else: `prob_add = max(max_size / n_entries_seen, min_prob)`; if drawn, overwrite a
     uniform-random slot. `min_prob=0.0` → vanilla reservoir; `min_prob>0` → exponential averaging.
5. **Anticipatory choice is per-episode, per-env** (`SeatActor._pick_training_policy` called on
   episode terminal). With multiple parallel envs, different envs can be in different modes
   simultaneously; actions are batched per mode.
6. **Random actions into M_SL**: `add_random_actions_to_buffer=True` stores even ε-random
   exploration actions in the averaging buffer (paper-faithful).
7. **Iteration-0 pretraining**: BR gets `training_multiplier_iter_0`× extra updates so the
   reservoir isn't filled with a clueless BR's behaviour without compensation.
8. **Training cadence**: play 128 steps × 128 envs per LA → then 2 BR grad steps and 2 AVG
   grad steps per iteration (grads computed on LAs, applied on per-seat ParameterServers).
9. Built on **PokerRL** (games: StandardLeduc, BigLeduc, FlatLimitPoker, HistoryRNN envs;
   bet sets; optional stack randomization), logging via PyCrayon, distribution via **Ray**.

## 7. Extension: Adaptive NFSP with Opponent Modeling [NFSP 2.pdf]

(Non-peer-reviewed preprint; trained vs a RandomAgent in RLCard NLHE. Demonstrates
*exploitation of weak opponents*, not Nash-level play.)

- **Hand-strength general strategy** (exploration): `strength = Σ card_value(c) / (len·14)`,
  A=14…2=2. If > 0.7 → raise (fallback half-pot/call), else fold. Threshold calibrated so
  those hands win ~70% vs random opponents.
- **Opponent model**: Random Forest (350 trees, depth 20) on **15 handcrafted features**
  (obs stats, chip ratio, stage weight, hand metrics: pair/straight indicators, fold tendency).
  62.9% accuracy, fold recall 0.91, raise recall only 0.22 (data imbalance).
- **Decision rule**: ε-greedy (ε = 0.8, ×0.9995 decay, min 0.1). Random < ε → general strategy;
  else predict opponent action: fold→raise, call→call, else → NFSP.
- **Results**: payoff −2.717 → −0.343 (−87% loss), win rate 40.3% → 47.0% vs random.
  Ablation: opponent modeling ≈ 80% of the gain; general strategy covers early adaptation.
- **Caveats**: handcrafted features limit scalability; faster ε decay converges faster but
  risks premature exploitation (< ~2000 episodes).

## 8. Implementation Checklist

1. Environment with imperfect-information info-states (Leduc = standard testbed; RLCard or PokerRL)
2. Encoding: k-of-n card vectors + betting history (flat vector; Leduc ≈ 30 dims)
3. Two networks per seat: Q (DQN/DDQN, MSE, target net) + Π (logits→softmax, CE, legal-action mask)
4. M_RL = FIFO circular buffer; M_SL = reservoir (formula above); **M_SL fed only in BR mode**
   (incl. ε-random actions if paper-faithful)
5. Per-episode anticipatory mode selection with η = 0.1
6. Cadence: 2 BR + 2 AVG SGD updates per ~128 steps (per LA), batch 128; target sync every
   300 NN updates; LR 0.1/0.005 SGD; grad-clip 10; ε: 0.06 → 0 with exponent 0.5
7. Iter-0 BR pretraining multiplier
8. Evaluate exploitability via best-response computation (repo: every 2000 iters);
   report mba/g or mbb/hand; 2δ exploitability ⇒ δ-Nash
9. Only after vanilla NFSP converges: add adaptive opponent-modeling layer for weak opponents

**Three things that separate converging NFSP from diverging DQN self-play:**
1. Reservoir-sampled M_SL fed only with best-response behaviour
2. η = 0.1 anticipatory mixture (not 0.5, not 1.0)
3. Stochastic average-policy play (evaluation and AVG-mode acting sample from Π)
