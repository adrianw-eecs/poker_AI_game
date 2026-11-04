# SD-CFR (Single Deep Counterfactual Regret Minimization) — Implementation Notes

Sources used for this summary:

1. **SD-CFR paper** — "Single Deep Counterfactual Regret Minimization", Eric Steinberger (arXiv:1901.07621) — `c:\Users\abcd1\Downloads\SDCFR 1.pdf`
2. **Deep CFR paper** — "Deep Counterfactual Regret Minimization", Brown, Lerer, Gross, Sandholm (arXiv:1811.00164, ICML 2019) — `c:\Users\abcd1\Downloads\SDCFR 2.pdf`
3. **Reference implementation** — https://github.com/EricSteinberger/Deep-CFR (by the SD-CFR author; implements both Deep CFR and SD-CFR on top of the [PokerRL](https://github.com/TinkeringCode/PokerRL) framework)

---

## 1. Background: Why SD-CFR exists

### The problem
- **CFR** (Counterfactual Regret Minimization, Zinkevich et al. 2008) is the leading framework for finding approximate Nash equilibria in two-player zero-sum imperfect-information games (e.g. poker).
- Tabular CFR requires **full game-tree traversals** or a **manually designed abstraction** of the game. Abstraction requires expert knowledge, isn't possible in all domains, and leads to exploitable policies in huge games (HUNL has ~10^160 states).
- **Deep CFR** (Brown et al. 2018) removes the need for abstraction by using deep neural networks to approximate CFR directly from samples, generalizing over the whole state space.

### Deep CFR's remaining inefficiency
Deep CFR trains **two kinds of networks**:
1. **Value / advantage networks** `V(I, a)` — approximating counterfactual regrets, one per player, retrained each iteration.
2. **A final average-strategy network** `S(I, a)` — trained **only once, at the very end of training**, to approximate the linear average strategy.

SD-CFR's core insight: **the average-strategy network `S` is redundant and introduces additional sampling + approximation error.** Instead, keep the buffer `B^M` of **all past value (advantage) networks** and derive the average strategy from them at play time. This gives:

- **Lower overall approximation error** (Theorem 1 vs Theorem 2 in the paper: even a *perfect* `S` approximator is not guaranteed to reproduce the true average strategy, whereas SD-CFR reproduces it *exactly* if the value networks are perfect).
- **Faster training** (no average-network training, no `B^S` strategy buffer — which is huge: ~25 GB in 5-FHP vs ~120 MB for SD-CFR's network buffer).
- Empirically: lower exploitability than Deep CFR in Leduc and head-to-head wins in 5-Flop Hold'em.

> Deep CFR paper itself endorses this (Section 4): "If the number of Deep CFR iterations and the size of each value network model is small, then one can avoid training the final policy network by instead storing each iteration's value network (Steinberger, 2019)."

---

## 2. Core algorithms and math

### 2.1 CFR background
Two-player zero-sum extensive-form game. Histories `h ∈ H`, terminals `Z`, utilities `u_i(z)`, imperfect information captured by **information sets** `I`. Strategy `σ_i(I, a)` = probability of action `a` in infoset `I`.

- **Instantaneous regret** (iteration `t`):
  `r_t^i(I, a) = π_{-i}^{σt}(I) · (v_i^{σt}(I, a) − v_i^{σt}(I))`
  (counterfactual reach-probability-weighted value difference of taking `a` vs the current strategy).
- **Regret matching** produces the iteration strategy:
  ```
  σt+1_i(I, a) = R_t^i(I,a)+ / Σ_a R_t^i(I,a)+   if Σ R+ > 0
                = 1/|A(I)|                        otherwise
  ```
- The quantity that converges to a Nash equilibrium is the **average strategy**:
  `σ̄_T^i(I, a) = Σ_t π^{σt}_i(I) σ^t_i(I, a) / Σ_t π^{σt}_i(I)`

### 2.2 Linear CFR weighting (used by both Deep CFR and SD-CFR)
Both algorithms approximate **Linear CFR (LCFR)**: every iteration `t` is weighted by `t` in both the regret accumulation and the average strategy:
```
D_T^i(I, a) = R_T^i,linear(I, a) / Σ_{t=1..T} (t · π_{-i}^{σt}(I))
σ̄_T^i(I, a) = Σ_t (t · π^{σt}_i(I) σ^t_i(I, a)) / Σ_t (t · π^{σt}_i(I))
```
Dividing the advantage by the total linear opponent/chance reach `Σ t·π_{-i}` keeps network targets in a stable numeric range (networks struggle across many orders of magnitude) and is safe because that denominator is identical for all `a ∈ A(I)`.

### 2.3 Deep CFR training loop (identical in SD-CFR)
For each iteration `t`, for each player `p` (alternating updates):

1. Run `K` **external-sampling traversals** with player `p` as the *traverser*:
   - At traverser infosets: **explore all actions**; back up the strategy-weighted average of child values.
   - At opponent/chance nodes: **sample one action**; back up its value unaltered.
   - At each traverser infoset visited, compute `r̃_t(I, a) = v(a) − Σ_a' σ(I,a')v(a')` and store `(I, t, r̃_t)` in the advantage memory `M^V,p` (reservoir-sampled when full).
   - While the *opponent* acts, their strategy vector `σ_t(I)` would be stored in the strategy memory `M^Π` — **this step is only needed for Deep CFR, not for SD-CFR.**
2. Train a **new value network from scratch** on the weighted MSE loss over the whole memory:
   `L(θ_p) = E_{(I,t',r̃)~M^V,p}[ t' · Σ_a ( r̃_t'(a) − V(I, a|θ_p) )² ]`
   (each sample weighted by the iteration number `t'` it was created on; rescale batch weights by `2/T`).
3. During traversals, the acting strategies `σ_t` are computed by regret matching on the outputs of the **current** value networks.

**Alternating updates**: only one player's regrets are updated per iteration (empirically faster convergence); an "iteration" can mean one update per player — note the repo counts one update for *both* players as one iteration.

### 2.4 SD-CFR: deriving the average strategy from the network buffer `B^M`
After training, SD-CFR holds a buffer `B^M_i` containing **every** per-iteration value network `D̂_t^i` (each ~400 KB; tens of thousands fit in ~10 GB). Two query modes:

**(a) Trajectory sampling (for playing / rollouts / one-on-one matches)**
- At the start of a hand, sample **one** value network `D̂_t ∈ B^M` with probability proportional to its iteration number `t`.
- Play the **whole trajectory** with the regret-matching policy that this single network induces.
- Correctness: the network is weighted by `t`, and its induced strategy `σ^t` naturally reaches each infoset with probability proportional to `π^{σt}_i(I)` — which exactly reproduces the linear-average-strategy weighting `t·π·σ`. Query cost is **constant** (one network), same as Deep CFR.

**(b) Explicit full distribution query (for exploitability evaluation)**
- To compute `σ̄_T(I)` at an arbitrary infoset: run every network in `B^M` on `I` and on every traverser infoset `I′` on the path to `I`, compute each iteration's reach `π^{σt}_i(I)` via equation (6), then form the `t·π`-weighted average of the regret-matching strategies. Caching step-wise reach probabilities along a played trajectory reduces this to at most `|B^M|` queries per step.

**Regret matching detail**: use positive-regret matching, but when `Σ_a D+(I, a) ≤ 0`, Deep CFR/SD-CFR **heuristically pick the argmax-advantage action** rather than uniform. The Deep CFR paper's ablation found this reduces final exploitability by ~50% vs uniform.

### 2.5 Key theorems (justifying the design)
- **Theorem 1 (Deep CFR flaw)**: with finite buffer `B^S` or finite traversals `K` per iteration, the average-strategy network is *not guaranteed* to represent `σ̄_T` correctly **even with perfect value networks** (external sampling may never visit some infosets for the strategy buffer; reservoir capacity / irrational reach ratios break exactness).
- **Theorem 2 (SD-CFR exactness)**: if all value networks are perfect, SD-CFR represents `σ̄_T` **without error**, for both query modes; and an opponent cannot distinguish trajectory-sampling from explicit `σ̄` queries from gameplay alone.

---

## 3. Network architecture (poker)

From the Deep CFR paper (Appendix C reference code, ~98,948 parameters):

- **Input encoding**:
  - **Cards**: for each card group (hole cards, flop, turn, river) the cards are represented as the **sum of three embeddings**: rank embedding (13), suit embedding (4), card embedding (52); masked/zeroed for missing cards. Permutation-invariant per group.
  - **Betting history**: for each of `6·N_rounds` possible betting positions, a binary "bet occurred" flag + a float bet size (clamped).
- **Two-branch trunk**:
  - Card branch: 3 fully-connected layers (e.g. 192 or 64 units each depending on game).
  - Bet branch: 2 fully-connected layers with a **skip connection** on the second.
  - Merged trunk: 3 fully-connected layers with **skip connections** where dimensions match; **last layer normalized** (zero mean / unit variance).
- **Output head**: linear layer → one value per action (advantages for value nets; logits for an average net in Deep CFR).
- Regularization found important: **skip connections + last-layer normalization** (needed for fast convergence in preliminary tests). Loss: **MSE** (any Bregman divergence works). Optimizer: **Adam, lr 1e-3, gradient-norm clipping**.
- The Deep-CFR repo also supports **recurrent** nets (`nn_type="recurrent"`, LSTM-based main module with 128 units) for games with arbitrarily long histories, plus a **dueling** head (`DuelingQNet`) for the advantage net.

---

## 4. Reference implementation layout (EricSteinberger/Deep-CFR)

```
DeepCFR/
├── TrainingProfile.py          # All hyperparameters (see §5)
├── EvalAgentDeepCFR.py         # Play/eval agent; EVAL_MODE_SINGLE vs EVAL_MODE_AVRG_NET
├── IterationStrategy.py        # Wraps one iteration's adv net (adds t-weighted sampling)
├── StrategyBuffer.py           # B^M: buffer of iteration strategies for SD-CFR
└── workers/
    ├── driver/Driver.py, _HighLevelAlgo.py   # Main loop: traversals → adv training → (avrg training) → eval
    ├── la/                     # Learner actors: generate traversals, compute grads
    │   ├── AdvWrapper.py / AvrgWrapper.py
    │   ├── buffers/AdvReservoirBuffer.py, AvrgReservoirBuffer.py, _ReservoirBufferBase.py
    │   └── sampling_algorithms/MultiOutcomeSampler.py   # batched external sampling
    └── ps/                     # Parameter servers (hold current adv/avrg weights)
```

Notable implementation details from `_HighLevelAlgo.py`:
- Main loop `run_one_iter_alternating_update(cfr_iter)`: for each player `p`: sync nets → `K` traversals (`_generate_traversals`) → train adv net from scratch (`_train_adv`: batched gradient computation across learner actors + Adam updates on a parameter server + LR-scheduler step) → **if SD-CFR mode: `_push_newest_adv_net_to_chief`** (store the iteration's adv net in the SD-CFR buffer) → re-sync.
- Deep CFR mode additionally calls `train_average_nets` (train the average net from a random init at the end).
- The eval agent supports both modes simultaneously on **shared value networks**, which is how the paper compared SD-CFR vs Deep CFR with low variance.

---

## 5. Hyperparameters (proven settings)

### Leduc (SD-CFR paper, Appendix A)
| Parameter | Value |
|---|---|
| Traversals per iteration | 1,500 external sampling |
| Adv net training | 750 updates, batch 2048, trained to convergence |
| Adv net init | random at t < 2, then **weights of iteration t−2** |
| Buffer capacity `B^V` | 1M per player |
| Architecture | 3 FC layers × 64 units (more params than Leduc has states) |
| Avg net (Deep CFR baseline) | 5,000 updates, batch 2048, always random init |

### 5-Flop Hold'em (SD-CFR paper §6.2, matches Brown et al.)
| Parameter | Value |
|---|---|
| Traversals per iteration | 300,000 external sampling |
| Adv net training | 4,000 updates, batch 10,240, **from scratch each iteration** |
| Avg net (Deep CFR) | 20,000 updates, batch 20,480 |
| Buffer capacity | 40M entries per player (`B^V`); `B^S` ~25 GB (averted by SD-CFR) |
| SD-CFR `B^M` size | all value nets ≈ 120 MB on disk |
| Eval | head-to-head mbb/g, 3M hands per point, 5 runs |

### Repo defaults (`TrainingProfile.py`) — general recipe
- `n_traversals_per_iter=30000`, `n_batches_adv_training=5000`, `mini_batch_size_adv=4096`
- Adam, `lr_adv=0.001`, `grad_norm_clipping_adv=10.0`, `loss_adv="weighted_mse"` (iteration-weighted MSE)
- `init_adv_model`: `"random"` (paper shows from-scratch beats fine-tuning by a large margin) or `"last"`
- `iter_weighting_exponent=1.0` → linear CFR weighting (`t^1`); set `0` for vanilla CFR weighting
- `max_buffer_size_adv=3e6` reservoir buffer
- `eval_modes_of_algo=(EvalAgentDeepCFR.EVAL_MODE_SINGLE,)` → SD-CFR
- Distributed: grads from `n_learner_actor_workers` learner actors are averaged on a parameter server; hyperparameters suffixed `_per_la` scale with worker count.

### Critical ablation findings (Deep CFR paper, Fig. 4)
1. **Retrain from scratch each iteration** — fine-tuning one model yields ~50% higher final exploitability (the objective shifts every iteration; a single model gets stuck in bad local minima).
2. **Reservoir sampling, not sliding window** — sliding-window buffers cause exploitability to stall/increase once full; reservoir sampling keeps an unbiased sample of all past iterations and keeps converging.
3. **Linear weighting** (`t`-weighted loss) — same asymptote as unweighted but converges faster. (No formal convergence proof for linear weighting in the MC case, but empirically better.)
4. **Argmax tie-break on all-negative regrets** — ~50% lower exploitability than uniform-random.

---

## 6. Results summary

| Experiment | Result |
|---|---|
| Leduc exploitability | SD-CFR beats Deep CFR across 10²–10³ iterations (mA/g scale) |
| Leduc, `B^M` reservoir-sampled at capacity 10–1000 | Plateauing + oscillation up to `|B^M|=1000`; **store all networks** |
| 5-FHP head-to-head (SD-CFR vs Deep CFR, shared value nets) | SD-CFR wins decisively (mean winnings positive across 5 runs, up to ~+30–40 mbb/g before `B^S` fills at iter 120; still clearly ahead afterwards), 3M hands per eval |
| Strategy agreement by depth (5-FHP) | Deep CFR's `Ŝ` matches SD-CFR's `σ̄` closely at depth 0–1 (L1 diff ~0.01) but diverges deeper (0.18 at depth 5) — Deep CFR's average-net error concentrates deep in the tree |
| Deep CFR vs NFSP (FHP) | Deep CFR reaches 37 mbb/g exploitability vs NFSP's 47 mbb/g, more sample-efficient |
| Deep CFR vs tabular abstractions (FHP) | Matches a 3.6M-cluster abstraction, converges 2–3 orders of magnitude fewer nodes touched than lossless abstraction |
| HULH head-to-head | Deep CFR loses only −11 ± 2 mbb/g to a 3.3×10⁸-bucket abstraction and beats NFSP by +43 ± 2 mbb/g |

Practical numbers: the network is tiny (<100 KB, ~400 KB on disk); 25,000 stored networks ≈ 10 GB — **storing `B^M` is never a practical problem**, while Deep CFR's `B^S` strategy buffer cost ~25 GB *and* added error.

---

## 7. How to implement SD-CFR — step-by-step

### Step 1 — Game engine
Implement (or use) an extensive-form game wrapper exposing:
- observation/information-set encoding per decision point,
- legal actions `A(I)`, chance sampling,
- terminal payoffs `u_i(z)` (two-player zero-sum assumed).

### Step 2 — Advantage network(s)
- One network per player `p` with the architecture from §3; output = one advantage per legal action.
- Pre-decide all nets share a fixed architecture; each iteration's net will be **fully retrained and archived**.

### Step 3 — Reservoir advantage buffer
- `M^V,p` per player: stores tuples `(infoset_features, iteration_t, r̃_t(I, a) vector)`.
- Reservoir sampling (Vitter 1985) once capacity is exceeded — **never** a sliding window.

### Step 4 — External-sampling traversal
Implement `TRAVERSE(h, p, θ1, θ2, t)` per Deep CFR Algorithm 2:
1. Terminal → return payoff. Chance → sample.
2. Traverser's turn: get advantages from current net → regret matching (with **argmax fallback when all regrets ≤ 0**) → recurse on **all** actions → store `(I, t, r̃)` in `M^V,p` → return `Σ_a σ(I,a)·v(a)`.
3. Opponent's turn: get advantages from opponent's net → regret matching → (Deep CFR only: store `σ_t(I)` in `M^Π`) → **sample one action** → recurse.
4. Optionally normalize advantage targets by `Σ t·π_{-i}(I)` (LCFR advantage definition) to keep targets well-scaled.

### Step 5 — Per-iteration advantage training
Each iteration `t`, for each player (alternating):
- Train a **fresh** net on the weighted MSE loss (weights = sample iteration `t′`; rescale by `2/T`); Adam, lr 1e-3, grad clipping; enough SGD steps that loss plateaus (convergence criterion used in the papers: fixed budget like 750–4,000 updates).
- **Archive the trained net** (weights + iteration number + architecture metadata) into `B^M_p` on disk. This replaces Deep CFR's average-net pipeline entirely.

### Step 6 — Acting (SD-CFR trajectory sampling)
- At the start of each hand, sample iteration index `t ~ P(t) ∝ t` from `B^M`.
- For every decision in the hand, feed the infoset to that **one** network and apply regret matching to its advantage vector to get `σ^t(·|I)`.
- (If you need explicit action distributions — e.g. for best-response/exploitability computation — enumerate all `t ∈ B^M`, weight each induced strategy by `t·π^{σt}_i(I)` computed along the trajectory, and normalize.)

### Step 7 — Evaluation
- Exploitability: best-response calculation against the explicit average strategy (mode 6b) in small games (Leduc).
- Head-to-head: play mode 6a against baselines, report mbb/g or mA/g with confidence intervals over many hands.
- Log adv-net training loss per iteration (a good convergence signal).

### Step 8 — Scaling up / engineering
- Traversals are CPU-bound and training is per-iteration sequential — the reference repo parallelizes traversal generation + gradient computation across learner actors (Ray), with parameter servers holding current weights.
- Store archived nets compressed; only load the sampled net at play time (or lazily for explicit queries).
- If games have huge action counts, external sampling becomes intractable — consider outcome sampling / average-strategy sampling instead (paper's future-work note).

---

## 8. Pitfalls & gotchas

1. **Never reservoir-sample `B^M`** (the SD-CFR network buffer) — Figure 1b shows plateauing/oscillation even with capacity 1,000. Store all value nets.
2. **Iteration-counting convention**: the repo defines an iteration as an update for *both* players, so "iteration 300" in the SD-CFR paper plots = 150 in the repo's TensorBoard logs.
3. **From-scratch training** each iteration is essential; initializing from the previous net (fine-tuning) converges worse. (Leduc exception in the paper: init from net `t−2` was used and worked — the from-scratch finding is from FHP-scale runs.)
4. Regret matching with uniform fallback instead of argmax-on-negative-regrets costs ~50% exploitability.
5. SD-CFR's advantage depends on **all** past networks being kept — deleting early networks skews the linear average (early iterations matter, weighted by `t`).
6. Iteration 0 eval results are meaningless (random net vs uniform).
7. Memory-format consistency: all archived nets must share architecture/observation encoding so any infoset can be fed to any iteration's net.

---

## 9. Quick reference — Deep CFR vs SD-CFR

| Aspect | Deep CFR | SD-CFR |
|---|---|---|
| Value/advantage nets | Per-player, retrained from scratch each iteration | **Identical** (same training) |
| Average strategy | Separate net `Ŝ` + strategy buffer `B^S`, trained at the end | Derived from buffer `B^M` of all past value nets |
| Extra buffers | `B^V` + `B^S` (huge, e.g. ~25 GB in 5-FHP) | Only `B^V` + disk archive of nets (~120 MB) |
| Approximation error | Advantage error + `B^S` sampling error + `Ŝ` fit error | Advantage error only |
| Theory | Theorem 1: `Ŝ` biased even with perfect value nets | Theorem 2: exact `σ̄_T` if value nets perfect |
| Play-time cost | One net query per decision | One net query per decision (sample `D̂_t ∝ t` once per hand) |
| Empirics | Baseline | Lower exploitability (Leduc); beats Deep CFR head-to-head (5-FHP) |
