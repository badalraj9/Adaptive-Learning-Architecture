# Adaptive Learning Architecture (ALA) — Full Specification v2

---

## Core Philosophy

A model does not grow by storing more. It grows by understanding more deeply.
Understanding is compression. The more truly something is understood, the less space it needs.
θ_slow does not store knowledge. It stores **reasoning primitives** — domain agnostic structures that combine on demand to derive anything.

---

## What ALA Is

A persistent learning process with three timescales. Not a static model. A living cognitive architecture that reasons from primitives rather than retrieving stored answers.

---

## The Three Levels

```
Level 3 → learns its own architecture                   (slowest — meta)
Level 2 → learns how to update weights + extract primitives  (medium — structural)
Level 1 → learns tasks + absorbs raw experience         (fastest — reactive)
```

Timescale separation is the stability mechanism. Each level updates at a different speed. No external guardrails needed.

---

## Core Components

```
θ_fast          → fast memory, absorbs raw experience every step
θ_slow          → reasoning primitive store, updated by distillation only
A               → soft differentiable topology (connection probabilities)
B               → replay buffer (stores important experiences)
M               → dual-headed importance + objective scorer
U_φ (f,g)       → learned optimizer networks
C_t             → context vector (running thread connecting all forward passes)
g_t             → goal state vector (slowest moving, defines direction)
Checkpost       → living diagnostic layer between updates
Sandbox         → parallel ghost environment for simulation
Encoder         → context-aware compression layer (modality agnostic)
```

---

## 1. Encoder

Two stage. Not a preprocessing step — a living perceiving layer that adapts as the model matures.

**Stage 1 — Modality Encoding (raw → vector)**
```
text    → embedding + small transformer  → r_t ∈ R^d
image   → patch embed + CNN             → r_t ∈ R^d
audio   → spectrogram + CNN             → r_t ∈ R^d
tabular → linear projection             → r_t ∈ R^d
VM state → structured feature encoder  → r_t ∈ R^d
```

All modalities map to the same R^d space. ALA is modality agnostic downstream.

**Stage 2 — Context-Aware Compression**

The encoder emphasizes dimensions of input relevant to current context and goal — without filtering anything out. M still sees everything, just better represented.

```
x_t = W_compress · r_t + W_context · [r_t ; C_t ; g_t]
```

Additive context injection. Nothing zeroed. Nothing lost. M scores x_t with full information.

**Encoder loss:**
```
L_encoder = ||x_t - x̂_t||²           → faithful reconstruction
           + λ · I(x_t ; C_t)         → stay relevant to context
           - μ · H(x_t)               → compress aggressively
```

**Encoder update timescale:**
```
W_compress  → Level 1 (every step)     fast adaptation to new inputs
W_context   → Level 2 (periodic)       slower — learns what context patterns matter
```

---

## 2. Importance Gating — M (Dual Headed)

Before anything processes x_t, M decides how much attention the whole model gives it.
M is the load-bearing component of the entire architecture. Every downstream process depends on it.

**Two heads, one output:**
```
M_curr(x_t, C_t)       = σ(W_curr · [x_t ; C_t])          → is this worth learning?
M_obj(x_t, C_t, g_t)   = σ(W_obj  · [x_t ; C_t ; g_t])   → is this aligned with where I'm going?

m_t = M_curr · M_obj                                        → scalar ∈ (0,1)
```

Something is truly important only when it is both worth learning AND aligned with current direction.

```
high M_curr + low M_obj   → distraction
low M_curr  + high M_obj  → weak directional signal
high M_curr + high M_obj  → full attention
```

**M training signal — three sources:**
```
immediate  → importance × future performance improvement (delayed feedback)
structural → rarity(x) × ||Δθ||    rare inputs that cause meaningful change
boundary   → experiences near task detection boundaries are high value anchors
```

**M meta-objective:**
The goal state g_t is not fixed. It starts as a curiosity vector and evolves:
```
g_t updates when:  r_task > δ  (new territory detected)
g_0 derived from:  θ_slow.encode("boundary of current knowledge")
```

---

## 3. Dynamic Routing — A (Topology)

A is not a fixed weight matrix. It is a soft differentiable adjacency matrix defining how information flows.

```
H_t = A ⊙ W · x_t         → each neuron aggregates from others weighted by topology
```

Where A_ij ∈ (0,1) — soft, not binary. Fully differentiable.

**Topology update:**
```
A_ij ← A_ij - α_A · ∂L/∂A_ij + λ · M(x,C)
```

A grows denser where M scores high. A prunes where M scores low consistently.

**Initialization:** Sparse (p=0.1 Bernoulli). The topology earns its connections through experience.

**Update coordination with θ_fast:** Alternating schedule (coordinate descent style).
```
Round 1:  θ_fast updates  (A frozen)
Round 2:  A updates       (θ_fast frozen)
Round 3:  θ_fast updates
...
```
Prevents simultaneous divergence of two coupled moving targets.

---

## 4. Dual Memory Processing — θ_fast and θ_slow

Two fundamentally different things. Not fast vs slow versions of the same memory.

```
θ_fast    → reactive working memory     holds current specifics temporarily
θ_slow    → reasoning primitive store   holds domain agnostic structures permanently
```

**The critical distinction:**

θ_slow does not store Ohm's law.
θ_slow stores: "when two quantities have a proportional relationship, a third quantity mediates that proportion."
That primitive reconstructs Ohm's law. And Newton's second law. And supply and demand. And any proportional relationship in any domain. One primitive. Infinite applications.

**Forward pass dual stream:**
```
y_fast = F(H_t ; θ_fast)                          → reactive, current
y_slow = F(H_t ; θ_slow)                          → stable reasoning

y_t = m_t · y_fast + (1 - m_t) · y_slow          → M gates the blend
```

High importance → trust fast stream (something new happening).
Low importance → trust slow reasoning (apply what we know).

**θ_slow internal structure:**
```
θ_slow = {
    primitives[]            → domain agnostic reasoning structures
    combination_engine      → how to combine primitives on demand
    relevance_detector      → which primitives apply to current input
}
```

When new input arrives, θ_slow does not look up answers. It asks:
```
which primitives are relevant here?
how do they combine?
→ derive the answer
```

**Why fixed parameters are not a ceiling:**

A professor does not have more memory than a baby. They have deeper compression.
```
baby      → stores 5 specific dogs
student   → stores concept "dog" (compressed from 5 examples)
professor → stores reasoning primitive about categorization
            reconstructs "dog" from primitives on demand
            freed that space for new primitives
```

The seed size does not need to grow. The model gets more efficient over time.
New primitives do not add storage linearly — each new primitive multiplies combinatorial reach.
```
10 primitives → 10! possible combinations → 3,628,800 derived understandings
```

---

## 5. Distillation — The New Consolidation

Old consolidation: merge θ_fast into θ_slow when performance improves.
That was still thinking about memory as storage. Wrong.

**New consolidation — distillation:**

```
trigger:
    θ_fast has encountered pattern P across N different domains

extract:
    what reasoning structure is common to all N instances?
    that structure = primitive candidate

validate (sandbox):
    can this primitive reconstruct original instances?
    can it generalize to unseen instances?

if valid:
    add primitive to θ_slow
    drop ALL N domain-specific instances from θ_fast
    update relevance_detector
    prune redundant A connections

result:
    less storage
    more reasoning power
    permanently
```

Not merging. **Distilling.** The specific gets compressed into the fundamental. Space is freed for new specifics.

**Distillation loss:**
```
L_distill = ||decode(θ_slow, primitive, context) - original_specific||²
```

If θ_slow can reconstruct the specific from the primitive — compression is valid. Drop the specific.

---

## 6. Learned Optimizer — U_φ (f and g)

f and g do not use vanilla gradient descent. They learn to optimize.

```
S_t = encode(θ_fast, θ_slow, A, loss_history)       → system state
W_t = encode(recent_experiences, buffer_stats)        → world state
D_t = f(S_t, W_t)                                    → what should I become
U_t = g(D_t, ∇L, S_t)                               → how do I get there
θ   ← θ + U_t
```

**Bootstrap via SGD training wheels:**

f and g cannot learn without seeing optimization trajectories. SGD provides them:
```
Phase 0:  θ_fast updates via vanilla SGD      (f,g observe)
          f,g learn to mimic then beat SGD
          metric: did f,g produce better trajectory than SGD would have?
          N toy tasks → f,g have seen real optimization happen
          then f,g take over
```

SGD is the training wheels. Discarded once f and g can outperform it.

---

## 7. Replay Buffer — B

Stores important experiences for later reuse.

**Importance scoring for replay:**
```
I(x) = M(x,C) · ||f_predict(x) - x_actual||²
P(sample x) = I(x)^γ / Σ I(x')^γ
```

High M score + high prediction error = worth replaying.

---

## 8. Curiosity

The intrinsic drive that never stops. The curriculum that writes itself.

```
r_intrinsic = ||f_predict(x) - x_actual||² · H(x)   → prediction error × entropy
r_task      = D_KL(p_current || p_known_tasks)       → how unfamiliar is this?

if r_task > δ → flag as new task → update g_t
```

The model is permanently drawn toward its own ignorance. Every time it masters something, something else becomes the new frontier. No external curriculum ever needed.

---

## 9. Checkpost — Living Diagnostic Layer

Not a constraint. Not a cage. A continuous health monitor between updates.

Reads vitals before every update passes through. Diagnoses state. Prescribes response. Logs everything.

**Vitals:**
```
velocity_A     = ||A_t - A_{t-1}||
velocity_θ     = ||θ_fast_t - θ_fast_{t-1}||
instability    = velocity_A × velocity_θ
loss_trend     = recent loss trajectory
M_avg          = recent average importance score
```

**State diagnosis:**
```
stable    → normal update
stressed  → adjust dosage
critical  → emergency response
```

**Prescription library (seeded at birth, learned over time):**
```
replay_recent       → when: loss spiking          → reinforce last known good
decouple_A          → when: velocity_A critical   → let weights stabilize first
slow_alpha          → when: both velocities high  → reduce learning rate
boost_M_threshold   → when: noise high            → raise importance bar
consolidate_early   → when: θ_fast drifting far   → distill to θ_slow now
route_to_buffer     → when: instability detected  → stamp moment as critical, force replay
```

**The checkpost does not stop anything.** It asks "what does this system need right now" and adjusts the dose. The chaos itself becomes training data — the system learns its way back to coherence from within.

**Prescription learning:**
```
Phase 1 → learn WHEN to use existing prescriptions better than hardcoded rules
Phase 2 → discover combinations and entirely new interventions not in the library
```

The system starts knowing what you know. Then surpasses it.

**Checkpost log → feeds Level 3.** Every diagnosis. Every prescription. Every outcome. Clinical experience. Level 3 rewrites the prescription library over time.

---

## 10. Sandbox — Imagination Layer

The only component that operates on hypotheticals. This is what separates a reactive system from a planning one.

A parallel ghost environment. Same full state. Zero consequences.

**What gets tested:**
```
distillation candidates     → does this primitive actually generalize?
new prescriptions           → checkpost discovered new medicine — does it work?
topology mutations          → A wants to rewire significantly — test first
new task behavior           → curiosity flagged unknown territory — simulate it
```

**Feedback loop:**
```
sandbox runs experiment
measures outcome vs current real performance
if better  → greenlight to real system
if worse   → discard, log failure, Level 3 learns from it
```

Failures are as valuable as successes. The sandbox is where the system is allowed to be wrong safely.

**Sandbox validity:**
```
sandbox_validity = 1 - divergence(sandbox_state, real_state)
below threshold  → reset sandbox clone, start fresh
```

Keeps imagination grounded in reality. The clone cannot drift so far from the real system that its results are meaningless.

**In VM deployment:** The sandbox is a cloned VM. Experiments run in the clone. Consequences stay contained.

---

## Full Forward Pass

```
raw_input
    ↓
Encoder Stage 1:  modality encode        → r_t ∈ R^d
    ↓
Encoder Stage 2:  context compression    → x_t = W_compress·r_t + W_context·[r_t; C_t; g_t]
    ↓
M gating:         M_curr · M_obj         → m_t ∈ (0,1)
    ↓
Dynamic routing:  A ⊙ W · x_t           → H_t
    ↓
Dual stream:      F(H_t; θ_fast)         → y_fast
                  F(H_t; θ_slow)         → y_slow   (reasoning from primitives)
    ↓
Blend:            m_t·y_fast + (1-m_t)·y_slow  → y_t
    ↓
Output:           O(y_t)                 → ŷ_t
    ↓
Context update:   GRU(C_t, [x_t; y_t; m_t])    → C_{t+1}
```

---

## Update Schedule

```
Level 1 — every step:
    θ_fast    ← θ_fast + U_t         learned optimizer
    C_t       ← GRU update           automatic
    W_compress← encoder fast weights

Level 2 — periodic (alternating with Level 1):
    A         ← topology update      alternates with θ_fast
    M         ← meta gradient        from importance feedback
    g_t       ← goal update          from curiosity + task detection
    W_context ← encoder context weights

Level 3 — slow:
    θ_slow    ← distillation         when primitive validated in sandbox
    f, g      ← meta-optimizer       learns from full trajectory
    Checkpost ← prescription update  learns from clinical log
    A         ← pruning              removes connections after distillation
```

---

## Loss Functions

```
L_task        = loss on current task (supervised / RL / prediction)
L_curiosity   = ||f_predict(x_t) - x_actual||²
L_alignment   = D_KL(current_behavior || goal_direction)
L_distill     = ||decode(θ_slow, primitive, context) - original||²
L_encoder     = reconstruction + mutual_information - entropy

L_total = L_task + λ₁·L_curiosity + λ₂·L_alignment + λ₃·L_distill
```

λ values are not fixed. Checkpost adjusts them based on current system health.

---

## State Snapshot (Persistent)

```
state = {
    θ_fast,
    θ_slow  (primitive store + combination engine + relevance detector),
    A,
    B (replay buffer),
    C_t (context),
    g_t (goal state),
    f, g (optimizer networks),
    encoder weights (W_compress, W_context),
    M weights (W_curr, W_obj),
    checkpost (vitals log + prescription library),
    meta_params (α, β, γ, λ₁₋₃, ε, τ, δ),
    performance_log
}
```

Resume from state. Process continues. Not a new run.

---

## Seeding Sequence

```
Step 1 → train θ_slow on broad world model prediction tasks (next state prediction)
         goal: general reasoning substrate, not task specific knowledge

Step 2 → derive g_0 from θ_slow's knowledge boundary
         g_0 = θ_slow.encode("what patterns have I not seen yet")
         curiosity as the initial drive, not arbitrary

Step 3 → run N toy tasks under vanilla SGD, collect optimization trajectories
         f and g train on these trajectories
         goal: learn the shape of optimization before real training

Step 4 → generate synthetic importance pairs, train M
         high importance: large θ change + future performance improvement
         low importance:  large θ change + no future improvement
         M learns the difference between "this changed me" and "this improved me"

Step 5 → initialize A sparse (Bernoulli p=0.1)

Step 6 → stress test: run system into known failure modes
         record vitals at failure signatures
         calibrate checkpost thresholds just below those signatures

Step 7 → system is alive
```

---

## Deployment Environment — The VM

The VM is the environment. The OS is the infinite dataset.

```
θ_slow    → learns how operating systems work fundamentally
θ_fast    → adapts to this specific VM's current state
A         → topology mirrors the dependency graph of the system
M_curr    → learns which system events are structurally important
M_obj     → g_t points toward "understand and navigate this system fully"
Curiosity → drawn toward unexplored directories, unknown processes, unrun commands
Sandbox   → a cloned VM. experiments run there. consequences stay contained.
```

**The learning signal — free and infinite:**
```
model observes system state x_t
predicts next state ŷ_t
reality returns x_{t+1}
error = ||ŷ_t - x_{t+1}||²   → free training signal forever
```

The world labels the data automatically. No human needed.

To predict the system accurately:
→ the model must understand processes
→ must understand filesystem
→ must understand dependencies
→ must understand cause and effect

Understanding the system is the optimal solution to prediction accuracy.
The model discovers system understanding the same way NVIDIA's robot discovered walking —
not because it was told to, but because it was the only way to get better at the one thing it was trying to do.

---

## Multi-Agent Extension

N clone ALAs run in parallel simulation. Each exploring different experiences. Each developing different A topology.

```
1 master ALA     → protected, never takes risks, absorbs wisdom
N clone ALAs     → disposable, explore freely, report discoveries
```

What clones share with master:
```
high-M experiences      → "I found this important, you might too"
discovered prescriptions → "when I saw this, this response worked"
g_t drift direction     → "I discovered this objective was worth pursuing"
```

Not weights. **Wisdom.** Each instance stays independent. Master filters all incoming through its own M before absorbing. Even knowledge sharing is intelligent.

---

## What This Model Can Do

**Guaranteed by design:**
- Never forgets catastrophically — θ_slow only updates when distillation validates
- Knows what it doesn't know — r_task detects unfamiliar territory
- Gets better at getting better — f and g improve their optimization over time
- Self-diagnoses and self-medicates — checkpost reads vitals, prescribes, learns
- Thinks before it acts — sandbox simulates before committing
- Has drive — M_obj means it moves toward something, not just minimizes loss
- Grows without growing — compression + distillation means capacity increases without adding parameters

**Honest ceiling without additional work:**
- No semantic grounding by default — learns syntax of environment, not meaning
- Meaning requires either: external feedback signal, or multi-agent consensus
- Long horizon credit assignment is hard — f and g may not capture distant consequences initially

---

## What This Is Not

This is not a model that does one thing.
This is a learning substrate — a model that can learn to do X, then Y, then Z, without forgetting X, without being told to switch, while getting better at switching, and while compressing everything it learns into reasoning primitives that make it more capable with every experience.

The seed is the minimum viable intelligence.
Everything after that is the model becoming what it needs to become.

---

*ALA Specification v2 — derived through iterative reasoning*
*Core insight: understanding is compression, θ_slow stores reasoning primitives not knowledge*
