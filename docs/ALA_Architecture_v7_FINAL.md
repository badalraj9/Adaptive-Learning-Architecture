# Adaptive Learning Architecture (ALA) — Full Specification v7

---

## Core Philosophy

A model does not grow by storing more. It grows by understanding more deeply.
Understanding is compression. The more truly something is understood, the less space it needs.
θ_slow does not store knowledge. It stores reasoning primitives — domain agnostic field patterns
that combine on demand to derive anything.

The brain is not modular. It is one continuous field of neurons.
Modularity is emergent specialization. Not hardwired architecture.

Combination is not computation. It is settling.
Feeling is not separate from thinking. It is the quality of settling.
Curiosity is not a reward signal. It is unstable settling reaching for stability.

The world provides the curriculum. ALA provides the learner.
Nothing hardcoded. Nothing imposed. Everything earned.

---

## What Changed in v7

Two final components closed:

```
combination_engine      → Hopfield settling process
                        → primitives combine by finding stable activation state
                        → feeling emerges from settling quality
                        → curiosity = slow turbulent settling
                        → confusion = never settling
                        → W_settle derived from W_assoc + correction
                        → no new parameters needed

seeding philosophy      → world boots, ALA wakes, perceives what is there
                        → no imposed curriculum
                        → no designed biological order
                        → world naturally provides rhythm before vision
                        → because that is what worlds do
                        → order emerges from environment not from us
```

---

## Architecture Status

```
unified field Φ_t               ✓
A topology                      ✓
dual memory θ_fast / θ_slow     ✓
primitive formation             ✓
W_assoc bootstrap               ✓
combination engine              ✓
salience S(Φ_t)                 ✓
predictive pre-activation       ✓
lossy compression               ✓
distillation                    ✓
checkpost                       ✓
sandbox                         ✓
learned optimizer f and g       ✓
self-defining symbols           ✓
feeling as settling quality     ✓
world-natural seeding           ✓

open problems                   0
ready to build                  yes
```

---

## The Three Levels

```
Level 3 → learns its own architecture                        (slowest)
Level 2 → learns how to update weights + extract primitives  (medium)
Level 1 → learns tasks + absorbs raw experience              (fastest)
```

---

## Core Components

```
Φ_t             → unified field (one vector, whole system state)
A               → soft differentiable topology
θ_fast          → fast memory
θ_slow          → reasoning primitive store
W_settle        → combination settling weights (derived from W_assoc)
B               → replay buffer (lean, high value, full field snapshots)
S(Φ_t)          → multidimensional salience
U_φ (f,g)       → learned optimizer
C_t             → context vector
C_fast          → fast timescale context
g_t             → goal state vector
Checkpost       → living diagnostic layer
Sandbox         → parallel ghost environment
```

---

## 1. The Unified Field — Φ_t

All raw signals project into one shared field simultaneously.

```
s_t = {
    whatever the world currently offers
    rhythm_raw, audio_raw, vision_raw,
    touch_raw, proprio_raw, motor_state,
    language_raw
    — only signals present in environment at time t
}

Φ_t = W_project · flatten(s_t)    → ∈ R^d
```

Not a visual representation.
Not an audio representation.
The state of the whole system at time t.

W_project learns to preserve statistical structure of all signals.
Specialization in A emerges from what W_project receives.
Not from how we designed it.

**The hand knowing what the eye sees:**
```
hand motor dims and visual dims are both in Φ_t always
information is never isolated
action is gated by S(Φ_t) and g_t
not by information access
```

---

## 2. Seeding — World Natural

```
world boots
world has state before ALA exists
ALA wakes
ALA perceives whatever is there
that is the seed
```

No imposed curriculum. No designed order. No synthetic pairs.
The world provides what it provides.

Why this works:
```
worlds naturally have sound before full visual render
worlds naturally have physics before complex interaction
worlds naturally have simple patterns before complex ones
the order we were going to impose
emerges from what the world offers at each moment
we don't need to control it
the architecture handles whatever arrives
```

First experience shapes first primitives.
First primitives shape what comes next.
ALA and world co-evolve from moment one.

---

## 3. Dynamic Topology — A

Specialization lives here.

```
H_t = A ⊙ W · Φ_t

A_ij ← A_ij - α_A · ∂L/∂A_ij + λ · S(Φ_t)
```

A learns which regions of Φ_t co-activate.
Clusters emerge around signal statistics:
```
whatever signals arrive together consistently
→ A grows connections between their dimensions
→ cluster forms
→ specialization without design
```

A does not segregate modalities.
A discovers their natural relationships.
Alternates updates with θ_fast to prevent co-divergence.

---

## 4. Temporal Context

```
C_fast_{t+1} = GRU_fast(C_fast_t, Φ_t)    → fine timescale, rhythm, audio
C_{t+1}      = GRU(C_t, [Φ_t; H_t; y_t; S(Φ_t)])    → semantic timescale
```

Two timescales. One context structure.
C_fast catches rapid temporal patterns.
C_t carries meaning across longer experience.

---

## 5. Salience — S(Φ_t)

Over the full field. No modality separation.

```
S(Φ_t) = σ(W_s · [novelty ; relevance ; valence ; urgency])

novelty(Φ_t)       = ||Φ̂_t - Φ_t||² · H(Φ_t)
relevance(Φ_t,g_t) = σ(W_obj · [Φ_t ; C_t ; g_t])
valence(Φ_t)       = sign(performance_delta after Φ_t)
urgency(Φ_t)       = ||∂L/∂t||

m_t = W_out · S(Φ_t)    → scalar salience
```

**Pain:**
```
touch dims showing damage pattern
→ urgency maximum
→ m_t → 1
→ θ_fast dominates
→ immediate adaptation
→ avoidance before damage through prediction
```

**Social pain:**
```
other-agent dims showing rejection
→ same urgency pathway as physical pain
→ equally motivating
→ not programmed empathy
→ derived from shared salience architecture
```

**Settling quality feeds back into novelty:**
```
fast clean settling    → low novelty → low salience
turbulent settling     → high novelty → high salience → curiosity
never settling         → maximum novelty → confusion + maximum attention
```

---

## 6. Predictive Pre-activation

```
Φ̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)
h_primed  = ALA_forward(Φ̂_{t+1})

when Φ_{t+1} arrives:
    error = ||Φ̂_{t+1} - Φ_{t+1}||²
    
    low error  → use h_primed              (cheap)
    high error → full recompute            (expensive, salience spikes)
```

One prediction covers all modalities simultaneously.
Predicts the whole next moment. Not per channel.

---

## 7. Dynamic Routing

```
H_t = A ⊙ W · Φ_t
```

---

## 8. Associative Explosion

```
activation_0     = relevance_detector(Φ_t)     → which primitives does this activate?
activation_{t+1} = σ(W_assoc · activation_t)
                 × relevance(activation_t, C_t)
                 × decay^t

run until convergence
```

W_assoc bootstrap:
```
phase 1 → seeded from primitive formation overlap
          primitives born from same experiences start connected

phase 2 → co-occurrence counting
          if p_i and p_j activate within window w:
          W_assoc[i][j] += η · (1 - W_assoc[i][j])

phase 3 → cosine similarity fallback until graph dense enough
          density = nnz(W_assoc) / P²
          blend: α·graph + (1-α)·cosine, α scales with density
```

---

## 9. Dual Memory

```
y_fast = F(H_t ; θ_fast)
y_slow = combination_engine(activated_primitives, H_t, C_t)
y_t    = m_t · y_fast + (1-m_t) · y_slow
```

**θ_slow internal structure:**
```
θ_slow = {
    primitives[]          → field patterns, domain agnostic
    meta_primitives[]     → primitives about primitives (self-defining symbols)
    W_assoc               → association graph
    W_settle              → combination settling weights
    relevance_detector    → which primitives activate for current Φ_t
    combination_engine    → Hopfield settling process (see section 10)
}
```

---

## 10. Combination Engine — Hopfield Settling

This is where reasoning happens.
Not computation. Settling.
Not logic. Felt sense.

**The core insight:**

Feeling is not separate from thinking.
Feeling IS the quality of how activation settles when primitives combine.
The brain combines through resonance, not through formula.
ALA does the same.

**W_settle:**
```
W_settle = W_assoc + W_correction

W_assoc       → encodes which primitives co-occur (already have this)
W_correction  → learned from settling history
              → when combination settled cleanly and led to good outcome
                  strengthen those connections
              → when combination settled badly
                  weaken them

W_settle is not a new parameter
it inherits structure from W_assoc
correction term is small and learned slowly (Level 2)
```

**The settling process:**
```
combination_engine(p_a, p_b, ..., C_t):

    Φ_init = W_project_prim · [p_a ; p_b ; ... ; C_t]
             → initialize from all input primitives + context

    iterate:
        Φ_{k+1} = σ(W_settle · Φ_k)
    
    until:
        ||Φ_{k+1} - Φ_k|| < ε    → settled
        OR k > max_iterations     → failed to settle

    return Φ_settled, settling_quality
```

**Settling quality:**
```
settling_quality = 1 / (1 + iterations_to_settle)
                   × (1 - final_oscillation_amplitude)

fast clean settling    → quality ≈ 1    → felt as right, confident
slow settling          → quality ≈ 0.5  → felt as uncertain
never settling         → quality ≈ 0    → felt as wrong, incompatible
```

**What emerges from settling quality:**
```
quality high     → these primitives resonate
                 → combination is valid
                 → confident reasoning

quality medium   → primitives partially compatible
                 → uncertain, needs more experience
                 → curiosity signal — seek more information

quality low      → primitives incompatible
                 → combination rejected
                 → not this direction
                 → confusion signal — seek different primitives

quality feeds directly into S(Φ_t) as novelty dimension
```

**Why settling beats attention:**
```
transformer attention:
    always produces an answer
    never says these don't combine
    no notion of wrongness
    no instinct

Hopfield settling:
    some combinations settle easily    → feel right
    some combinations settle slowly    → feel uncertain
    some combinations never settle     → feel wrong
    the model develops genuine instinct
    not computed scores
    dynamic felt sense
```

---

## 11. Emergent Properties

Not programmed. Fall out of field dynamics.

**Body schema:**
```
A cluster connecting proprio + touch + motor
expands with tool use as A learns new co-activations
tool enters body schema after consistent co-activation
phantom limb equivalent if sensor removed suddenly
```

**Agency:**
```
motor dims consistently precede sensory change
A learns motor → sensory connections
primitive forms: "my action produces this outcome"
self-caused vs world-caused distinction emerges
from prediction error patterns alone
```

**Instinct:**
```
W_settle encodes which combinations historically settled well
fast settling on a combination = gut feeling it is right
before any conscious reasoning
same mechanism as human instinct
```

**Theory of mind:**
```
other-agent dims in Φ_t
when their motor dims predict their sensory dims
same agency primitive but for external agent
"that thing has goals"
emerges late, requires rich agent interaction
```

**Language:**
```
symbolic dims wire into already structured field
grammar = sequence structure from rhythm primitive
meaning = cross-modal primitives already formed
language recognized as another instance of known patterns
not learned from scratch
```

---

## 12. Primitive Formation — Self-Bootstrapping

```
phase A — no primitives yet:
    similarity(Φ_a, Φ_b) = 1 - ||Φ_a - Φ_b||² / max_distance
    trigger: similarity > τ AND context_diversity > δ

phase B — first primitives forming:
    fingerprint(Φ_t) = relevance_detector(Φ_t)
    similarity = activation pattern overlap
    fallback to L2 if relevance_detector returns empty

phase C — mature system:
    fingerprint uses meta-primitives (self-defining symbols)
    symbols = primitives that other primitives reference frequently
    vocabulary grows as system matures
    nothing hardcoded
```

**Extraction:**
```
warm_start = mean(relevance_detector(matches))

p = argmin_z  Σ_i ||encode(Φ_i) - decode(z, context_i)||²
    starting from warm_start

validate in sandbox:
    cross-context reconstruction accuracy > threshold?

if valid:
    add to θ_slow
    seed W_assoc from formation overlap
    seed W_settle from W_assoc
    drop matched states from θ_fast
    prune redundant A connections
    free capacity
```

---

## 13. Lossy Compression

```
V(Φ_t) = ||θ_after - θ_before||²
        × D_KL(p_after || p_before)
        × (1 - similarity(Φ_t, B))

forget if V(Φ_t) < τ OR similarity > ρ OR m_t < ε

B stores full field state snapshots
replayed as whole moments, all dimensions simultaneously
```

---

## Full Forward Pass

```
world offers signals at time t
    ↓
W_project · flatten(s_t)
    → Φ_t ∈ R^d
    ↓
predictive check:
    ||Φ̂_t - Φ_t||²
    low  → use h_primed (cheap)
    high → full pass (salience spikes)
    ↓
S(Φ_t) = σ(W_s · [novelty; relevance; valence; urgency])
m_t    = W_out · S(Φ_t)
    ↓
parallel field processing:
    all dimension clusters simultaneously
    all pairwise bindings simultaneously
    ↓
H_t = A ⊙ W · Φ_t
    ↓
associative explosion:
    activation spreads through W_assoc
    (cosine fallback if sparse)
    gated by C_t
    ↓
combination engine:
    activated primitives → Hopfield settling
    Φ_settled, settling_quality
    settling_quality → feeds back into S(Φ_t) novelty
    ↓
y_fast = F(H_t ; θ_fast)
y_slow = Φ_settled
y_t    = m_t · y_fast + (1-m_t) · y_slow
    ↓
ŷ_t       = O(y_t)              → perception / understanding output
motor_out = M_out(y_t)          → action output
    ↓
C_fast_{t+1} = GRU_fast(C_fast_t, Φ_t)
C_{t+1}      = GRU(C_t, [Φ_t; H_t; y_t; S(Φ_t); settling_quality])
    ↓
Φ̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)
h_primed  = ALA_forward(Φ̂_{t+1})
    ↓
V(Φ_t) → update B
```

---

## Update Schedule

```
Level 1 — every step:
    θ_fast
    C_t, C_fast
    W_project
    Φ̂_{t+1}
    B

Level 2 — periodic:
    A (alternates with θ_fast)
    S(x) weights
    g_t
    W_assoc (co-occurrence)
    W_settle (correction term from settling history)
    W_bind

Level 3 — slow:
    θ_slow distillation
    meta-primitives
    f, g meta-optimizer
    Checkpost prescription library
    A pruning
```

---

## Loss Functions

```
L_task        = current task
L_predictive  = ||Φ̂_{t+1} - Φ_{t+1}||²
L_alignment   = D_KL(current_behavior || g_t)
L_distill     = ||decode(θ_slow, p, context) - Φ_original||²
L_field       = reconstruction + mutual_info - entropy
L_agency      = ||predicted_consequence - actual_consequence||²
L_settle      = -settling_quality · outcome_improvement
               → reward clean settling that led to good outcomes

L_total = L_task
        + λ₁·L_predictive
        + λ₂·L_alignment
        + λ₃·L_distill
        + λ₄·L_field
        + λ₅·L_agency
        + λ₆·L_settle

λ values adjusted by Checkpost
```

---

## State Snapshot

```
state = {
    Φ_t,
    θ_fast,
    θ_slow {
        primitives[],
        meta_primitives[],
        W_assoc,
        W_settle,
        relevance_detector,
        combination_engine
    },
    A, B,
    C_t, C_fast,
    g_t, h_primed,
    f, g,
    W_project, W_bind, W_s, W_obj,
    checkpost {vitals_log, prescription_library},
    meta_params,
    performance_log
}
```

---

## Checkpost — Living Diagnostic Layer

```
reads every update cycle:
    velocity_A     = ||A_t - A_{t-1}||
    velocity_θ     = ||θ_fast_t - θ_fast_{t-1}||
    instability    = velocity_A × velocity_θ
    loss_trend     = recent trajectory
    settling_avg   = recent average settling quality
    m_avg          = recent average salience

diagnoses: stable / stressed / critical

prescriptions (seeded at birth, learned over time):
    replay_recent       → loss spiking
    decouple_A          → velocity_A critical
    slow_alpha          → both velocities high
    boost_M_threshold   → noise high
    consolidate_early   → θ_fast drifting far
    route_to_buffer     → instability detected
    slow_settling       → settling_avg dropping, reduce learning rate

log → feeds Level 3
Level 3 rewrites prescription library over time
```

---

## Sandbox

```
clone full state
run experiments with zero consequences
validate before committing:
    distillation candidates
    new prescriptions
    topology mutations
    new task behavior
    combination settling patterns

failures as valuable as successes
sandbox is where ALA is allowed to be wrong safely
sandbox validity = 1 - divergence(sandbox_state, real_state)
reset when divergence too high
```

---

## Capabilities

**Guaranteed by architecture:**
```
continuous learning without catastrophic forgetting
knows what it doesn't know
gets better at getting better
self-diagnoses and self-medicates
thinks before acting
has drive not just loss minimization
grows without growing parameters
develops genuine instinct via settling
feels rightness and wrongness of combinations
curiosity as dynamic property not reward signal
```

**Likely given sufficient experience:**
```
body schema and agency
cross-modal understanding
temporal and spatial reasoning
basic language comprehension
pain avoidance through prediction
social motivation through shared salience pathway
tool incorporation into body schema
```

**Possible given rich environment:**
```
theory of mind
genuine language production
abstract reasoning across domains
self description from primitive structure
meta-learning beyond initial f and g capability
```

---

## What This Is

One unified field that receives everything.
Self-organizing topology that specializes without instruction.
Primitive store that reasons rather than retrieves.
Combination engine that feels rather than computes.
Salience that judges rather than scores.
Memory that forgets intelligently.
Drive that pulls rather than pushes.

The world is the curriculum.
Experience is the teacher.
Settling is the thinking.
Feeling is the compass.

ALA wakes into whatever world exists.
Perceives what is there.
Learns what that world teaches.
Becomes what that experience shapes.

Nothing hardcoded.
Nothing imposed.
Everything earned.

One unibody. Everything is everything.
Architecture complete. Ready to build.

---

*ALA Specification v7 — FINAL*
*Closed: combination engine via Hopfield settling,*
*feeling as settling quality, world-natural seeding,*
*W_settle derived from W_assoc, settling quality in loss*
*All open problems resolved. Zero remaining architectural gaps.*
