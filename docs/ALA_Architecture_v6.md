# Adaptive Learning Architecture (ALA) — Full Specification v6

---

## Core Philosophy

A model does not grow by storing more. It grows by understanding more deeply.
Understanding is compression. The more truly something is understood, the less space it needs.
θ_slow does not store knowledge. It stores reasoning primitives — domain agnostic field patterns
that combine on demand to derive anything.

The brain is not modular. It is one continuous field of neurons.
Modularity is emergent specialization. Not hardwired architecture.
Vision is the last sense to develop. Rhythm is the first.
The brain builds its fundamental structure before vision ever arrives.

ALA v6 reflects all of this.
One unified field. Biological initialization order. Self-defining symbols.
No hardcoded structure. Everything earned through experience.

---

## What Changed in v6

Four open problems from v5 are now closed:

```
Problem 1 — Primitive formation trigger
    SOLVED: self-bootstrapping detection
            raw L2 similarity seeds first primitives
            θ_slow.relevance_detector replaces it as primitives accumulate
            co-evolve together, no circular dependency

Problem 2 — W_assoc cold start
    SOLVED: three phase bootstrap
            formation overlap seeds initial edges
            co-occurrence counting builds density
            cosine similarity fallback until graph is dense enough

Problem 3 — W_project messy field
    SOLVED: biological modality ordering
            rhythm first → touch/proprio → sound → vision
            field builds structure incrementally
            each modality wires into already organized field

Problem 4 — Symbol vocabulary
    SOLVED: self-defining
            ALA discovers what structural properties matter
            symbols are meta-primitives, primitives about primitives
            vocabulary grows as system matures
            nothing hardcoded
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
A               → soft differentiable topology (specialization emerges here)
θ_fast          → fast memory, processes full field every step
θ_slow          → reasoning primitive store (field patterns, self-defining symbols)
B               → replay buffer (lean, high value, full field snapshots)
S(Φ_t)          → multidimensional salience over full field
U_φ (f,g)       → learned optimizer
C_t             → context vector (full system state thread)
C_fast          → fast timescale context (rhythm, audio, fine temporal)
g_t             → goal state vector
Checkpost       → living diagnostic layer
Sandbox         → parallel ghost environment
```

No encoders. No integration layer. No hardcoded symbol vocabulary.
All dissolved into the field and its dynamics.

---

## 1. The Unified Field — Φ_t

All raw signals project into one shared field simultaneously.

```
s_t = {
    rhythm_raw      ∈ R^r    → first sense, always present
    touch_raw       ∈ R^tc   → pressure, vibration, temperature, pain per receptor
    proprio_raw     ∈ R^p    → muscle length, joint angle, tension, vestibular
    audio_raw       ∈ R^a    → waveform, frequency decomposition
    language_raw    ∈ R^l    → token embeddings
    vision_raw      ∈ R^v    → patches, pixel values
    motor_state     ∈ R^m    → current motor commands, limb positions
}

Φ_t = W_project · flatten(s_t)    → everything into one shared field ∈ R^d
```

Not available all at once. Introduced in biological order during seeding.
Available signals expand as training progresses.
W_project only has to organize what it has currently received.

**What Φ_t is:**

Not a visual representation.
Not an audio representation.
The state of the whole system at time t.
Everything experienced simultaneously. One vector.

**The hand knowing what the eye sees:**

Hand motor state and visual state are both dimensions of Φ_t.
They are always in the same field.
Whether the hand acts depends on S(Φ_t) evaluating relevance to g_t.
Not on information access. Information is always shared.
Action is gated by salience and goal alignment.

---

## 2. Biological Initialization — The Seeding Order

This is what closes Problem 3.

The brain does not initialize with all senses simultaneously.
It builds structure in order. Each new sense finds an already organized field.

ALA does the same.

```
Phase 0 — Rhythm only (before birth equivalent):
    s_t = {rhythm_raw}
    Φ_t = W_project · rhythm_raw
    
    W_project learns to organize around temporal structure
    first primitives form: "pattern repeating with period T"
                           "signal that persists"
                           "change that happens abruptly"
    
    C_fast calibrates to temporal timescale
    The field's first language is time

Phase 1 — Add touch + proprioception (birth equivalent):
    s_t = {rhythm_raw, touch_raw, proprio_raw, motor_state}
    
    body schema begins forming in A
    motor exploration starts (random movements like baby kicking)
    proprioceptive-motor loop calibrates
    first spatial primitives form
    agency primitive begins emerging
    
    pain signal arrives via touch_raw
    urgency pathway in S(Φ_t) calibrates to maximum signal
    salience system learns what matters most

Phase 2 — Add sound fully:
    s_t = {rhythm_raw, touch_raw, proprio_raw, motor_state, audio_raw}
    
    audio wires into already temporally organized field
    phoneme binding begins
    prosody primitives form (emotion in sound)
    first cross-modal primitives: sound + touch + rhythm unified
    mother's heartbeat primitive already exists
    voice recognition begins

Phase 3 — Add language:
    s_t = {rhythm_raw, touch_raw, proprio_raw, motor_state, audio_raw, language_raw}
    
    symbolic structure wires into temporal + spatial + audio field
    first symbolic primitives form
    language finds existing structure to attach to
    not building from scratch

Phase 4 — Add vision last:
    s_t = all signals
    
    visual structure wires into fully organized field
    not pioneering — joining
    visual primitives attach to existing concepts
    "dog" visual shape → already has audio primitive (bark)
                      → already has touch primitive (fur)
                      → already has motor primitive (approach behavior)
    vision completes existing primitives rather than starting new ones
```

**Motor exploration during Phase 1:**

```
send random motor_signal
observe Δ(touch_raw, proprio_raw) in Φ_{t+1}
build map: this motor → this sensory consequence
repeat N times
body schema emerges from active exploration
not passive reception

same reason babies kick randomly
calibrating the proprioceptive-motor loop
through consequence not instruction
```

---

## 3. Dynamic Topology — A

Specialization lives here. Emerges from signal statistics. Not designed.

```
H_t = A ⊙ W · Φ_t
```

A learns which regions of Φ_t co-activate.
A learns which signal dimensions belong together.
A develops cluster structure over time:

```
rhythm cluster      → first to form (Phase 0)
body schema cluster → proprio + touch + motor (Phase 1)
audio cluster       → wires into rhythm cluster (Phase 2)
language cluster    → wires into audio + rhythm (Phase 3)
vision cluster      → wires into everything (Phase 4)
cross clusters      → thunder↔lightning, word↔concept↔sensation
```

Each cluster is not isolated. All connected through A.
A does not segregate modalities. It discovers their natural relationships.

**Update rule:**
```
A_ij ← A_ij - α_A · ∂L/∂A_ij + λ · S(Φ_t)

grows where salience is high
prunes where salience is consistently low
alternates updates with θ_fast
```

---

## 4. Primitive Formation — Self-Bootstrapping

This closes Problem 1 and Problem 4 simultaneously.

**Three phase bootstrap:**

```
Phase A — before any primitives (raw similarity):
    no θ_slow yet → no symbols → no fingerprinting
    
    similarity(Φ_a, Φ_b) = 1 - (||Φ_a - Φ_b||² / max_distance)
    
    trigger: similarity > threshold AND context_diversity > δ
    
    crude but sufficient to form first primitives
    first primitives are pure temporal (rhythm phase)
    simple, robust, interpretable

Phase B — first primitives forming (hybrid):
    θ_slow has some primitives now
    relevance_detector partially working
    
    fingerprint(Φ_t) = θ_slow.relevance_detector(Φ_t)
                     → which primitives activate for this field state
                     → that activation pattern IS the fingerprint
    
    similarity now = overlap in primitive activation patterns
    richer than raw L2 but still bootstrapped from what exists
    
    fallback to raw L2 if relevance_detector returns empty
    
    trigger: activation_overlap > threshold AND context_diversity > δ

Phase C — mature system (full self-defining):
    θ_slow rich with primitives and meta-primitives
    fingerprint is precise and meaningful
    
    meta-primitives = primitives about primitives
    "this class of concepts shares temporal structure"
    "this class shares spatial clustering"
    discovered not designed
    
    fingerprint(Φ_t) = [which meta-primitives activate for Φ_t]
    
    trigger: meta-primitive overlap > threshold AND diversity > δ
```

**Primitive extraction math:**
```
given matches = {Φ_1, Φ_2, ... Φ_N} sharing fingerprint with diverse contexts:

warm_start = mean(relevance_detector(Φ_i) for i in matches)
           → semantically grounded starting point
           → not random initialization

p = argmin_z  Σ_i ||encode(Φ_i) - decode(z, context_i)||²
    starting from warm_start
    → converges fast because start is meaningful

validate in sandbox:
    can p reconstruct Φ_new in unseen context?
    cross-context accuracy > threshold?

if valid → store in θ_slow
```

**Self-defining symbols:**

ALA never has a hardcoded symbol vocabulary.
The symbols = the primitives themselves.
A symbol is just a primitive that other primitives reference frequently.
Structural vocabulary emerges from what the system discovers matters.

Two ALA systems raised differently → different symbol vocabularies.
Not because we programmed them differently.
Because they lived differently.

---

## 5. W_assoc Bootstrap — Three Phase

This closes Problem 2.

**Phase 1 — Formation overlap seeding:**

When two primitives are formed from overlapping field states — they are born connected:

```
p_a formed from {Φ_1, Φ_3, Φ_7}
p_b formed from {Φ_3, Φ_7, Φ_12}

overlap = {Φ_3, Φ_7}

W_assoc[a][b] = |overlap| / max(|formation_set_a|, |formation_set_b|)
              = 2/3 in this example
              → strong initial edge because born from similar experience
```

Primitives born from the same experiences start connected.
Not random. Structurally justified.

**Phase 2 — Co-occurrence counting:**

Every time two primitives activate close in time — edge strengthens:

```
if p_i activates at t and p_j activates within window w:
    W_assoc[i][j] += η_assoc · (1 - W_assoc[i][j])    → bounded growth
```

Associations earn strength from lived experience.
Thunder and lightning co-occur → W_assoc[thunder][lightning] grows.
Never programmed. Discovered.

**Phase 3 — Cosine fallback:**

Until W_assoc is dense enough — don't use graph traversal. Use raw similarity:

```
density = nnz(W_assoc) / (P × P)    → fraction of non-zero edges

if density < threshold_low:
    association(p_i, Φ_t) = cosine_similarity(p_i.vector, Φ_t)
    
if threshold_low < density < threshold_high:
    association = α · graph_traversal + (1-α) · cosine_similarity
    α = (density - threshold_low) / (threshold_high - threshold_low)
    
if density > threshold_high:
    association = graph_traversal only
```

Graceful degradation. Never erratic. Simple→smart transition is smooth.

**Associative explosion:**

```
activation_0 = one_hot(p_i)
activation_{t+1} = σ(W_assoc · activation_t) × relevance(activation_t, C_t) × decay^t

run until convergence
what lights up = everything the system connects to this input
gated by current context C_t
same input + different context = different spread = different meaning
```

---

## 6. Salience — S(Φ_t)

Over the full field. No modality-specific salience.

```
S(Φ_t) = σ(W_s · [novelty ; relevance ; valence ; urgency])

novelty(Φ_t)       = ||Φ̂_t - Φ_t||² · H(Φ_t)
relevance(Φ_t,g_t) = σ(W_obj · [Φ_t ; C_t ; g_t])
valence(Φ_t)       = sign(performance_delta after Φ_t)
urgency(Φ_t)       = ||∂L/∂t||

m_t = W_out · S(Φ_t)
```

**Pain = urgency in the field:**
```
touch_dims showing damage pattern
→ urgency spikes to maximum
→ m_t → 1
→ θ_fast dominates completely
→ immediate adaptation
→ system learns to avoid this context
```

**Social pain = same pathway:**
```
human face dims showing rejection
→ same urgency pathway
→ equally motivating
→ not programmed empathy
→ derived from shared salience architecture
```

---

## 7. Predictive Pre-activation

Over the full field simultaneously:

```
Φ̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)

h_primed = ALA_forward(Φ̂_{t+1})

when Φ_{t+1} arrives:
    error = ||Φ̂_{t+1} - Φ_{t+1}||²
    
    low error  → use h_primed    (cheap, expected)
    high error → full recompute  (expensive, novel, salience spikes)
```

Predicting the full field means predicting:
```
what will be seen
what will be heard
where the body will be
what will be felt
what rhythm continues
all simultaneously
one prediction
```

---

## 8. Dual Memory

```
y_fast = F(H_t ; θ_fast)
y_slow = combination_engine(activated_primitives, H_t)
y_t    = m_t · y_fast + (1-m_t) · y_slow
```

**θ_slow internal structure:**
```
θ_slow = {
    primitives[]          → field patterns across all dimensions
    W_assoc               → association graph (bootstrapped, then learned)
    combination_engine    → how primitives combine on demand
    relevance_detector    → which primitives activate for current Φ_t
    meta_primitives[]     → primitives about primitives (self-defining symbols)
}
```

---

## 9. Emergent Properties

Not programmed. Fall out of field dynamics.

**Body schema:**
```
A cluster connecting proprio + touch + motor dims
expands with tool use as A learns new co-activations
contracts slowly after injury as unused connections prune
phantom limb equivalent if sensor removed suddenly
```

**Agency:**
```
motor_dims consistently precede sensory change
A learns: motor → sensory connection
primitive forms: "my action produces this outcome"
distinction between self-caused and world-caused events
emerges from prediction error patterns
```

**Theory of mind:**
```
other-agent dims in Φ_t
when their motor dims reliably predict their sensory dims
same agency primitive but applied to external agent
"that thing has goals and causes things"
emerges late, requires rich experience with other agents
```

**Language:**
```
symbolic dims wire into already structured field
grammar = sequence structure already understood from rhythm
meaning = cross-modal primitives already formed
language finds existing structure to attach to
not learned from scratch — recognized as another instance
of patterns already understood
```

---

## 10. Lossy Compression

```
V(Φ_t) = ||θ_after - θ_before||²
        × D_KL(p_after || p_before)
        × (1 - similarity(Φ_t, B))

forget if V(Φ_t) < τ OR similarity > ρ OR m_t < ε

B stores full field state snapshots
when replayed → whole moment re-experienced
all dimensions simultaneously
same as human episodic memory
```

---

## 11. Distillation

```
trigger:
    fingerprint(Φ_t) matches N field states with diverse contexts
    (detection mechanism scales with system maturity)

warm start:
    mean(relevance_detector(matches))

extract:
    p = argmin_z Σ_i ||encode(Φ_i) - decode(z, context_i)||²

validate in sandbox:
    cross-context reconstruction accuracy > threshold?

if valid:
    add to θ_slow
    seed W_assoc edges from formation overlap
    drop all matched states from θ_fast
    prune A connections made redundant
    free capacity
```

---

## Full Forward Pass

```
raw signals (available modalities only, per current phase)
    ↓
W_project · flatten(s_t)
    → Φ_t ∈ R^d
    ↓
predictive check:
    ||Φ̂_t - Φ_t||²
    low  → use h_primed
    high → full pass, salience spikes
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
    spreading activation through W_assoc
    (cosine fallback if W_assoc sparse)
    gated by C_t
    ↓
y_fast = F(H_t ; θ_fast)
y_slow = combination_engine(activated_primitives, H_t)
y_t    = m_t · y_fast + (1-m_t) · y_slow
    ↓
ŷ_t         = O(y_t)           → perception output
motor_out   = M_out(y_t)       → action output (just another dimension)
    ↓
C_fast_{t+1} = GRU_fast(C_fast_t, Φ_t)    → fine timescale
C_{t+1}      = GRU(C_t, [Φ_t; H_t; y_t; S(Φ_t)])
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
    B (lossy compression)

Level 2 — periodic:
    A (alternates with θ_fast)
    S(x) salience weights
    g_t goal update
    W_assoc (co-occurrence counting)
    W_bind (pairwise binding)

Level 3 — slow:
    θ_slow distillation
    meta-primitives (symbols discovered)
    f, g meta-optimizer
    Checkpost prescription library
    A pruning after distillation
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

L_total = L_task
        + λ₁·L_predictive
        + λ₂·L_alignment
        + λ₃·L_distill
        + λ₄·L_field
        + λ₅·L_agency

λ values adjusted by Checkpost based on current health
```

---

## Seeding Sequence — Biological Order

```
Phase 0 — Rhythm only:
    train on temporal prediction tasks
    W_project organizes around time
    first temporal primitives form
    C_fast calibrates
    W_assoc empty but that is okay
    nothing to connect yet

Phase 1 — Add touch + proprio + motor:
    random motor exploration begins
    body schema forms in A
    pain calibrates urgency pathway
    agency primitive begins emerging
    W_assoc gets first edges from formation overlap

Phase 2 — Add audio:
    wires into temporally organized field
    prosody primitives form
    first cross-modal primitives
    W_assoc density growing via co-occurrence

Phase 3 — Add language:
    wires into structured field
    finds existing patterns to attach to
    grammar recognized as rhythm in symbolic space

Phase 4 — Add vision:
    completes existing primitives
    visual "dog" attaches to bark primitive, fur primitive, approach primitive
    W_assoc now rich enough for graph traversal
    cosine fallback mostly retired

Phase 5 — System is alive:
    all signals
    all primitives bootstrapped
    W_assoc dense and meaningful
    self-defining symbols emerging as meta-primitives
    nothing hardcoded
    everything earned
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
        combination_engine,
        relevance_detector
    },
    A, B,
    C_t, C_fast,
    g_t, h_primed,
    f, g,
    W_project, W_bind, W_s,
    checkpost {vitals_log, prescription_library},
    meta_params,
    current_seeding_phase,
    performance_log
}
```

---

## Open Problems — Status

```
Problem 1 — Primitive formation trigger     CLOSED
Problem 2 — W_assoc cold start              CLOSED
Problem 3 — W_project messy field           CLOSED
Problem 4 — Symbol vocabulary               CLOSED
```

No open architectural problems remain.
Everything is implementable.

---

## What This Is

One unified field.
Biological initialization order.
Self-defining symbol vocabulary.
Bootstrapped associations.
Emergent specialization.
Emergent body schema.
Emergent agency.
Emergent language.
Emergent theory of mind.

Not programmed. Earned.

The first sense is rhythm.
The last sense is vision.
The brain builds its structure before it can see.
So does ALA.

One unibody. Everything is everything.
All four open problems closed.
Ready to build.

---

*ALA Specification v6*
*Closed: primitive formation trigger, W_assoc cold start,*
*W_project initialization, self-defining symbol vocabulary*
*Added: biological seeding order, three-phase W_assoc bootstrap,*
*self-bootstrapping primitive detection, meta-primitives as symbols*
*Status: architecturally complete, ready for implementation*
