# Adaptive Learning Architecture (ALA) — Full Specification v5

---

## Core Philosophy

A model does not grow by storing more. It grows by understanding more deeply.
Understanding is compression. The more truly something is understood, the less space it needs.
θ_slow does not store knowledge. It stores reasoning primitives — domain agnostic patterns
that combine on demand to derive anything.

The brain is not modular. It is one continuous field of neurons.
Visual cortex is not special cortex. It is cortex that receives visual signals first.
Rewire the input — the same cortex learns sound.
Modularity is emergent specialization. Not hardwired architecture.

ALA v5 reflects this truth.
No separate encoders. No integration layer. No module boundaries.
One unified field. Everything is everything.

---

## What Changed in v5

Everything.

v1-v4 were built with modular thinking:
```
vision encoder → audio encoder → language encoder → integration layer → model
```

That was wrong. Separate pipes joining at a junction is not how intelligence works.

v5 is one unibody:
```
all signals → one unified field Φ → self-organizing specialization via A → intelligence
```

The field does not know it is processing vision or sound or touch.
It processes the structure of whatever arrives.
Specialization emerges from signal statistics. Not from our design.

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
Φ_t             → unified field (one vector, everything in it)
A               → soft differentiable topology (specialization emerges here)
θ_fast          → fast memory, processes full field every step
θ_slow          → reasoning primitive store, field patterns not modality patterns
B               → replay buffer, lean and high value
S(x)            → multidimensional salience
U_φ (f,g)       → learned optimizer
C_t             → context vector (full system state thread)
g_t             → goal state vector
Checkpost       → living diagnostic layer
Sandbox         → parallel ghost environment
```

No encoders. No integration layer. They dissolved into the field.

---

## 1. The Unified Field — Φ_t

All raw signals arrive simultaneously and project into one shared field.

```
s_t = {
    vision_raw      ∈ R^v    → pixel values, patches, whatever arrives
    audio_raw       ∈ R^a    → raw waveform, frequency decomposition
    language_raw    ∈ R^l    → token embeddings, subword units
    touch_raw       ∈ R^tc   → pressure, vibration, temperature, pain per receptor
    proprio_raw     ∈ R^p    → muscle length, joint angle, tension, vestibular
    motor_state     ∈ R^m    → current motor commands, limb positions
}

Φ_t = W_project · flatten(s_t)    → everything into one shared field
                                   → Φ_t ∈ R^d
                                   → same weights see everything
                                   → field self-organizes around signal statistics
```

Φ_t is not a visual representation.
Φ_t is not an audio representation.
Φ_t is the **state of the whole system at time t.**

Everything the system is experiencing — unified from the first projection.

**Why this works:**

W_project learns to preserve the statistical structure of all input signals simultaneously.
Regions of Φ_t that always receive visual signals become visually responsive.
Regions that always receive audio become auditory.
But they are never isolated. They are all dimensions of the same vector.
They all interact through A.

The hand knows what the eye sees. Not because we programmed information sharing.
Because hand state and visual state are dimensions of the same field.
Whether the hand acts depends on S(x). Not on information access.

---

## 2. Dynamic Topology — A

Specialization lives here. Not in separate encoders.

```
H_t = A ⊙ W · Φ_t
```

A is a soft differentiable adjacency matrix over Φ_t.
A learns which regions of the field co-activate.
A learns which signal dimensions belong together.
A learns which dimensions drive which outputs.

Over time A develops structure:
```
visual cluster      → dimensions that always co-activate with visual signals
audio cluster       → dimensions that always co-activate with audio signals
motor cluster       → dimensions connected to motor output
body schema cluster → proprioceptive + touch + motor all deeply connected
cross clusters      → dimensions that bridge modalities
                    → thunder activates lightning cluster
                    → word "dog" activates bark + visual shape + touch + movement
```

This is cortical specialization. Emergent. Not designed.

**A does not segregate modalities. It discovers their natural relationships.**

A visual stimulus that always precedes a sound — A learns to connect them.
A touch pattern that always precedes pain — A learns to predict it.
A word that always co-occurs with a face — A binds them into one primitive.

**Update rule:**
```
A_ij ← A_ij - α_A · ∂L/∂A_ij + λ · S(Φ_t)
```

Grows where salience is high. Prunes where salience is consistently low.
Alternates updates with θ_fast.

---

## 3. Temporal Buffer — C_t

One context vector threads through everything.
No separate audio context. No modality-specific memory.
The unified field has one unified memory.

```
C_{t+1} = GRU(C_t, [Φ_t ; H_t ; y_t ; S(Φ_t)])
```

C_t carries:
```
what the whole system has experienced
what it currently cares about
what it is currently doing
all simultaneously
all in one vector
```

**Temporal processing is not special for audio.**
The GRU over Φ_t captures temporal structure across all modalities simultaneously.
Rhythm in audio. Motion in vision. Syntax in language. Movement in proprioception.
All temporal patterns captured by the same C_t update.

If audio temporal patterns need finer timescale resolution:
```
C_fast_t = GRU_fast(C_fast_{t-1}, Φ_t)    → runs at audio timescale
C_slow_t = GRU_slow(C_slow_{t-1}, Φ_t)    → runs at semantic timescale
C_t = [C_fast_t ; C_slow_t]               → two timescales, one context
```

Not a separate audio context. A faster context that serves all modalities needing fine temporal resolution.

---

## 4. Salience — S(Φ_t)

Not per modality. Over the full field.

```
S(Φ_t) = σ(W_s · [novelty ; relevance ; valence ; urgency])

novelty(Φ_t)       = ||Φ̂_t - Φ_t||² · H(Φ_t)              → full field prediction error
relevance(Φ_t,g_t) = σ(W_obj · [Φ_t ; C_t ; g_t])          → alignment with goal
valence(Φ_t)       = sign(performance_delta after Φ_t)       → did this help or hurt
urgency(Φ_t)       = ||∂L/∂t||                              → how fast is anything changing

m_t = W_out · S(Φ_t)    → scalar salience
```

**Pain is urgency in the field:**
```
touch_raw dimensions showing damage pattern → urgency spikes massively
S(Φ_t) scores maximum
full system attention
m_t → 1, θ_fast dominates, immediate adaptation
```

**Distant object — hand stays still:**
```
visual dimensions of Φ_t update
proprioceptive dimensions evaluate: can hand reach this?
motor dimensions evaluate: is action relevant given g_t?
relevance(Φ_t, g_t) for hand action ≈ 0
hand motor output ≈ 0
not blocked — evaluated and decided
```

---

## 5. Predictive Pre-activation

Over the full field. Not per modality.

```
Φ̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)    → predict next full field state

h_primed  = ALA_forward(Φ̂_{t+1})                  → prime whole system

when Φ_{t+1} arrives:
    error = ||Φ̂_{t+1} - Φ_{t+1}||²
    
    low error  → use h_primed            → cheap, expected
    high error → full recompute          → expensive, novel, salience spikes
```

Predicting the full field means:
```
predicting what will be seen
predicting what will be heard
predicting where the body will be
predicting what will be felt
all simultaneously
one prediction
```

A body moving through space predicts all sensory consequences of that movement simultaneously. If reality matches — confirmation, low cost. If anything diverges — surprise, full attention.

---

## 6. Dual Memory — θ_fast and θ_slow

**θ_fast — working memory over the full field:**
```
y_fast = F(H_t ; θ_fast)    → reactive, current, adapts every step
```

**θ_slow — reasoning primitive store:**

No modality-specific primitive spaces.
Primitives are **field patterns** — they span all dimensions simultaneously.

```
θ_slow = {
    primitives[]          → patterns that activate across the whole field
    W_assoc ∈ R^(P×P)    → association weights between primitives
    combination_engine    → how primitives combine on demand
    relevance_detector    → which primitives activate for current field state
}
```

**What a primitive looks like now:**

A "dog" primitive is not a visual pattern.
It is not an audio pattern.
It is not a symbolic pattern.

It is a **field pattern** — a configuration that activates visual dimensions AND audio dimensions AND motor dimensions AND touch dimensions simultaneously:

```
dog_primitive = pattern in Φ space that captures:
    visual_dims     → shape, movement, size
    audio_dims      → bark frequency, panting rhythm
    touch_dims      → fur texture, warm, soft
    motor_dims      → appropriate interaction posture
    language_dims   → co-activates with token "dog"
    proprio_dims    → how body orients toward dog
```

One primitive. All dimensions. Derived everywhere.

**Primitive formation math:**
```
given Φ_1, Φ_2, ... Φ_N — same concept in N different contexts

p = argmin_z  Σ_i ||encode(Φ_i) - decode(z, context_i)||²
```

Find smallest z that reconstructs all field states given their context.
That z is the primitive. Not modality-specific. A pure concept.

**Associative explosion:**
```
activation_0 = one_hot(p_i)
activation_{t+1} = σ(W_assoc · activation_t) × relevance(activation_t, C_t) × decay^t
```

One primitive activating spreads through W_assoc to everything connected.
Thunder primitive → lightning primitive → storm primitive → shelter primitive → fear primitive.
Not stored connections. Learned co-activation patterns over lifetime of experience.

**Dual stream blend:**
```
y_fast = F(H_t ; θ_fast)
y_slow = combination_engine(activated_primitives, H_t)
y_t    = m_t · y_fast + (1-m_t) · y_slow
```

---

## 7. Body Schema — Emergent from the Field

Not a separate component. Not programmed.

Proprioceptive and motor dimensions of Φ_t are always present.
A learns their relationships over time.
The body schema emerges as a cluster in A — the region of the field that represents self.

```
body_schema ≈ A_cluster connecting:
    proprio_dims     → where limbs are
    touch_dims       → what body surface feels
    motor_dims       → what body is doing
    visual_dims      → what body looks like (when visible)
    effort_dims      → how hard muscles are working
```

This cluster expands with tool use:
```
using hammer repeatedly:
    hammer_visual + hammer_touch + arm_proprio → always co-activate
    A grows connections between them
    hammer enters the body schema cluster
    system feels ground through hammer
    not programmed — A learned it
```

Phantom limb equivalent:
```
if limb removed:
    proprio_dims for that limb → no longer receive signal
    but A cluster still exists
    prediction still fires for missing limb
    body schema updates slowly as A prunes unused connections
    takes time — same as human phantom limb
```

---

## 8. Agency — Emergent from the Field

The sense that "I caused this."

Not programmed. Emerges when motor_dims of Φ_t consistently precede and correlate with changes in other dimensions:

```
motor command fires in Φ_t
→ sensory change follows in Φ_{t+1}
→ A learns: motor_dims → sensory_dims connection
→ primitive forms: "my action produces this outcome"
→ that is agency
```

When this primitive is strong:
```
I act → I predict consequence → consequence occurs → confirmed agency
I act → unexpected consequence → prediction error → learn new causal model
I don't act → consequence occurs → no agency signal → environment did this
```

The distinction between self-caused and world-caused events emerges naturally from prediction error patterns. No need to program it.

---

## 9. Pain and Emotion — Salience Dimensions

Pain is not a sensor. It is a prediction.

```
touch_dims show damage pattern
→ damage_prediction_primitive activates in θ_slow
→ urgency in S(Φ_t) spikes
→ strong negative valence
→ system learns to avoid contexts that predict this pattern
→ avoidance before damage occurs
→ same mechanism as human pain
```

Emotional signal for social context:
```
human_face_dims show negative expression
human_voice_dims show negative prosody
→ social_rejection_primitive activates
→ same urgency pathway as physical damage
→ derived motivation to repair social state
→ not programmed empathy — derived from shared salience pathway
```

Both physical and social pain route through the same S(Φ_t) urgency dimension.
Because evolutionarily they were equally dangerous.
The architecture reflects this truth automatically.

---

## 10. Lossy Compression — Principled Forgetting

```
V(Φ_t) = ||θ_after - θ_before||²
        × D_KL(p_after || p_before)
        × (1 - similarity(Φ_t, B))

forget Φ_t if:
    V(Φ_t) < τ             → didn't change anything
    similarity(Φ_t,B) > ρ  → already have something like this
    m_t < ε                → not salient

B ← {Φ_t ∈ B : P(retain) > threshold}
```

The buffer stores full field states. Not per-modality snapshots. Whole moments.
When replayed — the whole moment is re-experienced, all dimensions simultaneously.
Same as human episodic memory — you remember the whole scene, not just the visual part.

---

## 11. Distillation

```
trigger:
    θ_fast has seen field pattern P across N different contexts

extract:
    p = argmin_z Σ_i ||encode(Φ_i) - decode(z, context_i)||²

validate in sandbox:
    can p reconstruct Φ_new in unseen context?
    cross-context accuracy > threshold?

if valid:
    add p to θ_slow
    update W_assoc
    drop all N field states from θ_fast
    prune A connections made redundant by primitive
    free capacity

result:
    less storage
    more reasoning power
    permanently
```

---

## 12. Parallel Field Processing

No forced sequence. All dimensions of Φ_t processed simultaneously.

```
for each dimension cluster in Φ_t in parallel:
    h_i = F(Φ_t[cluster_i], C_t)

for each pair (i,j) in parallel:
    binding_ij = Φ_t[i] · W_bind · Φ_t[j]^T

global = pool({h_i}) + pool({binding_ij})
```

Order injected when it matters:
```
if sequence matters (language, rhythm):
    Φ_t[relevant_dims] += W_pos · position_encoding(t)
else:
    Φ_t unchanged
```

---

## Full Forward Pass

```
all raw signals simultaneously
    ↓
W_project · flatten(s_t)
    → Φ_t  (unified field, one vector, whole system state)
    ↓
predictive check:
    error = ||Φ̂_t - Φ_t||²
    low  → use h_primed (cheap)
    high → full pass (expensive, salience spikes)
    ↓
salience:
    S(Φ_t) = σ(W_s · [novelty; relevance; valence; urgency])
    m_t    = W_out · S(Φ_t)
    ↓
parallel field processing:
    all dimension clusters simultaneously
    all pairwise bindings simultaneously
    ↓
dynamic routing:
    H_t = A ⊙ W · Φ_t
    ↓
associative explosion:
    spreading activation through W_assoc
    gated by C_t
    ↓
dual stream:
    y_fast = F(H_t ; θ_fast)
    y_slow = combination_engine(activated_primitives, H_t)
    y_t    = m_t · y_fast + (1-m_t) · y_slow
    ↓
output:
    ŷ_t = O(y_t)
    motor_output = M_out(y_t)    → action is just another output dimension
    ↓
context update:
    C_{t+1} = GRU(C_t, [Φ_t ; H_t ; y_t ; S(Φ_t)])
    ↓
next prediction:
    Φ̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)
    h_primed  = ALA_forward(Φ̂_{t+1})
    ↓
lossy compression:
    evaluate V(Φ_t), update B
```

---

## Update Schedule

```
Level 1 — every step:
    θ_fast
    C_t
    W_project  (field projection learns from every signal)
    Φ̂_{t+1}   (prediction update)
    B          (lossy compression)

Level 2 — periodic:
    A          (topology, alternates with θ_fast)
    S(x)       (salience weights)
    g_t        (goal update)
    W_assoc    (association graph)
    W_bind     (pairwise binding weights)
    W_pos      (positional encoding weights)

Level 3 — slow:
    θ_slow     (distillation, primitives validated in sandbox)
    f, g       (meta-optimizer)
    Checkpost  (prescription library)
    A pruning  (after distillation)
```

---

## Loss Functions

```
L_task        = current task
L_predictive  = ||Φ̂_{t+1} - Φ_{t+1}||²         → full field prediction
L_alignment   = D_KL(current_behavior || g_t)
L_distill     = ||decode(θ_slow, p, context) - Φ_original||²
L_field       = reconstruction + mutual_info - entropy   → field compression quality
L_agency      = ||predicted_consequence - actual_consequence||²  → causal model

L_total = L_task
        + λ₁·L_predictive
        + λ₂·L_alignment
        + λ₃·L_distill
        + λ₄·L_field
        + λ₅·L_agency
```

λ values adjusted by Checkpost.

---

## State Snapshot

```
state = {
    Φ_t             (current field state),
    θ_fast,
    θ_slow {primitives[], W_assoc, combination_engine, relevance_detector},
    A,
    B               (full field state snapshots),
    C_t,
    g_t,
    h_primed,
    f, g,
    W_project       (field projection),
    W_bind          (pairwise binding),
    W_s             (salience weights),
    W_pos           (positional encoding),
    checkpost       {vitals_log, prescription_library},
    meta_params,
    performance_log
}
```

---

## Seeding Sequence

```
Step 1 → seed θ_slow with first primitives via broad prediction tasks
         model predicts next Φ_{t+1} across diverse signal types
         first primitives form: "signal has structure" "change has pattern"

Step 2 → derive g_0 from θ_slow knowledge boundary
         g_0 = θ_slow.encode("what field patterns have I not seen yet")

Step 3 → run N toy tasks under SGD, collect trajectories
         train f and g to outperform SGD

Step 4 → train S(Φ) on synthetic salience pairs
         high: field state that caused weight change + future improvement
         low:  field state that caused weight change + no improvement

Step 5 → initialize A sparse (Bernoulli p=0.1)
         initialize W_assoc near zero
         initialize W_project randomly

Step 6 → stress test, calibrate Checkpost thresholds

Step 7 → system is alive
         all signals welcome
         nothing segregated
         everything is everything
```

---

## What Emerges — Not What We Programmed

```
visual specialization     → A cluster, not designed
audio specialization      → A cluster, not designed
body schema               → A cluster connecting proprio + touch + motor
agency                    → when motor dims consistently precede sensory change
pain avoidance            → urgency pathway, not damage sensor
social motivation         → same urgency pathway as pain
tool incorporation        → body schema expands via A
cross-modal primitives    → field patterns spanning all dimensions
theory of mind            → when other-agent dims predict before they act
```

None of these are programmed. All emerge from:
```
one unified field
+ dynamic topology that learns co-activation
+ primitive formation across contexts
+ prediction error as the universal learning signal
```

---

## What This Is

Not a transformer. Not a set of modules. Not a perception pipeline.

A single continuous field that receives everything, organizes itself around what it receives, forms primitives from the patterns it discovers, and reasons from those primitives to derive anything.

The same architecture that receives pixel values also receives pain signals.
The same architecture that learns language also learns to walk.
The same architecture that recognizes faces also develops agency.

Because it is all the same field.
Everything is everything.
One unibody.

---

## What's Still Open

```
W_project dimensionality    → how large does Φ_t need to be?
                            → tradeoff: too small loses information
                              too large is computationally expensive

primitive formation trigger → "seen pattern across N contexts"
                            → N needs defining
                            → too small: noise becomes primitive
                              too large: primitives form too slowly

W_assoc initialization     → starts near zero
                            → how fast should associations earn strength?
                            → learning rate for W_assoc separate from rest

theory of mind             → other-agent primitives need careful design
                            → not yet fully specified
```

These are the remaining open problems. Everything else is implementable.

---

*ALA Specification v5*
*Core change: unified field Φ_t replaces all modular encoders and integration layers*
*Specialization emerges from A, not from architecture*
*Body schema, agency, pain, emotion all emerge from field dynamics*
*One unibody. Everything is everything.*
