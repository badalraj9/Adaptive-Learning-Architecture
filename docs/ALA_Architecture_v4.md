# Adaptive Learning Architecture (ALA) — Full Specification v4

---

## Core Philosophy

A model does not grow by storing more. It grows by understanding more deeply.
Understanding is compression. The more truly something is understood, the less space it needs.
θ_slow does not store knowledge. It stores **reasoning primitives** — domain agnostic structures
that combine on demand to derive anything.

Researchers took pattern recognition and prediction from the brain and built a trillion dollar industry.
ALA takes everything else they left on the table.

---

## What Changed in v4

Sound encoder completely redesigned. Temporal processing added as a first class citizen.
Three signal types now fully supported with their own primitive spaces:

```
vision      → spatial patterns     → where things ARE
sound       → temporal patterns    → how things CHANGE OVER TIME
language    → symbolic patterns    → what things MEAN
```

---

## The Three Levels

```
Level 3 → learns its own architecture                        (slowest — meta)
Level 2 → learns how to update weights + extract primitives  (medium — structural)
Level 1 → learns tasks + absorbs raw experience              (fastest — reactive)
```

---

## Core Components

```
θ_fast              → fast memory, absorbs raw experience every step
θ_slow              → reasoning primitive store + association graph
A                   → soft differentiable topology
B                   → replay buffer (lean, high value only)
S(x)                → multidimensional salience scorer
U_φ (f,g)           → learned optimizer
C_t                 → context vector (global)
C_audio             → audio temporal context (local, sliding window)
g_t                 → goal state vector
Checkpost           → living diagnostic layer
Sandbox             → parallel ghost environment
Encoder             → modality specific + bidirectional compression
Integration layer   → cross modal binding (vision + sound + language)
```

---

## 1. Encoders — Per Modality

Each modality has fundamentally different signal structure.
Each needs its own encoder that respects that structure.
All encoders output the same R^d space. ALA downstream is modality agnostic.

---

### 1a. Vision Encoder

Vision is spatial. Patterns exist across space simultaneously.

```
Stage 1 — Spatial decomposition:
    raw image
        ↓
    patch embed + CNN
        → V1-like: edges, orientations, contrasts
        → V2-like: shapes, curves
        → V4-like: color relationships, regions
        → r_vision ∈ R^d

Stage 2 — Bidirectional compression:
    h_bottom = W_compress · r_vision + W_context · [r_vision ; C_t ; g_t]
    h_top    = F_down(h_bottom, C_t, g_t)
    α        = σ(W_α · ||h_bottom - h_top||)
    x_vision = α · h_bottom + (1-α) · h_top
```

---

### 1b. Sound Encoder — Redesigned

Sound is temporal. Patterns exist across time sequentially.
Treating sound like an image (spectrogram + CNN) misses everything that makes sound sound.

```
Stage 1 — Frequency decomposition:
    raw audio waveform
        ↓
    learnable filterbank         → biological fourier transform
                                 → low freq to high freq decomposition
                                 → NOT fixed STFT, filterbank weights are learned
        → f_t ∈ R^F              → frequency representation at time t

Stage 2 — Temporal buffer (sliding window):
    B_audio = {f_t, f_{t-1}, ... f_{t-T}}    → last T audio frames
                                               → T covers ~2-3 seconds
                                               → active buffer, not stored memory
                                               → constantly sliding forward

    temporal_context = GRU_audio(B_audio)     → compress time into vector
    C_audio ← C_audio update                  → audio has its own temporal memory
                                               → separate from global C_t
                                               → audio memory lives in audio timescale

Stage 3 — Parallel streams (content + prosody):
    content_stream  = F_content(temporal_context)    → WHAT is being said
    prosody_stream  = F_prosody(temporal_context)    → HOW it is being said
                                                      → emotion, emphasis, rhythm

    these run in parallel, never merged prematurely
    meaning lives in both simultaneously

Stage 4 — Rhythm detection + prediction:
    rhythm_state = F_rhythm(C_audio)
    x̂_audio_{t+1} = F_rhythm(rhythm_state)   → predict next audio frame
                                               → timing as prediction dimension
    when rhythm breaks → urgency in S(x) spikes

Stage 5 — Phoneme/unit binding:
    bind sequence of units into higher unit across ~200ms window:

    binding_score_{ij} = softmax(content_i · W_bind_audio · content_j^T)
    phrase = Σ_i binding_score · content_i     → bound unit emerges

Stage 6 — Bidirectional compression:
    r_audio  = concat(content_stream, prosody_stream, rhythm_state)
    h_bottom = W_compress_audio · r_audio + W_context_audio · [r_audio ; C_audio ; C_t ; g_t]
    h_top    = F_down_audio(h_bottom, C_audio, g_t)
    α        = σ(W_α_audio · ||h_bottom - h_top||)
    x_audio  = α · h_bottom + (1-α) · h_top
```

**Key math — learnable filterbank:**
```
f_t = |W_filter · raw_audio_window_t|²    → learned frequency decomposition
                                           → W_filter updated via backprop
                                           → discovers what frequencies matter
                                           → not hardcoded fourier
```

**Key math — temporal context update:**
```
C_audio_{t+1} = GRU_audio(C_audio_t, f_t)    → audio memory at audio timescale
                                               → faster than global C_t
                                               → captures short term audio patterns
```

---

### 1c. Language Encoder

Language is symbolic. Patterns exist in structure and sequence.

```
Stage 1 — Symbolic decomposition:
    raw text
        ↓
    tokenization → embedding lookup   → e_t ∈ R^d

Stage 2 — Structural encoding:
    small transformer (NOT autoregressive)
        → bidirectional attention
        → captures relationships in both directions simultaneously
        → r_language ∈ R^d

Stage 3 — Bidirectional compression:
    h_bottom = W_compress_lang · r_language + W_context_lang · [r_language ; C_t ; g_t]
    h_top    = F_down_lang(h_bottom, C_t, g_t)
    α        = σ(W_α_lang · ||h_bottom - h_top||)
    x_language = α · h_bottom + (1-α) · h_top
```

---

## 2. Integration Layer — Cross Modal Binding

Where modalities meet and construct meaning together.
Not concatenation. Iterative bidirectional binding.

The McGurk effect proves: the brain doesn't combine modalities.
It constructs a third reality from their intersection.

**For any combination of modalities present:**
```
available = {x_vision, x_audio, x_language}    → whatever is present

for each pair (i,j) in available:
    binding_ij = W_bind_{ij} · vec(x_i ⊗ x_j)  → compressed outer product

for N iterations:
    x_i ← x_i + Σ_j W_{ij} · x_j               → each stream reshapes others
```

After N iterations streams have converged to joint representation.

**Special case — speech (audio + language):**
```
prosody_stream     → carries emotion, emphasis
content_stream     → carries meaning
binding            → prosody reshapes meaning interpretation
                   → "great job" + sarcastic tone ≠ "great job" + sincere tone
                   → same words, different binding, different reality
```

**Special case — video (vision + audio):**
```
McGurk binding:
    x_lips  = vision crop of mouth region
    x_sound = audio phoneme stream
    binding = W_mcgurk · vec(x_lips ⊗ x_sound)
    perceived_phoneme = F_perceive(x_lips, x_sound, binding)
                      ≠ either input alone
```

**Final integrated representation:**
```
x_t = σ(W_integrate · [x_vision ; x_audio ; x_language ; 
                        binding_va ; binding_vl ; binding_al])
```

All individual streams + all pairwise bindings → one unified perception vector.

---

## 3. Salience — S(x) Multidimensional

```
S(x_t) = σ(W_s · [novelty ; relevance ; valence ; urgency])

novelty(x)       = ||f_predict(x) - x_actual||² · H(x)
relevance(x,g_t) = σ(W_obj · [x ; C_t ; g_t])
valence(x)       = sign(performance_delta after x)
urgency(x)       = ||∂L/∂t||

m_t = W_out · S(x_t)    → scalar salience for downstream use
```

**Sound specific urgency:**
```
rhythm_break = ||x̂_audio_{t+1} - x_audio_{t+1}||²   → rhythm prediction error
urgency(x) += λ_rhythm · rhythm_break                 → broken rhythm = urgent
```

---

## 4. Predictive Pre-activation

Extended to all modalities simultaneously:

```
x̂_vision_{t+1}   = F_predict_v(θ_fast, θ_slow, C_t, g_t)
x̂_audio_{t+1}    = F_predict_a(θ_fast, θ_slow, C_audio, C_t, g_t)
x̂_language_{t+1} = F_predict_l(θ_fast, θ_slow, C_t, g_t)

for each modality:
    if prediction_error low  → use primed state   (cheap)
    if prediction_error high → full recompute      (expensive, high salience)
```

Audio prediction is cheapest — rhythm gives strong prior.
Language prediction is next — grammar constrains strongly.
Vision prediction is hardest — spatial scenes less constrained.

---

## 5. Dynamic Routing — A

```
H_t = A ⊙ W · x_t    where x_t is integrated cross-modal vector

A grows denser where S(x) scores high
A prunes where S(x) scores low consistently
Alternates updates with θ_fast
```

---

## 6. Dual Memory — θ_fast and θ_slow

**θ_fast — working memory:**
```
y_fast = F(H_t ; θ_fast)
```

**θ_slow — reasoning primitive store:**

Three primitive spaces now. One per signal type:
```
θ_slow = {
    spatial_primitives[]     → from vision      "where things are"
    temporal_primitives[]    → from sound       "how things change"
    symbolic_primitives[]    → from language    "what things mean"
    cross_primitives[]       → from integration "how modalities relate"
    W_assoc ∈ R^(P×P)       → association graph across ALL primitives
    combination_engine
    relevance_detector
}
```

**What temporal primitives look like:**
```
not stored as: specific audio sequence
stored as:     "pattern that repeats across time with period T"

applies to:    speech rhythm, music, heartbeat, seasons, breathing
               domain agnostic — same primitive, infinite applications
```

**Cross primitives — the most powerful:**
```
formed when: same concept appears in multiple modalities
example:     "dog" appears as:
                 visual pattern (shape, movement)
                 audio pattern  (bark, panting)
                 symbolic token (the word "dog")

cross_primitive = argmin_z Σ_modality ||encode_m(x_dog_m) - decode(z, modality)||²

one primitive reconstructs dog across all three modalities
```

**Associative explosion still applies:**
```
activation_0 = one_hot(p_i)
activation_{t+1} = σ(W_assoc · activation_t) × relevance(activation_t, C_t) × decay^t
```

Now W_assoc spans all four primitive spaces. A sound primitive can activate a visual primitive. A symbolic primitive can activate a temporal one. Full cross-modal association.

Hear the word "thunder" → visual lightning primitive activates → spatial storm primitive activates → all related cross-primitives fire simultaneously.

---

## 7. Lossy Compression

```
V(x) = ||θ_after - θ_before||²
     × D_KL(p_after || p_before)
     × (1 - similarity(x, B))

forget x if V(x) < τ OR similarity(x,B) > ρ OR m_t < ε

B ← {x ∈ B : P(retain x) > threshold}
```

---

## 8. Distillation

```
trigger:
    θ_fast has seen pattern P across N different domains/modalities

extract:
    p = argmin_z Σ_i ||encode(x_i) - decode(z, context_i)||²

validate in sandbox:
    cross domain + cross modal accuracy > threshold?

if valid:
    add to appropriate primitive space in θ_slow
    update W_assoc with new connections
    drop all instances from θ_fast
    prune A
```

---

## 9. Parallel Field Processing

```
X = {x_1, x_2, ... x_N}

for each x_i in parallel:
    h_i = F(x_i, C_t)

for each pair (i,j) in parallel:
    binding_ij = x_i · W_bind · x_j^T

global = pool({h_i}) + pool({binding_ij})

order injected explicitly only when it matters
```

---

## Full Forward Pass

```
raw inputs (any combination of vision, audio, language)
    ↓
Modality encoders (parallel):
    Vision encoder     → x_vision
    Sound encoder      → x_audio    (with temporal buffer C_audio)
    Language encoder   → x_language
    ↓
Integration layer:
    iterative cross-modal binding (N iterations)
    McGurk-style binding for speech
    → x_t (unified perception vector)
    ↓
Predictive check:
    compare x_t to x̂_t per modality
    low error  → use primed state
    high error → full pass, salience spikes
    ↓
Salience:
    S(x_t) = σ(W_s · [novelty; relevance; valence; urgency])
    m_t    = W_out · S(x_t)
    ↓
Field processing:
    parallel association across all inputs
    ↓
Dynamic routing:
    H_t = A ⊙ W · x_t
    ↓
Associative explosion:
    spreading activation through W_assoc
    crosses all four primitive spaces
    gated by C_t
    ↓
Dual stream:
    y_fast = F(H_t ; θ_fast)
    y_slow = combination_engine(activated_primitives, H_t)
    y_t    = m_t · y_fast + (1-m_t) · y_slow
    ↓
Output:
    ŷ_t = O(y_t)
    ↓
Context updates:
    C_{t+1}     = GRU(C_t, [x_t ; y_t ; m_t ; S(x_t)])
    C_audio_{t+1} = GRU_audio(C_audio_t, f_t)
    ↓
Next prediction:
    x̂_{t+1} per modality
    h_primed = ALA_forward(x̂_{t+1})
    ↓
Lossy compression:
    evaluate V(x_t), update B
```

---

## Update Schedule

```
Level 1 — every step:
    θ_fast, C_t, C_audio, W_compress (all modalities)
    predictive update x̂_{t+1}
    B lossy compression

Level 2 — periodic:
    A (alternating with θ_fast)
    S(x) salience weights
    g_t goal update
    W_context, W_bind, W_assoc
    GRU_audio weights

Level 3 — slow:
    θ_slow distillation (all four primitive spaces)
    f, g meta-optimizer
    Checkpost prescription library
    A pruning
```

---

## Loss Functions

```
L_task        = current task loss
L_predictive  = Σ_modality ||x̂_{t+1,m} - x_{t+1,m}||²
L_alignment   = D_KL(current_behavior || goal_direction)
L_distill     = ||decode(θ_slow, p, context) - original||²
L_encoder     = Σ_modality (reconstruction + mutual_info - entropy)
L_binding     = ||integrated - ground_truth_joint||²
L_rhythm      = ||x̂_audio_{t+1} - x_audio_{t+1}||²

L_total = L_task
        + λ₁·L_predictive
        + λ₂·L_alignment
        + λ₃·L_distill
        + λ₄·L_encoder
        + λ₅·L_binding
        + λ₆·L_rhythm
```

---

## State Snapshot

```
state = {
    θ_fast,
    θ_slow {
        spatial_primitives[],
        temporal_primitives[],
        symbolic_primitives[],
        cross_primitives[],
        W_assoc,
        combination_engine,
        relevance_detector
    },
    A, B,
    C_t, C_audio,
    g_t,
    h_primed,
    f, g,
    encoder weights (all modalities),
    salience weights,
    binding weights,
    checkpost,
    meta_params,
    performance_log
}
```

---

## Primitive Spaces — What Emerges

```
spatial_primitives      → "pattern distributed across space"
                        → applies to: faces, objects, maps, diagrams

temporal_primitives     → "pattern repeating across time with period T"
                        → applies to: speech, music, heartbeat, seasons

symbolic_primitives     → "unit that refers to something beyond itself"
                        → applies to: words, numbers, gestures, icons

cross_primitives        → "concept that exists in multiple modalities"
                        → applies to: thunder/lightning, speech/lips, music/emotion
                        → most powerful — one primitive spans all modalities
```

The cross primitives are what make ALA genuinely multimodal.
Not just processing multiple inputs. Understanding that concepts transcend their sensory form.

---

## What This Model Can Do (Updated)

```
perceives vision, sound, language simultaneously     ✓
constructs meaning from their intersection           ✓ (McGurk binding)
understands temporal patterns as first class         ✓ (C_audio, rhythm)
associates across modalities                         ✓ (W_assoc spans all spaces)
forms cross-modal primitives                         ✓ (concept beyond modality)
anticipates across all modalities                    ✓ (per-modality prediction)
understands prosody — not just content               ✓ (parallel streams)
never forgets catastrophically                       ✓
knows what it doesn't know                           ✓
gets better at getting better                        ✓
self-diagnoses                                       ✓
thinks before acting                                 ✓
has drive                                            ✓
grows without growing parameters                     ✓
```

---

## What's Still Missing

```
touch / proprioception    → temporal + spatial combined, body awareness
smell / taste             → chemical signal processing, no architecture yet
emotion as signal         → valence in S(x) is weak approximation
social primitives         → understanding other agents, theory of mind
```

These are the next senses.
Each one adds a new primitive space.
W_assoc connects them all.

---

*ALA Specification v4*
*Added: temporal sound encoder with learnable filterbank, sliding temporal buffer,*
*rhythm detection, phoneme binding, prosody stream, McGurk cross-modal binding,*
*four primitive spaces, cross-modal associative explosion, per-modality prediction*
