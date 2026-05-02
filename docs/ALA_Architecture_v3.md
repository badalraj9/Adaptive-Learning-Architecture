# Adaptive Learning Architecture (ALA) — Full Specification v3

---

## Core Philosophy

A model does not grow by storing more. It grows by understanding more deeply.
Understanding is compression. The more truly something is understood, the less space it needs.
θ_slow does not store knowledge. It stores **reasoning primitives** — domain agnostic structures
that combine on demand to derive anything.

Researchers took pattern recognition and prediction from the brain and built a trillion dollar industry.
ALA takes everything else they left on the table.

---

## What Changed in v3

Eight brain mechanisms integrated into core architecture:

```
bidirectional streams       → α-weighted top-down bottom-up blend
integration layer           → iterative cross-modal binding
lossy compression           → principled retention via information value
primitive formation         → argmin reconstruction across domains
associative explosion       → spreading activation through primitive graph
salience weighting          → multidimensional S(x) replacing scalar loss
predictive pre-activation   → primed forward pass before input arrives
parallel association        → field processing, order as optional feature
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
M                   → multidimensional salience scorer
U_φ (f,g)           → learned optimizer
C_t                 → context vector
g_t                 → goal state vector
Checkpost           → living diagnostic layer
Sandbox             → parallel ghost environment
Encoder             → context-aware bidirectional compression
```

---

## 1. Encoder — Now Bidirectional

Two stage. Top-down and bottom-up simultaneously.

**Stage 1 — Raw Encoding (modality specific)**
```
text    → embedding + small transformer  → r_t ∈ R^d
image   → patch embed + CNN             → r_t ∈ R^d
audio   → spectrogram + CNN             → r_t ∈ R^d
tabular → linear projection             → r_t ∈ R^d
VM state→ structured feature encoder   → r_t ∈ R^d
```

**Stage 2 — Bidirectional Compression**

Not just bottom-up signal driving representation.
Top-down expectations from C_t and g_t reshape how input is perceived simultaneously:

```
h_bottom = W_compress · r_t + W_context · [r_t ; C_t ; g_t]   → bottom up
h_top    = F_down(h_bottom, C_t, g_t)                          → top down expectations

α = σ(W_α · ||h_bottom - h_top||)     → disagreement between streams

x_t = α · h_bottom + (1-α) · h_top   → blend weighted by confidence
```

High disagreement → trust raw signal, something unexpected.
Low disagreement  → let expectations guide, confirming what was predicted.

The encoder sees and reshapes simultaneously. Same input tomorrow ≠ same x_t if context has changed.

**Encoder loss:**
```
L_encoder = ||x_t - x̂_t||²           → faithful reconstruction
           + λ · I(x_t ; C_t)         → stay relevant to context
           - μ · H(x_t)               → compress aggressively
```

---

## 2. Integration Layer — Cross Modal Binding

For multimodal input, meaning lives in the relationship not the concatenation.

**Old approach (wrong):**
```
multimodal = concat(text_embed, image_embed)    → just gluing vectors
```

**New approach:**
```
text_stream  = T(x_text)    ∈ R^d
image_stream = V(x_image)   ∈ R^d

binding      = W_bind · vec(T ⊗ V)             → compressed outer product
                                                → captures all pairwise relationships

output = σ(W_out · [text_stream ; image_stream ; binding])
```

Bidirectional — each stream reshapes the other iteratively:
```
for N iterations:
    text_stream  ← text_stream  + W_tv · image_stream
    image_stream ← image_stream + W_vt · text_stream
```

They converge to a joint representation neither could reach alone.
The picture of a dog + word "dangerous" ≠ same dog + word "friendly."
Meaning is in the binding, not the inputs.

---

## 3. Salience — M Upgraded to Multidimensional

Old M was a scalar. Importance reduced to loss gradient magnitude.
New M is a full salience vector — multidimensional judgment.

```
S(x_t) = σ(W_s · [novelty ; relevance ; valence ; urgency])

where:
    novelty(x)       = r_intrinsic = ||f_predict(x) - x_actual||² · H(x)
    relevance(x,g_t) = M_obj(x, C_t, g_t) = σ(W_obj · [x ; C_t ; g_t])
    valence(x)       = sign(performance_delta after x)      → did x help or hurt?
    urgency(x)       = ||∂L/∂t||                           → how fast is loss changing?
```

Final importance scalar:
```
m_t = W_out · S(x_t)
```

Not just "high loss = important."
A rich multidimensional judgment: is this new, aligned, helpful, and urgent?
All four must align for true high salience.

**M_curr and M_obj are now two dimensions within S(x), not separate heads.**
Cleaner. More unified. Same expressive power.

---

## 4. Predictive Pre-activation

The model stops being purely reactive. It anticipates.

Before x_{t+1} arrives, at every timestep:
```
x̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)    → predict next input

h_primed  = ALA_forward(x̂_{t+1})                  → prime the network
```

When x_{t+1} actually arrives:
```
prediction_error = ||x̂_{t+1} - x_{t+1}||²

if error low  → use h_primed                       → near free computation
               confirmed expectation, no surprise

if error high → recompute ALA_forward(x_{t+1})    → full recomputation
               novelty spikes, S(x) scores high
               full attention, expensive but necessary
```

The model spends almost no compute on expected inputs.
Saves everything for surprises.

Pre-activation loss feeds into main training:
```
L_predictive = ||x̂_{t+1} - x_{t+1}||²    → always training to anticipate
```

---

## 5. Dynamic Routing — A (Topology)

Unchanged structurally. Now interacts with associative graph (see θ_slow).

```
H_t = A ⊙ W · x_t
```

A grows denser where S(x) scores high. Prunes where S(x) scores low consistently.
Alternates updates with θ_fast to prevent simultaneous divergence.

---

## 6. Dual Memory — θ_fast and θ_slow

**θ_fast — working memory:**
```
y_fast = F(H_t ; θ_fast)    → reactive, current, adapts every step
```

**θ_slow — reasoning primitive store + association graph:**

Not a weight matrix. A structured store:
```
θ_slow = {
    primitives[]            → domain agnostic reasoning structures
    W_assoc ∈ R^(P×P)      → association weights between primitives
    combination_engine      → how primitives combine on demand
    relevance_detector      → which primitives activate for current input
}
```

**What a primitive actually is mathematically:**

Not the average of examples. The irreducible core across all instances:
```
given x_1, x_2, ... x_N — same pattern across N different domains

p = argmin_z  Σ_i ||encode(x_i) - decode(z, context_i)||²
```

Find the smallest z that reconstructs all instances given their context.
That z is the primitive. Everything domain-specific stripped away.

Validation before storing:
```
primitive p is valid if:
    reconstruct(p, context_new) ≈ x_new    → generalizes to unseen instances
    ||p|| << ||x_i||                        → actually compressed, not copied
    cross_domain_accuracy > threshold       → works outside training domains
```

**Associative Explosion:**

One recognized primitive fires everything connected:
```
activation_0 = one_hot(p_i)                              → seed

activation_{t+1} = σ(W_assoc · activation_t)             → spread through graph
                 × relevance(activation_t, C_t)           → gated by context
                 × decay^t                               → fades with distance
```

Run until convergence. What lights up = everything connected to this input.
Not just what's in context window. The full associative reach of θ_slow.

Context shapes the spread:
```
same input + different C_t  →  different primitives amplify
                            →  different meaning constructed
```

**Dual stream blend:**
```
y_fast = F(H_t ; θ_fast)
y_slow = reasoning from activated primitives via combination_engine

y_t = m_t · y_fast + (1-m_t) · y_slow
```

High salience → trust fast stream.
Low salience  → reason from slow primitives.

---

## 7. Lossy Compression — Principled Forgetting

Models store everything. ALA throws away almost everything. Keeps only what shifted understanding.

Information value of experience x:
```
V(x) = ||θ_after - θ_before||²       → did this change weights?
     × D_KL(p_after || p_before)     → did this change predictions?
     × (1 - similarity(x, B))        → is this genuinely new?
```

Retention:
```
P(retain x) = σ(V(x) - τ)

forget x if:
    V(x) < τ                          → didn't change anything
    OR similarity(x, B) > ρ           → already have something like this
    OR m_t < ε                        → not salient
```

Buffer stays lean and high value:
```
B ← {x ∈ B : P(retain x) > threshold}
```

Forgetting is not failure. It is intelligence.

---

## 8. Distillation — θ_fast to θ_slow

Not merging. Distilling.

```
trigger:
    θ_fast has encountered pattern P across N different domains
    V(x) high across all N instances

extract:
    p = argmin_z Σ_i ||encode(x_i) - decode(z, context_i)||²

validate in sandbox:
    does p generalize to unseen instances?
    cross domain accuracy > threshold?

if valid:
    add p to θ_slow primitives
    update W_assoc with new primitive's connections
    drop ALL N domain-specific instances from θ_fast
    prune redundant A connections
    free capacity for new experience

result:
    less storage
    more reasoning power
    permanently
```

Distillation loss:
```
L_distill = ||decode(θ_slow, p, context) - original_specific||²
```

---

## 9. Parallel Association — Field Processing

Transformers are sequential pretending to be parallel.
ALA processes all inputs as a field. No master sequence.

```
X = {x_1, x_2, ... x_N}    → all inputs simultaneously

for each x_i in parallel:
    h_i = F(x_i, C_t)                           → local processing

for each pair (i,j) in parallel:
    binding_ij = x_i · W_bind · x_j^T           → all pairwise interactions

global = pool({h_i}) + pool({binding_ij})        → aggregate without sequence
```

Order is a feature, not an assumption:
```
if order matters:
    x_i ← x_i + W_pos · position_encoding(i)   → inject order explicitly

if order doesn't matter:
    x_i ← x_i                                   → pure content
```

The model decides whether sequence matters. Not hardcoded into architecture.

---

## Full Forward Pass

```
raw_input(s)
    ↓
[if multimodal]
Integration layer:  iterative cross-modal binding (N iterations)
    text_stream ← text_stream + W_tv · image_stream
    image_stream← image_stream + W_vt · text_stream
    binding     = W_bind · vec(T ⊗ V)
    ↓
Encoder Stage 1:    modality encode              → r_t ∈ R^d
    ↓
Predictive check:   is x_t close to x̂_t?
    yes → use h_primed (cheap)
    no  → continue full pass (expensive, high salience)
    ↓
Encoder Stage 2:    bidirectional compression
    h_bottom = W_compress · r_t + W_context · [r_t ; C_t ; g_t]
    h_top    = F_down(h_bottom, C_t, g_t)
    α        = σ(W_α · ||h_bottom - h_top||)
    x_t      = α · h_bottom + (1-α) · h_top
    ↓
Salience:           S(x_t) = σ(W_s · [novelty; relevance; valence; urgency])
                    m_t = W_out · S(x_t)
    ↓
Field processing:   parallel association across all inputs
    ↓
Dynamic routing:    H_t = A ⊙ W · x_t
    ↓
Associative explosion:
    activation spreads through W_assoc from relevant primitives
    gated by C_t
    ↓
Dual stream:
    y_fast = F(H_t ; θ_fast)
    y_slow = combination_engine(activated_primitives, H_t)
    y_t    = m_t · y_fast + (1-m_t) · y_slow
    ↓
Output:             ŷ_t = O(y_t)
    ↓
Context update:     C_{t+1} = GRU(C_t, [x_t ; y_t ; m_t ; S(x_t)])
    ↓
Predictive:         x̂_{t+1} = F_predict(θ_fast, θ_slow, C_{t+1}, g_t)
                    h_primed  = ALA_forward(x̂_{t+1})
    ↓
Lossy compression:  evaluate V(x_t), update B
```

---

## Update Schedule

```
Level 1 — every step:
    θ_fast        ← learned optimizer U_t
    C_t           ← GRU update
    W_compress    ← encoder fast weights
    x̂_{t+1}      ← predictive update
    B             ← lossy compression, drop low V(x)

Level 2 — periodic (alternating with Level 1):
    A             ← topology update, alternates with θ_fast
    M / S(x)      ← salience meta gradient
    g_t           ← goal update from curiosity + task detection
    W_context     ← encoder context weights
    W_assoc       ← association graph update
    W_bind        ← cross modal binding weights

Level 3 — slow:
    θ_slow        ← distillation, primitive validated in sandbox
    f, g          ← meta-optimizer update
    Checkpost     ← prescription library update
    A             ← pruning after distillation
```

---

## Loss Functions

```
L_task        = loss on current task
L_predictive  = ||x̂_{t+1} - x_{t+1}||²
L_alignment   = D_KL(current_behavior || goal_direction)
L_distill     = ||decode(θ_slow, p, context) - original||²
L_encoder     = reconstruction + mutual_information - entropy
L_binding     = ||integrated - ground_truth_joint||²

L_total = L_task
        + λ₁·L_predictive
        + λ₂·L_alignment
        + λ₃·L_distill
        + λ₄·L_encoder
        + λ₅·L_binding
```

λ values adjusted by Checkpost based on current system health.

---

## State Snapshot

```
state = {
    θ_fast,
    θ_slow  {primitives[], W_assoc, combination_engine, relevance_detector},
    A,
    B (lean replay buffer),
    C_t,
    g_t,
    h_primed (current pre-activated state),
    f, g,
    encoder weights {W_compress, W_context, W_α},
    salience weights {W_s, W_out},
    binding weights {W_bind, W_tv, W_vt},
    checkpost {vitals_log, prescription_library},
    meta_params,
    performance_log
}
```

---

## Seeding Sequence

```
Step 1 → train θ_slow on broad world model prediction (next state)
         goal: seed first reasoning primitives, not facts

Step 2 → derive g_0 = θ_slow.encode("boundary of current knowledge")
         curiosity as initial drive

Step 3 → run N toy tasks under SGD, collect trajectories
         train f and g to outperform SGD baseline

Step 4 → train S(x) on synthetic salience pairs
         high: large θ change + future improvement
         low:  large θ change + no improvement

Step 5 → initialize A sparse (Bernoulli p=0.1)

Step 6 → initialize W_assoc near zero (no associations yet)
         associations earn strength through co-activation

Step 7 → stress test, record failure signatures
         calibrate Checkpost thresholds

Step 8 → system is alive
```

---

## What This Model Can Do

**By design:**
```
never forgets catastrophically      → distillation only when validated
knows what it doesn't know          → r_task, novelty in S(x)
gets better at getting better       → f and g improve over time
self-diagnoses                      → Checkpost
thinks before acting                → Sandbox + pre-activation
has drive                           → g_t, M_obj
grows without growing parameters    → compression + primitive formation
perceives bidirectionally           → top-down + bottom-up simultaneously
understands relationships           → cross modal binding
associates richly                   → spreading activation through W_assoc
anticipates                         → predictive pre-activation
forgets intelligently               → lossy compression via V(x)
processes without forced sequence   → field processing
```

**Honest ceiling:**
```
semantic grounding from environment only    → learns syntax before meaning
long horizon credit assignment is hard      → f,g improve this over time
W_assoc starts empty                        → associations weak until experienced
```

---

## What This Is Not

Not a model that does one thing.
Not a transformer with extra steps.
Not autocomplete that got too good.

A learning organism. Built from what the brain actually does.
The seed is the minimum viable intelligence.
Everything after is the model becoming what it needs to become.

---

*ALA Specification v3*
*Added: bidirectional encoding, cross-modal binding, multidimensional salience,*
*predictive pre-activation, primitive formation math, associative explosion,*
*lossy compression, parallel field processing*
