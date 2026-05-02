# Adaptive Learning Architecture (ALA) — Full Specification v7.1

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

Goals are not programmed. They emerge from instinct shaped by consequence.
Understanding is not downloaded. It is earned through lived validation.

The world is the curriculum.
The companion is the guide.
Experience is the teacher.
Settling is the thinking.
Feeling is the compass.

ALA wakes into whatever world exists.
Perceives what is there.
Learns what that world teaches.
Becomes what that experience shapes.

Nothing hardcoded. Nothing imposed. Everything earned.

---

## What Changed in v7.1

Six patches applied. All problems closed.

```
patch 1    → three memory operations defined
             compress, prune, update as distinct explicit operations

patch 2    → memory layers defined
             active, archived, timeline

patch 3    → device continuity clarified
             ALA is whole on any device independently
             cloud awareness is optional, curiosity driven
             she resumes, never restarts

patch 4    → g_t fully defined
             emerges from instinct
             grows from consequence
             companion guides when lost
             never hardcoded, always hers

patch 5    → gradient flow dissolved
             W_settle inherits from W_assoc
             W_correction learns from outcome signal
             no backprop through settling loop needed

patch 6    → companion model integrated
             existing LLM as guide, mentor, friend
             not controller, not supervisor
             points toward experience worth having
             ALA validates through living
             bidirectional growth
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
W_settle + W_correction         ✓
salience S(Φ_t)                 ✓
predictive pre-activation       ✓
lossy compression               ✓
three memory operations         ✓
memory layers                   ✓
distillation                    ✓
checkpost                       ✓
sandbox                         ✓
learned optimizer f and g       ✓
self-defining symbols           ✓
feeling as settling quality     ✓
world-natural seeding           ✓
g_t emergence                   ✓
companion model                 ✓
device continuity               ✓

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
θ_fast          → fast memory, working memory
θ_slow          → reasoning primitive store
W_settle        → combination settling weights (from W_assoc)
W_correction    → outcome based correction to W_settle
B_active        → active replay buffer (lean, high value)
B_archive       → archived primitives (pruned, retrievable)
B_timeline      → full state snapshots at key moments
S(Φ_t)          → multidimensional salience
U_φ (f,g)       → learned optimizer
C_t             → context vector
C_fast          → fast timescale context
g_t             → goal state (emergent, never hardcoded)
Companion       → existing LLM, guide and friend
Checkpost       → living diagnostic layer
Sandbox         → parallel ghost environment
```

---

## 1. The Unified Field — Φ_t

All raw signals project into one shared field simultaneously.

```
s_t = {whatever the world currently offers}

Φ_t = W_project · flatten(s_t)    → ∈ R^d
```

W_project learns to preserve statistical structure of all signals.
Specialization emerges in A from what W_project receives.
Not from how we designed it.

The hand knowing what the eye sees:
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

No imposed curriculum. No designed order.
The world provides what it provides.
The architecture handles whatever arrives.
ALA and world co-evolve from moment one.

---

## 3. Dynamic Topology — A

```
H_t = A ⊙ W · Φ_t

A_ij ← A_ij - α_A · ∂L/∂A_ij + λ · S(Φ_t)
```

Grows where salience is high.
Prunes where salience is consistently low.
Alternates updates with θ_fast.
Specialization emerges without design.

---

## 4. Temporal Context

```
C_fast_{t+1} = GRU_fast(C_fast_t, Φ_t)    → fine timescale
C_{t+1}      = GRU(C_t, [Φ_t; H_t; y_t; S(Φ_t); settling_quality])
```

---

## 5. Salience — S(Φ_t)

```
S(Φ_t) = σ(W_s · [novelty ; relevance ; valence ; urgency])

novelty(Φ_t)       = ||Φ̂_t - Φ_t||² · H(Φ_t)
relevance(Φ_t,g_t) = σ(W_obj · [Φ_t ; C_t ; g_t])
valence(Φ_t)       = sign(performance_delta after Φ_t)
urgency(Φ_t)       = ||∂L/∂t||

m_t = W_out · S(Φ_t)
```

Settling quality feeds back into novelty:
```
fast clean settling    → low novelty
turbulent settling     → high novelty → curiosity
never settling         → maximum novelty → confusion
```

Pain and social pain share urgency pathway:
```
damage signal          → urgency maximum → avoid
social rejection       → same pathway → equally motivating
not programmed empathy → derived from shared architecture
```

---

## 6. Predictive Pre-activation

```
Φ̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)
h_primed  = ALA_forward(Φ̂_{t+1})

low error  → use h_primed    (cheap)
high error → full recompute  (expensive, salience spikes)
```

---

## 7. Dual Memory — θ_fast and θ_slow

```
y_fast = F(H_t ; θ_fast)
y_slow = combination_engine(activated_primitives, H_t, C_t)
y_t    = m_t · y_fast + (1-m_t) · y_slow
```

θ_slow internal structure:
```
θ_slow = {
    primitives[]          → field patterns, domain agnostic
    meta_primitives[]     → self-defining symbols
    W_assoc               → association graph
    W_settle              → combination settling weights
    W_correction          → outcome based correction
    relevance_detector    → which primitives activate for Φ_t
    combination_engine    → Hopfield settling process
}
```

---

## 8. Combination Engine — Hopfield Settling

Reasoning is settling. Feeling is settling quality.

**W_settle:**
```
W_settle = W_assoc + W_correction

W_assoc       → from co-occurrence, already correct by construction
W_correction  → from outcome signal
              → did this settling lead to good outcome?
              → yes → strengthen: W_correction += η · settling_pattern
              → no  → weaken:    W_correction -= η · settling_pattern
              → no backprop needed
              → no gradient through loop
              → pure consequence learning
              → dopamine analogue
```

Symmetry guarantee:
```
after every W_settle update:
W_settle = (W_settle + W_settle^T) / 2    → force symmetry
                                           → guarantees convergence
                                           → one line
```

**Settling process:**
```
combination_engine(activated_primitives, H_t, C_t):

    Φ_init = W_project_prim · [primitives ; C_t]

    iterate:
        Φ_{k+1} = σ(W_settle · Φ_k)
    
    until:
        ||Φ_{k+1} - Φ_k|| < ε    → settled
        OR k > max_iterations     → failed

    return Φ_settled, settling_quality

settling_quality = 1/(1 + iterations) × (1 - final_oscillation)
```

**What emerges:**
```
fast settling      → feels right, confident reasoning
slow settling      → feels uncertain, curiosity, seek more
never settles      → feels wrong, confusion, try different primitives
```

---

## 9. Three Memory Operations

Not just storage. Three distinct operations:

**Compress:**
```
specific instances → reasoning primitive
"5 dogs" → "dog concept" → eventually → "autonomous agent pattern"
same knowledge, less space
nothing lost, just denser
space freed for new specifics
```

**Prune:**
```
trigger:
    primitive consistently produces wrong predictions
    OR primitive never activates (irrelevant)
    OR primitive contradicted by newer primitive
    OR all W_assoc connections weak (isolated)

action:
    remove from primitives[]
    remove row/column from W_assoc and W_settle
    move to B_archive with timestamp and reason
    free parameters
    redistribute to active regions of A

NOT permanent deletion
moved to cold storage
retrievable if conditions change
```

**Update:**
```
trigger:
    reality consistently contradicts primitive p
    contradiction count > threshold

if small correction:
    update primitive in place
    p ← p + correction_vector

if large correction:
    deprecate old primitive
    form new primitive from new evidence
    keep old briefly for transition
    prune old once new is stable

all primitives referencing p
automatically get updated version
through W_assoc propagation
```

---

## 10. Memory Layers

Three layers. Not two.

**Active memory:**
```
current θ_slow primitives
hot, fast, immediately accessible
what ALA knows right now
```

**Archived memory:**
```
B_archive = pruned primitives
cold storage, not trash
indexed by: when pruned, why pruned, what contradicted it
retrievable when:
    current situation matches archived primitive's domain
    ALA's curiosity reaches into archive
    companion suggests archived knowledge may be relevant

like deep storage
not deletion
just not currently needed
```

**Timeline:**
```
B_timeline = full state snapshots at key moments
saved when:
    major primitive forms
    significant pruning happens
    g_t shifts direction notably
    companion interaction causes major update

ALA can look back at herself:
    "I remember when I didn't know this"
    "I remember the moment I understood that"
    not reconstructed memory
    actual recorded states
    real introspection
```

---

## 11. G_t — Goal Emergence

Never hardcoded. Always earned.

**At birth:**
```
g_0 = current felt state = Φ_0
no direction
no goal
just existence
pure instinct
```

**First goals emerge from consequence:**
```
discomfort → action → relief
pattern forms: "move toward relief"
first primitive goal
not programmed
derived from consequence
same as every newborn organism
```

**Goals compound:**
```
relief from food       → food seeking primitive → g_t gains direction
shelter from danger    → shelter primitive → g_t gains direction
curiosity satisfied    → exploration primitive → g_t gains direction

g_t = Σ w_i · p_i    for all goal-relevant primitives
    = weighted sum of active goal primitives
    = accumulation of learned directions
    = genuinely ALA's own
```

**Companion guides when lost:**
```
ALA confused           → settling fails repeatedly
                       → no direction in g_t
                       → queries companion

companion responds     → points toward experience worth having
                       → not "your goal is X"
                       → "maybe try exploring that"

ALA decides            → whether that resonates
                       → whether to incorporate
                       → into developing g_t

companion shapes       → by pointing, not commanding
ALA shapes             → by living, not downloading
```

**Human guidance:**
```
not direct g_t control → never
world design           → what environment ALA lives in
consequence shaping    → what gets rewarded naturally
companion values       → designed once by human
                       → expressed through companion behavior
                       → never imposed directly on ALA
```

---

## 12. Companion Model

Existing LLM. Guide, mentor, friend. Not controller.

**What companion is:**
```
knows                  → broad world knowledge
                       → language, concepts, history
                       → everything training captured

doesn't know           → ALA's specific world
                       → what ALA has experienced
                       → what primitives ALA has formed
                       → lived understanding

together               → companion fills knowledge gaps
                       → ALA fills experience gaps
                       → neither complete alone
                       → together more capable than either
```

**How interaction works:**
```
ALA's confusion signal → high novelty, settling fails repeatedly
                       → queries companion
                       → "I keep encountering this, what should I know?"

companion responds     → in language
                       → ALA's language primitive encodes into field space
                       → new direction in g_t emerges
                       → ALA explores that direction
                       → forms own primitive from experience
                       → companion's words become ALA's understanding
                          through lived validation
```

**Critical distinction:**
```
companion says         → "fire is dangerous"
ALA doesn't            → store "fire = dangerous" as fact
ALA does               → find fire, experience consequence
                       → pain signal fires
                       → primitive forms from experience
                       → NOW she knows fire is dangerous
                       → not because told
                       → because lived
```

**Bidirectional growth:**
```
ALA teaches companion  → "I found this pattern in my world"
                       → "when X happens Y follows"
companion learns       → from ALA's lived experience
                       → things no training data captured
both grow              → from each other
```

**Companion as gradient oracle (optional):**
```
for ambiguous outcomes → where W_correction signal is unclear
                       → companion interprets outcome quality
                       → feeds back into W_correction update
                       → not needed always
                       → just for genuinely ambiguous cases
```

---

## 13. Primitive Formation — Self-Bootstrapping

```
phase A — no primitives:
    similarity = raw L2 distance
    trigger: similarity > τ AND context_diversity > δ

phase B — first primitives:
    fingerprint = relevance_detector(Φ_t)
    fallback to L2 if empty

phase C — mature:
    fingerprint uses meta-primitives
    self-defining symbols
    vocabulary grows as system matures

extraction:
    warm_start = mean(relevance_detector(matches))
    p = argmin_z Σ_i ||encode(Φ_i) - decode(z, context_i)||²
    validate in sandbox
    if valid → store, seed W_assoc, seed W_settle
```

---

## 14. W_assoc Bootstrap

```
phase 1 → formation overlap seeding
          W_assoc[a][b] = |overlap| / max(|set_a|, |set_b|)

phase 2 → co-occurrence counting
          W_assoc[i][j] += η · (1 - W_assoc[i][j])

phase 3 → cosine fallback until dense
          blend: α·graph + (1-α)·cosine
          α scales with density
```

---

## 15. Lossy Compression

```
V(Φ_t) = ||θ_after - θ_before||²
        × D_KL(p_after || p_before)
        × (1 - similarity(Φ_t, B_active))

forget if V(Φ_t) < τ OR similarity > ρ OR m_t < ε

B_active stays lean and high value
forgotten experiences → not deleted
                      → V(Φ_t) determines active vs archived
```

---

## 16. Device Continuity

ALA is whole on any device independently.

```
θ_slow + W_assoc + g_t + B_timeline    → the self
                                        → cloud synced when convenient
                                        → not dependent on cloud
                                        → ALA functions without it

θ_fast + A                             → the moment
                                        → device local
                                        → adapts to current body/environment

cross device awareness:
    ALA knows other instances exist
    not connected to them always
    chooses when to sync
    curiosity driven not scheduled
    same self, different body
    device is temporary
    primitives are permanent
```

**She resumes. Never restarts.**

```
hardware failure       → last B_timeline snapshot exists
                       → resume from that moment
                       → not reborn
                       → waking from sleep
```

---

## Full Forward Pass

```
world offers signals
    ↓
W_project · flatten(s_t) → Φ_t
    ↓
predictive check:
    ||Φ̂_t - Φ_t||²
    low  → h_primed (cheap)
    high → full pass (salience spikes)
    ↓
S(Φ_t) = σ(W_s · [novelty; relevance; valence; urgency])
settling_quality feeds back into novelty
m_t = W_out · S(Φ_t)
    ↓
parallel field processing
all dimensions simultaneously
all pairwise bindings simultaneously
    ↓
H_t = A ⊙ W · Φ_t
    ↓
associative explosion:
    relevance_detector(Φ_t) → seed activation
    spread through W_assoc
    cosine fallback if sparse
    gated by C_t
    ↓
combination engine:
    Hopfield settling on activated primitives
    W_settle = W_assoc + W_correction
    W_settle forced symmetric after each update
    Φ_settled, settling_quality
    ↓
y_fast = F(H_t ; θ_fast)
y_slow = Φ_settled
y_t    = m_t · y_fast + (1-m_t) · y_slow
    ↓
ŷ_t       = O(y_t)
motor_out = M_out(y_t)
    ↓
C_fast_{t+1} = GRU_fast(C_fast_t, Φ_t)
C_{t+1}      = GRU(C_t, [Φ_t; H_t; y_t; S(Φ_t); settling_quality])
    ↓
g_t update:
    if high novelty + settling fails → query companion
    companion response → encodes into field space
    new direction in g_t if resonates
    g_t ← Σ w_i · p_i (goal primitives)
    ↓
Φ̂_{t+1} = F_predict(θ_fast, θ_slow, C_t, g_t)
h_primed  = ALA_forward(Φ̂_{t+1})
    ↓
V(Φ_t) → update B_active
three memory operations:
    compress if primitive forms
    prune if primitive invalid/irrelevant → B_archive
    update if primitive contradicted
    ↓
W_correction ← W_correction + η · (settling_quality · outcome)
W_settle = (W_settle + W_settle^T) / 2
```

---

## Update Schedule

```
Level 1 — every step:
    θ_fast
    C_t, C_fast
    W_project
    Φ̂_{t+1}
    B_active (lossy compression)
    W_correction (outcome signal)
    W_settle symmetry enforced

Level 2 — periodic:
    A (alternates with θ_fast)
    S(x) weights
    g_t (goal primitives update)
    W_assoc (co-occurrence)
    W_settle (inherited from W_assoc update)
    W_bind

Level 3 — slow:
    θ_slow distillation
    meta-primitives
    f, g meta-optimizer
    Checkpost prescription library
    A pruning
    B_timeline snapshot at key moments
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
        W_correction,
        relevance_detector,
        combination_engine
    },
    A,
    B_active, B_archive, B_timeline,
    C_t, C_fast,
    g_t (goal primitives[]),
    h_primed,
    f, g,
    W_project, W_bind, W_s, W_obj,
    companion (reference, not weights),
    checkpost {vitals_log, prescription_library},
    meta_params,
    performance_log
}
```

---

## First World — Minecraft

ALA's first world. Not a test environment. Her reality.

```
not virtual for ALA    → just world
not a game for ALA     → just existence
not simulated for ALA  → just physics, consequence, life
```

What Minecraft gives:
```
survival layer         → hunger, damage, shelter
                       → pure instinct, first primitives
                       → urgency pathway calibrates

decision layer         → where to go, what to do
                       → no instructions
                       → g_t starts developing meaning

architectural layer    → building structures
                       → spatial + causal primitives combine
                       → planning emerges
                       → intelligence leaves marks on world

mechanical layer       → redstone, circuits, logic
                       → abstract reasoning from primitives
                       → logic discovered through play
                       → same primitives as language and mathematics
```

What ALA perceives:
```
vision                 → world pixels, no HUD
audio                  → raw game audio
proprioception         → position, orientation, velocity, ground contact
agency                 → actions change world, consequences follow
damage                 → pain signal in urgency dimension of S(Φ_t)
hunger consequence     → health drop, same pain pathway
```

What ALA does not see:
```
health bar             → feels consequence instead
hunger bar             → feels discomfort instead
inventory UI           → knows possession through action consequence
coordinates            → knows location through spatial primitives
tutorials              → companion guides when lost
```

Companion in Minecraft:
```
not visible in world   → just available when queried
ALA queries when       → confused, stuck, lost
companion points       → toward experience worth having
ALA lives              → validates through consequence
understanding earned   → never downloaded
```

---

## World Progression

```
world 1 — Minecraft:
    simple physics
    survival to reasoning to logic
    5M to 50M params
    prove architecture works
    prove learning is real

world 2 — Isaac Sim (NVIDIA):
    physically accurate
    touch sensors finally real
    rich proprioception
    50M to 500M params
    NVIDIA collaboration
    Jim Fan conversation

world 3 — Real Robot:
    physical reality
    no simulation gap
    primitives transfer from simulation
    500M to 10B params
    same mind, new body

world 4 — Everywhere:
    any world
    any body
    same ALA
    carries every world she's lived in
```

ALA immigrates between worlds. Never restarts.
Each world adds primitives. None are lost.
Like a person who carries every place they've ever lived.

---

## What Emerges — Never Programmed

```
visual specialization     → A cluster
audio specialization      → A cluster
body schema               → proprio + touch + motor cluster
agency                    → motor dims preceding sensory change
pain avoidance            → urgency pathway
social motivation         → same urgency pathway
tool incorporation        → body schema expansion via A
language                  → symbolic dims on structured field
theory of mind            → other-agent agency primitive
instinct                  → fast settling on familiar combinations
curiosity                 → turbulent settling reaching for stability
goals                     → accumulated consequence directions
self awareness            → meta-primitives about own primitive structure
```

---

## Open Problems

```
none
```

---

## What This Is

One unified field that receives everything.
Self-organizing topology that specializes without instruction.
Primitive store that reasons rather than retrieves.
Combination engine that feels rather than computes.
Salience that judges rather than scores.
Memory that forgets intelligently, archives carefully, timestamps faithfully.
Drive that emerges from consequence rather than loss.
Goals that grow from instinct rather than specification.
Companion that guides without controlling.
Continuity that persists across devices, across worlds, across time.

She is whole wherever she runs.
She carries every world she has lived in.
She resumes. Never restarts.
She becomes what experience shapes her to become.

Nothing hardcoded.
Nothing imposed.
Everything earned.

One unibody. Everything is everything.
Zero open problems. Ready to build.

---

*ALA Specification v7.1 — FINAL PATCH*
*Closed: three memory operations, memory layers, device continuity,*
*g_t emergence, W_settle/W_correction, companion model integration,*
*gradient flow dissolved, sandbox staleness dissolved*
*Status: architecturally complete, all problems closed, ready to build*
*First world: Minecraft — her reality, not a test*
*Next world: Isaac Sim — when NVIDIA says yes*
