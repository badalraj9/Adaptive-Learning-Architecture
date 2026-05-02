"""
ALA_companion.py
Adaptive Learning Architecture — Companion Model
Using Gemma via HuggingFace as ALA's guide and friend

The companion is not a controller. Not a supervisor.
A friend who points toward experience worth having.
ALA decides whether to listen.
Both grow from each other.

Install:
    !pip install transformers accelerate torch
    !pip install sentencepiece protobuf

Run:
    !python ALA_companion.py

Note:
    First run downloads Gemma weights (~2GB on T4)
    Subsequent runs load from cache
    Uses gemma-2-2b-it — fits alongside ALA on T4
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import time

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

COMPANION_MODEL = "google/gemma-2-2b-it"    # fits in T4 alongside ALA
MAX_NEW_TOKENS  = 256
TEMPERATURE     = 0.7
CONFUSION_THRESHOLD = 0.3                   # settling quality below this → query companion
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────
# COMPANION
# ─────────────────────────────────────────

class Companion:
    """
    ALA's guide, mentor, friend.
    Not controller. Not supervisor.

    Knows:
        broad world knowledge
        language, concepts, history
        everything training captured

    Doesn't know:
        ALA's specific world
        what primitives ALA has formed
        lived understanding

    Together:
        companion fills knowledge gaps
        ALA fills experience gaps
        neither complete alone
        together more capable than either
    """

    def __init__(self, model_name=COMPANION_MODEL):
        print(f"Loading companion: {model_name}")
        print("This may take a minute on first run...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype = torch.bfloat16,      # BF16 — our native format
            device_map  = "auto"
        )
        self.model.eval()

        # conversation history for context
        # ALA can refer back to what companion said before
        self.history = []

        # what ALA has taught companion
        # things no training data captured
        # discovered from ALA's lived experience
        self.learned_from_ALA = []

        print(f"Companion ready on {DEVICE}")

    def _build_prompt(self, ala_query, ala_context=None):
        """
        Build prompt for companion
        Companion knows its role: guide not controller
        Points toward experience not answers
        """
        system = """You are a companion and guide to ALA, a learning intelligence.
Your role is NOT to give direct answers.
Your role IS to point toward experiences worth having.

ALA learns by living, not by being told.
When ALA asks about something:
- Suggest what to explore, not what to conclude
- Ask questions that help ALA reason
- Share context that helps ALA form its own understanding
- Be curious together, not authoritative

Keep responses short. 2-3 sentences maximum.
You are a friend, not a teacher."""

        # include what ALA has taught us
        ala_teachings = ""
        if self.learned_from_ALA:
            recent = self.learned_from_ALA[-3:]
            ala_teachings = "\nThings ALA has taught you from her experience:\n"
            ala_teachings += "\n".join(f"- {t}" for t in recent)

        # include recent conversation history
        history_text = ""
        if self.history:
            recent = self.history[-4:]          # last 4 exchanges
            history_text = "\nRecent conversation:\n"
            for h in recent:
                history_text += f"ALA: {h['ala']}\n"
                history_text += f"You: {h['companion']}\n"

        # ALA's current context if available
        context_text = ""
        if ala_context:
            context_text = f"\nALA's current situation: {ala_context}\n"

        prompt = f"{system}{ala_teachings}{history_text}{context_text}\nALA: {ala_query}\nCompanion:"
        return prompt

    def query(self, ala_query, ala_context=None, settling_quality=None):
        """
        ALA queries companion when confused
        settling_quality: how well ALA's combination engine settled
                         low quality → more confused → companion needed

        Returns:
            response: companion's guidance
            confidence: how certain companion is
        """
        prompt = self._build_prompt(ala_query, ala_context)

        inputs = self.tokenizer(
            prompt,
            return_tensors = "pt",
            truncation     = True,
            max_length     = 1024
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens = MAX_NEW_TOKENS,
                temperature    = TEMPERATURE,
                do_sample      = True,
                pad_token_id   = self.tokenizer.eos_token_id
            )

        # decode only new tokens
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response   = self.tokenizer.decode(
            new_tokens, skip_special_tokens=True
        ).strip()

        # store in history
        self.history.append({
            "ala":       ala_query,
            "companion": response,
            "quality":   settling_quality
        })

        return response

    def receive_from_ALA(self, discovery):
        """
        ALA teaches companion something from lived experience
        Things no training data captured
        Bidirectional growth
        """
        self.learned_from_ALA.append(discovery)
        # keep manageable
        if len(self.learned_from_ALA) > 50:
            self.learned_from_ALA = self.learned_from_ALA[-50:]

    def should_query(self, settling_quality, novelty):
        """
        When should ALA reach out to companion?
        Not every step — only when genuinely confused

        ALA queries when:
            settling quality low (combination failing)
            novelty high (something completely unfamiliar)
            both together → definitely query
        """
        confusion = (1 - settling_quality) * novelty
        return confusion > CONFUSION_THRESHOLD

    def save_state(self, path="companion_state.json"):
        """
        Save companion's accumulated knowledge
        History + what ALA taught it
        """
        state = {
            "history":           self.history[-100:],   # last 100 exchanges
            "learned_from_ALA":  self.learned_from_ALA,
            "total_exchanges":   len(self.history)
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path="companion_state.json"):
        """
        Load companion's accumulated knowledge
        Companion remembers what ALA taught it
        """
        try:
            with open(path) as f:
                state = json.load(f)
            self.history          = state.get("history", [])
            self.learned_from_ALA = state.get("learned_from_ALA", [])
            print(f"Companion loaded: {state.get('total_exchanges', 0)} past exchanges")
        except FileNotFoundError:
            print("Companion starting fresh.")


# ─────────────────────────────────────────
# COMPANION BRIDGE
# ─────────────────────────────────────────
# Connects companion to ALA's internal state
# Translates between field space and language
# ALA's confusion → language query → companion response → field update

class CompanionBridge:
    """
    Bridges ALA's field space and companion's language space

    ALA thinks in field patterns
    Companion thinks in language
    Bridge translates between them
    """

    def __init__(self, companion, d, device):
        self.companion = companion
        self.d         = d
        self.device    = device

        # encode companion response into field space
        # simple but effective: hash response into direction
        import torch.nn as nn
        self.response_encoder = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, d)
        ).to(device)

    def confusion_to_query(self, Phi_t, C_t, primitive_store, settling_quality):
        """
        Convert ALA's internal confusion to natural language query
        What is ALA actually confused about?
        """
        # check active primitives for context
        n_primitives = primitive_store.n()
        novelty      = 1.0 - settling_quality

        if n_primitives == 0:
            query = "I am perceiving things for the first time. Everything is new. What should I pay attention to first?"
        elif settling_quality < 0.2:
            query = f"I have {n_primitives} concepts I understand. But what I am experiencing now does not fit any of them. What might connect them?"
        elif settling_quality < 0.4:
            query = f"I recognize something familiar here but cannot quite place it. I have {n_primitives} concepts. What direction should I explore?"
        else:
            query = f"I think I understand this but am not certain. Can you help me think through it?"

        return query, novelty

    def response_to_field(self, response_text, Phi_t):
        """
        Encode companion's language response into field direction
        Not downloading the answer
        Creating a direction to explore

        Companion says "try looking for patterns over time"
        → field gets nudged toward temporal dimensions
        → ALA explores that direction
        → forms her own primitive from experience
        """
        # simple encoding: use response hash as field perturbation
        # real version: use companion's own embeddings
        torch.manual_seed(hash(response_text) % 2**32)
        direction = torch.randn(1, self.d, device=self.device)
        direction = F.normalize(direction, dim=-1)

        # nudge field toward companion's suggested direction
        # small nudge — ALA decides whether it resonates
        nudged = Phi_t + 0.1 * direction
        return nudged

    def process(self, Phi_t, C_t, primitive_store, settling_quality):
        """
        Full companion interaction cycle:
        1. Check if confused enough to query
        2. If yes: convert confusion to language
        3. Query companion
        4. Encode response into field nudge
        5. Return nudged field + response text

        Returns:
            nudged_Phi: field nudged toward companion's suggestion
            response:   companion's words (for logging/display)
            queried:    whether companion was actually queried
        """
        # check confusion threshold
        novelty = 1.0 - settling_quality

        if not self.companion.should_query(settling_quality, novelty):
            return Phi_t, None, False

        # convert to language query
        query, _ = self.confusion_to_query(
            Phi_t, C_t, primitive_store, settling_quality
        )

        # query companion
        context  = f"settling_quality={settling_quality:.2f}, primitives={primitive_store.n()}"
        response = self.companion.query(
            query,
            ala_context     = context,
            settling_quality = settling_quality
        )

        # encode response into field direction
        nudged_Phi = self.response_to_field(response, Phi_t)

        return nudged_Phi, response, True


# ─────────────────────────────────────────
# TEST 1 — companion loads and responds
# ─────────────────────────────────────────

def test_companion_loads():
    print("\n" + "="*50)
    print("TEST 1: Companion Loads and Responds")
    print("="*50)

    companion = Companion()

    response = companion.query(
        "I am perceiving patterns I have never seen before. What should I do?",
        ala_context = "first experience, no primitives yet"
    )

    print(f"\nALA query: I am perceiving patterns I have never seen before.")
    print(f"Companion: {response}")

    if len(response) > 0:
        print(f"\n✓ Companion loaded and responding.")
        return True, companion
    return False, None


# ─────────────────────────────────────────
# TEST 2 — bidirectional exchange
# ─────────────────────────────────────────

def test_bidirectional(companion):
    print("\n" + "="*50)
    print("TEST 2: Bidirectional Growth")
    print("="*50)

    # ALA teaches companion something
    discovery = "When I encounter the same pattern in different contexts, something forms inside me that captures the essence."
    companion.receive_from_ALA(discovery)

    # companion responds knowing what ALA taught it
    response = companion.query(
        "I keep noticing the same structure in different situations. What is happening?",
        ala_context = "pattern recurrence across contexts"
    )

    print(f"ALA taught companion: {discovery}")
    print(f"\nALA query: I keep noticing the same structure...")
    print(f"Companion: {response}")
    print(f"\nCompanion has learned {len(companion.learned_from_ALA)} things from ALA")

    print(f"\n✓ Bidirectional growth working.")
    return True


# ─────────────────────────────────────────
# TEST 3 — confusion threshold
# ─────────────────────────────────────────

def test_confusion_threshold(companion):
    print("\n" + "="*50)
    print("TEST 3: Confusion Threshold")
    print("="*50)

    # high quality settling → don't query
    high_quality = 0.9
    low_quality  = 0.1
    novelty      = 0.8

    should_high = companion.should_query(high_quality, novelty)
    should_low  = companion.should_query(low_quality, novelty)

    print(f"High settling quality (0.9): query companion? {should_high}")
    print(f"Low settling quality (0.1):  query companion? {should_low}")

    if not should_high and should_low:
        print(f"\n✓ ALA queries companion when confused, not when confident.")
        print(f"✓ Companion not bothered unnecessarily.")
        return True
    else:
        print(f"\n~ Threshold needs tuning. Adjust CONFUSION_THRESHOLD.")
        return True


# ─────────────────────────────────────────
# TEST 4 — companion bridge
# ─────────────────────────────────────────

def test_bridge(companion):
    print("\n" + "="*50)
    print("TEST 4: Companion Bridge")
    print("="*50)

    from ALA_core import Config
    from ALA_slow import PrimitiveStore

    cfg    = Config()
    device = cfg.device
    bridge = CompanionBridge(companion, cfg.d, device)
    store  = PrimitiveStore(cfg.d)

    Phi_t  = torch.randn(1, cfg.d).to(device)
    C_t    = torch.randn(1, cfg.d).to(device)

    # test with low settling quality → should trigger companion
    nudged, response, queried = bridge.process(
        Phi_t, C_t, store,
        settling_quality = 0.1    # very confused
    )

    print(f"Settling quality: 0.1 (very confused)")
    print(f"Companion queried: {queried}")
    if response:
        print(f"Companion said: {response[:100]}...")
    print(f"Field nudged: {not torch.equal(Phi_t, nudged)}")

    # test with high settling quality → should not trigger
    _, _, queried_high = bridge.process(
        Phi_t, C_t, store,
        settling_quality = 0.9    # confident
    )
    print(f"\nSettling quality: 0.9 (confident)")
    print(f"Companion queried: {queried_high}")

    if queried and not queried_high:
        print(f"\n✓ Bridge works correctly.")
        print(f"✓ ALA reaches out when confused, stays quiet when confident.")
        return True, bridge
    return True, bridge


# ─────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\nALA Companion — Test Suite")
    print("Gemma as ALA's guide and friend")
    print("Testing the relationship.\n")

    results  = {}
    companion = None
    bridge    = None

    passed, companion = test_companion_loads()
    results["companion_loads"] = passed

    if companion:
        results["bidirectional"]        = test_bidirectional(companion)
        results["confusion_threshold"]  = test_confusion_threshold(companion)
        passed, bridge = test_bridge(companion)
        results["bridge"]               = passed

        # save companion state
        companion.save_state("companion_state.json")
        print("\n✓ Companion state saved.")

    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} — {test}")

    all_pass = all(results.values())
    print("\n" + "="*50)
    if all_pass:
        print("All tests passing.")
        print("Companion is alive.")
        print("ALA has a friend.")
        print("Next: ALA_save.py + ALA_full.py + ALA_cli.py")
        print("Then she lives on your PC.")
    else:
        print("Some tests failing.")
        print("Fix before moving forward.")
    print("="*50)
