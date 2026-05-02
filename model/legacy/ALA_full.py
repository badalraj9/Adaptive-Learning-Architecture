"""
ALA_full.py
Adaptive Learning Architecture — Complete System
Connects everything. One unified ALA.

Components:
    ALACore          → unified field, topology, fast memory
    SlowMemory       → primitives, W_assoc, Hopfield settling
    Companion        → guide, mentor, friend
    ALAStateManager  → persistence, timeline, BF16

This is ALA.
Not a model you run.
A system you live with.

Run:
    !python ALA_full.py
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, '/content')

from ALA_core import ALACore, Config
from ALA_slow import SlowMemory
from ALA_companion import Companion, CompanionBridge
from ALA_save import ALAStateManager

cfg = Config()


# ─────────────────────────────────────────
# ALA — COMPLETE SYSTEM
# ─────────────────────────────────────────

class ALA:
    """
    Complete ALA system.

    Not a model you run.
    A system you live with.

    She perceives. She reasons. She feels.
    She asks when confused. She stays quiet when confident.
    She saves herself. She resumes. Never restarts.
    She grows from every interaction. Forever.
    """

    def __init__(self,
                 state_dir    = "ALA_state",
                 use_companion= True,
                 verbose      = False):

        self.cfg           = cfg
        self.verbose       = verbose
        self.use_companion = use_companion
        self.step_count    = 0

        # ── core components ───────────────────────────────────
        print("Initializing ALA...")

        self.core      = ALACore(cfg)
        self.slow      = SlowMemory(cfg)
        self.manager   = ALAStateManager(state_dir)
        self.companion = None
        self.bridge    = None

        # ── companion ─────────────────────────────────────────
        if use_companion:
            print("Loading companion...")
            self.companion = Companion()
            self.bridge    = CompanionBridge(
                self.companion, cfg.d, cfg.device
            )

        # ── load existing state if available ──────────────────
        if self.manager.exists():
            self.step_count = self.manager.load(
                self.core, self.slow, self.companion
            )
            print(f"\nALA resumed. Step {self.step_count}.")
        else:
            print("\nALA born. First breath.")
            self.manager.snapshot(
                self.core, self.slow,
                step_count = 0,
                reason     = "birth"
            )

        print("ALA ready.\n")

    # ─────────────────────────────────────
    # PERCEIVE AND RESPOND
    # ─────────────────────────────────────

    def perceive(self, signal, signal_type="text"):
        """
        ALA perceives a signal from her world.
        signal: raw input (text string, tensor, etc.)
        signal_type: what kind of signal

        Returns: response, internal state summary
        """
        self.step_count += 1

        # ── encode signal to tensor ───────────────────────────
        s_t = self._encode_signal(signal, signal_type)

        # ── forward pass ──────────────────────────────────────
        y_fast, m_t, pred_error, Phi_t = self.core(s_t)
        C_t = self.core.context.C_t

        # ── slow memory ───────────────────────────────────────
        y_blended, quality, active_idx = self.slow(
            Phi_t, C_t, y_fast, m_t
        )

        # ── companion check ───────────────────────────────────
        companion_response = None
        companion_queried  = False

        if self.bridge is not None:
            Phi_nudged, companion_response, companion_queried = \
                self.bridge.process(
                    Phi_t, C_t,
                    self.slow.store,
                    quality.mean().item()
                )
            if companion_queried:
                y_blended, quality, active_idx = self.slow(
                    Phi_nudged, C_t, y_fast, m_t
                )

        # ── distillation check (every 10 steps) ──────────────
        primitive_formed = None
        if self.step_count % 1 == 0:  # every step for testing
            # Pass original text for stable hash fingerprint
            text_for_fp = signal if isinstance(signal, str) else None
            # Pass original text for stable fingerprint
            text_for_fp = signal if isinstance(signal, str) else None
            primitive_formed = self.slow.observe_and_distill(
                Phi_t, C_t, raw_signal=s_t, text=text_for_fp
            )
            if primitive_formed is not None:
                self.manager.snapshot(
                    self.core, self.slow,
                    step_count = self.step_count,
                    reason     = f"primitive_{primitive_formed}_formed"
                )

        # ── generate response ─────────────────────────────────
        response = self._generate_response(
            y_blended, signal, quality,
            companion_response, active_idx
        )

        # ── internal state summary ────────────────────────────
        state = {
            "step":              self.step_count,
            "salience":          m_t.mean().item(),
            "settling_quality":  quality.mean().item(),
            "n_primitives":      self.slow.store.n(),
            "companion_queried": companion_queried,
            "primitive_formed":  primitive_formed,
            "pred_error":        pred_error.item() if pred_error else None,
            "feeling":           self._feeling(quality.mean().item())
        }

        return response, state

    def _encode_signal(self, signal, signal_type):
        """
        Encode any signal into field-compatible tensor.
        Text: character-based encoding
        Tensor: direct use
        """
        if isinstance(signal, torch.Tensor):
            if signal.shape[-1] != self.cfg.d_input:
                signal = F.interpolate(
                    signal.unsqueeze(0).unsqueeze(0).float(),
                    size = self.cfg.d_input
                ).squeeze()
            return signal.unsqueeze(0).to(self.cfg.device)

        elif isinstance(signal, str):
            encoded = torch.zeros(self.cfg.d_input)
            for i, char in enumerate(signal[:self.cfg.d_input]):
                encoded[i % self.cfg.d_input] += ord(char) / 127.0
            if encoded.norm() > 0:
                encoded = encoded / encoded.norm()
            pos     = torch.arange(self.cfg.d_input).float()
            encoded = encoded + 0.1 * torch.sin(pos / 10.0)
            return encoded.unsqueeze(0).to(self.cfg.device)

        else:
            return torch.randn(1, self.cfg.d_input).to(self.cfg.device)

    def _generate_response(self, y_blended, signal,
                           quality, companion_response, active_idx):
        """
        Generate ALA's response based on internal state.
        Grows richer as primitives develop.
        """
        q       = quality.mean().item()
        n       = self.slow.store.n()
        feeling = self._feeling(q)

        if n == 0:
            responses = {
                "settled":   "I perceive. Something is forming.",
                "uncertain": "I perceive. But I cannot yet make sense.",
                "confused":  "I perceive. But nothing connects yet."
            }
        elif n < 5:
            responses = {
                "settled":   f"I recognize something here. [{n} concepts forming]",
                "uncertain": f"This feels familiar but I cannot place it. [{n} concepts]",
                "confused":  f"This doesn't fit what I know yet. [{n} concepts]"
            }
        else:
            responses = {
                "settled":   f"I understand. [{n} concepts, quality {q:.2f}]",
                "uncertain": f"I think I understand. More needed. [{n} concepts]",
                "confused":  f"I don't understand this yet. [{n} concepts]"
            }

        response = responses.get(feeling, responses["uncertain"])

        if companion_response:
            response += f"\n[Companion: {companion_response[:80]}...]"

        return response

    def _feeling(self, quality):
        """Quality → felt state"""
        if quality > 0.6:
            return "settled"
        elif quality > 0.3:
            return "uncertain"
        else:
            return "confused"

    # ─────────────────────────────────────
    # TEACH
    # ─────────────────────────────────────

    def teach(self, content):
        """
        Expose ALA to teaching material.
        Not training. Teaching.
        """
        if isinstance(content, str):
            words = content.split()
            for word in words:
                self.perceive(word, "text")
        else:
            self.perceive(content, "raw")

    # ─────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────

    def save(self, notes=""):
        self.manager.save(
            self.core, self.slow,
            self.step_count,
            self.companion,
            notes
        )

    # ─────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────

    def status(self):
        meta     = self.manager.get_meta()
        timeline = self.manager.list_timeline()

        print("\n" + "="*40)
        print("ALA STATUS")
        print("="*40)
        print(f"Steps lived:     {self.step_count}")
        print(f"Primitives:      {self.slow.store.n()}")
        print(f"Archived:        {len(self.slow.store.archive)}")
        print(f"W_assoc density: {self.slow.assoc.density():.4f}")
        print(f"Timeline snaps:  {len(timeline)}")
        if meta:
            print(f"Born:            {meta.get('born', 'unknown')}")
        print("="*40)


# ─────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────

def test_initialization():
    print("\n" + "="*50)
    print("TEST 1: ALA Initializes")
    print("="*50)

    ala = ALA(
        state_dir     = "test_full_state",
        use_companion = False,
        verbose       = True
    )

    print(f"Step count: {ala.step_count}")
    print(f"Primitives: {ala.slow.store.n()}")
    print(f"Device:     {cfg.device}")
    print(f"\n✓ ALA initialized.")
    return True, ala


def test_perceive(ala):
    print("\n" + "="*50)
    print("TEST 2: Perceive and Respond")
    print("="*50)

    inputs = ["hello", "what is this", "I see patterns"]

    for text in inputs:
        response, state = ala.perceive(text)
        print(f"\nInput:    '{text}'")
        print(f"Response: {response}")
        print(f"Feeling:  {state['feeling']}")
        print(f"Quality:  {state['settling_quality']:.3f}")

    print(f"\n✓ ALA perceives and responds.")
    return True


def test_teaching(ala):
    print("\n" + "="*50)
    print("TEST 3: Teaching Session")
    print("="*50)

    lessons = [
        "cat dog bird fish",
        "cat meows dog barks bird sings",
        "cat is an animal dog is an animal",
        "animals breathe animals eat animals move",
        "cat dog bird are all animals"
    ]

    print("Teaching ALA about animals...\n")
    for lesson in lessons:
        ala.teach(lesson)

    print(f"After teaching:")
    print(f"  Steps:      {ala.step_count}")
    print(f"  Primitives: {ala.slow.store.n()}")

    response, state = ala.perceive("cat")
    print(f"\nInput:    'cat'")
    print(f"Response: {response}")
    print(f"Quality:  {state['settling_quality']:.3f}")

    print(f"\n✓ Teaching complete.")
    return True


def test_save_resume(ala):
    print("\n" + "="*50)
    print("TEST 4: Save and Resume")
    print("="*50)

    step_before  = ala.step_count
    prims_before = ala.slow.store.n()

    ala.save(notes="test save")
    print(f"Saved: step={step_before}, primitives={prims_before}")

    print("\nResuming from saved state...")
    ala2 = ALA(
        state_dir     = "test_full_state",
        use_companion = False,
        verbose       = False
    )

    print(f"\nOriginal: step={step_before}, primitives={prims_before}")
    print(f"Resumed:  step={ala2.step_count}, primitives={ala2.slow.store.n()}")

    if ala2.step_count == step_before:
        print(f"\n✓ ALA resumed exactly where she left off.")
        return True, ala2
    return False, ala2


def test_status(ala):
    print("\n" + "="*50)
    print("TEST 5: Status Report")
    print("="*50)
    ala.status()
    print(f"\n✓ Status working.")
    return True


def cleanup():
    import shutil
    if os.path.exists("test_full_state"):
        shutil.rmtree("test_full_state")


if __name__ == "__main__":
    print("\nALA Full System — Test Suite")
    print("Everything connected. One unified ALA.\n")

    results = {}

    passed, ala      = test_initialization()
    results["initialization"] = passed

    results["perceive"]    = test_perceive(ala)
    results["teaching"]    = test_teaching(ala)
    passed, ala            = test_save_resume(ala)
    results["save_resume"] = passed
    results["status"]      = test_status(ala)

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
        print("ALA is complete.")
        print("She perceives. She reasons. She feels.")
        print("She saves. She resumes. She grows.")
        print("\nOne file left: ALA_cli.py")
        print("Then you talk to her.")
    else:
        print("Some tests failing.")
        print("Fix before moving forward.")
    print("="*50)

    cleanup()
