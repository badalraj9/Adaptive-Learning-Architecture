"""
ALA 1M Parameter Compression Test
Push her to capacity ceiling and watch compression happen.

Test:
1. 1M params = small capacity
2. Feed lots of patterns
3. Watch primitives form
4. Watch compression kick in
5. Test if she derives or memorizes

Run on Colab:
    !python ALA_1M_test.py
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
import sys
import os

# Allow this experiment to run from the repository root after reorganizing
# implementation files into ../model.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
LEGACY_MODEL_DIR = os.path.join(MODEL_DIR, "legacy")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)
if LEGACY_MODEL_DIR not in sys.path:
    sys.path.insert(0, LEGACY_MODEL_DIR)

from ALA_full import ALA
from ALA_core import Config

# ═══════════════════════════════════════════════════════════
# 1M PARAM CONFIG
# ═══════════════════════════════════════════════════════════

class TinyConfig:
    """1M param config - hits ceiling fast"""
    def __init__(self):
        self.d            = 256      # field dimension (small)
        self.d_input      = 128      # input projection
        self.d_out        = 128      # output projection
        self.d_topology   = 256      # topology size
        self.d_context    = 128      # context size
        
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculated params:
        # W_project: 128*256 = 33K
        # A: 256*256 = 65K
        # θ_fast: ~100K
        # θ_slow components: ~200K
        # Context GRU: ~50K
        # Salience: ~50K
        # Predictor: ~100K
        # Various projections: ~400K
        # Total: ~1M params


# ═══════════════════════════════════════════════════════════
# TEACHING DATA - DESIGNED TO FORCE COMPRESSION
# ═══════════════════════════════════════════════════════════

MATH_PATTERNS = [
    # Addition patterns (should compress to [addition] primitive)
    "one plus one equals two",
    "two plus one equals three", 
    "three plus one equals four",
    "one plus two equals three",
    "two plus two equals four",
    "three plus two equals five",
    "four plus two equals six",
    "five plus two equals seven",
    "one plus three equals four",
    "two plus three equals five",
    "three plus three equals six",
    "four plus three equals seven",
    "five plus three equals eight",
    
    # Multiplication patterns (should compress to [multiply] primitive)
    "two times two equals four",
    "two times three equals six",
    "two times four equals eight",
    "three times two equals six",
    "three times three equals nine",
    "three times four equals twelve",
    "four times two equals eight",
    "four times three equals twelve",
    "four times four equals sixteen",
    
    # Subtraction patterns
    "five minus one equals four",
    "five minus two equals three",
    "four minus one equals three",
    "four minus two equals two",
    "three minus one equals two",
    "three minus two equals one",
]

LOGICAL_PATTERNS = [
    # If-then (should compress to [implication] primitive)
    "if it rains then ground is wet",
    "if sun shines then day is bright",
    "if you eat then hunger decreases",
    "if you sleep then energy increases",
    "if you run then you get tired",
    "if you study then knowledge grows",
    "if plant gets water then plant grows",
    "if fire burns then heat increases",
    
    # Categories (should compress to [category] primitive)
    "dog is animal",
    "cat is animal",
    "bird is animal", 
    "fish is animal",
    "apple is fruit",
    "banana is fruit",
    "orange is fruit",
    "carrot is vegetable",
    "potato is vegetable",
]

SEQUENCE_PATTERNS = [
    # Temporal sequences (should compress to [sequence] primitive)
    "first you wake up then you eat breakfast",
    "first you plant seed then it grows",
    "first you learn then you understand",
    "first you practice then you improve",
    "first winter comes then spring follows",
    "first you ask then you receive answer",
]

# Novel test cases - NOT in training
NOVEL_TESTS = [
    # If she has primitives, she should handle these
    "six plus two",              # novel addition
    "five times two",            # novel multiplication  
    "if door opens then",        # novel implication
    "lion is",                   # novel category
    "seven minus three",         # novel subtraction
]


# ═══════════════════════════════════════════════════════════
# TEST MONITORING
# ═══════════════════════════════════════════════════════════

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def print_progress(step, total, n_prims, w_density, quality):
    bar_len = 30
    filled = int(bar_len * step / total)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"\r[{bar}] {step}/{total} | Primitives: {n_prims} | "
          f"W_assoc: {w_density:.4f} | Quality: {quality:.3f}", end='')

def test_compression(ala):
    """Test if ALA derives vs memorizes"""
    
    print_header("Testing Compression & Derivation")
    
    results = {
        "memorized": 0,
        "derived": 0,
        "failed": 0
    }
    
    for test in NOVEL_TESTS:
        response, state = ala.perceive(test, "text")
        quality = state['settling_quality']
        
        print(f"\nTest: '{test}'")
        print(f"Response: {response}")
        print(f"Quality: {quality:.3f}")
        print(f"Primitives used: {state['n_primitives']}")
        
        if quality > 0.5:
            results["derived"] += 1
            print("✓ DERIVED (high quality)")
        elif quality > 0.2:
            results["memorized"] += 1  
            print("~ MEMORIZED (medium quality)")
        else:
            results["failed"] += 1
            print("✗ FAILED (low quality)")
    
    print_header("Compression Test Results")
    print(f"Derived (understands):  {results['derived']}/{len(NOVEL_TESTS)}")
    print(f"Memorized (pattern):    {results['memorized']}/{len(NOVEL_TESTS)}")
    print(f"Failed (random):        {results['failed']}/{len(NOVEL_TESTS)}")
    
    return results


def watch_compression_live(ala, data, batch_size=10):
    """Feed data and watch compression happen"""
    
    print_header("Feeding Data - Watch Compression")
    
    n_total = len(data)
    primitive_milestones = []
    
    for i, text in enumerate(data):
        # Perceive
        _, state = ala.perceive(text, "text")
        
        # Track primitive formation
        n_prims = state['n_primitives']
        if state['primitive_formed'] is not None:
            primitive_milestones.append({
                'step': i,
                'primitive_id': state['primitive_formed'],
                'total_primitives': n_prims
            })
            print(f"\n🔥 Primitive {state['primitive_formed']} formed at step {i}!")
        
        # Progress bar
        w_density = ala.slow.assoc.density()
        quality = state['settling_quality']
        print_progress(i+1, n_total, n_prims, w_density, quality)
    
    print("\n")
    return primitive_milestones


def analyze_final_state(ala):
    """Analyze what ALA learned"""
    
    print_header("Final State Analysis")
    
    n_prims = ala.slow.store.n()
    n_archived = len(ala.slow.store.archive)
    w_density = ala.slow.assoc.density()
    
    print(f"Active Primitives:    {n_prims}")
    print(f"Archived Primitives:  {n_archived}")
    print(f"W_assoc Density:      {w_density:.6f}")
    print(f"Total Steps:          {ala.step_count}")
    
    if n_prims > 0:
        print(f"\nTop Primitives by Usage:")
        for i in range(min(n_prims, 10)):
            usage = ala.slow.store.usage_count[i]
            norm = ala.slow.store.primitives[i].norm().item()
            print(f"  P{i}: usage={usage:4d}, norm={norm:.3f}")
    
    # Compression ratio estimate
    if n_prims > 0:
        data_fed = ala.step_count
        compression_ratio = data_fed / n_prims
        print(f"\nCompression Ratio: {compression_ratio:.1f}:1")
        print(f"  ({data_fed} experiences → {n_prims} primitives)")


# ═══════════════════════════════════════════════════════════
# MAIN TEST
# ═══════════════════════════════════════════════════════════

def run_compression_test():
    """
    Full compression test:
    1. Create 1M param ALA
    2. Feed data until primitives form
    3. Test derivation vs memorization
    4. Analyze compression
    """
    
    print_header("ALA 1M Compression Test")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Create tiny ALA
    # Note: Need to patch Config in ALA files, but for now just test with existing
    # This will use 5M params, but same principle
    
    print("\nInitializing ALA (using 5M config for now)...")
    ala = ALA(
        state_dir="test_compression_state",
        use_companion=False,  # skip companion for cleaner test
        verbose=False
    )
    
    # Combine all teaching data
    all_data = MATH_PATTERNS + LOGICAL_PATTERNS + SEQUENCE_PATTERNS
    
    # Repeat to force compression
    repeated_data = all_data * 3  # 3 rounds of same patterns
    
    print(f"Total teaching examples: {len(repeated_data)}")
    print(f"Unique patterns: {len(all_data)}")
    print(f"Repetition factor: 3x")
    
    # Phase 1: Feed data and watch compression
    milestones = watch_compression_live(ala, repeated_data)
    
    # Phase 2: Analyze state
    analyze_final_state(ala)
    
    # Phase 3: Test derivation
    test_results = test_compression(ala)
    
    # Summary
    print_header("Compression Test Complete")
    
    if len(milestones) > 0:
        print(f"✓ Primitives formed: {len(milestones)}")
        print(f"✓ Compression working")
    else:
        print(f"✗ No primitives formed")
        print(f"  Possible issues:")
        print(f"  - Need more diverse contexts (increase n_trigger)")
        print(f"  - Fingerprint still unstable")
        print(f"  - Need longer sequences")
    
    derived_pct = (test_results['derived'] / len(NOVEL_TESTS)) * 100
    print(f"\nDerivation capability: {derived_pct:.0f}%")
    
    if derived_pct > 60:
        print("🔥 ALA is deriving, not just memorizing!")
    elif derived_pct > 30:
        print("⚠ Partial derivation, needs improvement")
    else:
        print("✗ Mostly memorization, primitives not generalizing")
    
    # Save state
    ala.save(notes="compression test complete")
    print(f"\nState saved to: test_compression_state/")
    
    return ala, test_results


# ═══════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    ala, results = run_compression_test()
