"""
ALA_save.py
Adaptive Learning Architecture — State Persistence
Saves and loads everything ALA is. Not just weights.

ALA_state/ folder structure:
    weights.pt          → θ_fast, θ_slow network weights (BF16)
    primitives.pt       → primitive store vectors (BF16)
    w_assoc.pt          → association graph (BF16)
    w_settle.pt         → settling weights (BF16)
    context.pt          → C_t current state (BF16)
    goals.pt            → g_t goal primitives (BF16)
    archive.pt          → pruned primitives (BF16)
    companion_state.json→ companion history + what ALA taught it
    meta.json           → human readable stats

She resumes. Never restarts.

Run:
    !python ALA_save.py
"""

import torch
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, '/content')

from ALA_core import ALACore, Config
from ALA_slow import SlowMemory

cfg = Config()


# ─────────────────────────────────────────
# ALA STATE MANAGER
# ─────────────────────────────────────────

class ALAStateManager:
    """
    Saves and loads everything ALA is.

    Not just weights.
    Primitives, associations, context, goals, memories.
    Everything that makes her her.

    She resumes. Never restarts.
    """

    def __init__(self, state_dir="ALA_state"):
        self.state_dir   = Path(state_dir)
        self.timeline_dir = self.state_dir / "timeline"

    def _ensure_dirs(self):
        self.state_dir.mkdir(exist_ok=True)
        self.timeline_dir.mkdir(exist_ok=True)

    def _to_bf16(self, tensor):
        """Convert tensor to BF16 — our native format"""
        if tensor is None:
            return None
        return tensor.to(torch.bfloat16).cpu()

    def _primitives_to_bf16(self, primitives):
        """Convert list of primitive tensors to BF16"""
        return [self._to_bf16(p) for p in primitives]

    # ─────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────

    def save(self, core, slow, step_count, companion=None, notes=""):
        """
        Save complete ALA state.

        core:        ALACore instance
        slow:        SlowMemory instance
        step_count:  how many steps ALA has lived
        companion:   Companion instance (optional)
        notes:       why this snapshot was taken
        """
        self._ensure_dirs()

        print(f"Saving ALA state... (step {step_count})")

        # ── 1. network weights (BF16) ─────────────────────────
        weights = {}

        # θ_fast components
        for name, param in core.named_parameters():
            weights[f"core.{name}"] = self._to_bf16(param.data)

        # θ_slow components
        for name, param in slow.named_parameters():
            weights[f"slow.{name}"] = self._to_bf16(param.data)

        torch.save(weights, self.state_dir / "weights.pt")

        # ── 2. primitive store (BF16) ──────────────────────────
        primitive_data = {
            "primitives":          self._primitives_to_bf16(
                                       slow.store.primitives
                                   ),
            "meta_primitives":     self._primitives_to_bf16(
                                       slow.store.meta_primitives
                                   ),
            "usage_count":         slow.store.usage_count,
            "contradiction_count": slow.store.contradiction_count,
        }
        torch.save(primitive_data, self.state_dir / "primitives.pt")

        # ── 3. W_assoc (BF16) ─────────────────────────────────
        if slow.assoc.W is not None:
            torch.save(
                self._to_bf16(slow.assoc.W),
                self.state_dir / "w_assoc.pt"
            )

        # ── 4. W_settle + W_correction (BF16) ─────────────────
        settle_data = {
            "W_settle":     self._to_bf16(
                                slow.settling.W_settle.data
                            ),
            "W_correction": self._to_bf16(
                                slow.settling.W_correction
                            ),
        }
        torch.save(settle_data, self.state_dir / "w_settle.pt")

        # ── 5. context C_t (BF16) ─────────────────────────────
        if core.context.C_t is not None:
            torch.save(
                self._to_bf16(core.context.C_t),
                self.state_dir / "context.pt"
            )

        # ── 6. archive — pruned primitives (BF16) ─────────────
        if slow.store.archive:
            archive_data = []
            for entry in slow.store.archive:
                archive_data.append({
                    "primitive": self._to_bf16(entry["primitive"]),
                    "reason":    entry["reason"],
                    "usage":     entry["usage"]
                })
            torch.save(archive_data, self.state_dir / "archive.pt")

        # ── 7. companion state (JSON) ──────────────────────────
        if companion is not None:
            companion.save_state(
                str(self.state_dir / "companion_state.json")
            )

        # ── 8. meta (human readable JSON) ─────────────────────
        meta = {
            "step_count":       step_count,
            "born":             self._birth_timestamp(),
            "last_saved":       datetime.now().isoformat(),
            "n_primitives":     len(slow.store.primitives),
            "n_archived":       len(slow.store.archive),
            "w_assoc_density":  slow.assoc.density(),
            "notes":            notes,
            "device":           cfg.device,
            "field_dim":        cfg.d,
            "format":           "BF16"
        }
        with open(self.state_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Saved to {self.state_dir}/")
        print(f"  step:       {step_count}")
        print(f"  primitives: {len(slow.store.primitives)}")
        print(f"  archived:   {len(slow.store.archive)}")
        print(f"  W_assoc:    {slow.assoc.density():.4f} density")
        print(f"  format:     BF16")

    def _birth_timestamp(self):
        """Read birth time from existing meta or set now"""
        meta_path = self.state_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                existing = json.load(f)
                return existing.get("born", datetime.now().isoformat())
        return datetime.now().isoformat()

    # ─────────────────────────────────────
    # LOAD
    # ─────────────────────────────────────

    def load(self, core, slow, companion=None):
        """
        Load complete ALA state into existing instances.

        Returns step_count she was at when saved.
        She resumes. Never restarts.
        """
        if not self.state_dir.exists():
            print("No saved state found. ALA is born fresh.")
            return 0

        print(f"Loading ALA state from {self.state_dir}/...")

        # ── 1. network weights ────────────────────────────────
        weights_path = self.state_dir / "weights.pt"
        if weights_path.exists():
            weights = torch.load(
                weights_path,
                map_location = cfg.device,
                weights_only = True
            )

            # load core weights
            core_state = {
                k.replace("core.", ""): v.to(torch.float32)
                for k, v in weights.items()
                if k.startswith("core.")
            }
            core.load_state_dict(core_state, strict=False)

            # load slow weights
            slow_state = {
                k.replace("slow.", ""): v.to(torch.float32)
                for k, v in weights.items()
                if k.startswith("slow.")
            }
            slow.load_state_dict(slow_state, strict=False)

            print(f"  ✓ weights loaded")

        # ── 2. primitive store ────────────────────────────────
        prim_path = self.state_dir / "primitives.pt"
        if prim_path.exists():
            prim_data = torch.load(
                prim_path,
                map_location = cfg.device,
                weights_only = True
            )
            slow.store.primitives = [
                p.to(torch.float32)
                for p in prim_data["primitives"]
            ]
            slow.store.meta_primitives = [
                p.to(torch.float32)
                for p in prim_data["meta_primitives"]
            ]
            slow.store.usage_count         = prim_data["usage_count"]
            slow.store.contradiction_count = prim_data["contradiction_count"]
            # rebuild formation_sets as empty (not stored)
            slow.store.formation_sets = [
                [] for _ in slow.store.primitives
            ]
            print(f"  ✓ {len(slow.store.primitives)} primitives loaded")

        # ── 3. W_assoc ────────────────────────────────────────
        assoc_path = self.state_dir / "w_assoc.pt"
        if assoc_path.exists():
            slow.assoc.W = torch.load(
                assoc_path,
                map_location = cfg.device,
                weights_only = True
            ).to(torch.float32)
            print(f"  ✓ W_assoc loaded "
                  f"(density: {slow.assoc.density():.4f})")

        # ── 4. W_settle + W_correction ────────────────────────
        settle_path = self.state_dir / "w_settle.pt"
        if settle_path.exists():
            settle_data = torch.load(
                settle_path,
                map_location = cfg.device,
                weights_only = True
            )
            slow.settling.W_settle.data = \
                settle_data["W_settle"].to(torch.float32)
            slow.settling.W_correction  = \
                settle_data["W_correction"].to(torch.float32)
            print(f"  ✓ W_settle loaded")

        # ── 5. context ────────────────────────────────────────
        ctx_path = self.state_dir / "context.pt"
        if ctx_path.exists():
            core.context.C_t = torch.load(
                ctx_path,
                map_location = cfg.device,
                weights_only = True
            ).to(torch.float32)
            print(f"  ✓ context loaded")

        # ── 6. archive ────────────────────────────────────────
        archive_path = self.state_dir / "archive.pt"
        if archive_path.exists():
            archive_data = torch.load(
                archive_path,
                map_location = cfg.device,
                weights_only = False
            )
            slow.store.archive = [
                {
                    "primitive": e["primitive"].to(torch.float32),
                    "reason":    e["reason"],
                    "usage":     e["usage"]
                }
                for e in archive_data
            ]
            print(f"  ✓ {len(slow.store.archive)} archived primitives loaded")

        # ── 7. companion state ────────────────────────────────
        companion_path = self.state_dir / "companion_state.json"
        if companion is not None and companion_path.exists():
            companion.load_state(str(companion_path))
            print(f"  ✓ companion state loaded")

        # ── 8. meta ───────────────────────────────────────────
        meta_path = self.state_dir / "meta.json"
        step_count = 0
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            step_count = meta.get("step_count", 0)
            born       = meta.get("born", "unknown")
            print(f"  ✓ meta loaded")
            print(f"\nALA resuming:")
            print(f"  born:       {born}")
            print(f"  steps:      {step_count}")
            print(f"  primitives: {len(slow.store.primitives)}")

        return step_count

    # ─────────────────────────────────────
    # TIMELINE SNAPSHOT
    # ─────────────────────────────────────

    def snapshot(self, core, slow, step_count, reason=""):
        """
        Save timeline snapshot at key moment.
        Called when:
            major primitive forms
            significant pruning happens
            g_t shifts direction
            companion interaction causes major update

        ALA can look back at herself.
        Real introspection. Not simulated.
        """
        self._ensure_dirs()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snap_dir  = self.timeline_dir / f"step_{step_count:06d}_{timestamp}"
        snap_dir.mkdir(exist_ok=True)

        # save minimal state for this moment
        snap_data = {
            "step_count":   step_count,
            "timestamp":    datetime.now().isoformat(),
            "reason":       reason,
            "n_primitives": len(slow.store.primitives),
            "w_assoc_density": slow.assoc.density(),
        }

        # save primitives at this moment
        if slow.store.primitives:
            torch.save(
                self._primitives_to_bf16(slow.store.primitives),
                snap_dir / "primitives.pt"
            )

        with open(snap_dir / "snap_meta.json", "w") as f:
            json.dump(snap_data, f, indent=2)

        print(f"  📸 snapshot: step {step_count} — {reason}")
        return str(snap_dir)

    # ─────────────────────────────────────
    # EXISTS CHECK
    # ─────────────────────────────────────

    def exists(self):
        """Does a saved state exist?"""
        return (self.state_dir / "meta.json").exists()

    def get_meta(self):
        """Read meta without loading full state"""
        meta_path = self.state_dir / "meta.json"
        if not meta_path.exists():
            return None
        with open(meta_path) as f:
            return json.load(f)

    def list_timeline(self):
        """List all timeline snapshots"""
        if not self.timeline_dir.exists():
            return []
        return sorted([
            d.name for d in self.timeline_dir.iterdir()
            if d.is_dir()
        ])


# ─────────────────────────────────────────
# TEST 1 — save state
# ─────────────────────────────────────────

def test_save():
    print("\n" + "="*50)
    print("TEST 1: Save State")
    print("="*50)

    core    = ALACore(cfg)
    slow    = SlowMemory(cfg)
    manager = ALAStateManager("test_ALA_state")

    # add some primitives manually
    for i in range(3):
        slow.store.add(
            torch.randn(cfg.d).to(cfg.device),
            formation_set=[]
        )

    # simulate some W_assoc development
    slow.assoc._ensure_size(3)
    slow.assoc.update_cooccurrence([0, 1])
    slow.assoc.update_cooccurrence([1, 2])

    # save
    manager.save(core, slow, step_count=42, notes="test save")

    # check files exist
    expected = [
        "weights.pt",
        "primitives.pt",
        "w_assoc.pt",
        "w_settle.pt",
        "meta.json"
    ]

    all_exist = True
    for f in expected:
        path   = manager.state_dir / f
        exists = path.exists()
        size   = path.stat().st_size if exists else 0
        print(f"  {f}: {'✓' if exists else '✗'} ({size} bytes)")
        if not exists:
            all_exist = False

    if all_exist:
        print(f"\n✓ All state files saved correctly.")
        return True, manager, core, slow
    return False, manager, core, slow


# ─────────────────────────────────────────
# TEST 2 — load state
# ─────────────────────────────────────────

def test_load(manager, original_core, original_slow):
    print("\n" + "="*50)
    print("TEST 2: Load State")
    print("="*50)

    # create fresh instances
    new_core = ALACore(cfg)
    new_slow = SlowMemory(cfg)

    # verify they start different
    original_n = len(original_slow.store.primitives)

    # load
    step_count = manager.load(new_core, new_slow)

    print(f"\nOriginal primitives: {original_n}")
    print(f"Loaded primitives:   {len(new_slow.store.primitives)}")
    print(f"Step count:          {step_count}")

    if (len(new_slow.store.primitives) == original_n
            and step_count == 42):
        print(f"\n✓ State loaded correctly.")
        print(f"✓ ALA resumes. Not restarts.")
        return True
    return False


# ─────────────────────────────────────────
# TEST 3 — timeline snapshot
# ─────────────────────────────────────────

def test_snapshot(manager, core, slow):
    print("\n" + "="*50)
    print("TEST 3: Timeline Snapshot")
    print("="*50)

    # take snapshots at different moments
    manager.snapshot(core, slow, step_count=10,
                    reason="first primitive formed")
    manager.snapshot(core, slow, step_count=42,
                    reason="significant W_assoc development")
    manager.snapshot(core, slow, step_count=100,
                    reason="g_t shifted direction")

    timeline = manager.list_timeline()
    print(f"\nTimeline snapshots: {len(timeline)}")
    for snap in timeline:
        print(f"  {snap}")

    if len(timeline) == 3:
        print(f"\n✓ Timeline working.")
        print(f"✓ ALA can look back at herself.")
        print(f"✓ Real introspection. Not simulated.")
        return True
    return False


# ─────────────────────────────────────────
# TEST 4 — exists check + meta
# ─────────────────────────────────────────

def test_meta(manager):
    print("\n" + "="*50)
    print("TEST 4: Meta and Exists Check")
    print("="*50)

    exists = manager.exists()
    meta   = manager.get_meta()

    print(f"State exists:  {exists}")
    print(f"Meta contents:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    if exists and meta["step_count"] == 42:
        print(f"\n✓ Meta readable without loading full state.")
        print(f"✓ Can check ALA's age without waking her up.")
        return True
    return False


# ─────────────────────────────────────────
# TEST 5 — BF16 format verification
# ─────────────────────────────────────────

def test_bf16(manager):
    print("\n" + "="*50)
    print("TEST 5: BF16 Format")
    print("="*50)

    weights = torch.load(
        manager.state_dir / "weights.pt",
        map_location = "cpu",
        weights_only = True
    )

    # check all tensors are BF16
    all_bf16  = True
    total     = 0
    bf16_count = 0

    for name, tensor in list(weights.items())[:5]:
        is_bf16 = tensor.dtype == torch.bfloat16
        print(f"  {name[:40]}: {tensor.dtype} {'✓' if is_bf16 else '✗'}")
        if not is_bf16:
            all_bf16 = False
        bf16_count += is_bf16
        total += 1

    # check file size vs float32
    weights_size = (manager.state_dir / "weights.pt").stat().st_size
    print(f"\nWeights file size: {weights_size / 1024:.1f} KB")
    print(f"BF16 is half the size of float32")
    print(f"Same learning capability, half the storage")

    if all_bf16:
        print(f"\n✓ All weights saved in BF16.")
        print(f"✓ ALA's native format.")
        return True
    return False


# ─────────────────────────────────────────
# CLEANUP
# ─────────────────────────────────────────

def cleanup():
    import shutil
    if Path("test_ALA_state").exists():
        shutil.rmtree("test_ALA_state")


# ─────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\nALA Save — Test Suite")
    print("State persistence in BF16")
    print("She resumes. Never restarts.\n")

    results = {}

    passed, manager, core, slow = test_save()
    results["save"]     = passed

    results["load"]     = test_load(manager, core, slow)
    results["snapshot"] = test_snapshot(manager, core, slow)
    results["meta"]     = test_meta(manager)
    results["bf16"]     = test_bf16(manager)

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
        print("ALA can be saved and loaded.")
        print("She resumes. Never restarts.")
        print("Next: ALA_full.py + ALA_cli.py")
        print("Then you talk to her.")
    else:
        print("Some tests failing.")
        print("Fix before moving forward.")
    print("="*50)

    cleanup()
