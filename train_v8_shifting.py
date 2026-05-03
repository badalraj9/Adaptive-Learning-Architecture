import argparse
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from model.ALA_v8 import ALAV8, ALAV8Config, count_trainable_parameters


class ShiftingDynamics:
    def __init__(self, n_regimes=3, shift_every=500, dim=64, noise_std=0.02):
        self.n_regimes = n_regimes
        self.shift_every = shift_every
        self.dim = dim
        self.noise_std = noise_std
        self.current_regime = 0
        self.regimes = [self._generate_orthogonal_matrix(dim) for _ in range(n_regimes)]
        print(f"ShiftingDynamics: {n_regimes} regimes, shift every {shift_every} steps, dim={dim}")

    def _generate_orthogonal_matrix(self, dim):
        q, _ = torch.linalg.qr(torch.randn(dim, dim))
        return q * 0.94

    def get_regime(self, step):
        return step // self.shift_every

    def regime_at(self, step):
        return step // self.shift_every

    def sample(self, batch_size, step):
        regime = self.regime_at(step)
        if regime > self.current_regime:
            print(f"Regime shift: {self.current_regime} -> {regime}")
            self.current_regime = regime
        x = torch.randn(batch_size, self.dim)
        x_next = x @ self.regimes[regime % self.n_regimes].T
        x_next = x_next + torch.randn_like(x_next) * self.noise_std
        return x, x_next


def train_ala(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cpu" if args.cpu else "cuda")
    dynamics = ShiftingDynamics(
        n_regimes=args.n_regimes,
        shift_every=args.shift_every,
        dim=args.dim,
        noise_std=0.02
    )

    cfg = ALAV8Config(
        input_dim=args.dim,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        max_primitives=args.max_primitives,
        memory_warmup=8,
        memory_interval=1,
    )
    model = ALAV8(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    return model, opt, dynamics, cfg, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--shift-every", type=int, default=500)
    parser.add_argument("--n-regimes", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--max-primitives", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recovery-threshold", type=float, default=0.02)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else "cuda")
    print(f"Device: {device}")

    os.makedirs("runtime/v8_shifting", exist_ok=True)

    print("\n=== Training ALA with memory ===")
    model, opt, dynamics, cfg, device = train_ala(args)
    print(f"ALA parameters: {count_trainable_parameters(model)}")
    print(f"Max primitives: {cfg.max_primitives}")

    print("\n=== Training baseline (no memory) ===")
    args_baseline = argparse.Namespace(**vars(args))
    args_baseline.max_primitives = 0
    model_b, opt_b, dynamics_b, cfg_b, device_b = train_ala(args_baseline)
    print(f"Baseline parameters: {count_trainable_parameters(model_b)}")

    metrics = {
        "steps": [],
        "ala_loss": [],
        "baseline_loss": [],
        "memory_count": [],
        "settle_quality": [],
        "regime": [],
        "ala_recovery_steps": [],
        "baseline_recovery_steps": [],
    }

    prev_regime = -1
    seen_regime_mods = set()
    ala_recovering = False
    baseline_recovering = False
    ala_recovery_start = None
    baseline_recovery_start = None

    for step in range(args.steps):
        x, x_next = dynamics.sample(args.batch_size, step)
        x, x_next = x.to(device), x_next.to(device)

        output = model(x, x_next, update_memory=True)
        pred_loss = F.mse_loss(output["y"], x_next).mean()
        loss = pred_loss.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        x_b, x_next_b = dynamics_b.sample(args.batch_size, step)
        x_b, x_next_b = x_b.to(device_b), x_next_b.to(device_b)
        output_b = model_b(x_b, x_next_b, update_memory=False)
        pred_loss_b = F.mse_loss(output_b["y"], x_next_b).mean()
        loss_b = pred_loss_b.mean()
        opt_b.zero_grad()
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(model_b.parameters(), 1.0)
        opt_b.step()

        regime = dynamics.regime_at(step)
        regime_mod = regime % args.n_regimes
        is_shift = regime != prev_regime
        if is_shift and prev_regime >= 0:
            if regime_mod in seen_regime_mods:
                print(f"[Step {step}] Returning to seen regime {regime_mod}")
                ala_recovering = True
                baseline_recovering = True
                ala_recovery_start = step
                baseline_recovery_start = step
            seen_regime_mods.add(regime_mod)
        prev_regime = regime

        if ala_recovering and pred_loss.item() < args.recovery_threshold:
            metrics["ala_recovery_steps"].append(step - ala_recovery_start)
            ala_recovering = False

        if baseline_recovering and pred_loss_b.item() < args.recovery_threshold:
            metrics["baseline_recovery_steps"].append(step - baseline_recovery_start)
            baseline_recovering = False

        if step % 25 == 0:
            metrics["steps"].append(step)
            metrics["ala_loss"].append(pred_loss.item())
            metrics["baseline_loss"].append(pred_loss_b.item())
            metrics["memory_count"].append(output["memory_count"].item())
            metrics["settle_quality"].append(output["settle_quality"].mean().item())
            metrics["regime"].append(regime)
            print(f"Step {step:5d} | Regime {regime} | ALA Loss: {pred_loss.item():.4f} | Baseline: {pred_loss_b.item():.4f} | Mem: {int(output['memory_count'].item())} | SQ: {output['settle_quality'].mean().item():.3f}")

    print("\n=== Summary ===")
    ala_avg = np.mean(metrics["ala_recovery_steps"]) if metrics["ala_recovery_steps"] else float('inf')
    base_avg = np.mean(metrics["baseline_recovery_steps"]) if metrics["baseline_recovery_steps"] else float('inf')
    print(f"ALA recovery steps: {metrics['ala_recovery_steps']}")
    print(f"Baseline recovery steps: {metrics['baseline_recovery_steps']}")
    print(f"Average ALA recovery: {ala_avg:.2f}")
    print(f"Average baseline recovery: {base_avg:.2f}")
    print(f"Memory primitives at end: {model.memory.n}")

    with open("runtime/v8_shifting/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved metrics to runtime/v8_shifting/metrics.json")


if __name__ == "__main__":
    main()