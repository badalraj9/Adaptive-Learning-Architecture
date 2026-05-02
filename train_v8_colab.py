"""
Colab-friendly smoke training for ALA v8.1.

Run:
    python train_v8_colab.py --steps 1000 --batch-size 64

This trains next-state prediction on synthetic dynamical sequences. It is not
the final benchmark; it is the sturdy first test that proves the 5M prototype
can run, learn, save checkpoints, and report memory behavior.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from model.ALA_v8 import ALAV8Config, ALAV8, count_trainable_parameters


class SyntheticDynamics:
    def __init__(self, dim: int, device: str, noise: float = 0.02):
        self.dim = dim
        self.device = device
        self.noise = noise
        basis = torch.randn(dim, dim, device=device)
        q, _ = torch.linalg.qr(basis)
        self.transition = 0.94 * q + 0.04 * torch.eye(dim, device=device)

    def sample(self, batch_size: int):
        x = torch.randn(batch_size, self.dim, device=self.device)
        x = F.normalize(x, dim=-1)
        drift = torch.sin(x * 2.0) * 0.05
        nxt = torch.tanh(x @ self.transition.T + drift)
        nxt = nxt + self.noise * torch.randn_like(nxt)
        return x, nxt


def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    cfg = ALAV8Config(
        input_dim=args.input_dim,
        output_dim=args.input_dim,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        max_primitives=args.max_primitives,
        device=device,
    )
    model = ALAV8(cfg)
    params = count_trainable_parameters(model)
    print(f"device={device}")
    print(f"trainable_params={params:,}")

    data = SyntheticDynamics(cfg.input_dim, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    context = None
    for step in range(1, args.steps + 1):
        x, x_next = data.sample(args.batch_size)
        result = model(x, x_next=x_next, context=context, update_memory=True)
        context = result["context"].detach()

        pred_loss = F.mse_loss(result["pred_phi_next"], result["target_phi_next"])
        output_loss = F.mse_loss(result["y"], x_next)
        settle_reg = (1.0 - result["settle_quality"]).mean() * args.settle_weight
        loss = pred_loss + args.output_weight * output_loss + settle_reg

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pred_loss": float(pred_loss.detach().cpu()),
                "output_loss": float(output_loss.detach().cpu()),
                "salience": float(result["salience"].mean().detach().cpu()),
                "alpha_fast": float(result["alpha_fast"].mean().detach().cpu()),
                "settle_quality": float(result["settle_quality"].mean().detach().cpu()),
                "memory_count": int(result["memory_count"].item()),
            }
            history.append(row)
            print(
                f"step={step:05d} loss={row['loss']:.5f} "
                f"pred={row['pred_loss']:.5f} out={row['output_loss']:.5f} "
                f"mem={row['memory_count']} settle={row['settle_quality']:.3f}"
            )

        if step % args.save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "config": cfg.__dict__,
                "step": step,
                "params": params,
            }
            torch.save(ckpt, out_dir / f"ala_v8_step_{step}.pt")

    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "step": args.steps, "params": params}, out_dir / "ala_v8_final.pt")
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"saved={out_dir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--max-primitives", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--output-weight", type=float, default=0.25)
    parser.add_argument("--settle-weight", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--out-dir", type=str, default="runtime/v8_runs")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
