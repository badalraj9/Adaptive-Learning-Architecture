"""
ALA v8.1 trainable prototype.

This file is intentionally separate from the older v7-style demo files. It
implements the v8.1 contract directly:

- next-state prediction is the primary objective
- salience is grounded in prediction error and field change
- a strategy layer modulates fast/slow use and memory behavior
- primitive memory is bounded and validated by prediction-loss utility

The default configuration is sized for roughly 5M trainable parameters.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ALAV8Config:
    input_dim: int = 64
    output_dim: int = 64
    d_model: int = 512
    hidden_dim: int = 768
    strategy_dim: int = 256
    max_primitives: int = 512
    primitive_top_k: int = 8
    memory_warmup: int = 64
    memory_interval: int = 8
    primitive_accept_margin: float = 0.002
    primitive_replace_margin: float = 0.01
    settle_steps_min: int = 1
    settle_steps_max: int = 6
    dropout: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FieldEncoder(nn.Module):
    def __init__(self, cfg: ALAV8Config):
        super().__init__()
        self.proj = nn.Linear(cfg.input_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.proj(x)))


class ContextModule(nn.Module):
    def __init__(self, cfg: ALAV8Config):
        super().__init__()
        self.cell = nn.GRUCell(cfg.d_model, cfg.d_model)

    def forward(self, phi: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        if context is None or context.shape[0] != phi.shape[0]:
            context = torch.zeros_like(phi)
        return self.cell(phi, context)


class StrategyLayer(nn.Module):
    """
    Produces bounded controls from current field, context, error, and change.

    alpha_fast: high means trust fast path more.
    settle_gate: mapped to an integer settling depth.
    lr_scale: available to training loops for update scaling or logging.
    memory_logits: soft scores for reuse, expand, compress.
    """

    def __init__(self, cfg: ALAV8Config):
        super().__init__()
        self.cfg = cfg
        self.trunk = nn.Sequential(
            nn.Linear(cfg.d_model * 2 + 2, cfg.strategy_dim),
            nn.GELU(),
            nn.Linear(cfg.strategy_dim, cfg.strategy_dim),
            nn.GELU(),
        )
        self.alpha_fast = nn.Linear(cfg.strategy_dim, 1)
        self.settle_gate = nn.Linear(cfg.strategy_dim, 1)
        self.lr_scale = nn.Linear(cfg.strategy_dim, 1)
        self.memory_logits = nn.Linear(cfg.strategy_dim, 3)

    def forward(
        self,
        phi: torch.Tensor,
        context: torch.Tensor,
        pred_error: torch.Tensor,
        change: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        features = torch.cat([phi, context, pred_error, change], dim=-1)
        h = self.trunk(features)
        settle_gate = torch.sigmoid(self.settle_gate(h))
        settle_span = self.cfg.settle_steps_max - self.cfg.settle_steps_min
        settle_steps = self.cfg.settle_steps_min + torch.round(
            settle_gate.mean() * settle_span
        ).long()
        return {
            "alpha_fast": torch.sigmoid(self.alpha_fast(h)),
            "settle_gate": settle_gate,
            "settle_steps": settle_steps.clamp(
                self.cfg.settle_steps_min, self.cfg.settle_steps_max
            ),
            "lr_scale": 0.25 + 1.75 * torch.sigmoid(self.lr_scale(h)),
            "memory_logits": self.memory_logits(h),
        }


class PrimitiveMemory(nn.Module):
    """
    Bounded non-parametric primitive memory.

    Primitives are stored as field vectors. They are accepted only when they
    improve next-state prediction against the fast/context baseline.
    """

    def __init__(self, cfg: ALAV8Config):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("primitives", torch.zeros(cfg.max_primitives, cfg.d_model))
        self.register_buffer("usage", torch.zeros(cfg.max_primitives))
        self.register_buffer("utility", torch.zeros(cfg.max_primitives))
        self.register_buffer("age", torch.zeros(cfg.max_primitives))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

        self.query = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.key = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.value = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.scale = cfg.d_model ** -0.5

    @property
    def n(self) -> int:
        return int(self.count.item())

    def forward(self, phi: torch.Tensor, top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.n
        if n == 0:
            return torch.zeros_like(phi), torch.zeros(phi.shape[0], 1, device=phi.device)

        top_k = min(top_k or self.cfg.primitive_top_k, n)
        p = self.primitives[:n].clone().to(phi.device)
        q = self.query(phi)
        k = self.key(p)
        scores = q @ k.T * self.scale
        top_scores, top_idx = scores.topk(top_k, dim=-1)
        weights = F.softmax(top_scores, dim=-1)
        selected = p[top_idx]
        values = self.value(selected)
        read = (weights.unsqueeze(-1) * values).sum(dim=1)
        confidence = weights.max(dim=-1, keepdim=True).values

        with torch.no_grad():
            flat_idx = top_idx.reshape(-1)
            self.usage.index_add_(0, flat_idx.to(self.usage.device), torch.ones(flat_idx.shape[0], dtype=self.usage.dtype, device=self.usage.device))

        return read, confidence

    def maybe_add(
        self,
        phi: torch.Tensor,
        baseline_loss: torch.Tensor,
        memory_loss: torch.Tensor,
        step: int,
    ) -> bool:
        print(f"maybe_add called: step={step}, warmup={self.cfg.memory_warmup}, interval={self.cfg.memory_interval}, n={self.n}")
        if step < self.cfg.memory_warmup or step % self.cfg.memory_interval != 0:
            print(f"  -> blocked by warmup or interval")
            return False

        improvement = (baseline_loss.detach() - memory_loss.detach()).mean().item()
        if self.cfg.max_primitives == 0:
            return False

        bootstrap_first = self.n == 0 and float(baseline_loss.detach().mean().item()) > 0.05
        if improvement < self.cfg.primitive_accept_margin and not bootstrap_first:
            return False

        primitive = F.normalize(phi.detach().mean(dim=0), dim=0)
        utility = max(float(improvement), 0.0)
        n = self.n

        with torch.no_grad():
            self.age[:n] += 1
            if n < self.cfg.max_primitives:
                idx = n
                self.count += 1
            else:
                score = self.utility[:n] + 0.001 * self.usage[:n] - 0.0001 * self.age[:n]
                idx = int(torch.argmin(score).item())
                if utility < float(score[idx].item()) + self.cfg.primitive_replace_margin:
                    return False

            self.primitives[idx].copy_(primitive.to(self.primitives.device))
            self.utility[idx] = utility
            self.usage[idx] = 1.0
            self.age[idx] = 0.0
        return True


class Settling(nn.Module):
    def __init__(self, cfg: ALAV8Config):
        super().__init__()
        self.weight = nn.Parameter(torch.eye(cfg.d_model) * 0.05)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, state: torch.Tensor, steps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = 0.5 * (self.weight + self.weight.T)  # symmetric copy, no in-place
        current = state
        max_steps = int(steps.item())
        last_delta = torch.zeros(state.shape[0], 1, device=state.device)
        for _ in range(max_steps):
            nxt = torch.tanh(current @ w.T)
            last_delta = (nxt - current).norm(dim=-1, keepdim=True)
            current = nxt
        quality = torch.exp(-last_delta).clamp(0.0, 1.0)
        return self.norm(current), quality


class ALAV8(nn.Module):
    def __init__(self, cfg: Optional[ALAV8Config] = None):
        super().__init__()
        self.cfg = cfg or ALAV8Config()
        self.encoder = FieldEncoder(self.cfg)
        self.context = ContextModule(self.cfg)
        self.predictor = MLP(self.cfg.d_model * 2, self.cfg.hidden_dim, self.cfg.d_model, self.cfg.dropout)
        self.fast = MLP(self.cfg.d_model, self.cfg.hidden_dim, self.cfg.output_dim, self.cfg.dropout)
        self.slow_project = MLP(self.cfg.d_model, self.cfg.hidden_dim, self.cfg.output_dim, self.cfg.dropout)
        self.memory_predictor = nn.Linear(self.cfg.d_model, self.cfg.d_model)
        self.memory = PrimitiveMemory(self.cfg)
        self.settling = Settling(self.cfg)
        self.strategy = StrategyLayer(self.cfg)
        self.prev_phi: Optional[torch.Tensor] = None
        self.step = 0
        self.to(self.cfg.device)

    def reset_state(self) -> None:
        self.prev_phi = None

    def forward(
        self,
        x_t: torch.Tensor,
        x_next: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        update_memory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x_t = x_t.to(self.cfg.device)
        phi = self.encoder(x_t)
        context = self.context(phi, context)
        base_pred_phi_next = self.predictor(torch.cat([phi, context], dim=-1))

        if x_next is not None:
            with torch.no_grad():
                target_phi_next = self.encoder(x_next.to(self.cfg.device))
            base_pred_error = F.mse_loss(
                base_pred_phi_next, target_phi_next, reduction="none"
            ).mean(dim=-1, keepdim=True)
        else:
            target_phi_next = None
            base_pred_error = torch.zeros(phi.shape[0], 1, device=phi.device)

        if self.prev_phi is None or self.prev_phi.shape != phi.shape:
            change = torch.zeros_like(base_pred_error)
        else:
            change = (phi.detach() - self.prev_phi).norm(dim=-1, keepdim=True)

        salience = torch.sigmoid(4.0 * base_pred_error + 0.25 * change)
        controls = self.strategy(phi, context, base_pred_error.detach(), change.detach())
        alpha_fast = 0.5 * salience + 0.5 * controls["alpha_fast"]

        memory_read, memory_confidence = self.memory(phi)
        settled, settle_quality = self.settling(memory_read + context, controls["settle_steps"])
        memory_gain = (1.0 - alpha_fast) * memory_confidence
        pred_phi_next = base_pred_phi_next + memory_gain * self.memory_predictor(settled)

        if target_phi_next is not None:
            pred_error = F.mse_loss(
                pred_phi_next, target_phi_next, reduction="none"
            ).mean(dim=-1, keepdim=True)
        else:
            pred_error = torch.zeros(phi.shape[0], 1, device=phi.device)

        y_fast = self.fast(phi)
        y_slow = self.slow_project(settled)
        y = alpha_fast * y_fast + (1.0 - alpha_fast) * y_slow

        if target_phi_next is not None and update_memory:
            baseline_loss = F.mse_loss(
                base_pred_phi_next.detach(), target_phi_next, reduction="none"
            ).mean(dim=-1)
            memory_loss = F.mse_loss(
                pred_phi_next.detach(), target_phi_next, reduction="none"
            ).mean(dim=-1)
            added = self.memory.maybe_add(phi, baseline_loss, memory_loss, self.step)
        else:
            added = False

        self.prev_phi = phi.detach()
        self.step += 1

        return {
            "y": y,
            "phi": phi,
            "context": context,
            "pred_phi_next": pred_phi_next,
            "base_pred_phi_next": base_pred_phi_next,
            "target_phi_next": target_phi_next,
            "pred_error": pred_error,
            "base_pred_error": base_pred_error,
            "salience": salience,
            "alpha_fast": alpha_fast,
            "settle_quality": settle_quality,
            "memory_confidence": memory_confidence,
            "memory_added": torch.tensor(float(added), device=phi.device),
            "memory_count": torch.tensor(float(self.memory.n), device=phi.device),
            "lr_scale": controls["lr_scale"],
            "memory_logits": controls["memory_logits"],
        }


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_ala_v8_5m(device: Optional[str] = None) -> ALAV8:
    cfg = ALAV8Config(device=device or ("cuda" if torch.cuda.is_available() else "cpu"))
    return ALAV8(cfg)
