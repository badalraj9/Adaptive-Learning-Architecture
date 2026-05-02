import argparse
import os
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ALA_v8 import ALAV8, ALAV8Config, count_trainable_parameters


class CharTokenizer:
    def __init__(self, text: Optional[str] = None):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        if text:
            self.build_vocab(text)

    def build_vocab(self, text: str):
        chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for ch, i in self.char_to_idx.items()}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(ch, 0) for ch in text]

    def decode(self, indices: List[int]) -> str:
        return ''.join([self.idx_to_char.get(i, '') for i in indices])

    def save(self, path: str):
        data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size
        }
        torch.save(data, path)

    def load(self, path: str):
        data = torch.load(path, map_location='cpu')
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.vocab_size = data['vocab_size']


def download_tinyshakespeare(path: str = "data/tinyshakespeare.txt"):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return f.read()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"Downloading TinyShakespeare...")
    response = requests.get(url)
    text = response.text
    with open(path, 'w') as f:
        f.write(text)
    print(f"Downloaded and saved to {path}")
    return text


class ALALanguageModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, d_model: int = 256,
                 hidden_dim: int = 512, max_primitives: int = 256, device: str = "cpu"):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        cfg = ALAV8Config(
            input_dim=embed_dim,
            output_dim=embed_dim,
            d_model=d_model,
            hidden_dim=hidden_dim,
            max_primitives=max_primitives,
            device=device
        )
        self.ala = ALAV8(cfg)
        self.embed_scale = embed_dim ** 0.5

    def forward(self, token_ids: torch.Tensor, update_memory: bool = True) -> Dict[str, torch.Tensor]:
        x = self.embedding(token_ids) * self.embed_scale

        batch_size, seq_len, _ = x.shape
        x = x.reshape(-1, self.embed_dim)

        outputs = []
        contexts = []
        memory_counts = []
        settle_qualities = []
        alpha_fasts = []
        saliences = []

        context = None
        for i in range(seq_len):
            x_t = x[i::seq_len]

            if i < seq_len - 1:
                x_next = x[i+1::seq_len]
            else:
                x_next = None

            result = self.ala(x_t, x_next=x_next, context=context, update_memory=update_memory)
            context = result['context'].detach()

            outputs.append(result['y'])
            contexts.append(result['context'])
            memory_counts.append(result['memory_count'])
            settle_qualities.append(result['settle_quality'])
            alpha_fasts.append(result['alpha_fast'])
            saliences.append(result['salience'])

        y = torch.stack(outputs, dim=1)
        logits = self.lm_head(y)

        return {
            'logits': logits,
            'context': context,
            'memory_count': torch.stack(memory_counts).mean(),
            'settle_quality': torch.stack(settle_qualities).mean(),
            'alpha_fast': torch.stack(alpha_fasts).mean(),
            'salience': torch.stack(saliences).mean(),
        }

    def generate(self, tokenizer: CharTokenizer, prompt: str = "", max_new_tokens: int = 200,
                 temperature: float = 1.0, top_k: int = 50) -> str:
        self.eval()
        with torch.no_grad():
            if prompt:
                tokens = tokenizer.encode(prompt)
            else:
                tokens = [np.random.randint(0, self.vocab_size)]

            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
            generated = tokens[0].tolist()

            context = None
            for _ in range(max_new_tokens):
                x = self.embedding(tokens[:, -1:]) * self.embed_scale
                x = x.squeeze(1)

                result = self.ala(x, x_next=None, context=context, update_memory=False)
                context = result['context'].detach()

                logits = self.lm_head(result['y']) / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('inf')
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated.append(next_token.item())
                tokens = next_token.T

            return tokenizer.decode(generated)


def get_batches(text: str, tokenizer: CharTokenizer, seq_len: int, batch_size: int,
               device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer.encode(text)
    total_tokens = len(encoded)

    num_batches = (total_tokens - 1) // (seq_len * batch_size)
    if num_batches == 0:
        raise ValueError(f"Text too short for given seq_len={seq_len} and batch_size={batch_size}")

    trimmed_len = num_batches * seq_len * batch_size
    inputs = torch.tensor(encoded[:trimmed_len], dtype=torch.long, device=device)
    targets = torch.tensor(encoded[1:trimmed_len+1], dtype=torch.long, device=device)

    inputs = inputs.reshape(batch_size, -1, seq_len)
    targets = targets.reshape(batch_size, -1, seq_len)

    return inputs, targets


def train(args):
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading TinyShakespeare...")
    text = download_tinyshakespeare()
    print(f"Dataset length: {len(text)} characters")

    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("Building model...")
    model = ALALanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        d_model=args.d_model,
        hidden_dim=args.hidden_dim,
        max_primitives=args.max_primitives,
        device=device
    ).to(device)

    total_params = count_trainable_parameters(model)
    print(f"Total trainable parameters: {total_params:,}")
    if total_params > 5_000_000:
        print(f"Warning: Parameter count ({total_params:,}) exceeds 5M target")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    inputs, targets = get_batches(text, tokenizer, args.seq_len, args.batch_size, device)
    num_batches = inputs.shape[1]
    print(f"Training with {num_batches} batches per epoch")

    checkpoint_dir = "runtime/v8_language"
    os.makedirs(checkpoint_dir, exist_ok=True)

    step = 0
    for epoch in range(1000):
        model.train()
        model.ala.reset_state()

        epoch_loss = 0
        batch_count = 0

        for batch_idx in range(num_batches):
            batch_inputs = inputs[:, batch_idx, :]
            batch_targets = targets[:, batch_idx, :]

            optimizer.zero_grad()

            result = model(batch_inputs, update_memory=True)
            logits = result['logits']

            loss = F.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size),
                batch_targets.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            step += 1

            if step % 100 == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | "
                      f"Memory: {result['memory_count'].item():.0f} | "
                      f"Settle Quality: {result['settle_quality'].item():.4f} | "
                      f"Alpha Fast: {result['alpha_fast'].item():.4f}")

            if step % 500 == 0:
                sample = model.generate(
                    tokenizer,
                    prompt=text[:50],
                    max_new_tokens=200,
                    temperature=args.temperature
                )
                print(f"\n--- Sample (step {step}) ---")
                print(sample[:200])
                print("------------------------\n")

            if step % 1000 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'tokenizer': tokenizer.char_to_idx,
                }, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

            if step >= args.steps:
                print(f"Reached {args.steps} steps. Training complete.")
                return

        if batch_count > 0:
            print(f"Epoch {epoch+1} | Avg Loss: {epoch_loss/batch_count:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ALA v8 Language Model")
    parser.add_argument("--steps", type=int, default=10000, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--cpu", action="store_true", help="Use CPU")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--embed-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--max-primitives", type=int, default=256, help="Max primitives")
    args = parser.parse_args()

    train(args)
