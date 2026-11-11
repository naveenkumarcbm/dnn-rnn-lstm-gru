"""Transformer components and experiments implemented with tensor operations."""

from __future__ import annotations

import math
import random
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from problem1.solution import RESULTS_PATH, RESULT_COLUMNS, _append_rows, load_har, set_seed


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "problem2" / "figures"
GENERATED_PATH = ROOT / "problem2" / "generated_samples.txt"


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    allowed_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention for Q,K,V shaped (batch, heads, time, head_dim)."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, v must have shape (batch, heads, time, head_dim)")
    if q.shape[:2] != k.shape[:2] or k.shape[:2] != v.shape[:2]:
        raise ValueError("batch and head dimensions must match")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("attention dimensions are incompatible")
    # Scores and probabilities are (batch, heads, query_time, key_time).
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if allowed_mask is not None:
        mask = allowed_mask.to(device=scores.device, dtype=torch.bool)
        try:
            mask = torch.broadcast_to(mask, scores.shape)
        except RuntimeError as error:
            raise ValueError("allowed_mask is not broadcastable to attention scores") from error
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout_p:
        weights = F.dropout(weights, p=dropout_p, training=training)
    return weights @ v, weights


def test_attention() -> dict[str, float]:
    torch.manual_seed(21)
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    v = torch.randn(2, 3, 5, 8)
    actual, weights = scaled_dot_product_attention(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v)
    error = (actual - expected).abs().max().item()
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)
    assert error < 1e-5, error
    causal = torch.tril(torch.ones(5, 5, dtype=torch.bool))
    _, causal_weights = scaled_dot_product_attention(q, k, v, causal)
    assert torch.count_nonzero(causal_weights.masked_select(~causal)) == 0
    # Hand-checkable 3-token example: zero Q/K makes each allowed row uniform.
    q3 = torch.zeros(1, 1, 3, 2)
    k3 = torch.zeros_like(q3)
    v3 = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]])
    allowed = torch.tril(torch.ones(3, 3, dtype=torch.bool))
    out3, w3 = scaled_dot_product_attention(q3, k3, v3, allowed)
    expected3 = torch.tensor([[[[1.0, 0.0], [0.5, 0.5], [2 / 3, 2 / 3]]]])
    assert torch.allclose(out3, expected3, atol=1e-6)
    assert torch.allclose(w3[0, 0], torch.tensor([[1., 0., 0.], [.5, .5, 0.], [1/3, 1/3, 1/3]]))
    return {"max_sdpa_error": error}


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.last_weights: torch.Tensor | None = None

    def forward(
        self, x: torch.Tensor, allowed_mask: torch.Tensor | None = None,
        causal: bool = False, return_weights: bool = False,
    ):
        # Input/output are (batch, time, d_model); projections reshape to
        # (batch, heads, time, head_dim) for independent attention heads.
        batch, steps, _ = x.shape
        if causal:
            causal_mask = torch.tril(torch.ones(steps, steps, device=x.device, dtype=torch.bool))
            allowed_mask = causal_mask if allowed_mask is None else allowed_mask & causal_mask
        def split(projection):
            return projection(x).view(batch, steps, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        context, weights = scaled_dot_product_attention(
            q, k, v, allowed_mask, self.dropout, self.training,
        )
        self.last_weights = weights.detach()
        context = context.transpose(1, 2).contiguous().view(batch, steps, self.d_model)
        output = self.o_proj(context)
        return (output, weights) if return_weights else output


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, allowed_mask: torch.Tensor | None = None,
                causal: bool = False):
        x = x + self.dropout(self.attention(self.norm1(x), allowed_mask, causal))
        return x + self.dropout(self.feed_forward(self.norm2(x)))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("encoding", pe.unsqueeze(0), persistent=True)

    def forward(self, x: torch.Tensor):
        return x + self.encoding[:, :x.shape[1]].to(dtype=x.dtype)


def encoder_block_parameter_count(d_model: int = 128, d_ff: int = 512) -> dict[str, int]:
    # Weights: four attention matrices and two feed-forward matrices.
    weights = 4 * d_model * d_model + 2 * d_model * d_ff
    biases = 4 * d_model + d_ff + d_model
    layer_norm = 4 * d_model
    model = EncoderBlock(d_model, 4, d_ff, dropout=0.0)
    return {
        "formula_weights": weights,
        "formula_biases": biases,
        "formula_layer_norm": layer_norm,
        "formula_total": weights + biases + layer_norm,
        "module_total": sum(p.numel() for p in model.parameters()),
    }


def test_encoder_block() -> dict[str, int]:
    block = EncoderBlock(32, 4, 128)
    x = torch.randn(3, 11, 32)
    y = block(x)
    assert y.shape == x.shape
    counts = encoder_block_parameter_count(32, 128)
    assert counts["formula_total"] == counts["module_total"], counts
    return counts


class HARTransformerClassifier(nn.Module):
    def __init__(self, positional: str = "sinusoidal", d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Linear(9, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.positional = positional
        if positional == "sinusoidal":
            self.position = SinusoidalPositionalEncoding(d_model, 130)
        elif positional == "learned":
            self.position = nn.Embedding(129, d_model)
        elif positional != "none":
            raise ValueError("positional must be none, sinusoidal, or learned")
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, 4 * d_model, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 6)

    def forward(self, x: torch.Tensor):
        # Add CLS at index 0; a learned or fixed encoding is applied to tokens.
        batch = x.shape[0]
        tokens = self.embedding(x)
        tokens = torch.cat([self.cls.expand(batch, -1, -1), tokens], dim=1)
        if self.positional == "sinusoidal":
            tokens = self.position(tokens)
        elif self.positional == "learned":
            indices = torch.arange(tokens.shape[1], device=tokens.device)
            tokens = tokens + self.position(indices).unsqueeze(0)
        for block in self.blocks:
            tokens = block(tokens)
        return self.classifier(self.norm(tokens[:, 0]))


class CharDataset(Dataset):
    def __init__(self, data: torch.Tensor, context: int):
        self.data = data
        self.context = context

    def __len__(self):
        return max(0, self.data.numel() - self.context)

    def __getitem__(self, index):
        return self.data[index:index + self.context], self.data[index + 1:index + self.context + 1]


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context: int = 128, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 4, dropout: float = 0.1,
                 causal: bool = True):
        super().__init__()
        self.context = context
        self.causal = causal
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = nn.Embedding(context, d_model)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, 4 * d_model, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        batch, steps = idx.shape
        if steps > self.context:
            raise ValueError(f"input length {steps} exceeds context {self.context}")
        pos = torch.arange(steps, device=idx.device)
        x = self.embedding(idx) + self.position(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, causal=self.causal)
        return self.lm_head(self.norm(x))


def load_tiny_shakespeare() -> tuple[str, dict[str, int], dict[int, str], torch.Tensor, torch.Tensor]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "input.txt"
    if not path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, path)
    text = path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {char: i for i, char in enumerate(chars)}
    itos = {i: char for char, i in stoi.items()}
    encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(encoded))
    return text, stoi, itos, encoded[:split], encoded[split:]


def train_language_model(
    context: int = 128, n_heads: int = 4, n_layers: int = 4, causal: bool = True,
    max_steps: int = 1000, batch_size: int = 64, eval_interval: int = 100,
    seed: int = 0, device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    _, stoi, itos, train_data, val_data = load_tiny_shakespeare()
    train_ds, val_ds = CharDataset(train_data, context), CharDataset(val_data, context)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    model = CausalTransformerLM(
        len(stoi), context, 128, n_heads, n_layers, dropout=0.1, causal=causal,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    iterator = iter(train_loader)
    steps, train_losses, val_losses = [], [], []
    start = time.perf_counter()
    for step in range(1, max_steps + 1):
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        xb, yb = xb.to(device), yb.to(device)
        model.train()
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % eval_interval == 0 or step == 1 or step == max_steps:
            model.eval()
            vals = []
            with torch.no_grad():
                for i, (vx, vy) in enumerate(val_loader):
                    if i >= 16:
                        break
                    logits_val = model(vx.to(device))
                    vals.append(F.cross_entropy(
                        logits_val.reshape(-1, logits_val.shape[-1]), vy.to(device).reshape(-1)
                    ).item())
            steps.append(step)
            train_losses.append(float(loss.item()))
            val_losses.append(float(np.mean(vals)))
    elapsed = time.perf_counter() - start
    return model, {"steps": steps, "train_loss": train_losses, "val_loss": val_losses,
                   "elapsed_sec": elapsed, "stoi": stoi, "itos": itos}


@torch.no_grad()
def generate(model: CausalTransformerLM, start: torch.Tensor, itos: dict[int, str],
             length: int = 500, temperature: float = 0.8, top_k: int = 20) -> str:
    if temperature <= 0 or top_k <= 0:
        raise ValueError("temperature and top_k must be positive")
    device = next(model.parameters()).device
    idx = start.to(device)
    for _ in range(length):
        context = idx[:, -model.context:]
        logits = model(context)[:, -1, :] / temperature
        k = min(top_k, logits.shape[-1])
        cutoff = torch.topk(logits, k).values[:, [-1]]
        logits = logits.masked_fill(logits < cutoff, -torch.inf)
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1)
        idx = torch.cat([idx, next_token], dim=1)
    return "".join(itos[int(i)] for i in idx[0].tolist())


def plot_lm_curve(history: dict, name: str = "causal"):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["steps"], history["train_loss"], label="train")
    ax.plot(history["steps"], history["val_loss"], label="validation")
    ax.set(xlabel="Training step", ylabel="Cross-entropy (nats)",
           title="Figure 1. Character LM learning curve")
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / f"p2_lm_learning_curve_{name}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_attention_heatmaps(model: CausalTransformerLM, sentence: str, stoi: dict[str, int]):
    tokens = torch.tensor([[stoi.get(c, 0) for c in sentence]], device=next(model.parameters()).device)
    model.eval()
    with torch.no_grad():
        _ = model(tokens)
    weights = model.blocks[0].attention.last_weights
    if weights is None:
        raise RuntimeError("attention weights were not recorded")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for head, ax in enumerate(axes):
        ax.imshow(weights[0, head, :len(sentence), :len(sentence)].cpu(), cmap="viridis")
        ax.set(title=f"Figure 2.{head + 1}. Layer 1, head {head + 1}",
               xlabel="Key character position", ylabel="Query character position")
    fig.tight_layout()
    path = FIG_DIR / "p2_attention_heads.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_heads_ablation(heads=(1, 4, 8), max_steps: int = 500):
    rows = []
    for count in heads:
        model, history = train_language_model(n_heads=count, max_steps=max_steps, seed=0)
        val = history["val_loss"][-1]
        rows.append({"heads": count, "val_loss_nats": val,
                     "bits_per_char": val / math.log(2),
                     "params": sum(p.numel() for p in model.parameters())})
    result = pd.DataFrame(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ax = result.plot(x="heads", y="val_loss_nats", marker="o", legend=False, figsize=(6, 4))
    ax.set(title="Figure 3. Number-of-heads ablation", ylabel="Validation loss (nats)")
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "p2_heads_ablation.png", dpi=160)
    plt.close(ax.figure)
    return result


def run_context_benchmark(contexts=(64, 128, 256, 512), batch_size: int = 4,
                          d_model: int = 128, n_heads: int = 4, repeats: int = 5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for context in contexts:
        model = CausalTransformerLM(65, context, d_model, n_heads, 2, dropout=0.0).to(device)
        x = torch.randint(0, 65, (batch_size, context), device=device)
        times = []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            start = time.perf_counter()
            logits = model(x)
            logits.square().mean().backward()
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            model.zero_grad(set_to_none=True)
        memory_mb = (
            torch.cuda.max_memory_allocated() / (1024 ** 2)
            if device == "cuda" else float("nan")
        )
        rows.append({"context": context, "seconds": float(np.mean(times)),
                     "peak_memory_mb": memory_mb, "device": device})
        del logits, model
    result = pd.DataFrame(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(result.context, result.seconds, marker="o", label="time")
    ax.set(xlabel="Context length", ylabel="Seconds per forward/backward pass",
           title="Figure 4. Context-length runtime and memory")
    ax2 = ax.twinx()
    ax2.plot(result.context, result.peak_memory_mb, marker="s", color="tab:red", label="peak memory")
    ax2.set_ylabel("Peak allocated memory (MiB)")
    fig.tight_layout()
    path = FIG_DIR / "p2_context_scaling.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return result


def train_har_transformer(positional: str, seed: int = 0, epochs: int = 8,
                          batch_size: int = 128, device: str | None = None):
    from problem1.solution import _loader
    from sklearn.metrics import accuracy_score, f1_score
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    train, val, test, _ = load_har(seed)
    model = HARTransformerClassifier(positional=positional).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    train_loader = _loader(train, batch_size, True)
    val_loader = _loader(val, batch_size, False)
    epoch_times, history = [], {"train_loss": [], "val_loss": []}
    for _ in range(epochs):
        begin = time.perf_counter()
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        if device == "cuda":
            torch.cuda.synchronize()
        epoch_times.append(time.perf_counter() - begin)
        history["train_loss"].append(float(np.mean(losses)))
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vals.append(criterion(model(xb.to(device)), yb.to(device)).item())
        history["val_loss"].append(float(np.mean(vals)))
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for xb, yb in _loader(test, batch_size, False):
            preds.extend(model(xb.to(device)).argmax(-1).cpu().tolist())
            ys.extend(yb.tolist())
    row = {
        "problem": "P2-HAR", "model": f"Transformer-{positional}", "seed": seed,
        "seq_len": 128, "test_acc": accuracy_score(ys, preds),
        "macro_f1": f1_score(ys, preds, average="macro"),
        "test_mse": "", "params": sum(p.numel() for p in model.parameters()),
        "sec_per_epoch": float(np.mean(epoch_times)), "val_loss_nats": "",
        "bits_per_char": "", "notes": f"position={positional}; epochs={epochs}",
    }
    _append_rows([row])
    return row, model, history


def record_language_result(history: dict, model: nn.Module, seed: int = 0, notes: str = ""):
    val = float(history["val_loss"][-1])
    row = {
        "problem": "P2-LM", "model": "Transformer-LM", "seed": seed,
        "seq_len": getattr(model, "context", ""), "test_acc": "", "macro_f1": "",
        "test_mse": "", "params": sum(p.numel() for p in model.parameters()),
        "sec_per_epoch": "", "val_loss_nats": val,
        "bits_per_char": val / math.log(2), "notes": notes,
    }
    _append_rows([row])
    return row
