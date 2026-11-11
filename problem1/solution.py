"""Reproducible recurrent baselines and experiments for Problem 1."""

from __future__ import annotations

import random
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "problem1" / "figures"
RESULTS_PATH = ROOT / "results.csv"
RESULT_COLUMNS = [
    "problem", "model", "seed", "seq_len", "test_acc", "macro_f1",
    "test_mse", "params", "sec_per_epoch", "val_loss_nats",
    "bits_per_char", "notes",
]
CHANNELS = [
    "body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x",
    "body_gyro_y", "body_gyro_z", "total_acc_x", "total_acc_y",
    "total_acc_z",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_adding(n: int, T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return n adding examples with exactly one marker in each half."""
    if n <= 0 or T < 2:
        raise ValueError("n must be positive and T must be at least 2")
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((n, T))
    i1 = rng.integers(0, T // 2, n)
    i2 = rng.integers(T // 2, T, n)
    rows = np.arange(n)
    x[rows, i1, 1] = 1
    x[rows, i2, 1] = 1
    y = x[rows, i1, 0] + x[rows, i2, 0]
    return x, y.astype(np.float32)


def test_make_adding() -> None:
    x, y = make_adding(128, 50, 7)
    assert x.shape == (128, 50, 2) and y.shape == (128,)
    markers = x[:, :, 1]
    assert np.all(markers.sum(axis=1) == 2)
    assert np.all(markers[:, :25].sum(axis=1) == 1)
    assert np.all(markers[:, 25:].sum(axis=1) == 1)
    np.testing.assert_allclose((x[:, :, 0] * markers).sum(axis=1), y)


class MyRNNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.input_linear(x) + self.hidden_linear(h))


class MyLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size, 4 * hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        # Gate tensors are (batch, 4 * hidden); chunks are i, f, g, o.
        i, f, g, o = (self.input_linear(x) + self.hidden_linear(h)).chunk(4, dim=-1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next


class MyGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_linear = nn.Linear(input_size, 3 * hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, 3 * hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # PyTorch gate order is reset, update, new; reset gates the recurrent candidate.
        ir, iz, inn = self.input_linear(x).chunk(3, dim=-1)
        hr, hz, hn = self.hidden_linear(h).chunk(3, dim=-1)
        reset = torch.sigmoid(ir + hr)
        update = torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + reset * hn)
        return candidate + update * (h - candidate)


def _copy_linear_cell_weights(source: nn.Module, target: nn.Module, gates: int) -> None:
    with torch.no_grad():
        target.input_linear.weight.copy_(source.weight_ih_l0)
        target.hidden_linear.weight.copy_(source.weight_hh_l0)
        target.input_linear.bias.copy_(source.bias_ih_l0)
        target.hidden_linear.bias.copy_(source.bias_hh_l0)
        if gates == 4:
            target.hidden_linear.bias.add_(source.bias_hh_l0)
        elif gates == 3:
            target.hidden_linear.bias.copy_(source.bias_hh_l0)


def _copy_lstm_weights(source: nn.LSTM, target: MyLSTMCell) -> None:
    with torch.no_grad():
        target.input_linear.weight.copy_(source.weight_ih_l0)
        target.hidden_linear.weight.copy_(source.weight_hh_l0)
        target.input_linear.bias.copy_(source.bias_ih_l0)
        target.hidden_linear.bias.copy_(source.bias_hh_l0)


def test_cells_match_torch() -> dict[str, float]:
    """Compare single-layer batch-first sequence results against PyTorch."""
    torch.manual_seed(19)
    batch, steps, d, h = 4, 7, 9, 13
    x = torch.randn(batch, steps, d)
    max_errors: dict[str, float] = {}

    ref_rnn = nn.RNN(d, h, batch_first=True)
    ours_rnn = MyRNNCell(d, h)
    with torch.no_grad():
        ours_rnn.input_linear.weight.copy_(ref_rnn.weight_ih_l0)
        ours_rnn.hidden_linear.weight.copy_(ref_rnn.weight_hh_l0)
        ours_rnn.input_linear.bias.copy_(ref_rnn.bias_ih_l0)
        ours_rnn.hidden_linear.bias.copy_(ref_rnn.bias_hh_l0)
        hr = torch.zeros(batch, h)
        out = []
        for t in range(steps):
            hr = ours_rnn(x[:, t], hr)
            out.append(hr)
        actual = torch.stack(out, dim=1)
        expected, _ = ref_rnn(x)
        max_errors["RNN"] = (actual - expected).abs().max().item()

    ref_lstm = nn.LSTM(d, h, batch_first=True)
    ours_lstm = MyLSTMCell(d, h)
    _copy_lstm_weights(ref_lstm, ours_lstm)
    with torch.no_grad():
        h0 = torch.zeros(batch, h)
        c0 = torch.zeros(batch, h)
        out = []
        for t in range(steps):
            h0, c0 = ours_lstm(x[:, t], (h0, c0))
            out.append(h0)
        actual = torch.stack(out, dim=1)
        expected, _ = ref_lstm(x)
        max_errors["LSTM"] = (actual - expected).abs().max().item()

    ref_gru = nn.GRU(d, h, batch_first=True)
    ours_gru = MyGRUCell(d, h)
    with torch.no_grad():
        ours_gru.input_linear.weight.copy_(ref_gru.weight_ih_l0)
        ours_gru.hidden_linear.weight.copy_(ref_gru.weight_hh_l0)
        ours_gru.input_linear.bias.copy_(ref_gru.bias_ih_l0)
        ours_gru.hidden_linear.bias.copy_(ref_gru.bias_hh_l0)
        hg = torch.zeros(batch, h)
        out = []
        for t in range(steps):
            hg = ours_gru(x[:, t], hg)
            out.append(hg)
        actual = torch.stack(out, dim=1)
        expected, _ = ref_gru(x)
        max_errors["GRU"] = (actual - expected).abs().max().item()
    assert all(error < 1e-5 for error in max_errors.values()), max_errors
    return max_errors


def hand_parameter_counts(input_size: int = 9, hidden_size: int = 64) -> dict[str, int]:
    """Counts for one bias per gate affine transform, then PyTorch's two biases."""
    d, h = input_size, hidden_size
    return {
        "RNN_hand": h * d + h * h + h,
        "LSTM_hand": 4 * (h * d + h * h + h),
        "GRU_hand": 3 * (h * d + h * h + h),
        "RNN_torch": h * d + h * h + 2 * h,
        "LSTM_torch": 4 * (h * d + h * h + 2 * h),
        "GRU_torch": 3 * (h * d + h * h + 2 * h),
    }


class RecurrentClassifier(nn.Module):
    def __init__(self, kind: str, input_size: int = 9, hidden_size: int = 64, classes: int = 6):
        super().__init__()
        self.kind = kind
        recurrent_type = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.recurrent = recurrent_type(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, classes)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Built-in recurrent layers are used for the fair, efficient model comparison;
        # their custom-cell equivalents are independently verified above.
        outputs, _ = self.recurrent(x)
        return self.classifier(outputs[:, -1])


class AddingRegressor(nn.Module):
    def __init__(self, kind: str, hidden_size: int = 64):
        super().__init__()
        self.kind = kind
        recurrent_type = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.recurrent = recurrent_type(2, hidden_size, batch_first=True)
        self.readout = nn.Linear(hidden_size, 1)
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each step has (value, marker); predict from the final recurrent state.
        outputs, _ = self.recurrent(x)
        return self.readout(outputs[:, -1]).squeeze(-1)


def _append_rows(rows: list[dict], path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def load_har(seed: int = 0, download: bool = True):
    """Load official train/test windows and create a subject-wise validation set."""
    base = DATA_DIR / "UCI HAR Dataset"
    if not base.exists():
        if not download:
            raise FileNotFoundError(f"Place the extracted UCI HAR Dataset under {DATA_DIR}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        archive = DATA_DIR / "har.zip"
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
            archive,
        )
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(DATA_DIR)
        nested = DATA_DIR / "UCI HAR Dataset.zip"
        if nested.exists() and not base.exists():
            with zipfile.ZipFile(nested) as zf:
                zf.extractall(DATA_DIR)
    def load_split(split: str):
        folder = base / split / "Inertial Signals"
        arrays = [np.loadtxt(folder / f"{name}_{split}.txt") for name in CHANNELS]
        x = np.stack(arrays, axis=-1).astype(np.float32)
        y = np.loadtxt(base / split / f"y_{split}.txt", dtype=np.int64) - 1
        subjects = np.loadtxt(base / split / f"subject_{split}.txt", dtype=np.int64)
        return x, y, subjects

    x_train_all, y_train_all, subjects = load_split("train")
    x_test, y_test, _ = load_split("test")
    unique = np.unique(subjects)
    val_subjects = np.random.default_rng(seed).choice(unique, 4, replace=False)
    val_mask = np.isin(subjects, val_subjects)
    x_train, y_train = x_train_all[~val_mask], y_train_all[~val_mask]
    x_val, y_val = x_train_all[val_mask], y_train_all[val_mask]
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.maximum(std, 1e-6)
    transform = lambda arr: ((arr - mean) / std).astype(np.float32)
    return (
        (transform(x_train), y_train),
        (transform(x_val), y_val),
        (transform(x_test), y_test),
        val_subjects,
    )


def plot_har_class_balance(y_train: np.ndarray, y_test: np.ndarray) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = ["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"]
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(6)
    ax.bar(idx - 0.18, np.bincount(y_train, minlength=6), 0.36, label="train")
    ax.bar(idx + 0.18, np.bincount(y_test, minlength=6), 0.36, label="test")
    ax.set_xticks(idx, labels, rotation=20, ha="right")
    ax.set_ylabel("Windows")
    ax.set_title("Figure 1. UCI HAR class balance")
    ax.legend()
    fig.tight_layout()
    path = FIG_DIR / "p1_class_balance.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _loader(pair, batch_size: int, shuffle: bool):
    x, y = pair
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size, shuffle=shuffle,
    )


def train_har(seed: int, epochs: int = 8, batch_size: int = 128, hidden_size: int = 64,
              device: str | None = None) -> tuple[pd.DataFrame, dict]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train, val, test, _ = load_har(seed)
    rows, histories = [], {}
    test_loader = _loader(test, batch_size, False)
    for kind in ("RNN", "LSTM", "GRU"):
        set_seed(seed)
        model = RecurrentClassifier(kind, hidden_size=hidden_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        train_loader = _loader(train, batch_size, True)
        val_loader = _loader(val, batch_size, False)
        history = {"train_loss": [], "val_loss": []}
        epoch_times = []
        for _ in range(epochs):
            start = time.perf_counter()
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
            epoch_times.append(time.perf_counter() - start)
            history["train_loss"].append(float(np.mean(losses)))
            model.eval()
            vals = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    vals.append(criterion(model(xb.to(device)), yb.to(device)).item())
            history["val_loss"].append(float(np.mean(vals)))
        model.eval()
        predictions, truths = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                predictions.extend(model(xb.to(device)).argmax(-1).cpu().tolist())
                truths.extend(yb.tolist())
        rows.append({
            "problem": "P1-HAR", "model": kind, "seed": seed, "seq_len": 128,
            "test_acc": accuracy_score(truths, predictions),
            "macro_f1": f1_score(truths, predictions, average="macro"),
            "test_mse": "", "params": sum(p.numel() for p in model.parameters()),
            "sec_per_epoch": float(np.mean(epoch_times)), "val_loss_nats": "",
            "bits_per_char": "", "notes": f"hidden={hidden_size}; epochs={epochs}",
        })
        histories[kind] = {
            "history": history, "model": model,
            "confusion": confusion_matrix(truths, predictions),
            "accuracy": accuracy_score(truths, predictions),
        }
    _append_rows(rows)
    return pd.DataFrame(rows), histories


def train_adding(kind: str, T: int, seed: int, epochs: int = 20, n_train: int = 10_000,
                 n_val: int = 1_000, n_test: int = 1_000, batch_size: int = 128,
                 hidden_size: int = 64, clip: float | None = 1.0,
                 device: str | None = None) -> tuple[dict, list[float]]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train = make_adding(n_train, T, 0)
    val = make_adding(n_val, T, 1)
    test = make_adding(n_test, T, 2)
    set_seed(seed)
    model = AddingRegressor(kind, hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_loader = _loader(train, batch_size, True)
    val_loader = _loader(val, batch_size, False)
    losses = []
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vals.append(criterion(model(xb.to(device)), yb.to(device)).item())
        losses.append(float(np.mean(vals)))
    elapsed = time.perf_counter() - start
    model.eval()
    test_loader = _loader(test, batch_size, False)
    errors = []
    with torch.no_grad():
        for xb, yb in test_loader:
            errors.extend(((model(xb.to(device)) - yb.to(device)) ** 2).cpu().tolist())
    row = {
        "problem": "P1-adding", "model": kind, "seed": seed, "seq_len": T,
        "test_acc": "", "macro_f1": "", "test_mse": float(np.mean(errors)),
        "params": sum(p.numel() for p in model.parameters()),
        "sec_per_epoch": elapsed / epochs, "val_loss_nats": "", "bits_per_char": "",
        "notes": f"hidden={hidden_size}; epochs={epochs}; clip={clip}",
    }
    return row, losses


def run_adding_sweep(epochs: int = 20, hidden_size: int = 64, device: str | None = None):
    rows = []
    histories = {}
    for T in (50, 100, 200, 400):
        for kind in ("RNN", "LSTM", "GRU"):
            for seed in (0, 1, 2):
                row, loss = train_adding(kind, T, seed, epochs=epochs, hidden_size=hidden_size, device=device)
                rows.append(row)
                histories[(T, kind, seed)] = loss
                _append_rows([row])
    return pd.DataFrame(rows), histories


def plot_adding_results(results: pd.DataFrame, histories: dict, out_dir: Path = FIG_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for kind in ("RNN", "LSTM", "GRU"):
        series = np.array([histories[(200, kind, seed)] for seed in (0, 1, 2)])
        ax.plot(np.arange(1, series.shape[1] + 1), series.mean(0), label=kind)
    ax.axhline(1 / 6, color="black", linestyle="--", label="constant baseline ≈ 0.167")
    ax.set(xlabel="Epoch", ylabel="Validation MSE", title="Figure 2. Adding problem learning curves (T=200)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "p1_adding_t200_learning_curves.png", dpi=160)
    plt.close(fig)
    grouped = results.groupby(["model", "seq_len"])["test_mse"].mean().unstack(0)
    ax = grouped.plot(marker="o", figsize=(7, 4), logy=True)
    ax.axhline(1 / 6, color="black", linestyle="--")
    ax.set(xlabel="Sequence length T", ylabel="Mean test MSE (log scale)",
           title="Figure 3. Adding problem error against sequence length")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "p1_adding_mse_by_length.png", dpi=160)
    plt.close(ax.figure)
    return results.assign(solved=results["test_mse"].astype(float) < 0.05).groupby(
        ["model", "seq_len"]
    )["solved"].sum().unstack(0)


def plot_gradient_norm(seed: int = 0, T: int = 200, hidden_size: int = 64):
    set_seed(seed)
    x, y = make_adding(128, T, seed)
    cell = MyLSTMCell(2, hidden_size)
    readout = nn.Linear(hidden_size, 1)
    inputs = torch.from_numpy(x)
    batch = inputs.shape[0]
    h = inputs.new_zeros(batch, hidden_size)
    c = inputs.new_zeros(batch, hidden_size)
    states = []
    for t in range(T):
        h, c = cell(inputs[:, t], (h, c))
        states.append(h)
    pred = readout(h).squeeze(-1)
    loss = nn.MSELoss()(pred, torch.from_numpy(y))
    state_grads = torch.autograd.grad(loss, states)
    grad = np.asarray([g.norm().item() for g in state_grads])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(T), grad)
    ax.set(xlabel="Time step", ylabel="Gradient norm of hidden state",
           title="Figure 4. Backpropagated hidden-state gradient by time step")
    fig.tight_layout()
    path = FIG_DIR / "p1_gradient_norm.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_hidden_size_sweep(sizes=(16, 32, 64, 128), epochs: int = 8):
    rows = []
    for size in sizes:
        row, _ = train_adding("LSTM", 200, 0, epochs=epochs, hidden_size=size)
        rows.append({"hidden_size": size, "test_mse": row["test_mse"], "params": row["params"]})
    result = pd.DataFrame(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ax = result.plot(x="hidden_size", y="test_mse", marker="o", logx=True, logy=True,
                     secondary_y="params", figsize=(7, 4))
    ax.set_title("Figure 5. Hidden-size sweep on adding problem (T=200)")
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "p1_hidden_size_sweep.png", dpi=160)
    plt.close(ax.figure)
    return result


def run_clipping_ablation(epochs: int = 20):
    """Task 5(c): isolate gradient clipping on a fixed LSTM adding setup."""
    rows = []
    for clipping in (None, 1.0):
        row, _ = train_adding("LSTM", 200, 0, epochs=epochs, clip=clipping)
        row["notes"] += "; ablation=gradient clipping"
        rows.append(row)
    result = pd.DataFrame(rows)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ax = result.assign(condition=["No clipping", "Clip norm=1"]).plot(
        x="condition", y="test_mse", kind="bar", legend=False, figsize=(6, 4)
    )
    ax.axhline(1 / 6, color="black", linestyle="--")
    ax.set(ylabel="Test MSE", title="Figure 6. Gradient-clipping ablation (LSTM, T=200)")
    ax.figure.tight_layout()
    ax.figure.savefig(FIG_DIR / "p1_clipping_ablation.png", dpi=160)
    plt.close(ax.figure)
    return result


def plot_har_confusion(history: dict):
    best = max(history.values(), key=lambda item: item["accuracy"])
    labels = ["Walk", "Upstairs", "Downstairs", "Sit", "Stand", "Lay"]
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(best["confusion"], cmap="Blues")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted", ylabel="True", title="Figure 7. Best HAR model confusion matrix")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path = FIG_DIR / "p1_best_confusion_matrix.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
