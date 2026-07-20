# Manual Bidirectional RNN (biRNN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `biRNN/` — a bidirectional vanilla RNN whose forward/backward recurrence is implemented by hand (via `nn.RNNCell` loops) rather than `nn.RNN(bidirectional=True)`, as the scaffold the future modular RNN will extend with restricted connectivity. Verify it's numerically identical to `RNN/model.py`'s `SimpleRNN`, then run the same MNIST and CartPole tests against it.

**Architecture:** `BidirectionalRNN` mirrors `SimpleRNN`'s interface exactly (`input_size, hidden_size, output_size, output_mode`), but runs two independent `nn.RNNCell`s in explicit Python loops (forward t=0→T-1, backward t=T-1→0) instead of one fused `nn.RNN` call.

**Tech Stack:** Same `.venv` as `RNN/` (PyTorch 2.13, torchvision 0.28, Gymnasium 1.3, pytest) — no new dependencies.

## Global Constraints

- Use the existing `.venv` at repo root — no new dependencies, no new `requirements.txt`.
- `BidirectionalRNN`'s public interface (constructor args, `forward` input/output shapes per `output_mode`) must exactly match `RNN/model.py`'s `SimpleRNN`, so the two are drop-in comparable.
- Must include a parity test proving the manual implementation matches `nn.RNN(bidirectional=True)` numerically — this is the entire point of the exercise (correctness of the scaffold the modular RNN will build on).
- MNIST/CartPole test scripts use the same hyperparameters and accuracy thresholds (>90%) as `RNN/`'s versions, for direct comparability.
- Each folder (`RNN/`, `biRNN/`) stays independently runnable — no shared base class; small cross-folder reuse (e.g. `heuristic_policy.py`) goes through explicit file-based imports, not package restructuring.

---

### Task 1: BidirectionalRNN model with shape + parity tests

**Files:**
- Create: `biRNN/model.py`
- Test: `biRNN/test_model.py`

**Interfaces:**
- Consumes: `RNN/model.py`'s `SimpleRNN` (read-only, via `importlib` file-loading — not a package import, to avoid both folders' `model.py` colliding under the same module name).
- Produces:
  - `get_device() -> torch.device`
  - `BidirectionalRNN(input_size: int, hidden_size: int, output_size: int, output_mode: str = "last")` — an `nn.Module` with the same shape contract as `RNN/model.py`'s `SimpleRNN`:
    - `forward(x)` where `x` is `(batch, seq_len, input_size)`.
    - `output_mode="last"` → `(batch, output_size)`.
    - `output_mode="all"` → `(batch, seq_len, output_size)`.
    - Invalid `output_mode` raises `ValueError` at construction.
    - Internal attributes `input_proj`, `fwd_cell`, `bwd_cell`, `output_proj` (named exactly this — Task 1's own parity test copies weights into them by name).

  These exact names are used by Task 2 (`test_mnist.py`) and Task 3 (`test_cartpole.py`).

- [ ] **Step 1: Write the failing tests**

Create `biRNN/test_model.py`:

```python
import importlib.util
import os

import torch

from model import BidirectionalRNN, get_device

_rnn_model_path = os.path.join(os.path.dirname(__file__), "..", "RNN", "model.py")
_spec = importlib.util.spec_from_file_location("rnn_model_for_parity_test", _rnn_model_path)
_rnn_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rnn_model)
SimpleRNN = _rnn_model.SimpleRNN


def _copy_weights(simple: SimpleRNN, manual: BidirectionalRNN) -> None:
    manual.input_proj.weight.data.copy_(simple.input_proj.weight.data)
    manual.input_proj.bias.data.copy_(simple.input_proj.bias.data)
    manual.output_proj.weight.data.copy_(simple.output_proj.weight.data)
    manual.output_proj.bias.data.copy_(simple.output_proj.bias.data)

    manual.fwd_cell.weight_ih.data.copy_(simple.rnn.weight_ih_l0.data)
    manual.fwd_cell.weight_hh.data.copy_(simple.rnn.weight_hh_l0.data)
    manual.fwd_cell.bias_ih.data.copy_(simple.rnn.bias_ih_l0.data)
    manual.fwd_cell.bias_hh.data.copy_(simple.rnn.bias_hh_l0.data)

    manual.bwd_cell.weight_ih.data.copy_(simple.rnn.weight_ih_l0_reverse.data)
    manual.bwd_cell.weight_hh.data.copy_(simple.rnn.weight_hh_l0_reverse.data)
    manual.bwd_cell.bias_ih.data.copy_(simple.rnn.bias_ih_l0_reverse.data)
    manual.bwd_cell.bias_hh.data.copy_(simple.rnn.bias_hh_l0_reverse.data)


def test_output_shape_last_mode():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 2)


def test_output_shape_all_mode():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="all")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 10, 2)


def test_default_output_mode_is_last():
    model = BidirectionalRNN(input_size=4, hidden_size=8, output_size=2)
    x = torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 2)


def test_invalid_output_mode_raises():
    try:
        BidirectionalRNN(input_size=4, hidden_size=8, output_size=2, output_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_parity_with_nn_rnn_last_mode():
    torch.manual_seed(0)
    simple = SimpleRNN(input_size=4, hidden_size=8, output_size=3, output_mode="last")
    manual = BidirectionalRNN(input_size=4, hidden_size=8, output_size=3, output_mode="last")
    _copy_weights(simple, manual)

    x = torch.randn(5, 6, 4)
    simple.eval()
    manual.eval()
    with torch.no_grad():
        out_simple = simple(x)
        out_manual = manual(x)

    assert torch.allclose(out_simple, out_manual, atol=1e-5)


def test_parity_with_nn_rnn_all_mode():
    torch.manual_seed(1)
    simple = SimpleRNN(input_size=4, hidden_size=8, output_size=3, output_mode="all")
    manual = BidirectionalRNN(input_size=4, hidden_size=8, output_size=3, output_mode="all")
    _copy_weights(simple, manual)

    x = torch.randn(5, 6, 4)
    simple.eval()
    manual.eval()
    with torch.no_grad():
        out_simple = simple(x)
        out_manual = manual(x)

    assert torch.allclose(out_simple, out_manual, atol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest test_model.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'model'` (file doesn't exist yet). (`biRNN/` doesn't exist yet either — create it as part of this step: `mkdir -p /Users/hoyeon/Codes/modularRNN/biRNN`.)

- [ ] **Step 3: Implement `biRNN/model.py`**

```python
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BidirectionalRNN(nn.Module):
    """Hand-rolled bidirectional Elman RNN: explicit forward/backward loops over nn.RNNCell."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_mode: str = "last"):
        super().__init__()
        if output_mode not in ("last", "all"):
            raise ValueError(f"output_mode must be 'last' or 'all', got {output_mode!r}")
        self.output_mode = output_mode
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.fwd_cell = nn.RNNCell(hidden_size, hidden_size)
        self.bwd_cell = nn.RNNCell(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        batch_size, seq_len, _ = x.shape

        h_fwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        fwd_states = []
        for t in range(seq_len):
            h_fwd = self.fwd_cell(x[:, t, :], h_fwd)
            fwd_states.append(h_fwd)

        h_bwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        bwd_states = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_bwd = self.bwd_cell(x[:, t, :], h_bwd)
            bwd_states[t] = h_bwd

        if self.output_mode == "all":
            combined = torch.stack(
                [torch.cat([fwd_states[t], bwd_states[t]], dim=1) for t in range(seq_len)],
                dim=1,
            )
            return self.output_proj(combined)

        combined = torch.cat([fwd_states[-1], bwd_states[0]], dim=1)
        return self.output_proj(combined)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest test_model.py -v`

Expected: all 7 tests PASS, including both parity tests (proving the manual loop matches `nn.RNN(bidirectional=True)` to `atol=1e-5`).

- [ ] **Step 5: Commit**

```bash
git add biRNN/model.py biRNN/test_model.py
git commit -m "Add hand-rolled BidirectionalRNN with nn.RNN parity tests"
```

---

### Task 2: MNIST test for biRNN

**Files:**
- Create: `biRNN/test_mnist.py`

**Interfaces:**
- Consumes: `BidirectionalRNN`, `get_device` from `biRNN/model.py` (Task 1).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Implement `biRNN/test_mnist.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import BidirectionalRNN, get_device


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def train(model, loader, device, epochs: int, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in loader:
            images = to_sequence(images).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss {total_loss / len(loader):.4f}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = to_sequence(images).to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = get_device()
    print(f"using device: {device}")

    train_loader, test_loader = load_data()
    model = BidirectionalRNN(input_size=28, hidden_size=64, output_size=10, output_mode="last").to(device)

    train(model, train_loader, device, epochs=5)
    accuracy = evaluate(model, test_loader, device)
    print(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test with 1 epoch**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && python3 -c "
import test_mnist
device = test_mnist.get_device()
train_loader, test_loader = test_mnist.load_data()
model = test_mnist.BidirectionalRNN(input_size=28, hidden_size=64, output_size=10, output_mode='last').to(device)
test_mnist.train(model, train_loader, device, epochs=1)
acc = test_mnist.evaluate(model, test_loader, device)
print('smoke test accuracy:', acc)
"`

Expected: MNIST reuses `RNN/`'s already-downloaded data if `biRNN/data/` doesn't exist yet it downloads fresh; one epoch completes without error and prints a smoke-test accuracy (no threshold check here — this only confirms the pipeline runs).

- [ ] **Step 3: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && python3 test_mnist.py`

Expected: prints per-epoch loss for 5 epochs, then `test accuracy: 0.9xxx`, no `AssertionError`.

- [ ] **Step 4: Add `biRNN/data/` to `.gitignore`**

Append to `/Users/hoyeon/Codes/modularRNN/.gitignore`:

```
biRNN/data/
```

- [ ] **Step 5: Commit**

```bash
git add biRNN/test_mnist.py .gitignore
git commit -m "Add sequential MNIST test for hand-rolled BidirectionalRNN"
```

---

### Task 3: CartPole test for biRNN

**Files:**
- Create: `biRNN/test_cartpole.py`

**Interfaces:**
- Consumes: `BidirectionalRNN`, `get_device` from `biRNN/model.py` (Task 1); `heuristic_action` from `RNN/heuristic_policy.py` (existing file, imported via `sys.path` insert — not duplicated).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Implement `biRNN/test_cartpole.py`**

```python
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "RNN"))
from heuristic_policy import heuristic_action

from model import BidirectionalRNN, get_device


def collect_episodes(num_episodes: int, seed: int):
    env = gym.make("CartPole-v1")
    episodes = []
    for i in range(num_episodes):
        state, _ = env.reset(seed=seed + i)
        states, actions = [], []
        done = False
        while not done:
            action = heuristic_action(state)
            states.append(state)
            actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        episodes.append((np.array(states, dtype=np.float32), np.array(actions, dtype=np.int64)))
    env.close()
    return episodes


class CartPoleSequenceDataset(Dataset):
    def __init__(self, episodes):
        self.episodes = episodes

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        states, actions = self.episodes[idx]
        return torch.from_numpy(states), torch.from_numpy(actions)


def collate_pad(batch):
    lengths = [states.shape[0] for states, _ in batch]
    max_len = max(lengths)
    padded_states = torch.zeros(len(batch), max_len, 4)
    padded_actions = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (states, actions) in enumerate(batch):
        length = states.shape[0]
        padded_states[i, :length] = states
        padded_actions[i, :length] = actions
        mask[i, :length] = True
    return padded_states, padded_actions, mask


def train(model, loader, device, epochs: int, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for states, actions, mask in loader:
            states, actions, mask = states.to(device), actions.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss_per_step = criterion(logits.transpose(1, 2), actions)
            loss = (loss_per_step * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs} loss {total_loss / len(loader):.4f}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for states, actions, mask in loader:
        states, actions, mask = states.to(device), actions.to(device), mask.to(device)
        logits = model(states)
        preds = logits.argmax(dim=-1)
        correct += ((preds == actions) & mask).sum().item()
        total += mask.sum().item()
    return correct / total


def main():
    device = get_device()
    print(f"using device: {device}")

    train_episodes = collect_episodes(num_episodes=200, seed=0)
    test_episodes = collect_episodes(num_episodes=50, seed=1000)

    train_loader = DataLoader(
        CartPoleSequenceDataset(train_episodes), batch_size=16, shuffle=True, collate_fn=collate_pad
    )
    test_loader = DataLoader(
        CartPoleSequenceDataset(test_episodes), batch_size=16, shuffle=False, collate_fn=collate_pad
    )

    model = BidirectionalRNN(input_size=4, hidden_size=32, output_size=2, output_mode="all").to(device)

    train(model, train_loader, device, epochs=10)
    accuracy = evaluate(model, test_loader, device)
    print(f"per-timestep action accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% action accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && python3 test_cartpole.py`

Expected: prints per-epoch loss for 10 epochs, then `per-timestep action accuracy: 0.9xxx`, no `AssertionError`. This will run noticeably slower than `RNN/test_cartpole.py` (Python-level per-timestep loop over up to 500 steps × 2 directions, vs. one fused `nn.RNN` call) — allow it several minutes; if it hasn't finished in 10 minutes, stop and report rather than assuming a hang.

- [ ] **Step 3: Commit**

```bash
git add biRNN/test_cartpole.py
git commit -m "Add CartPole behavior-cloning test for hand-rolled BidirectionalRNN"
```

---

## Final check

- [ ] Run `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest test_model.py -v` — all 7 pass, including both parity tests.
- [ ] Confirm `biRNN/` contains: `model.py`, `test_model.py`, `test_mnist.py`, `test_cartpole.py`.
- [ ] Confirm both `python3 test_mnist.py` and `python3 test_cartpole.py` (run from inside `biRNN/`) complete without `AssertionError`.
