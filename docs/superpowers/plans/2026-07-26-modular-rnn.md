# Modular Bidirectional RNN (modRNN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `modRNN/` — a bidirectional RNN whose hidden layer is split into `input`/`intermediate`/`output` modules with restricted connectivity (dense within module, sparse between adjacent modules, none between the two extreme modules), per `CLAUDE.md`'s spec and `docs/superpowers/specs/2026-07-26-modular-rnn-design.md`.

**Architecture:** `ModularBidirectionalRNN` mirrors `biRNN.BidirectionalRNN`'s interface (`input_size, hidden_size, output_size, output_mode`) plus a new `near_module_sparsity` arg. A custom `ModularRNNCell` replaces `nn.RNNCell`, applying three fixed 0/1 masks (`ih_mask`, `hh_mask`, and an `output_mask` on the final projection) via elementwise multiply before each matmul.

**Tech Stack:** Same `.venv` as `RNN/`/`biRNN/` (PyTorch, torchvision, Gymnasium, pytest, matplotlib) — no new dependencies.

## Global Constraints

- Use the existing `.venv` at repo root — no new dependencies, no new `requirements.txt`.
- `hidden_size` must be evenly divisible by 3; constructor raises `ValueError` otherwise.
- `ModularBidirectionalRNN`'s public interface (constructor args, `forward` input/output shapes per `output_mode`) matches `biRNN.BidirectionalRNN`'s, plus the added `near_module_sparsity: float = 0.1` arg.
- No parity test against `biRNN`/`RNN` — the input↔output block is always zero regardless of sparsity, so the two can never compute the same function. Verification is structural (mask assertions) instead.
- MNIST/CartPole test scripts use the same hyperparameters and `>90%` accuracy thresholds as `biRNN`'s versions, for direct comparability. If `>90%` doesn't hold, the fix is to raise `hidden_size` (kept divisible by 3), not lower the threshold.
- Each folder (`RNN/`, `biRNN/`, `modRNN/`) stays independently runnable — no shared base class; small cross-folder reuse (`heuristic_policy.py`, `live_plot.py`) goes through explicit file-based imports/copies, not package restructuring — consistent with how `biRNN/` already duplicates `live_plot.py` and imports `heuristic_policy.py` via `sys.path`.

---

### Task 1: ModularBidirectionalRNN model with structural + shape tests

**Files:**
- Create: `modRNN/model.py`
- Test: `modRNN/test_model.py`

**Interfaces:**
- Produces:
  - `get_device() -> torch.device`
  - `ModularRNNCell(hidden_size: int, near_module_sparsity: float)` — an `nn.Module` with:
    - `weight_ih`, `weight_hh`: `nn.Parameter` of shape `(hidden_size, hidden_size)`
    - `bias_ih`, `bias_hh`: `nn.Parameter` of shape `(hidden_size,)`
    - `ih_mask`, `hh_mask`: registered buffers of shape `(hidden_size, hidden_size)`
    - `forward(z: torch.Tensor, h: torch.Tensor) -> torch.Tensor` — one RNN step
  - `ModularBidirectionalRNN(input_size: int, hidden_size: int, output_size: int, output_mode: str = "last", near_module_sparsity: float = 0.1)`:
    - `forward(x)` where `x` is `(batch, seq_len, input_size)`.
    - `output_mode="last"` → `(batch, output_size)`.
    - `output_mode="all"` → `(batch, seq_len, output_size)`.
    - Invalid `output_mode` raises `ValueError` at construction.
    - `hidden_size` not divisible by 3 raises `ValueError` at construction.
    - Internal attributes `input_proj`, `fwd_cell`, `bwd_cell`, `output_proj`, `output_mask` (named exactly this — Task 2/3 scripts and this task's own tests rely on these names).

  These exact names are used by Task 2 (`test_mnist.py`) and Task 3 (`test_cartpole.py`).

- [ ] **Step 1: Write the failing tests**

Create `modRNN/test_model.py`:

```python
import torch

from model import ModularBidirectionalRNN, ModularRNNCell, get_device


def _module_bounds(hidden_size: int) -> tuple[int, int]:
    third = hidden_size // 3
    return third, 2 * third


def test_output_shape_last_mode():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2, output_mode="last")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 2)


def test_output_shape_all_mode():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2, output_mode="all")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 10, 2)


def test_default_output_mode_is_last():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2)
    x = torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 2)


def test_invalid_output_mode_raises():
    try:
        ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2, output_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_hidden_size_not_divisible_by_3_raises():
    try:
        ModularBidirectionalRNN(input_size=4, hidden_size=10, output_size=2)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_ih_mask_restricts_external_input_to_input_module():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.ih_mask[:lo, :] == 1.0)
    assert torch.all(cell.ih_mask[lo:, :] == 0.0)


def test_hh_mask_same_module_blocks_are_dense():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.hh_mask[:lo, :lo] == 1.0)
    assert torch.all(cell.hh_mask[lo:hi, lo:hi] == 1.0)
    assert torch.all(cell.hh_mask[hi:, hi:] == 1.0)


def test_hh_mask_input_output_blocks_are_zero():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    assert torch.all(cell.hh_mask[:lo, hi:] == 0.0)
    assert torch.all(cell.hh_mask[hi:, :lo] == 0.0)


def test_hh_mask_near_module_density_matches_sparsity():
    torch.manual_seed(0)
    cell = ModularRNNCell(hidden_size=300, near_module_sparsity=0.1)
    lo, hi = _module_bounds(300)
    density = cell.hh_mask[:lo, lo:hi].mean().item()
    assert 0.05 < density < 0.15


def test_forbidden_hh_blocks_never_contribute_regardless_of_weight():
    cell = ModularRNNCell(hidden_size=9, near_module_sparsity=0.1)
    lo, hi = _module_bounds(9)
    masked = cell.weight_hh * cell.hh_mask
    assert torch.all(masked[:lo, hi:] == 0.0)
    assert torch.all(masked[hi:, :lo] == 0.0)


def test_output_mask_restricts_to_output_module_both_directions():
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=2)
    lo, hi = _module_bounds(9)
    mask = model.output_mask
    assert torch.all(mask[hi:9] == 1.0)
    assert torch.all(mask[9 + hi : 18] == 1.0)
    assert torch.all(mask[:hi] == 0.0)
    assert torch.all(mask[9 : 9 + hi] == 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && mkdir -p /Users/hoyeon/Codes/modularRNN/modRNN && cd /Users/hoyeon/Codes/modularRNN/modRNN && pytest test_model.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'model'` (file doesn't exist yet).

- [ ] **Step 3: Implement `modRNN/model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _module_slices(hidden_size: int) -> tuple[slice, slice, slice]:
    third = hidden_size // 3
    return slice(0, third), slice(third, 2 * third), slice(2 * third, hidden_size)


def _build_ih_mask(hidden_size: int) -> torch.Tensor:
    input_sl, _, _ = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size, hidden_size)
    mask[input_sl, :] = 1.0
    return mask


def _build_hh_mask(hidden_size: int, near_module_sparsity: float) -> torch.Tensor:
    input_sl, inter_sl, output_sl = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size, hidden_size)

    for sl in (input_sl, inter_sl, output_sl):
        mask[sl, sl] = 1.0

    for row_sl, col_sl in (
        (input_sl, inter_sl),
        (inter_sl, input_sl),
        (inter_sl, output_sl),
        (output_sl, inter_sl),
    ):
        rows = row_sl.stop - row_sl.start
        cols = col_sl.stop - col_sl.start
        mask[row_sl, col_sl] = (torch.rand(rows, cols) < near_module_sparsity).float()

    return mask


def _build_output_mask(hidden_size: int) -> torch.Tensor:
    _, _, output_sl = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size * 2)
    mask[output_sl] = 1.0
    mask[hidden_size + output_sl.start : hidden_size + output_sl.stop] = 1.0
    return mask


class ModularRNNCell(nn.Module):
    """Hand-rolled RNNCell restricted to modular connectivity: dense within module, sparse near, none far."""

    def __init__(self, hidden_size: int, near_module_sparsity: float):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        bound = hidden_size ** -0.5
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        nn.init.uniform_(self.bias_ih, -bound, bound)
        nn.init.uniform_(self.bias_hh, -bound, bound)

        self.register_buffer("ih_mask", _build_ih_mask(hidden_size))
        self.register_buffer("hh_mask", _build_hh_mask(hidden_size, near_module_sparsity))

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            F.linear(z, self.weight_ih * self.ih_mask, self.bias_ih)
            + F.linear(h, self.weight_hh * self.hh_mask, self.bias_hh)
        )


class ModularBidirectionalRNN(nn.Module):
    """Bidirectional RNN whose hidden layer is split into input/intermediate/output modules with restricted connectivity."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        output_mode: str = "last",
        near_module_sparsity: float = 0.1,
    ):
        super().__init__()
        if output_mode not in ("last", "all"):
            raise ValueError(f"output_mode must be 'last' or 'all', got {output_mode!r}")
        if hidden_size % 3 != 0:
            raise ValueError(f"hidden_size must be divisible by 3, got {hidden_size}")
        self.output_mode = output_mode
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.fwd_cell = ModularRNNCell(hidden_size, near_module_sparsity)
        self.bwd_cell = ModularRNNCell(hidden_size, near_module_sparsity)
        self.output_proj = nn.Linear(hidden_size * 2, output_size)
        self.register_buffer("output_mask", _build_output_mask(hidden_size))

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

        masked_weight = self.output_proj.weight * self.output_mask

        if self.output_mode == "all":
            combined = torch.stack(
                [torch.cat([fwd_states[t], bwd_states[t]], dim=1) for t in range(seq_len)],
                dim=1,
            )
            return F.linear(combined, masked_weight, self.output_proj.bias)

        combined = torch.cat([fwd_states[-1], bwd_states[0]], dim=1)
        return F.linear(combined, masked_weight, self.output_proj.bias)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/modRNN && pytest test_model.py -v`

Expected: all 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add modRNN/model.py modRNN/test_model.py
git commit -m "Add modular BidirectionalRNN with input/intermediate/output connectivity masks"
```

---

### Task 2: MNIST test for modRNN

**Files:**
- Create: `modRNN/live_plot.py` (copy of `biRNN/live_plot.py`, unchanged)
- Create: `modRNN/test_mnist.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `ModularBidirectionalRNN`, `get_device` from `modRNN/model.py` (Task 1); `LiveTrainingPlot` from `modRNN/live_plot.py`.
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Copy `live_plot.py`**

```bash
cp /Users/hoyeon/Codes/modularRNN/biRNN/live_plot.py /Users/hoyeon/Codes/modularRNN/modRNN/live_plot.py
```

- [ ] **Step 2: Implement `modRNN/test_mnist.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from live_plot import LiveTrainingPlot
from model import ModularBidirectionalRNN, get_device


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def train(model, train_loader, test_loader, device, epochs: int, lr: float = 1e-3, live_plot=None) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = to_sequence(images).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f}")
        if live_plot is not None:
            live_plot.update(epoch + 1, avg_loss, accuracy)
    return accuracy


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
    model = ModularBidirectionalRNN(input_size=28, hidden_size=63, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="modRNN/test_mnist.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=5, live_plot=live_plot)
    print(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

Note: `hidden_size=63` (divisible by 3, close to `biRNN`'s 64) — each module gets 21 units.

- [ ] **Step 3: Smoke-test with 1 epoch**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/modRNN && python3 -c "
import test_mnist
device = test_mnist.get_device()
train_loader, test_loader = test_mnist.load_data()
model = test_mnist.ModularBidirectionalRNN(input_size=28, hidden_size=63, output_size=10, output_mode='last').to(device)
acc = test_mnist.train(model, train_loader, test_loader, device, epochs=1)
print('smoke test accuracy:', acc)
"`

Expected: one epoch completes without error and prints a smoke-test accuracy (no threshold check here — this only confirms the pipeline runs). If accuracy after 1 epoch looks far below what 5 epochs could plausibly reach >90% from, stop and report rather than proceeding to Step 4.

- [ ] **Step 4: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/modRNN && python3 test_mnist.py`

Expected: prints per-epoch loss/accuracy for 5 epochs, then `test accuracy: 0.9xxx`, no `AssertionError`. If it doesn't clear 90%, raise `hidden_size` to the next multiple of 3 (e.g. 96) and retry rather than lowering the threshold.

- [ ] **Step 5: Add `modRNN/data/` to `.gitignore`**

Append to `/Users/hoyeon/Codes/modularRNN/.gitignore`:

```
modRNN/data/
```

- [ ] **Step 6: Commit**

```bash
git add modRNN/live_plot.py modRNN/test_mnist.py .gitignore
git commit -m "Add sequential MNIST test for modular BidirectionalRNN"
```

---

### Task 3: CartPole test for modRNN

**Files:**
- Create: `modRNN/test_cartpole.py`

**Interfaces:**
- Consumes: `ModularBidirectionalRNN`, `get_device` from `modRNN/model.py` (Task 1); `LiveTrainingPlot` from `modRNN/live_plot.py` (Task 2); `heuristic_action` from `RNN/heuristic_policy.py` (existing file, imported via `sys.path` insert — not duplicated).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Implement `modRNN/test_cartpole.py`**

```python
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "RNN"))
from heuristic_policy import heuristic_action

from live_plot import LiveTrainingPlot
from model import ModularBidirectionalRNN, get_device


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


def train(model, train_loader, test_loader, device, epochs: int, lr: float = 1e-3, live_plot=None) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    accuracy = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for states, actions, mask in train_loader:
            states, actions, mask = states.to(device), actions.to(device), mask.to(device)
            optimizer.zero_grad()
            logits = model(states)
            loss_per_step = criterion(logits.transpose(1, 2), actions)
            loss = (loss_per_step * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f}")
        if live_plot is not None:
            live_plot.update(epoch + 1, avg_loss, accuracy)
    return accuracy


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

    model = ModularBidirectionalRNN(input_size=4, hidden_size=33, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="modRNN/test_cartpole.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=10, live_plot=live_plot)
    print(f"per-timestep action accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% action accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

Note: `hidden_size=33` (divisible by 3, close to `biRNN`'s 32) — each module gets 11 units.

- [ ] **Step 2: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/modRNN && python3 test_cartpole.py`

Expected: prints per-epoch loss/accuracy for 10 epochs, then `per-timestep action accuracy: 0.9xxx`, no `AssertionError`. This will run noticeably slower than `RNN/test_cartpole.py` (Python-level per-timestep loop, same as `biRNN`) — allow several minutes; if it hasn't finished in 10 minutes, stop and report rather than assuming a hang. If accuracy doesn't clear 90%, raise `hidden_size` to the next multiple of 3 (e.g. 63) and retry rather than lowering the threshold.

- [ ] **Step 3: Commit**

```bash
git add modRNN/test_cartpole.py
git commit -m "Add CartPole behavior-cloning test for modular BidirectionalRNN"
```

---

## Final check

- [ ] Run `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/modRNN && pytest test_model.py -v` — all 12 tests pass.
- [ ] Confirm `modRNN/` contains: `model.py`, `test_model.py`, `live_plot.py`, `test_mnist.py`, `test_cartpole.py`.
- [ ] Confirm both `python3 test_mnist.py` and `python3 test_cartpole.py` (run from inside `modRNN/`) complete without `AssertionError`.
