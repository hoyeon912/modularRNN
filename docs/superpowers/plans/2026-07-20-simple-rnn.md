# Simple RNN (non-modular baseline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a plain, non-modular, bidirectional RNN baseline in `RNN/`, tested on sequential MNIST (simple test) and a CartPole behavior-cloning task (hard test).

**Architecture:** One `SimpleRNN` class (`Linear -> bidirectional nn.RNN -> Linear`) with an `output_mode` switch (`"last"` for sequence classification, `"all"` for per-timestep prediction). MNIST feeds 28-row sequences through `"last"` mode; CartPole feeds heuristic-labeled state trajectories through `"all"` mode.

**Tech Stack:** Python 3.14 (existing `.venv`), PyTorch 2.13, torchvision 0.28 (MNIST dataset), Gymnasium 1.3 (CartPole-v1), pytest.

## Global Constraints

- Use the existing `.venv` at repo root — do not create a new venv (per `CLAUDE.md`).
- GPU acceleration required: device selection must try `cuda`, then `mps`, then fall back to `cpu` (per `CLAUDE.md`).
- Model must be a vanilla `nn.RNN`-based bidirectional network with no module/connectivity restrictions (per spec: this is the "no modulation" baseline, contrasted with the project's modular RNN).
- No CLI framework, config files, or checkpointing — hardcoded hyperparameters in scripts (per design doc scope).
- Scripts run directly: `python RNN/test_mnist.py`, `python RNN/test_cartpole.py` (per design doc).

---

### Task 1: Environment setup

**Files:**
- Create: `RNN/requirements.txt`

**Interfaces:**
- Produces: an activated `.venv` with `torch`, `torchvision`, `gymnasium`, `numpy`, `pytest` importable — every later task depends on this.

- [ ] **Step 1: Create `RNN/requirements.txt`**

```
torch
torchvision
gymnasium
numpy
pytest
```

- [ ] **Step 2: Install into the existing venv**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && pip install -r /Users/hoyeon/Codes/modularRNN/RNN/requirements.txt`

Expected: all packages install successfully (wheels for `cp314-macosx_14_0_arm64` are available as of this plan — confirmed via `pip install --dry-run` before writing this plan). If any package fails to find a compatible wheel for this Python/platform combination, STOP and report it — do not silently switch interpreters or add workarounds.

- [ ] **Step 3: Verify imports and device selection**

Run:
```bash
source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && python3 -c "
import torch, torchvision, gymnasium, numpy, pytest
print('torch', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('mps available:', torch.backends.mps.is_available())
"
```

Expected: prints versions with no import errors. On Apple Silicon, `mps available: True` is expected; `cuda available: False` is expected (no NVIDIA GPU on this machine).

- [ ] **Step 4: Commit**

```bash
git add RNN/requirements.txt
git commit -m "Add RNN folder with environment requirements"
```

---

### Task 2: SimpleRNN model

**Files:**
- Create: `RNN/model.py`
- Test: `RNN/test_model.py`

**Interfaces:**
- Consumes: `torch`, `torch.nn` (from Task 1's installed packages).
- Produces:
  - `get_device() -> torch.device`
  - `SimpleRNN(input_size: int, hidden_size: int, output_size: int, output_mode: str = "last")` — an `nn.Module`.
    - `forward(x: torch.Tensor) -> torch.Tensor` where `x` has shape `(batch, seq_len, input_size)`.
    - `output_mode="last"` → returns shape `(batch, output_size)`.
    - `output_mode="all"` → returns shape `(batch, seq_len, output_size)`.
    - Invalid `output_mode` raises `ValueError` at construction time.

  These exact names/signatures are used by Task 3 (`test_mnist.py`) and Task 4 (`test_cartpole.py`).

- [ ] **Step 1: Write the failing tests**

Create `RNN/test_model.py`:

```python
import torch

from model import SimpleRNN, get_device


def test_output_shape_last_mode():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="last")
    x = torch.randn(3, 10, 4)  # batch=3, seq_len=10, input_size=4
    out = model(x)
    assert out.shape == (3, 2)


def test_output_shape_all_mode():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="all")
    x = torch.randn(3, 10, 4)
    out = model(x)
    assert out.shape == (3, 10, 2)


def test_default_output_mode_is_last():
    model = SimpleRNN(input_size=4, hidden_size=8, output_size=2)
    x = torch.randn(2, 5, 4)
    out = model(x)
    assert out.shape == (2, 2)


def test_invalid_output_mode_raises():
    try:
        SimpleRNN(input_size=4, hidden_size=8, output_size=2, output_mode="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_model.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'model'` (file doesn't exist yet).

- [ ] **Step 3: Implement `RNN/model.py`**

```python
import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleRNN(nn.Module):
    """Plain bidirectional Elman RNN baseline: dense hidden layer, no module structure."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_mode: str = "last"):
        super().__init__()
        if output_mode not in ("last", "all"):
            raise ValueError(f"output_mode must be 'last' or 'all', got {output_mode!r}")
        self.output_mode = output_mode

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        outputs, hidden = self.rnn(x)
        # outputs: (batch, seq_len, hidden_size*2)
        # hidden: (2, batch, hidden_size) -- num_layers=1, bidirectional=True -> [forward, backward]

        if self.output_mode == "all":
            return self.output_proj(outputs)

        forward_last = hidden[0]
        backward_last = hidden[1]
        combined = torch.cat([forward_last, backward_last], dim=1)
        return self.output_proj(combined)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_model.py -v`

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add RNN/model.py RNN/test_model.py
git commit -m "Add SimpleRNN baseline model with unit tests"
```

---

### Task 3: MNIST simple test

**Files:**
- Create: `RNN/test_mnist.py`

**Interfaces:**
- Consumes: `SimpleRNN`, `get_device` from `RNN/model.py` (Task 2).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Implement `RNN/test_mnist.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SimpleRNN, get_device


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    # images: (batch, 1, 28, 28) -> (batch, 28, 28), each image read as 28 rows of 28 pixels
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
    model = SimpleRNN(input_size=28, hidden_size=64, output_size=10, output_mode="last").to(device)

    train(model, train_loader, device, epochs=5)
    accuracy = evaluate(model, test_loader, device)
    print(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the pipeline with a tiny run**

Before committing to a full 5-epoch run, verify the script's plumbing (data loading, shapes, training loop) works by temporarily running with 1 epoch:

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 -c "
import test_mnist
device = test_mnist.get_device()
train_loader, test_loader = test_mnist.load_data()
model = test_mnist.SimpleRNN(input_size=28, hidden_size=64, output_size=10, output_mode='last').to(device)
test_mnist.train(model, train_loader, device, epochs=1)
acc = test_mnist.evaluate(model, test_loader, device)
print('smoke test accuracy:', acc)
"`

Expected: MNIST downloads to `RNN/data/` on first run, one epoch of training completes, prints a smoke-test accuracy (will likely be 80-95% after just 1 epoch — no threshold assertion here, this step only confirms the pipeline runs end-to-end without errors).

- [ ] **Step 3: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 test_mnist.py`

Expected: prints per-epoch loss for 5 epochs, then `test accuracy: 0.9xxx`, and exits with no `AssertionError` (accuracy above 0.90).

- [ ] **Step 4: Add `RNN/data/` to `.gitignore`**

Create or append to `/Users/hoyeon/Codes/modularRNN/.gitignore`:

```
RNN/data/
```

- [ ] **Step 5: Commit**

```bash
git add RNN/test_mnist.py .gitignore
git commit -m "Add sequential MNIST test for SimpleRNN"
```

---

### Task 4: CartPole hard test (behavior cloning)

**Files:**
- Create: `RNN/heuristic_policy.py`
- Create: `RNN/test_cartpole.py`

**Interfaces:**
- Consumes: `SimpleRNN`, `get_device` from `RNN/model.py` (Task 2); `heuristic_action` from `RNN/heuristic_policy.py` (this task).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Implement `RNN/heuristic_policy.py`**

```python
def heuristic_action(state) -> int:
    """Bang-bang pole-balancing controller: push toward the direction the pole is falling."""
    _, _, pole_angle, pole_angular_velocity = state
    return 1 if pole_angle + 0.5 * pole_angular_velocity > 0 else 0
```

- [ ] **Step 2: Verify the heuristic balances the pole**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 -c "
import gymnasium as gym
from heuristic_policy import heuristic_action

env = gym.make('CartPole-v1')
state, _ = env.reset(seed=0)
steps = 0
done = False
while not done:
    action = heuristic_action(state)
    state, _, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    steps += 1
print('episode length:', steps)
env.close()
"`

Expected: `episode length:` close to 500 (CartPole-v1's max episode length), confirming the heuristic keeps the pole up rather than failing immediately. If it prints a small number (e.g. under 50), the heuristic's coefficients need adjusting before proceeding — do not move on with a broken heuristic, since Task 4's training data depends on it producing long, informative episodes.

- [ ] **Step 3: Implement `RNN/test_cartpole.py`**

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from heuristic_policy import heuristic_action
from model import SimpleRNN, get_device


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
            logits = model(states)  # (batch, seq_len, 2)
            loss_per_step = criterion(logits.transpose(1, 2), actions)  # (batch, seq_len)
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

    model = SimpleRNN(input_size=4, hidden_size=32, output_size=2, output_mode="all").to(device)

    train(model, train_loader, device, epochs=10)
    accuracy = evaluate(model, test_loader, device)
    print(f"per-timestep action accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% action accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 test_cartpole.py`

Expected: prints per-epoch loss for 10 epochs, then `per-timestep action accuracy: 0.9xxx`, and exits with no `AssertionError` (accuracy above 0.90).

- [ ] **Step 5: Commit**

```bash
git add RNN/heuristic_policy.py RNN/test_cartpole.py
git commit -m "Add CartPole behavior-cloning hard test for SimpleRNN"
```

---

## Final check

- [ ] Run `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_model.py -v` — all pass.
- [ ] Confirm `RNN/` contains: `requirements.txt`, `model.py`, `test_model.py`, `test_mnist.py`, `heuristic_policy.py`, `test_cartpole.py`.
- [ ] Confirm both `python3 test_mnist.py` and `python3 test_cartpole.py` complete without `AssertionError`.
