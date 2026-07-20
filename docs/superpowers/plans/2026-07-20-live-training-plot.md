# Live Training Progress GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a live-updating matplotlib window (loss + held-out accuracy, per epoch) to all four existing test scripts, without ever risking their accuracy assertions if no GUI backend is available.

**Architecture:** A small `LiveTrainingPlot` class (duplicated as `RNN/live_plot.py` and `biRNN/live_plot.py`) wraps `matplotlib.pyplot` in interactive mode with a try/except fallback. Each `train()` function is restructured to evaluate on the held-out set every epoch (not just at the end) and feed loss+accuracy to the plot, returning the final accuracy for the threshold assertion.

**Tech Stack:** Same `.venv` as before, plus `matplotlib` (new dependency).

## Global Constraints

- `LiveTrainingPlot` must never be able to crash a test script — if matplotlib can't get a GUI backend, it silently disables itself (prints one warning) and `update()` becomes a no-op.
- No shared module between `RNN/` and `biRNN/` — `live_plot.py` is duplicated, per the design doc.
- Every script's accuracy threshold assertion (>90%) must still pass after the change.
- `biRNN/test_cartpole.py` will get noticeably slower (per-epoch eval added on top of its already-long ~28 minute runtime) — expected and accepted.

---

### Task 1: `RNN/live_plot.py` with unit tests

**Files:**
- Modify: `RNN/requirements.txt`
- Create: `RNN/live_plot.py`
- Test: `RNN/test_live_plot.py`

**Interfaces:**
- Produces: `LiveTrainingPlot(title: str)` with `.enabled: bool`, `.update(epoch: int, loss: float, accuracy: float) -> None`. Used by Task 3 (`RNN/test_mnist.py`) and Task 4 (`RNN/test_cartpole.py`).

- [ ] **Step 1: Add matplotlib to requirements and install it**

Edit `RNN/requirements.txt` to add a line:

```
matplotlib
```

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && pip install matplotlib`

Expected: installs successfully into the existing venv.

- [ ] **Step 2: Write the failing tests**

Create `RNN/test_live_plot.py`:

```python
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, deterministic for tests

from live_plot import LiveTrainingPlot


def test_enabled_on_working_backend():
    plot = LiveTrainingPlot(title="test")
    assert plot.enabled is True


def test_update_appends_data():
    plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)
    plot.update(2, 0.3, 0.9)

    assert plot.epochs == [1, 2]
    assert plot.losses == [0.5, 0.3]
    assert plot.accuracies == [0.8, 0.9]


def test_disabled_when_backend_unavailable():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    assert plot.enabled is False


def test_update_is_noop_when_disabled():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)  # must not raise
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_live_plot.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'live_plot'`.

- [ ] **Step 4: Implement `RNN/live_plot.py`**

```python
import matplotlib.pyplot as plt


class LiveTrainingPlot:
    """Live-updating loss/accuracy window. Disables itself (no crash) if no GUI backend is available."""

    def __init__(self, title: str):
        self.enabled = True
        self.epochs = []
        self.losses = []
        self.accuracies = []
        try:
            plt.ion()
            self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
            self.fig.suptitle(title)

            self.ax_loss.set_xlabel("epoch")
            self.ax_loss.set_ylabel("loss")
            (self.loss_line,) = self.ax_loss.plot([], [], marker="o")

            self.ax_acc.set_xlabel("epoch")
            self.ax_acc.set_ylabel("accuracy")
            self.ax_acc.set_ylim(0, 1)
            (self.acc_line,) = self.ax_acc.plot([], [], marker="o", color="tab:green")

            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.001)
        except Exception as e:
            self.enabled = False
            print(f"live plot disabled (no GUI backend available): {e}")

    def update(self, epoch: int, loss: float, accuracy: float) -> None:
        if not self.enabled:
            return

        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

        self.loss_line.set_data(self.epochs, self.losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.acc_line.set_data(self.epochs, self.accuracies)
        self.ax_acc.set_xlim(0.5, max(self.epochs) + 0.5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_live_plot.py -v`

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add RNN/requirements.txt RNN/live_plot.py RNN/test_live_plot.py
git commit -m "Add live training-progress plot helper for RNN/"
```

---

### Task 2: `biRNN/live_plot.py` (duplicate) with unit tests

**Files:**
- Create: `biRNN/live_plot.py`
- Test: `biRNN/test_live_plot.py`

**Interfaces:**
- Produces: identical `LiveTrainingPlot` class as Task 1, in `biRNN/`. Used by Task 5 (`biRNN/test_mnist.py`) and Task 6 (`biRNN/test_cartpole.py`).

- [ ] **Step 1: Write the failing tests**

Create `biRNN/test_live_plot.py`:

```python
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, deterministic for tests

from live_plot import LiveTrainingPlot


def test_enabled_on_working_backend():
    plot = LiveTrainingPlot(title="test")
    assert plot.enabled is True


def test_update_appends_data():
    plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)
    plot.update(2, 0.3, 0.9)

    assert plot.epochs == [1, 2]
    assert plot.losses == [0.5, 0.3]
    assert plot.accuracies == [0.8, 0.9]


def test_disabled_when_backend_unavailable():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    assert plot.enabled is False


def test_update_is_noop_when_disabled():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest test_live_plot.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'live_plot'`.

- [ ] **Step 3: Implement `biRNN/live_plot.py`**

```python
import matplotlib.pyplot as plt


class LiveTrainingPlot:
    """Live-updating loss/accuracy window. Disables itself (no crash) if no GUI backend is available."""

    def __init__(self, title: str):
        self.enabled = True
        self.epochs = []
        self.losses = []
        self.accuracies = []
        try:
            plt.ion()
            self.fig, (self.ax_loss, self.ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
            self.fig.suptitle(title)

            self.ax_loss.set_xlabel("epoch")
            self.ax_loss.set_ylabel("loss")
            (self.loss_line,) = self.ax_loss.plot([], [], marker="o")

            self.ax_acc.set_xlabel("epoch")
            self.ax_acc.set_ylabel("accuracy")
            self.ax_acc.set_ylim(0, 1)
            (self.acc_line,) = self.ax_acc.plot([], [], marker="o", color="tab:green")

            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.001)
        except Exception as e:
            self.enabled = False
            print(f"live plot disabled (no GUI backend available): {e}")

    def update(self, epoch: int, loss: float, accuracy: float) -> None:
        if not self.enabled:
            return

        self.epochs.append(epoch)
        self.losses.append(loss)
        self.accuracies.append(accuracy)

        self.loss_line.set_data(self.epochs, self.losses)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        self.acc_line.set_data(self.epochs, self.accuracies)
        self.ax_acc.set_xlim(0.5, max(self.epochs) + 0.5)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest test_live_plot.py -v`

Expected: all 4 tests PASS. (`matplotlib` is already installed in the shared venv from Task 1 — no reinstall needed.)

- [ ] **Step 5: Commit**

```bash
git add biRNN/live_plot.py biRNN/test_live_plot.py
git commit -m "Add live training-progress plot helper for biRNN/"
```

---

### Task 3: Wire live plot into `RNN/test_mnist.py`

**Files:**
- Modify: `RNN/test_mnist.py`

**Interfaces:**
- Consumes: `LiveTrainingPlot` from `RNN/live_plot.py` (Task 1).

- [ ] **Step 1: Replace `train()` and `main()`**

In `RNN/test_mnist.py`, add the import:

```python
from live_plot import LiveTrainingPlot
```

Replace the `train` function:

```python
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
```

Replace `main`:

```python
def main():
    device = get_device()
    print(f"using device: {device}")

    train_loader, test_loader = load_data()
    model = SimpleRNN(input_size=28, hidden_size=64, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="RNN/test_mnist.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=5, live_plot=live_plot)
    print(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"
```

`evaluate()` is unchanged.

- [ ] **Step 2: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 test_mnist.py`

Expected: prints per-epoch `loss` **and** `accuracy` for 5 epochs (accuracy now shown every epoch, not just at the end), then `test accuracy: 0.9xxx`, no `AssertionError`. A live plot window should appear and update each epoch — verify this visually if running with a display attached (Claude cannot see this, only that the script exits cleanly and reports a passing accuracy).

- [ ] **Step 3: Commit**

```bash
git add RNN/test_mnist.py
git commit -m "Show live training progress in RNN/test_mnist.py"
```

---

### Task 4: Wire live plot into `RNN/test_cartpole.py`

**Files:**
- Modify: `RNN/test_cartpole.py`

**Interfaces:**
- Consumes: `LiveTrainingPlot` from `RNN/live_plot.py` (Task 1).

- [ ] **Step 1: Replace `train()` and `main()`**

Add the import:

```python
from live_plot import LiveTrainingPlot
```

Replace `train`:

```python
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
```

Replace `main`:

```python
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

    live_plot = LiveTrainingPlot(title="RNN/test_cartpole.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=10, live_plot=live_plot)
    print(f"per-timestep action accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% action accuracy, got {accuracy:.4f}"
```

`evaluate()`, `collect_episodes`, `CartPoleSequenceDataset`, `collate_pad` are unchanged.

- [ ] **Step 2: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 test_cartpole.py`

Expected: prints per-epoch `loss` and `accuracy` for 10 epochs, then `per-timestep action accuracy: 0.9xxx`, no `AssertionError`.

- [ ] **Step 3: Commit**

```bash
git add RNN/test_cartpole.py
git commit -m "Show live training progress in RNN/test_cartpole.py"
```

---

### Task 5: Wire live plot into `biRNN/test_mnist.py`

**Files:**
- Modify: `biRNN/test_mnist.py`

**Interfaces:**
- Consumes: `LiveTrainingPlot` from `biRNN/live_plot.py` (Task 2).

- [ ] **Step 1: Replace `train()` and `main()`**

`biRNN/test_mnist.py` already imports `BidirectionalRNN` and `get_device` from `model` — keep that line. Add:

```python
from live_plot import LiveTrainingPlot
```

Replace the `train` function:

```python
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
```

Replace `main`:

```python
def main():
    device = get_device()
    print(f"using device: {device}")

    train_loader, test_loader = load_data()
    model = BidirectionalRNN(input_size=28, hidden_size=64, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="biRNN/test_mnist.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=5, live_plot=live_plot)
    print(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"
```

`evaluate()`, `load_data()`, `to_sequence()` are unchanged.

- [ ] **Step 2: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && python3 test_mnist.py`

Expected: prints per-epoch `loss` and `accuracy` for 5 epochs, then `test accuracy: 0.9xxx`, no `AssertionError`.

- [ ] **Step 3: Commit**

```bash
git add biRNN/test_mnist.py
git commit -m "Show live training progress in biRNN/test_mnist.py"
```

---

### Task 6: Wire live plot into `biRNN/test_cartpole.py`

**Files:**
- Modify: `biRNN/test_cartpole.py`

**Interfaces:**
- Consumes: `LiveTrainingPlot` from `biRNN/live_plot.py` (Task 2).

- [ ] **Step 1: Replace `train()` and `main()`**

`biRNN/test_cartpole.py` already imports `BidirectionalRNN`, `get_device`, and `heuristic_action` (via the `sys.path.append(...)` line) — keep those. Add:

```python
from live_plot import LiveTrainingPlot
```

Replace the `train` function:

```python
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
```

Replace `main`:

```python
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

    live_plot = LiveTrainingPlot(title="biRNN/test_cartpole.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=10, live_plot=live_plot)
    print(f"per-timestep action accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% action accuracy, got {accuracy:.4f}"
```

`evaluate()`, `collect_episodes`, `CartPoleSequenceDataset`, `collate_pad` are unchanged.

- [ ] **Step 2: Run the full script**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && python3 test_cartpole.py`

Expected: prints per-epoch `loss` and `accuracy` for 10 epochs, then `per-timestep action accuracy: 0.9xxx`, no `AssertionError`. **This will take noticeably longer than the ~28 minutes the unmodified version took** (per-epoch eval adds extra forward passes on top of the already-slow per-timestep loop) — run this in the background and check back rather than assuming a hang.

- [ ] **Step 3: Commit**

```bash
git add biRNN/test_cartpole.py
git commit -m "Show live training progress in biRNN/test_cartpole.py"
```

---

## Final check

- [ ] Run `pytest test_live_plot.py -v` in both `RNN/` and `biRNN/` — all pass.
- [ ] Run `pytest test_model.py -v` in both `RNN/` and `biRNN/` — still all pass (unaffected by this change).
- [ ] Confirm all four `test_*.py` scripts still complete without `AssertionError`.
