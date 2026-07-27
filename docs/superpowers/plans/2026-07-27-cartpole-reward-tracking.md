# CartPole Reward Tracking + Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add closed-loop reward tracking (live-plotted every epoch) and a rendered episode
at the end of training to `RNN/test_cartpole.py`, `biRNN/test_cartpole.py`, and
`modRNN/test_cartpole.py`, per `docs/superpowers/specs/2026-07-27-cartpole-reward-tracking-design.md`.

**Architecture:** A new `rollout_episode(model, env, device, max_steps=500)` function
(duplicated per folder) actually drives the CartPole environment with the model's own
greedy action at every step and returns total reward. `evaluate_reward()` averages 3 such
rollouts per epoch for the live-plotted metric. `LiveTrainingPlot` becomes metric-generic
(`metrics` tuple + `*values` in `update()`) instead of hardcoded to loss/accuracy, so it can
grow a third "reward" subplot without touching `test_mnist.py`'s call sites. A final
unconditional rendered episode (`render_mode="human"`) runs after training completes.

**Tech Stack:** Same `.venv` as the rest of the repo (PyTorch, Gymnasium, matplotlib,
pytest) — no new dependencies.

## Global Constraints

- Use the existing `.venv` at repo root — no new dependencies.
- `LiveTrainingPlot(title: str, metrics: tuple[str, ...] = ("loss", "accuracy"))` — default
  unchanged so `test_mnist.py` in all three folders needs zero code changes.
- `update(self, epoch: int, *values: float)` — values map positionally to `self.metrics`.
- Fixed y-axis ranges: `"accuracy"` → `(0, 1)`, `"reward"` → `(0, 500)`. Any other metric
  name (e.g. `"loss"`) autoscales via `relim()`/`autoscale_view()`.
- `rollout_episode` and `evaluate_reward` are duplicated per folder's `test_cartpole.py`
  (no shared base class), consistent with the project's existing convention.
- Rendering is unconditional (no CLI flag) and wrapped in `try/except` so a headless
  machine prints a message instead of crashing — matching `LiveTrainingPlot`'s existing
  no-GUI fallback pattern.
- Accepted tradeoff: closed-loop rollout is O(T²) per episode (bidirectional models must
  recompute the full backward pass every step) — `test_cartpole.py` runs will take
  noticeably longer than before. Do not reduce eval frequency/episode count below "every
  epoch, 3 episodes" to compensate — this was an explicit, informed choice.
- **Long-running steps:** the full-script runs in Tasks 2–4 will take longer than the
  plan's earlier `biRNN`/`modRNN` CartPole tasks did (which already took several minutes
  and caused subagent connection stalls on synchronous waits, including one case where a
  raw shell `&` background process died silently when its spawning tool call ended).
  Launch these with your Bash tool's `run_in_background: true` option specifically (not a
  shell `&`), and check on progress with short, separate calls rather than one long
  blocking wait.

---

### Task 1: Generalize LiveTrainingPlot to N metrics

**Files:**
- Modify: `RNN/live_plot.py`, `biRNN/live_plot.py`, `modRNN/live_plot.py` (all three are
  currently byte-identical — apply the same new content to all three)
- Modify: `RNN/test_live_plot.py`, `biRNN/test_live_plot.py` (currently byte-identical)
- Create: `modRNN/test_live_plot.py` (didn't exist before; same content as the other two)

**Interfaces:**
- Produces: `LiveTrainingPlot(title: str, metrics: tuple[str, ...] = ("loss", "accuracy"))`
  with `.enabled: bool`, `.epochs: list[int]`, `.history: dict[str, list[float]]`, and
  `update(self, epoch: int, *values: float) -> None`. Task 2/3/4 depend on being able to
  construct it with `metrics=("loss", "accuracy", "reward")` and call
  `update(epoch, avg_loss, accuracy, reward)`.

- [ ] **Step 1: Write the failing tests**

Replace the contents of `RNN/test_live_plot.py` with:

```python
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, deterministic for tests

from live_plot import LiveTrainingPlot


def test_enabled_on_working_backend():
    plot = LiveTrainingPlot(title="test")
    assert plot.enabled is True


def test_update_appends_data_default_metrics():
    plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)
    plot.update(2, 0.3, 0.9)

    assert plot.epochs == [1, 2]
    assert plot.history["loss"] == [0.5, 0.3]
    assert plot.history["accuracy"] == [0.8, 0.9]


def test_update_appends_data_three_metrics():
    plot = LiveTrainingPlot(title="test", metrics=("loss", "accuracy", "reward"))
    plot.update(1, 0.5, 0.8, 120.0)
    plot.update(2, 0.3, 0.9, 250.0)

    assert plot.epochs == [1, 2]
    assert plot.history["loss"] == [0.5, 0.3]
    assert plot.history["accuracy"] == [0.8, 0.9]
    assert plot.history["reward"] == [120.0, 250.0]


def test_disabled_when_backend_unavailable():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    assert plot.enabled is False


def test_update_is_noop_when_disabled():
    with patch("live_plot.plt.subplots", side_effect=RuntimeError("no display")):
        plot = LiveTrainingPlot(title="test")
    plot.update(1, 0.5, 0.8)  # must not raise
```

Copy the exact same content to `biRNN/test_live_plot.py` and `modRNN/test_live_plot.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_live_plot.py -v`

Expected: `test_update_appends_data_default_metrics` and `test_update_appends_data_three_metrics`
FAIL with `AttributeError: 'LiveTrainingPlot' object has no attribute 'history'` (current
implementation has `.losses`/`.accuracies`, not `.history`, and `update()` takes fixed
`(epoch, loss, accuracy)` args, not `*values`). The other tests currently pass against the
old implementation — that's fine, they'll continue to pass after the rewrite too.

- [ ] **Step 3: Implement the new `live_plot.py`**

Replace the contents of `RNN/live_plot.py` with:

```python
import matplotlib.pyplot as plt

_FIXED_YLIM = {"accuracy": (0, 1), "reward": (0, 500)}
_COLOR = {"accuracy": "tab:green", "reward": "tab:orange"}


class LiveTrainingPlot:
    """Live-updating metrics window. Disables itself (no crash) if no GUI backend is available."""

    def __init__(self, title: str, metrics: tuple[str, ...] = ("loss", "accuracy")):
        self.enabled = True
        self.metrics = list(metrics)
        self.epochs = []
        self.history = {metric: [] for metric in self.metrics}
        try:
            plt.ion()
            self.fig, axes = plt.subplots(1, len(self.metrics), figsize=(5 * len(self.metrics), 4))
            self.fig.suptitle(title)
            if len(self.metrics) == 1:
                axes = [axes]

            self.axes = {}
            self.lines = {}
            for ax, metric in zip(axes, self.metrics):
                ax.set_xlabel("epoch")
                ax.set_ylabel(metric)
                if metric in _FIXED_YLIM:
                    ax.set_ylim(*_FIXED_YLIM[metric])
                (line,) = ax.plot([], [], marker="o", color=_COLOR.get(metric))
                self.axes[metric] = ax
                self.lines[metric] = line

            self.fig.tight_layout()
            self.fig.canvas.draw()
            plt.pause(0.001)
        except Exception as e:
            self.enabled = False
            print(f"live plot disabled (no GUI backend available): {e}")

    def update(self, epoch: int, *values: float) -> None:
        if not self.enabled:
            return

        self.epochs.append(epoch)
        for metric, value in zip(self.metrics, values):
            self.history[metric].append(value)
            self.lines[metric].set_data(self.epochs, self.history[metric])
            ax = self.axes[metric]
            if metric in _FIXED_YLIM:
                ax.set_xlim(0.5, max(self.epochs) + 0.5)
            else:
                ax.relim()
                ax.autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
```

Copy the exact same content to `biRNN/live_plot.py` and `modRNN/live_plot.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_live_plot.py -v && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest test_live_plot.py -v && cd /Users/hoyeon/Codes/modularRNN/modRNN && pytest test_live_plot.py -v`

Expected: all 5 tests pass in all three folders (15 total).

- [ ] **Step 5: Run the existing MNIST tests' smoke path to confirm no regression**

`test_mnist.py` in all three folders calls `LiveTrainingPlot(title=...)` with no `metrics`
arg and `live_plot.update(epoch + 1, avg_loss, accuracy)` — this must keep working
unchanged. Don't run the full multi-epoch MNIST scripts (slow); just confirm by reading
`RNN/test_mnist.py`, `biRNN/test_mnist.py`, `modRNN/test_mnist.py` that their `LiveTrainingPlot`
construction and `update()` calls pass exactly 2 positional values after `epoch`, matching
the new default `metrics=("loss", "accuracy")`. No code changes needed there.

- [ ] **Step 6: Commit**

```bash
git add RNN/live_plot.py RNN/test_live_plot.py biRNN/live_plot.py biRNN/test_live_plot.py modRNN/live_plot.py modRNN/test_live_plot.py
git commit -m "Generalize LiveTrainingPlot to support an arbitrary metrics tuple"
```

---

### Task 2: Reward tracking + rendered episode for RNN/test_cartpole.py

**Files:**
- Modify: `RNN/test_cartpole.py`

**Interfaces:**
- Consumes: `LiveTrainingPlot(title, metrics=...)` / `update(epoch, *values)` from Task 1.
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Add `rollout_episode` and `evaluate_reward`, wire into `train()` and `main()`**

In `RNN/test_cartpole.py`, insert these two functions after `collate_pad` and before `train`:

```python
def rollout_episode(model, env, device, max_steps: int = 500) -> float:
    model.eval()
    state, _ = env.reset()
    states = [state]
    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = logits[0, -1].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            states.append(state)
            if terminated or truncated:
                break
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes
```

Replace the body of `train()` (keep the same signature) with:

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
            logits = model(states)  # (batch, seq_len, 2)
            loss_per_step = criterion(logits.transpose(1, 2), actions)  # (batch, seq_len)
            loss = (loss_per_step * mask).sum() / mask.sum()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)
        reward = evaluate_reward(model, device)
        print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f} reward {reward:.1f}")
        if live_plot is not None:
            live_plot.update(epoch + 1, avg_loss, accuracy, reward)
    return accuracy
```

Replace `main()` with:

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

    live_plot = LiveTrainingPlot(title="RNN/test_cartpole.py", metrics=("loss", "accuracy", "reward"))
    accuracy = train(model, train_loader, test_loader, device, epochs=10, live_plot=live_plot)
    print(f"per-timestep action accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% action accuracy, got {accuracy:.4f}"

    try:
        render_env = gym.make("CartPole-v1", render_mode="human")
        reward = rollout_episode(model, render_env, device)
        render_env.close()
        print(f"rendered episode reward: {reward:.0f}")
    except Exception as e:
        print(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
```

Everything else in the file (`collect_episodes`, `CartPoleSequenceDataset`, `collate_pad`,
`evaluate`) is unchanged.

- [ ] **Step 2: Run the full script**

Launch in the background rather than one long blocking call (per Global Constraints):

Use your Bash tool's `run_in_background: true` option (not a shell `&`, which can die when the tool call that spawned it ends) to run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 test_cartpole.py > /tmp/rnn_cartpole_reward.log 2>&1`. Then check on it with short, separate `tail -30 /tmp/rnn_cartpole_reward.log` calls rather than one long blocking wait — you'll get a notification when the backgrounded command completes.

Expected: per-epoch lines now include `reward {X}` (an average CartPole episode length,
0–500) in addition to loss/accuracy; final `per-timestep action accuracy: 0.9xxx`, no
`AssertionError` (this assertion is on action accuracy, unaffected by the new reward
logic — `RNN/`'s existing hyperparameters already cleared 90% before this change and
nothing about training changed, only added evaluation/rendering); then either
`rendered episode reward: X` or a `render skipped (no display available): ...` message
(both are acceptable outcomes — do not treat a "render skipped" message as a failure).
This run will take noticeably longer than before (see Global Constraints) — allow
significant time before concluding something is wrong; if genuinely stuck (no epoch
progress for 15+ minutes), stop and report rather than continuing to wait.

- [ ] **Step 3: Commit**

```bash
git add RNN/test_cartpole.py
git commit -m "Add closed-loop reward tracking and rendered episode to RNN/test_cartpole.py"
```

---

### Task 3: Reward tracking + rendered episode for biRNN/test_cartpole.py

**Files:**
- Modify: `biRNN/test_cartpole.py`

**Interfaces:**
- Consumes: `LiveTrainingPlot(title, metrics=...)` / `update(epoch, *values)` from Task 1.
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Add `rollout_episode` and `evaluate_reward`, wire into `train()` and `main()`**

Identical to Task 2's Step 1, except:
- `biRNN/test_cartpole.py` already imports `gym`, `heuristic_action` (via
  `sys.path.append(os.path.join(os.path.dirname(__file__), "..", "RNN"))`), and
  `BidirectionalRNN, get_device` from local `model` — keep those imports as they are.
- In `main()`, use `model = BidirectionalRNN(input_size=4, hidden_size=32, output_size=2, output_mode="all").to(device)`
  (unchanged from the current file) and
  `live_plot = LiveTrainingPlot(title="biRNN/test_cartpole.py", metrics=("loss", "accuracy", "reward"))`.
  Keep `epochs=10` (unchanged).

Insert `rollout_episode`/`evaluate_reward` (same bodies as Task 2's Step 1) after
`collate_pad` and before `train`. Update `train()`'s body and `main()` the same way as
Task 2's Step 1, adapted to `biRNN`'s model class and title string. Add the same
try/except render block at the end of `main()`, using `BidirectionalRNN`'s already-trained
`model`.

- [ ] **Step 2: Run the full script**

Use your Bash tool's `run_in_background: true` option (not a shell `&`) to run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && python3 test_cartpole.py > /tmp/birnn_cartpole_reward.log 2>&1`. Then check on it with short, separate `tail -30 /tmp/birnn_cartpole_reward.log` calls rather than one long blocking wait.

Expected: same as Task 2's Step 2 (per-epoch reward printed, `per-timestep action accuracy: 0.9xxx`
with no `AssertionError`, then a render success or skip message). This will run slower than
`RNN/`'s equivalent (hand-rolled `nn.RNNCell` loops instead of fused `nn.RNN`) — allow
more time; same "stop and report after 15+ minutes of no progress" rule applies.

- [ ] **Step 3: Commit**

```bash
git add biRNN/test_cartpole.py
git commit -m "Add closed-loop reward tracking and rendered episode to biRNN/test_cartpole.py"
```

---

### Task 4: Reward tracking + rendered episode for modRNN/test_cartpole.py

**Files:**
- Modify: `modRNN/test_cartpole.py`

**Interfaces:**
- Consumes: `LiveTrainingPlot(title, metrics=...)` / `update(epoch, *values)` from Task 1.
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Add `rollout_episode` and `evaluate_reward`, wire into `train()` and `main()`**

Identical to Task 2's Step 1, except:
- `modRNN/test_cartpole.py` already imports `gym`, `heuristic_action` (via `sys.path`),
  and `ModularBidirectionalRNN, get_device` from local `model` — keep those imports.
- In `main()`, use the file's current model construction line
  `model = ModularBidirectionalRNN(input_size=4, hidden_size=63, output_size=2, output_mode="all").to(device)`
  and `live_plot = LiveTrainingPlot(title="modRNN/test_cartpole.py", metrics=("loss", "accuracy", "reward"))`.
  Keep `epochs=25` (unchanged — this is the value already tuned in the prior plan's
  Task 3 to clear >90% action accuracy; do not change it here).

Insert `rollout_episode`/`evaluate_reward` and update `train()`/`main()` the same way as
Task 2's Step 1, adapted to `modRNN`'s model class and title string. Add the same
try/except render block at the end of `main()`.

- [ ] **Step 2: Run the full script**

Use your Bash tool's `run_in_background: true` option (not a shell `&`) to run: `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/modRNN && python3 test_cartpole.py > /tmp/modrnn_cartpole_reward.log 2>&1`. Then check on it with short, separate `tail -40 /tmp/modrnn_cartpole_reward.log` calls rather than one long blocking wait.

Expected: same shape of output as Tasks 2–3, at 25 epochs, ending in
`per-timestep action accuracy: 0.9xxx` (should still land near the previously-achieved
93.60%, since training itself is unchanged) with no `AssertionError`, then a render
success or skip message. This is the slowest of the three (hand-rolled cell loops + most
epochs + rollouts growing to the largest episode lengths once the model performs well) —
allow the most time here; same "stop and report after 15+ minutes of no progress" rule
applies, but don't be surprised if this one legitimately takes well beyond that — check
the log file's timestamps/progress rather than assuming a hang from wall-clock alone.

- [ ] **Step 3: Commit**

```bash
git add modRNN/test_cartpole.py
git commit -m "Add closed-loop reward tracking and rendered episode to modRNN/test_cartpole.py"
```

---

## Final check

- [ ] Run `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest test_live_plot.py test_model.py -v && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest test_live_plot.py test_model.py -v && cd /Users/hoyeon/Codes/modularRNN/modRNN && pytest test_live_plot.py test_model.py -v` — all pass (5+5=10 in RNN, 5+7=12 in biRNN, 5+12=17 in modRNN).
- [ ] Confirm all three `test_cartpole.py` scripts ran to completion with per-epoch reward output, a passing `>90%` action-accuracy assertion, and either a rendered episode or a graceful skip message.
- [ ] Confirm `test_mnist.py` in all three folders is untouched (no diff) — Task 1 was designed to require zero changes there.
