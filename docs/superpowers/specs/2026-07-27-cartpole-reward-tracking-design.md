# CartPole Reward Tracking + Visualization — Design

## Purpose

`RNN/test_cartpole.py`, `biRNN/test_cartpole.py`, and `modRNN/test_cartpole.py` currently
train via behavior cloning against a heuristic policy and only report "per-timestep action
accuracy" — how often the model's predicted action matches the heuristic's, evaluated on
heuristic-generated trajectories. This never actually runs the model as a controller, so
there's no way to see how well it balances the pole under its own control. This design adds:

1. A live-plotted reward metric (actual CartPole reward under the model's own control),
   tracked every epoch alongside the existing loss/accuracy.
2. A rendered episode at the end of training so the pole balancing can be watched directly.

Both are added independently to all three folders (`RNN/`, `biRNN/`, `modRNN/`), each
duplicating the logic per the project's existing no-shared-base-class convention.

## Architecture

### Closed-loop rollout (new, per folder's `test_cartpole.py`)

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
```

This differs fundamentally from the existing `evaluate()`: `evaluate()` checks the model's
predictions against a heuristic-generated trajectory (the states were never influenced by
the model). `rollout_episode` actually drives the environment with the model's own greedy
(`argmax`) action at every step, so the resulting reward reflects real control performance.

Because the models are bidirectional, each step's forward pass must recompute the full
backward pass over the sequence-so-far — there is no incremental way to extend a backward
RNN pass as new timesteps are appended at the tail. This makes each rollout episode O(T²)
in the episode length T, instead of O(T). This is an accepted, known cost (see Non-goals).

### Per-epoch reward metric

```python
def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes
```

Called once per epoch, after the existing `evaluate()` (action-accuracy) call, inside
`train()`. Averages 3 closed-loop episodes per epoch.

### Live plot: `LiveTrainingPlot` becomes metric-generic

Current constructor is `LiveTrainingPlot(title: str)`, hardcoded to 2 subplots
(loss, accuracy). New constructor: `LiveTrainingPlot(title: str, metrics: tuple[str, ...] =
("loss", "accuracy"))`, building one subplot per metric name in a `1 × len(metrics)` grid.
Each known metric name gets a fixed y-axis policy:

- `"loss"` — autoscaled (current behavior)
- `"accuracy"` — fixed `ylim(0, 1)` (current behavior)
- `"reward"` — fixed `ylim(0, 500)` (CartPole-v1's max episode length)

`update(self, epoch: int, *values: float)` replaces the current fixed-arity
`update(self, epoch, loss, accuracy)` — `values` is zipped positionally against
`self.metrics`, updating each subplot's line and axis.

`test_mnist.py` in all three folders needs **no changes**: it doesn't pass `metrics`, so
the default `("loss", "accuracy")` preserves current behavior, and its existing
`live_plot.update(epoch + 1, avg_loss, accuracy)` call matches the new `*values` signature
positionally (2 values → 2 metrics, same as before).

`test_cartpole.py` in all three folders passes `metrics=("loss", "accuracy", "reward")`
and, inside `train()`, computes `reward = evaluate_reward(model, device)` each epoch and
calls `live_plot.update(epoch + 1, avg_loss, accuracy, reward)`.

### Rendered episode at the end of training

Unconditional (no CLI flag) — after training completes and the existing `>90%` assertion
passes, each `test_cartpole.py`'s `main()` does:

```python
try:
    render_env = gym.make("CartPole-v1", render_mode="human")
    reward = rollout_episode(model, render_env, device)
    render_env.close()
    print(f"rendered episode reward: {reward:.0f}")
except Exception as e:
    print(f"render skipped (no display available): {e}")
```

Wrapped in try/except so a headless machine (no display) degrades to a printed message
instead of crashing — matching `LiveTrainingPlot`'s existing no-GUI fallback pattern
(`live_plot.py`'s `try/except` around `plt.ion()`/`plt.subplots()`).

## Non-goals / accepted tradeoffs

- **Performance:** the O(T²)-per-rollout-episode cost (inherent to bidirectional
  architectures — no incremental backward-pass extension is possible) will noticeably
  increase `test_cartpole.py`'s total runtime beyond its current several-minutes baseline,
  especially in later epochs once episodes survive longer. Accepted tradeoff, confirmed
  with the user; not addressed by reducing eval frequency or episode count below what was
  requested (every epoch, 3 episodes).
- No change to the training procedure itself (still open-loop behavior cloning against the
  heuristic's trajectories) — only new evaluation/visualization is added.
- No CLI flags — rendering is unconditional, per user's explicit choice.
- `RNN/test_cartpole.py`'s `SimpleRNN` and `biRNN`/`modRNN`'s hand-rolled equivalents all
  expose the same `forward(x)` → `(batch, seq_len, output_size)` contract under
  `output_mode="all"`, so `rollout_episode` is structurally identical across all three
  folders (only the imported model class differs) — duplicated per file, not shared.
