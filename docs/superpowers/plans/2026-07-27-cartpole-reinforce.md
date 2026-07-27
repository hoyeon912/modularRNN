# CartPole REINFORCE Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace behavior-cloning training (supervised imitation of a heuristic policy) with
REINFORCE (vanilla Monte-Carlo policy gradient) in all three folders' `test_cartpole.py`,
per `docs/superpowers/specs/2026-07-27-cartpole-reinforce-design.md`. The model learns
purely from its own environment interaction and reward signal — no heuristic policy, no
pre-collected dataset.

**Architecture:** Each folder's `test_cartpole.py` drops `collect_episodes`,
`CartPoleSequenceDataset`, `collate_pad`, `evaluate`, and the `heuristic_action` import
entirely. New functions: `collect_episode_stochastic` (causal growing-prefix rollout with
sampled actions and gradient-tracked log-probs/entropies), `compute_returns` (discounted
reward-to-go), `reinforce_update` (policy-gradient loss with a batch-mean baseline and an
entropy bonus). The existing `rollout_episode`/`evaluate_reward` (greedy, `torch.no_grad()`)
are kept unchanged for periodic eval and the final rendered episode.

**Tech Stack:** Same `.venv` as the rest of the repo (PyTorch, Gymnasium, matplotlib) — no
new dependencies.

## Global Constraints

- Use the existing `.venv` at repo root — no new dependencies.
- `RNN/heuristic_policy.py` is **not deleted** — only unused by `test_cartpole.py` going
  forward in all three folders.
- `live_plot.py`, `model.py`, `test_model.py`, `test_mnist.py`, `test_live_plot.py` are
  **not touched** by this plan in any folder — `LiveTrainingPlot` already supports an
  arbitrary `metrics` tuple, so `metrics=("loss", "reward")` requires no changes there.
- Hyperparameters, identical across all three folders except `hidden_size` (as before) and
  `episodes_per_update` (see MPS fix below — `RNN=8`, `biRNN=4`, `modRNN=4`): `gamma=0.99`,
  `num_updates=100`, `max_steps=500`, `entropy_coef=0.01`, `lr=1e-3` (Adam).
- Success bar: `assert avg_reward > 150` (out of a 500 max) after the fixed 100-update
  budget — deliberately modest relative to CartPole-v1's official 475+ "solved" bar, given
  the compute cost noted below and vanilla REINFORCE's high sample complexity.
- **Early stopping (added mid-execution, user request):** `train()` breaks out of the
  update loop as soon as `avg_reward >= 500` (the environment's max possible reward) is
  reached, printing a message and returning immediately, rather than continuing to the
  full `num_updates` budget. This matters a lot in practice: once the policy is solved,
  every remaining update's rollouts run at (or near) the full 500-step cap, which is by
  far the most expensive regime for the O(T²) causal-rollout cost — RNN's actual first run
  hit reward 500 at update 38/100 and, without this check, would have spent the other 62
  updates at maximum per-episode cost for zero additional benefit.
- **MPS out-of-memory crash (added mid-execution, recurring on `biRNN`):** `biRNN` crashed
  with `Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)` across four
  consecutive attempts, at update numbers 26, 38, 32, and 37 — always once episodes started
  regularly exceeding ~150-300 steps. This tracks *peak* memory during a single update's
  batched backward pass (which grows with episode length as the policy improves), not
  cross-update accumulation. `torch.mps.empty_cache()` between updates was tried at three
  frequencies (never, every 10th, every single update) and **none reliably prevented the
  crash** — clearing between updates cannot reduce the peak a single update's own backward
  pass needs while it's running, and "every single update" also made training far slower
  for no proven benefit (confirmed only by chance timing in one earlier run, not by
  actually surviving to completion). `train()` keeps the every-10th-update
  `torch.mps.empty_cache()` call as a harmless minor hygiene measure (RNN's original,
  never-crashed setting) but it is not the actual fix.

  **The actual fix: reduce `episodes_per_update` from 8 to 4 for `biRNN` and `modRNN`**
  (RNN keeps 8 — its one run completed without ever crashing). This directly halves how
  many episodes' log_probs/entropies are held in one `reinforce_update` batch and
  backpropped together in a single call, directly shrinking the peak memory that call
  needs — unlike cache-clearing, which only affects memory *between* update calls, not
  the peak *during* one.
- **If 150 isn't cleared in 100 updates:** report the observed reward trend and ask the
  user before changing the budget or any hyperparameter — do not silently raise
  `num_updates`/`episodes_per_update` repeatedly or lower the threshold. This mirrors the
  escalation discipline already used for `modRNN`'s CartPole hyperparameter tuning.
- **No new automated tests for the REINFORCE math** (`compute_returns`, `reinforce_update`) —
  consistent with this repo's existing convention that `test_cartpole.py`'s training logic
  is verified only by running the full script and checking the printed/asserted result, not
  by unit tests (unlike `model.py`/`live_plot.py`, which do have `pytest` coverage).
- **Long-running steps:** every training episode now costs the same O(T²) causal-rollout
  compute that was previously only paid for periodic eval — expect these runs to take
  substantially longer than any prior `test_cartpole.py` run in this repo, especially for
  `biRNN`/`modRNN`'s hand-rolled cell loops. Launch with your Bash tool's
  `run_in_background: true` (not a shell `&`) and `python3 -u` (unbuffered), and check
  progress with short, separate `tail` calls rather than one long blocking wait.

---

### Task 1: REINFORCE training for RNN/test_cartpole.py

**Files:**
- Modify: `RNN/test_cartpole.py` (full rewrite of most of the file)

**Interfaces:**
- Consumes: `SimpleRNN`, `get_device` from `RNN/model.py` (unchanged); `LiveTrainingPlot`
  from `RNN/live_plot.py` (unchanged, already supports `metrics` tuples).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Replace `RNN/test_cartpole.py`'s contents**

```python
import gymnasium as gym
import torch

from live_plot import LiveTrainingPlot
from model import SimpleRNN, get_device


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


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
    model.train()
    state, _ = env.reset()
    states = [state]
    log_probs = []
    entropies = []
    rewards = []
    for _ in range(max_steps):
        x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        dist = torch.distributions.Categorical(logits=logits[0, -1])
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())
        state, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        states.append(state)
        if terminated or truncated:
            break
    return log_probs, entropies, rewards


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def reinforce_update(model, optimizer, episode_batch, gamma: float = 0.99, entropy_coef: float = 0.01) -> float:
    all_log_probs = []
    all_entropies = []
    all_returns = []
    for log_probs, entropies, rewards in episode_batch:
        all_log_probs.extend(log_probs)
        all_entropies.extend(entropies)
        all_returns.extend(compute_returns(rewards, gamma))

    returns_tensor = torch.tensor(all_returns, dtype=torch.float32, device=all_log_probs[0].device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    log_probs_tensor = torch.stack(all_log_probs)
    entropy_tensor = torch.stack(all_entropies)
    policy_loss = -(log_probs_tensor * advantages).mean() - entropy_coef * entropy_tensor.mean()

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    return policy_loss.item()


def train(model, device, num_updates: int, episodes_per_update: int = 8, live_plot=None) -> float:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    env = gym.make("CartPole-v1")
    avg_reward = 0.0
    for update in range(num_updates):
        batch = [collect_episode_stochastic(model, env, device) for _ in range(episodes_per_update)]
        loss = reinforce_update(model, optimizer, batch)
        avg_reward = evaluate_reward(model, device)
        print(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {avg_reward:.1f}")
        if live_plot is not None:
            live_plot.update(update + 1, loss, avg_reward)
        if device.type == "mps" and (update + 1) % 10 == 0:
            torch.mps.empty_cache()
        if avg_reward >= 500:
            print(f"reached max reward (500) at update {update + 1}, stopping early")
            break
    env.close()
    return avg_reward


def main():
    device = get_device()
    print(f"using device: {device}")

    model = SimpleRNN(input_size=4, hidden_size=32, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="RNN/test_cartpole.py", metrics=("loss", "reward"))
    avg_reward = train(model, device, num_updates=100, live_plot=live_plot)
    print(f"average reward: {avg_reward:.1f}")
    assert avg_reward > 150, f"expected average reward > 150, got {avg_reward:.1f}"

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

This removes `import numpy as np`, `import torch.nn as nn`,
`from torch.utils.data import DataLoader, Dataset`, and
`from heuristic_policy import heuristic_action`, along with `collect_episodes`,
`CartPoleSequenceDataset`, `collate_pad`, and the old heuristic-based `evaluate`/`train`/`main`.

- [ ] **Step 2: Run the full script**

Use your Bash tool's `run_in_background: true` option (not a shell `&`) to run:
`source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && python3 -u test_cartpole.py > /tmp/rnn_cartpole_reinforce.log 2>&1`.
Then check on it with short, separate `tail -30 /tmp/rnn_cartpole_reinforce.log` calls.

Expected: 100 `update N/100 loss ... reward ...` lines, reward generally trending upward
(noisy — REINFORCE is high-variance, don't expect monotonic improvement), then
`average reward: X.X` with no `AssertionError`, then a render success or graceful skip
message. This will take longer than any previous `test_cartpole.py` run in this repo — do
not assume a hang without checking the log's actual progress; if genuinely no update has
completed in 20+ minutes, stop and report rather than continuing to wait.

**If `avg_reward` does not exceed 150:** stop, report the full per-update reward trend to
the user, and ask how to proceed (more updates? larger `episodes_per_update`? accept as a
reported result?) — per Global Constraints, do not silently retune.

- [ ] **Step 3: Commit**

```bash
git add RNN/test_cartpole.py
git commit -m "Replace behavior cloning with REINFORCE training in RNN/test_cartpole.py"
```

---

### Task 2: REINFORCE training for biRNN/test_cartpole.py

**Files:**
- Modify: `biRNN/test_cartpole.py` (full rewrite of most of the file)

**Interfaces:**
- Consumes: `BidirectionalRNN`, `get_device` from `biRNN/model.py` (unchanged);
  `LiveTrainingPlot` from `biRNN/live_plot.py` (unchanged).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Replace `biRNN/test_cartpole.py`'s contents**

Identical to Task 1's Step 1, except:
- No `sys.path`/`os`/`sys` imports needed at all (they existed only to import
  `heuristic_action`, which is being removed) — the new file's imports are exactly
  `import gymnasium as gym`, `import torch`, `from live_plot import LiveTrainingPlot`,
  `from model import BidirectionalRNN, get_device`.
- In `main()`, use `model = BidirectionalRNN(input_size=4, hidden_size=32, output_size=2, output_mode="all").to(device)`
  and `live_plot = LiveTrainingPlot(title="biRNN/test_cartpole.py", metrics=("loss", "reward"))`,
  and call `train(model, device, num_updates=100, episodes_per_update=4, live_plot=live_plot)`
  (per the Global Constraints "MPS out-of-memory crash" note: `episodes_per_update=4`, not
  `8`, is the actual fix for `biRNN`'s recurring crash — cache-clearing frequency alone
  did not prevent it across four attempts).
- `train()`'s MPS cache-clearing keeps the every-10th-update condition
  (`if device.type == "mps" and (update + 1) % 10 == 0:`), same as Task 1 — harmless minor
  hygiene, not the fix.
- All other functions (`rollout_episode`, `evaluate_reward`, `collect_episode_stochastic`,
  `compute_returns`, `reinforce_update`) are byte-identical to Task 1's.

- [ ] **Step 2: Run the full script**

Use your Bash tool's `run_in_background: true` option to run:
`source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/biRNN && python3 -u test_cartpole.py > /tmp/birnn_cartpole_reinforce.log 2>&1`.
Then check on it with short, separate `tail -30 /tmp/birnn_cartpole_reinforce.log` calls.

Expected: same shape of output as Task 1. This will run slower than `RNN/`'s equivalent
(hand-rolled `nn.RNNCell` loops instead of fused `nn.RNN`) — allow more time; same
"stop and report after 20+ minutes of no update progress" rule applies. Same
"ask before retuning if <150" rule applies too.

- [ ] **Step 3: Commit**

```bash
git add biRNN/test_cartpole.py
git commit -m "Replace behavior cloning with REINFORCE training in biRNN/test_cartpole.py"
```

---

### Task 3: REINFORCE training for modRNN/test_cartpole.py

**Files:**
- Modify: `modRNN/test_cartpole.py` (full rewrite of most of the file)

**Interfaces:**
- Consumes: `ModularBidirectionalRNN`, `get_device` from `modRNN/model.py` (unchanged);
  `LiveTrainingPlot` from `modRNN/live_plot.py` (unchanged).
- Produces: a standalone script; no other task depends on its internals.

- [ ] **Step 1: Replace `modRNN/test_cartpole.py`'s contents**

Identical to Task 2's Step 1, except:
- In `main()`, use the file's current model construction line exactly as-is:
  `model = ModularBidirectionalRNN(input_size=4, hidden_size=63, output_size=2, output_mode="all").to(device)`
  (keep `hidden_size=63` — this was tuned in an earlier, separate plan; do not change it)
  and `live_plot = LiveTrainingPlot(title="modRNN/test_cartpole.py", metrics=("loss", "reward"))`,
  and call `train(model, device, num_updates=300, episodes_per_update=4, live_plot=live_plot)`
  — same `episodes_per_update=4` fix as `biRNN` (Task 2), applied preemptively since
  `modRNN`'s hand-rolled cells share the same memory-scaling characteristics.
  **`num_updates=300`, not 100 (user-approved deviation):** a first run at `num_updates=100`
  completed cleanly (no crash) but ended at `avg_reward=18.3`, far short of `>150` — reward
  stayed near chance level (~9-12) for the first 63/100 updates, briefly peaked at 80 around
  update 77, then declined. modRNN's restricted/sparse connectivity plausibly needs
  substantially more updates to find a working policy than `RNN`/`biRNN`'s dense
  connections did (both solved in ~35-51 updates) — consistent with modRNN's
  already-observed weaker closed-loop control in the prior reward-tracking work. The user
  chose to raise the budget to 300 rather than accept the result or retune other
  hyperparameters first.
- `train()`'s MPS cache-clearing keeps the every-10th-update condition, same as Task 1/2.
- All other functions are byte-identical to Task 1/2's.

- [ ] **Step 2: Run the full script**

Use your Bash tool's `run_in_background: true` option to run:
`source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/modRNN && python3 -u test_cartpole.py > /tmp/modrnn_cartpole_reinforce.log 2>&1`.
Then check on it with short, separate `tail -40 /tmp/modrnn_cartpole_reinforce.log` calls.

Expected: same shape of output as Tasks 1-2. This is the slowest of the three (hand-rolled
cells + modRNN's restricted/sparser connectivity, which the reward-tracking work already
showed struggles more at closed-loop control than RNN/biRNN) — allow the most time here.
Same "stop and report after 20+ minutes of no update progress" rule applies, and the same
"ask before retuning if <150" rule applies — given modRNN's previously-observed weaker
closed-loop reward (~11-37 under behavior cloning), it would not be surprising if REINFORCE
also lands lower here than in RNN/biRNN; report the actual trend rather than assuming
something is broken.

- [ ] **Step 3: Commit**

```bash
git add modRNN/test_cartpole.py
git commit -m "Replace behavior cloning with REINFORCE training in modRNN/test_cartpole.py"
```

---

## Final check

- [ ] Confirm all three `test_cartpole.py` files no longer import `heuristic_action`,
  `numpy`, `torch.nn`, or `torch.utils.data`, and no longer define `collect_episodes`,
  `CartPoleSequenceDataset`, `collate_pad`, or the heuristic-based `evaluate`.
- [ ] Confirm `RNN/heuristic_policy.py` still exists on disk, unmodified.
- [ ] Confirm all three scripts ran to completion with per-update loss/reward output, a
  passing `>150` average-reward assertion (or an explicitly user-approved deviation,
  documented the same way the `modRNN` CartPole hyperparameter deviation was), and either
  a rendered episode or a graceful skip message.
- [ ] Run `source /Users/hoyeon/Codes/modularRNN/.venv/bin/activate && cd /Users/hoyeon/Codes/modularRNN/RNN && pytest -q && cd /Users/hoyeon/Codes/modularRNN/biRNN && pytest -q && cd /Users/hoyeon/Codes/modularRNN/modRNN && pytest -q` — all existing tests (`test_model.py`, `test_live_plot.py`) still pass unchanged (39 total), confirming this plan didn't touch anything it wasn't supposed to.
