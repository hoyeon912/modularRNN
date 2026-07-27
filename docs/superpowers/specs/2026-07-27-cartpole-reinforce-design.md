# CartPole REINFORCE Training — Design

## Purpose

`RNN/test_cartpole.py`, `biRNN/test_cartpole.py`, and `modRNN/test_cartpole.py` currently
train via behavior cloning: `collect_episodes()` generates trajectories using a fixed
hand-coded `heuristic_action` policy, and the model is trained with supervised
cross-entropy loss to imitate the heuristic's action at each timestep. The model never
learns from the environment's actual reward signal, and "per-timestep action accuracy"
(how often it matches the heuristic) is a proxy metric, not a measure of control quality —
this was the motivation for the closed-loop reward tracking added previously.

This design replaces behavior cloning with genuine reinforcement learning: REINFORCE
(vanilla Monte-Carlo policy gradient). The model learns purely from its own environment
interaction and the reward it receives — no heuristic policy, no pre-collected dataset.

## Architecture

### Removed from each folder's `test_cartpole.py`

`collect_episodes`, `CartPoleSequenceDataset`, `collate_pad`, `evaluate` (the heuristic
action-accuracy check), and the `heuristic_action` import
(`sys.path.append(...) / from heuristic_policy import heuristic_action` in `biRNN`/`modRNN`).
`RNN/heuristic_policy.py` itself is left on disk unmodified — it's simply no longer used by
`test_cartpole.py` in any folder.

### Causal rollout is bidirectional-safe

Feeding the *growing* state-history prefix (`states[0..t]`) into a bidirectional model at
each step is causally valid: the tensor never contains real future data, it's both-direction
processing of only what's been observed so far. This is exactly what the pre-existing
`rollout_episode` (from the reward-tracking feature) already does, and REINFORCE reuses the
same mechanism for both training rollouts and eval rollouts — just with different action
selection at the final step (stochastic sampling vs. greedy argmax).

### Training rollout — `collect_episode_stochastic`

```python
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
```

Gradients flow through `log_probs` and `entropies` (no `torch.no_grad()`), unlike the
existing greedy `rollout_episode` used for eval, which stays exactly as-is (unchanged,
still `torch.no_grad()`, still `argmax`).

### Policy-gradient update — `reinforce_update`

```python
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
```

- **Reward-to-go:** each timestep's return is the discounted sum of *future* rewards from
  that point on (`compute_returns`), not the whole episode's total — standard variance
  reduction, cheap to add.
- **Baseline:** the batch's mean return, subtracted from every timestep's return before
  weighting the log-prob. No separate value network — keeps the "no shared base class /
  no new dependency" spirit of the rest of the repo.
- **Entropy bonus:** `entropy_coef * mean(entropy)` is subtracted from the loss (i.e. added
  as a reward for higher-entropy/more-exploratory policies), discouraging premature
  collapse to a deterministic policy — a standard, cheap stabilizer for vanilla REINFORCE
  that adds no new network or dependency.

### Training loop

"Epoch" is renamed to "update" (matches RL terminology, and avoids implying a fixed
dataset is being iterated, since there isn't one anymore):

```python
def train(model, device, num_updates, episodes_per_update=8, live_plot=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    avg_reward = 0.0
    for update in range(num_updates):
        batch = [collect_episode_stochastic(model, gym.make("CartPole-v1"), device) for _ in range(episodes_per_update)]
        loss = reinforce_update(model, optimizer, batch)
        avg_reward = evaluate_reward(model, device)  # existing function, unchanged, greedy/no_grad
        print(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {avg_reward:.1f}")
        if live_plot is not None:
            live_plot.update(update + 1, loss, avg_reward)
    return avg_reward
```

(Illustrative — the real implementation reuses one `gym.make("CartPole-v1")` env across
the batch rather than remaking it per episode, for efficiency.)

### Live plot

`LiveTrainingPlot(title=..., metrics=("loss", "reward"))` — 2 metrics instead of the
previous 3 ("accuracy" is gone, since there's no heuristic to compare against). No changes
needed to `live_plot.py` itself — it already supports an arbitrary `metrics` tuple.

### Success bar and training budget

`assert avg_reward > 150, ...` after a fixed `num_updates=100` (`episodes_per_update=8`) —
a deliberately modest target relative to CartPole-v1's official "solved" bar of 475+ over
100 consecutive episodes, given the compute cost of O(T²) causal rollouts (now paid on
*every* training episode, not just periodic eval) and vanilla REINFORCE's known high
sample complexity. If 150 isn't cleared in 100 updates, that's a reportable result — the
same escalation discipline as prior tuning work applies: surface the observed trend and
ask before changing the budget, rather than silently lowering the bar or looping
indefinitely.

### Rendered episode

Unchanged from the existing reward-tracking feature: one greedy (`argmax`, `torch.no_grad()`)
rollout with `render_mode="human"` after training, wrapped in try/except.

## Hyperparameters (identical across all three folders, only `hidden_size` differs as before)

- `gamma = 0.99`
- `episodes_per_update = 8`
- `num_updates = 100`
- `max_steps = 500`
- `entropy_coef = 0.01`
- `lr = 1e-3` (Adam, unchanged)

## Non-goals / accepted tradeoffs

- No new dependencies (no external RL library) — REINFORCE implemented from scratch,
  consistent with the rest of the repo's hand-rolled style.
- No learned value-function baseline (Actor-Critic) — batch-mean baseline only, to keep
  the change scoped to "replace the training algorithm," not "add a second network."
- Training will be substantially slower than the previous behavior-cloning approach: every
  training episode now costs the same O(T²) causal-rollout compute that was previously
  only paid for 3 periodic eval episodes per epoch. Explicitly accepted, not optimized
  around, in this iteration.
- `RNN/heuristic_policy.py` is not deleted, only unused by `test_cartpole.py` going forward.
- `test_model.py` / `test_mnist.py` / `live_plot.py` in all three folders are unaffected —
  this change is scoped entirely to each folder's `test_cartpole.py`.
