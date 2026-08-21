# CartPole RNN-DQN Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `tests/cartpole.ipynb` as an executable RNN-DQN tutorial that learns from the eight latest CartPole observations and stops at a 475 rolling-average score or 1,000 episodes.

**Architecture:** Keep the implementation in one notebook, split into small ordered cells. Store complete `(state_sequence, action, reward, next_state_sequence, done)` transitions in a bounded replay buffer; train an online RNN against a periodically copied target RNN.

**Tech Stack:** Python, PyTorch, Gymnasium, NumPy, Matplotlib, Jupyter

## Global Constraints

- Modify only `tests/cartpole.ipynb`; do not change `models/RNN.py` or create reusable modules or test files.
- Every model input has shape `(batch, 8, 4)` and every model output has shape `(batch, 2)`.
- Pad each new episode by repeating its initial observation eight times.
- Treat both `terminated` and `truncated` as terminal for DQN targets.
- Stop after a latest-100 return mean of 475 or after 1,000 episodes.
- Keep the design and implementation-plan documents untracked by Git.

---

### Task 1: Replace the exploratory cells with deterministic setup and sequence helpers

**Files:**
- Modify: `tests/cartpole.ipynb`

**Interfaces:**
- Produces: constants `SEQUENCE_LENGTH`, `OBSERVATION_SIZE`, `ACTION_SIZE`, `DEVICE`; functions `make_sequence(observation) -> np.ndarray` and `append_observation(sequence, observation) -> np.ndarray`.

- [ ] **Step 1: Add the imports, seed, environment, device, and hyperparameter cell**

```python
import math
import random
from collections import deque, namedtuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
SEQUENCE_LENGTH = 8
HIDDEN_SIZE = 128
BATCH_SIZE = 64
REPLAY_CAPACITY = 50_000
GAMMA = 0.99
LEARNING_RATE = 1e-3
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 20_000
TARGET_UPDATE_FREQUENCY = 1_000
LEARNING_STARTS = 1_000
MAX_EPISODES = 1_000
SOLVED_SCORE = 475.0
SOLVED_WINDOW = 100
EVALUATION_EPISODES = 10

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make("CartPole-v1")
env.action_space.seed(SEED)
initial_observation, _ = env.reset(seed=SEED)

assert env.observation_space.shape == (4,)
assert isinstance(env.action_space, gym.spaces.Discrete)
OBSERVATION_SIZE = env.observation_space.shape[0]
ACTION_SIZE = env.action_space.n
print(f"device={DEVICE}, observation={OBSERVATION_SIZE}, actions={ACTION_SIZE}")
```

- [ ] **Step 2: Add assertions that initially fail because sequence helpers are undefined**

```python
test_observation = np.arange(OBSERVATION_SIZE, dtype=np.float32)
test_sequence = make_sequence(test_observation)
assert test_sequence.shape == (SEQUENCE_LENGTH, OBSERVATION_SIZE)
assert np.all(test_sequence == test_observation)

new_observation = np.full(OBSERVATION_SIZE, 9.0, dtype=np.float32)
shifted = append_observation(test_sequence, new_observation)
assert np.array_equal(shifted[:-1], test_sequence[1:])
assert np.array_equal(shifted[-1], new_observation)
assert np.array_equal(test_sequence[0], test_observation), "helper must not mutate input"
```

- [ ] **Step 3: Run through the assertion cell and verify the intended failure**

Run the setup cell and assertion cell.

Expected: `NameError: name 'make_sequence' is not defined`.

- [ ] **Step 4: Insert the minimal sequence helpers before their assertions**

```python
def make_sequence(observation):
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"expected observation shape {(OBSERVATION_SIZE,)}, got {observation.shape}")
    return np.repeat(observation[None, :], SEQUENCE_LENGTH, axis=0)


def append_observation(sequence, observation):
    sequence = np.asarray(sequence, dtype=np.float32)
    observation = np.asarray(observation, dtype=np.float32)
    if sequence.shape != (SEQUENCE_LENGTH, OBSERVATION_SIZE):
        raise ValueError(f"unexpected sequence shape {sequence.shape}")
    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"unexpected observation shape {observation.shape}")
    return np.concatenate((sequence[1:], observation[None, :]), axis=0)
```

- [ ] **Step 5: Re-run the cells**

Expected: all sequence assertions pass and no array is mutated.

### Task 2: Add the RNN Q-network and replay buffer contracts

**Files:**
- Modify: `tests/cartpole.ipynb`

**Interfaces:**
- Consumes: `SEQUENCE_LENGTH`, `OBSERVATION_SIZE`, `ACTION_SIZE`, `DEVICE`.
- Produces: `VanillaRNN`, `Transition`, `ReplayBuffer`, `online_network`, `target_network`, `optimizer`, `loss_function`, and `replay_buffer`.

- [ ] **Step 1: Add a failing network contract cell**

```python
network = VanillaRNN(OBSERVATION_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(DEVICE)
dummy_states = torch.zeros(4, SEQUENCE_LENGTH, OBSERVATION_SIZE, device=DEVICE)
dummy_q_values = network(dummy_states)
assert dummy_q_values.shape == (4, ACTION_SIZE)

try:
    network(torch.zeros(4, OBSERVATION_SIZE, device=DEVICE))
except ValueError:
    pass
else:
    raise AssertionError("rank-2 input must be rejected")
```

- [ ] **Step 2: Run it and verify the intended failure**

Expected: `NameError: name 'VanillaRNN' is not defined`.

- [ ] **Step 3: Insert the Q-network before the contract cell**

```python
class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"expected (batch, sequence, features), got {tuple(x.shape)}")
        if x.shape[1:] != (SEQUENCE_LENGTH, OBSERVATION_SIZE):
            raise ValueError(f"unexpected sequence dimensions {tuple(x.shape[1:])}")
        h0 = torch.zeros(1, x.shape[0], self.hidden_size, device=x.device, dtype=x.dtype)
        output, _ = self.rnn(x, h0)
        return self.fc(output[:, -1, :])
```

- [ ] **Step 4: Add the replay buffer**

```python
Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done"]
)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            Transition(state.copy(), int(action), float(reward), next_state.copy(), bool(done))
        )

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
```

- [ ] **Step 5: Add replay isolation assertions**

```python
buffer_check = ReplayBuffer(2)
state_check = make_sequence(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
next_check = append_observation(state_check, np.ones(OBSERVATION_SIZE, dtype=np.float32))
buffer_check.push(state_check, 1, 1.0, next_check, False)
state_check[:] = 99.0
stored = buffer_check.sample(1)[0]
assert stored.state.shape == (SEQUENCE_LENGTH, OBSERVATION_SIZE)
assert not np.any(stored.state == 99.0), "buffer must own copies"
```

- [ ] **Step 6: Initialize training objects and re-run all Task 2 cells**

```python
online_network = VanillaRNN(OBSERVATION_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(DEVICE)
target_network = VanillaRNN(OBSERVATION_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(DEVICE)
target_network.load_state_dict(online_network.state_dict())
target_network.eval()

optimizer = optim.Adam(online_network.parameters(), lr=LEARNING_RATE)
loss_function = nn.SmoothL1Loss()
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
```

Expected: network shape, rejected rank-2 input, and replay-copy assertions all pass.

### Task 3: Implement epsilon-greedy selection and one DQN update

**Files:**
- Modify: `tests/cartpole.ipynb`

**Interfaces:**
- Consumes: networks, optimizer, loss function, replay buffer, DQN constants.
- Produces: `epsilon_at(step) -> float`, `select_action(sequence, step) -> int`, and `optimize_model() -> float | None`.

- [ ] **Step 1: Add failing epsilon assertions**

```python
assert epsilon_at(0) == EPSILON_START
assert EPSILON_END < epsilon_at(EPSILON_DECAY_STEPS) < EPSILON_START
assert abs(epsilon_at(1_000_000) - EPSILON_END) < 1e-6
```

- [ ] **Step 2: Run and confirm `epsilon_at` is undefined, then implement it**

```python
def epsilon_at(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(
        -step / EPSILON_DECAY_STEPS
    )
```

- [ ] **Step 3: Add action selection**

```python
def select_action(sequence, step):
    if random.random() < epsilon_at(step):
        return env.action_space.sample()

    state_tensor = torch.as_tensor(sequence, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q_values = online_network(state_tensor)
    if not torch.isfinite(q_values).all():
        raise FloatingPointError("non-finite Q-value during action selection")
    return int(q_values.argmax(dim=1).item())
```

- [ ] **Step 4: Populate synthetic replay data before implementing optimization**

```python
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
base = make_sequence(np.zeros(OBSERVATION_SIZE, dtype=np.float32))
for index in range(LEARNING_STARTS):
    next_state = append_observation(
        base, np.full(OBSERVATION_SIZE, index / LEARNING_STARTS, dtype=np.float32)
    )
    replay_buffer.push(base, index % ACTION_SIZE, 1.0, next_state, index % 7 == 0)

smoke_loss = optimize_model()
assert smoke_loss is not None
assert math.isfinite(smoke_loss)
```

- [ ] **Step 5: Run and confirm `optimize_model` is undefined, then implement it**

```python
def optimize_model():
    if len(replay_buffer) < max(BATCH_SIZE, LEARNING_STARTS):
        return None

    transitions = replay_buffer.sample(BATCH_SIZE)
    states = torch.as_tensor(
        np.stack([item.state for item in transitions]), dtype=torch.float32, device=DEVICE
    )
    actions = torch.as_tensor(
        [item.action for item in transitions], dtype=torch.long, device=DEVICE
    )
    rewards = torch.as_tensor(
        [item.reward for item in transitions], dtype=torch.float32, device=DEVICE
    )
    next_states = torch.as_tensor(
        np.stack([item.next_state for item in transitions]), dtype=torch.float32, device=DEVICE
    )
    dones = torch.as_tensor(
        [item.done for item in transitions], dtype=torch.float32, device=DEVICE
    )

    assert states.shape == (BATCH_SIZE, SEQUENCE_LENGTH, OBSERVATION_SIZE)
    assert next_states.shape == states.shape

    chosen_q_values = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_network(next_states).max(dim=1).values
        targets = rewards + GAMMA * (1.0 - dones) * next_q_values

    if not torch.isfinite(chosen_q_values).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("non-finite value in DQN update")

    loss = loss_function(chosen_q_values, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite DQN loss")

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()
    return float(loss.item())
```

- [ ] **Step 6: Run one smoke-test backward pass**

```python
smoke_loss = optimize_model()
assert smoke_loss is not None and math.isfinite(smoke_loss)
print(f"smoke loss: {smoke_loss:.4f}")
```

Expected: one finite optimization loss and no shape assertion failure.

### Task 4: Add bounded training, plots, and greedy evaluation

**Files:**
- Modify: `tests/cartpole.ipynb`

**Interfaces:**
- Consumes: every Task 1-3 interface.
- Produces: `episode_returns`, `rolling_means`, `losses`, trained networks, and `evaluate_policy(episodes, seed_offset) -> list[float]`.

- [ ] **Step 1: Reinitialize all mutable training state after the synthetic smoke check**

```python
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
env.action_space.seed(SEED)

online_network = VanillaRNN(OBSERVATION_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(DEVICE)
target_network = VanillaRNN(OBSERVATION_SIZE, HIDDEN_SIZE, ACTION_SIZE).to(DEVICE)
target_network.load_state_dict(online_network.state_dict())
target_network.eval()
optimizer = optim.Adam(online_network.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
```

- [ ] **Step 2: Add the training loop**

```python
episode_returns = []
rolling_means = []
losses = []
global_step = 0

for episode in range(MAX_EPISODES):
    observation, _ = env.reset(seed=SEED + episode)
    state_sequence = make_sequence(observation)
    episode_return = 0.0

    while True:
        action = select_action(state_sequence, global_step)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_sequence = append_observation(state_sequence, next_observation)

        replay_buffer.push(state_sequence, action, reward, next_state_sequence, done)
        loss = optimize_model()
        if loss is not None:
            losses.append(loss)

        global_step += 1
        episode_return += reward
        state_sequence = next_state_sequence

        if global_step % TARGET_UPDATE_FREQUENCY == 0:
            target_network.load_state_dict(online_network.state_dict())

        if done:
            break

    episode_returns.append(episode_return)
    rolling_mean = float(np.mean(episode_returns[-SOLVED_WINDOW:]))
    rolling_means.append(rolling_mean)

    if (episode + 1) % 10 == 0:
        print(
            f"episode={episode + 1:4d} return={episode_return:6.1f} "
            f"mean={rolling_mean:6.1f} epsilon={epsilon_at(global_step):.3f}"
        )

    if len(episode_returns) >= SOLVED_WINDOW and rolling_mean >= SOLVED_SCORE:
        print(f"Solved at episode {episode + 1}: latest-100 mean={rolling_mean:.1f}")
        break

env.close()
assert 1 <= len(episode_returns) <= MAX_EPISODES
assert len(rolling_means) == len(episode_returns)
assert all(math.isfinite(value) for value in episode_returns + rolling_means + losses)
```

- [ ] **Step 3: Plot learning signals**

```python
fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
axes[0].plot(episode_returns, alpha=0.35, label="episode return")
axes[0].plot(rolling_means, label="latest-100 mean")
axes[0].axhline(SOLVED_SCORE, color="tab:red", linestyle="--", label="target")
axes[0].set(xlabel="Episode", ylabel="Return", title="CartPole RNN-DQN training")
axes[0].legend()

axes[1].plot(losses, alpha=0.6)
axes[1].set(xlabel="Optimization step", ylabel="Smooth L1 loss", title="DQN loss")
plt.show()
```

- [ ] **Step 4: Add greedy evaluation**

```python
def evaluate_policy(episodes=EVALUATION_EPISODES, seed_offset=10_000):
    evaluation_env = gym.make("CartPole-v1")
    returns = []
    online_network.eval()

    for episode in range(episodes):
        observation, _ = evaluation_env.reset(seed=SEED + seed_offset + episode)
        sequence = make_sequence(observation)
        total_reward = 0.0

        while True:
            state = torch.as_tensor(sequence, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action = int(online_network(state).argmax(dim=1).item())
            observation, reward, terminated, truncated, _ = evaluation_env.step(action)
            sequence = append_observation(sequence, observation)
            total_reward += reward
            if terminated or truncated:
                break

        returns.append(total_reward)

    evaluation_env.close()
    online_network.train()
    return returns


evaluation_returns = evaluate_policy()
assert len(evaluation_returns) == EVALUATION_EPISODES
assert all(math.isfinite(value) for value in evaluation_returns)
print("evaluation returns:", evaluation_returns)
print(f"evaluation mean: {np.mean(evaluation_returns):.1f}")
```

- [ ] **Step 5: Execute the notebook from a fresh kernel**

Run:

```bash
jupyter nbconvert --to notebook --execute tests/cartpole.ipynb \
  --output /tmp/cartpole-rnn-dqn-executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Expected: exit code 0; every assertion passes; training ends at the solved threshold or at episode 1,000; plots and evaluation output are present in `/tmp/cartpole-rnn-dqn-executed.ipynb`.

## Final Review Checklist

- [ ] Confirm `git diff -- tests/cartpole.ipynb` contains only the intended notebook rewrite.
- [ ] Confirm `models/RNN.py` remains unchanged.
- [ ] Confirm `git status --short` still lists both design documents as untracked and neither is staged.
- [ ] Record whether the run stopped by score or episode limit; do not describe reaching 475 unless the executed output proves it.
