# CartPole RNN-DQN Notebook Design

## Goal

Extend `tests/cartpole.ipynb` into an educational, cell-by-cell DQN training notebook for `CartPole-v1`. The Q-network uses the eight most recent observations as one sequence. Training stops when either the latest 100 episode returns average at least 475 or 1,000 episodes have run.

This change is limited to the notebook. Reusable modules, command-line scripts, automated test files, Double DQN, prioritized replay, dueling networks, checkpointing, and rendering are outside this design.

## Sequence Semantics

Each CartPole observation has four features. At an episode reset, the initial observation is repeated eight times to form the first state sequence. After each environment step, the oldest observation is discarded and the newest observation is appended. Resetting an episode replaces the whole sequence, so observations from different episodes never mix.

Each replay entry contains:

```text
state_sequence:      (8, 4)
action:              scalar integer
reward:              scalar float
next_state_sequence: (8, 4)
done:                scalar boolean
```

The replay buffer stores these complete transitions. This duplicates observations between adjacent entries, but keeps sampling and the DQN update explicit and easy to inspect.

## Notebook Structure

The notebook will contain cells in this order:

1. Imports, random seeds, device selection, and hyperparameters.
2. A `VanillaRNN` Q-network that accepts `(batch, 8, 4)` and returns `(batch, 2)`.
3. Replay-buffer and sequence initialization/update helpers.
4. Online and target network initialization.
5. Epsilon-greedy action selection.
6. A minibatch DQN optimization function.
7. The training loop for at most 1,000 episodes.
8. Reward, rolling-average, and loss plots.
9. Greedy policy evaluation over multiple episodes.

The RNN hidden state is initialized to zero on each network call. Temporal information is nevertheless processed within each supplied eight-observation sequence; hidden state is not persisted across replay entries.

## DQN Update

For a sampled batch of size `B`, the state and next-state tensors have shape `(B, 8, 4)`. The online network produces `(B, 2)` Q-values. The Q-value corresponding to each stored action is selected with `gather`.

The target is:

```text
target = reward + gamma * (1 - done) * max_a Q_target(next_state_sequence, a)
```

Target values are computed without gradient tracking. Both Gymnasium `terminated` and `truncated` signals are treated as `done`; consequently, no future value is bootstrapped after either condition.

Optimization uses Adam, Smooth L1 loss, and gradient-norm clipping at 10.0. The target network receives a hard copy of the online network weights every 1,000 environment steps.

## Hyperparameters

```text
sequence_length         = 8
hidden_size             = 128
batch_size              = 64
replay_capacity         = 50,000
gamma                   = 0.99
learning_rate           = 0.001
epsilon_start           = 1.0
epsilon_end             = 0.05
epsilon_decay_steps     = 20,000
target_update_frequency = 1,000 environment steps
learning_starts         = 1,000 transitions
max_episodes            = 1,000
solved_score            = 475.0
solved_window           = 100 episodes
```

Epsilon decays exponentially by environment step rather than episode so that exploration depends on the amount of collected experience. Optimization begins only after 1,000 transitions have been collected.

## Training and Evaluation Flow

At every environment step, the notebook:

1. Converts the current sequence to a batch of one.
2. Selects a random action with probability epsilon; otherwise selects the online network's largest-Q action.
3. Steps the environment and constructs the next sequence.
4. Stores the completed transition.
5. Samples and optimizes one minibatch if warm-up is complete.
6. Copies online weights to the target network when the update interval is reached.
7. Resets both the environment and sequence on episode termination or truncation.

After each episode, its return is recorded. Once at least 100 returns exist, training stops early if their latest-100 mean reaches 475. Otherwise, training ends after episode 1,000.

Evaluation disables exploration and runs the greedy policy over multiple fresh episodes. The notebook reports the individual returns and their mean. Rendering is omitted because notebook environments differ in GUI support.

## Validation and Failure Handling

Before full training, a smoke check will:

1. Pass a synthetic `(4, 8, 4)` batch through the RNN.
2. Assert that Q-values have shape `(4, 2)`.
3. Select stored-action values and assert shape `(4,)`.
4. Compute a finite loss and complete one backward pass.

Runtime checks will verify expected observation/action-space shapes, minibatch tensor shapes, and finite Q-values, targets, and losses. A non-finite value raises an error immediately instead of allowing silent corruption.

Python, NumPy, PyTorch, the Gymnasium environment, and the action space receive explicit seeds. A single seed does not establish learning reliability; failure to reach 475 in one run is not by itself evidence of an implementation defect.

## Success Criteria

The notebook is successful when:

- all cells execute in order without undefined variables or shape errors;
- replay samples never cross episode boundaries;
- the smoke check completes a backward pass with finite values;
- training records returns, rolling averages, and losses for plotting;
- training stops at the performance threshold or at 1,000 episodes; and
- greedy evaluation runs independently of epsilon exploration.
