"""CNN-ModularRNN-DQN on CartPole-v1: identical to scripts/test_cnn_rnn_dqn_cartpole.py's
CNN-RNN-DQN (same conv front-end, Double DQN target, n-step returns, LR decay, decoupled
greedy evaluation, stored-state hidden threading) except the recurrent core is
models.modular_rnn.ModularRNN instead of a plain nn.RNNCell.

Per CLAUDE.md's spec, ModularRNN's hidden layer is split into three equal-sized modules --
input, intermediate, output -- with restricted connectivity: fully connected within a module,
10%-connected between adjacent modules (`near_module_sparsity`, fixed in-degree per row, see
models/modular_rnn.py), and disconnected between input and output ("don't link with the
further module"). Q-values are read only from the output module (ModularRNN's masked
output_proj), matching "output is determined by environments" -- the CNN and every timestep's
raw conv features only ever reach the network's action-values through that same restricted
pathway. All of the masking/connectivity behavior itself is already unit-tested in
models/test_modular_rnn.py; this file only tests the DQN-specific plumbing built on top
(replay memory, Double-DQN target, hidden-state threading, training loop).

Kept in its own file (reusing scripts/test_dqn_cartpole.py's frame rendering/stacking
helpers and get_device, and models/modular_rnn.py's ModularRNN) rather than folded into
test_cnn_rnn_dqn_cartpole.py, so the two recurrent architectures -- and their results --
stay clearly distinguishable.
"""

import json
import math
import random
import sys
from collections import deque, namedtuple
from itertools import count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from models.common import get_device
from models.modular_rnn import ModularRNN
from scripts.cartpole_render import render_cartpole_frame
from scripts.test_dqn_cartpole import frame_to_tensor, stack_frames

ModularTransition = namedtuple(
    "ModularTransition", ("hidden", "state", "action", "next_hidden", "next_state", "reward")
)


class ModularReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(ModularTransition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class CNNModularRNNDQN(nn.Module):
    """Same conv front-end as test_dqn_cartpole.DQN / test_cnn_rnn_dqn_cartpole.CNNRNNDQN
    (matching CNNEncoder's kernel schedule), but the recurrent core is
    models.modular_rnn.ModularRNN instead of a stateless Linear layer or a plain nn.RNNCell:
    `rnn_hidden_size` units split into input/intermediate/output modules, dense within a
    module, `near_module_sparsity`-connected between adjacent modules, disconnected between
    input and output. `hidden_sizes` sets any Linear+ReLU layers *before* the recurrent
    core -- () feeds conv features straight into it, matching CNNRNNDQN's default head shape
    but with the RNNCell replaced by a ModularRNN. `recurrent_gain`/`input_gain` are passed
    straight through to ModularRNN (default 1.4/1.0, matching its own defaults) -- exposed
    here rather than left hardcoded so a search can sweep them; ModularRNN's recurrent_gain
    deliberately pushes the recurrent weight's spectral radius past 1 (see
    models/common.py's scaled_recurrent_init_ docstring), which is a reasonable prior for
    BPTT-trained sequence classification but is untested for a Q-network whose hidden state
    is threaded step-by-step through online TD updates -- expansive dynamics there could
    plausibly block learning rather than help it."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        image_size: int = 64,
        hidden_sizes: tuple[int, ...] = (),
        rnn_hidden_size: int = 129,
        near_module_sparsity: float = 0.1,
        recurrent_gain: float = 1.4,
        input_gain: float = 1.0,
        conv_channels: tuple[int, int, int] = (16, 32, 32),
    ):
        super().__init__()
        c1, c2, c3 = conv_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            flat_dim = self.conv(dummy).flatten(1).shape[1]

        pre_layers = []
        in_dim = flat_dim
        for h in hidden_sizes:
            pre_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.pre = nn.Sequential(*pre_layers) if pre_layers else nn.Identity()

        self.rnn = ModularRNN(
            input_size=in_dim,
            hidden_size=rnn_hidden_size,
            output_size=n_actions,
            output_mode="last",
            near_module_sparsity=near_module_sparsity,
            recurrent_gain=recurrent_gain,
            input_gain=input_gain,
        )

    def init_hidden(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        return self.rnn.init_hidden(batch_size, device, dtype)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pre(self.conv(x).flatten(1))
        return self.rnn.step(features, h)


def select_action_modular(
    state: torch.Tensor,
    hidden: torch.Tensor,
    policy_net: CNNModularRNNDQN,
    n_actions: int,
    steps_done: int,
    device: torch.device,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: float = 1000,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The hidden-state transition depends only on the observation, not on which action
    gets taken, so it's computed unconditionally before the epsilon-greedy branch."""
    with torch.no_grad():
        q, next_hidden = policy_net.step(state, hidden)

    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
    if random.random() > eps_threshold:
        action = q.max(1).indices.view(1, 1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action, next_hidden, eps_threshold


def optimize_model_modular(
    memory: ModularReplayMemory,
    policy_net: CNNModularRNNDQN,
    target_net: CNNModularRNNDQN,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int = 128,
    gamma: float = 0.99,
) -> float | None:
    """Double-Q target (van Hasselt et al., 2016): the next-state action is chosen by
    `policy_net` (argmax), but its value is read from `target_net`, same as
    test_dqn_cartpole.optimize_model -- see that module's docstring for why."""
    if len(memory) < batch_size:
        return None
    transitions = memory.sample(batch_size)
    batch = ModularTransition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    non_final_next_hidden = torch.cat([h for h, s in zip(batch.next_hidden, batch.next_state) if s is not None])

    hidden_batch = torch.cat(batch.hidden)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values, _ = policy_net.step(state_batch, hidden_batch)
    state_action_values = q_values.gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        best_actions = policy_net.step(non_final_next_states, non_final_next_hidden)[0].argmax(1)
        target_q, _ = target_net.step(non_final_next_states, non_final_next_hidden)
        next_state_values[non_final_mask] = target_q.gather(1, best_actions.unsqueeze(1)).squeeze(1)
    expected_state_action_values = reward_batch + gamma * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


def soft_update_modular(policy_net: CNNModularRNNDQN, target_net: CNNModularRNNDQN, tau: float) -> None:
    policy_state = policy_net.state_dict()
    target_state = target_net.state_dict()
    for key in policy_state:
        target_state[key] = policy_state[key] * tau + target_state[key] * (1 - tau)
    target_net.load_state_dict(target_state)


def evaluate_greedy_modular(
    policy_net: CNNModularRNNDQN,
    device: torch.device,
    image_size: int,
    frame_stack: int,
    num_episodes: int = 5,
    max_steps: int = 500,
) -> list[int]:
    """Separate deterministic (argmax, no epsilon) rollout, decoupled from the noisy
    training rollout -- see test_dqn_cartpole.evaluate_greedy's docstring for why judging
    "solved" from training-time reward can't work."""
    policy_net.eval()
    env = gym.make("CartPole-v1")
    durations = []
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            frames = deque([render_cartpole_frame(obs, size=image_size)] * frame_stack, maxlen=frame_stack)
            state = frame_to_tensor(stack_frames(frames), device)
            hidden = policy_net.init_hidden(1, device)
            for t in count():
                q, hidden = policy_net.step(state, hidden)
                action = q.max(1).indices.view(1, 1)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                if terminated or truncated or t + 1 >= max_steps:
                    durations.append(t + 1)
                    break
                frames.append(render_cartpole_frame(obs, size=image_size))
                state = frame_to_tensor(stack_frames(frames), device)
    env.close()
    policy_net.train()
    return durations


def train_dqn_modular(
    env: gym.Env,
    device: torch.device,
    num_episodes: int,
    image_size: int = 64,
    hidden_sizes: tuple[int, ...] = (),
    rnn_hidden_size: int = 129,
    near_module_sparsity: float = 0.1,
    recurrent_gain: float = 1.4,
    input_gain: float = 1.0,
    conv_channels: tuple[int, int, int] = (16, 32, 32),
    frame_stack: int = 4,
    batch_size: int = 128,
    gamma: float = 0.99,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: float = 1000,
    tau: float = 0.005,
    lr: float = 1e-4,
    memory_capacity: int = 10000,
    results_path: str | None = None,
    verbose: bool = True,
    eval_every: int | None = None,
    eval_episodes: int = 5,
    solved_mean_reward: float | None = None,
    lr_decay_every: int | None = None,
    lr_decay_factor: float = 0.5,
    n_step: int = 1,
) -> tuple[list[int], list[dict]]:
    """CNNModularRNNDQN counterpart of test_cnn_rnn_dqn_cartpole.train_dqn_recurrent -- same
    algorithm (Double DQN, n-step returns, LR decay, decoupled greedy evaluation, stored-state
    hidden threading), but the recurrent core is ModularRNN instead of nn.RNNCell.

    `n_step` and its end-of-episode flush behave exactly as in train_dqn_recurrent (see that
    docstring); the only difference here is the network whose hidden state gets threaded
    through every step is modular-connectivity restricted rather than fully recurrent."""
    n_actions = env.action_space.n
    in_channels = frame_stack
    gamma_n = gamma**n_step

    policy_net = CNNModularRNNDQN(
        in_channels,
        n_actions,
        image_size=image_size,
        hidden_sizes=hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        near_module_sparsity=near_module_sparsity,
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
        conv_channels=conv_channels,
    ).to(device)
    target_net = CNNModularRNNDQN(
        in_channels,
        n_actions,
        image_size=image_size,
        hidden_sizes=hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        near_module_sparsity=near_module_sparsity,
        recurrent_gain=recurrent_gain,
        input_gain=input_gain,
        conv_channels=conv_channels,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ModularReplayMemory(memory_capacity)

    steps_done = 0
    episode_durations = []
    history = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        frames = deque([render_cartpole_frame(obs, size=image_size)] * frame_stack, maxlen=frame_stack)
        state = frame_to_tensor(stack_frames(frames), device)
        hidden = policy_net.init_hidden(1, device)
        episode_losses = []
        eps = eps_start
        n_step_buffer = deque()  # (hidden, state, action, reward), oldest first
        for t in count():
            action, next_hidden, eps = select_action_modular(
                state, hidden, policy_net, n_actions, steps_done, device, eps_start, eps_end, eps_decay
            )
            steps_done += 1
            obs, reward, terminated, truncated, _ = env.step(action.item())

            if terminated:
                next_state = None
            else:
                frames.append(render_cartpole_frame(obs, size=image_size))
                next_state = frame_to_tensor(stack_frames(frames), device)

            n_step_buffer.append((hidden, state, action, reward))
            if len(n_step_buffer) >= n_step:
                h0, s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][2]
                n_step_return = sum(gamma**i * r for i, (_, _, _, r) in enumerate(n_step_buffer))
                memory.push(h0, s0, a0, next_hidden, next_state, torch.tensor([n_step_return], device=device))
                n_step_buffer.popleft()

            state, hidden = next_state, next_hidden

            loss = optimize_model_modular(memory, policy_net, target_net, optimizer, device, batch_size, gamma_n)
            if loss is not None:
                episode_losses.append(loss)
            soft_update_modular(policy_net, target_net, tau)

            if terminated or truncated:
                final_next_state = None if terminated else next_state
                final_next_hidden = next_hidden
                while n_step_buffer:
                    h0, s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][2]
                    n_step_return = sum(gamma**i * r for i, (_, _, _, r) in enumerate(n_step_buffer))
                    memory.push(h0, s0, a0, final_next_hidden, final_next_state, torch.tensor([n_step_return], device=device))
                    n_step_buffer.popleft()
                episode_reward = t + 1
                break

        episode_durations.append(episode_reward)
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None

        if lr_decay_every is not None and (ep + 1) % lr_decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor

        eval_mean_reward = None
        if eval_every is not None and (ep + 1) % eval_every == 0:
            eval_durations = evaluate_greedy_modular(policy_net, device, image_size, frame_stack, num_episodes=eval_episodes)
            eval_mean_reward = sum(eval_durations) / len(eval_durations)

        history.append(
            {
                "episode": ep + 1,
                "reward": episode_reward,
                "loss": avg_loss,
                "epsilon": eps,
                "eval_mean_reward": eval_mean_reward,
            }
        )
        if verbose:
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "n/a"
            eval_str = f" eval_mean_reward={eval_mean_reward:.1f}" if eval_mean_reward is not None else ""
            print(
                f"episode {ep + 1}/{num_episodes} reward {episode_reward} loss {loss_str} eps {eps:.3f}{eval_str}",
                flush=True,
            )
        if results_path is not None:
            with open(results_path, "w") as f:
                json.dump(history, f, indent=2)

        if solved_mean_reward is not None and eval_mean_reward is not None and eval_mean_reward > solved_mean_reward:
            if verbose:
                print(f"solved: greedy eval mean reward {eval_mean_reward:.1f} > {solved_mean_reward}", flush=True)
            break

    return episode_durations, history


# --- unit tests ---


def test_modular_replay_memory_push_and_sample():
    memory = ModularReplayMemory(capacity=5)
    for i in range(3):
        memory.push(f"h{i}", f"state{i}", f"action{i}", f"nh{i}", f"next{i}", f"reward{i}")
    assert len(memory) == 3
    sample = memory.sample(2)
    assert len(sample) == 2
    assert all(isinstance(t, ModularTransition) for t in sample)


def test_modular_replay_memory_respects_capacity():
    memory = ModularReplayMemory(capacity=3)
    for i in range(5):
        memory.push(i, i, i, i, i, i)
    assert len(memory) == 3
    assert [t.state for t in memory.memory] == [2, 3, 4]


def test_cnn_modular_rnn_dqn_uses_modular_rnn_core_with_requested_sparsity():
    net = CNNModularRNNDQN(in_channels=4, n_actions=2, image_size=32, rnn_hidden_size=9, near_module_sparsity=0.1)
    assert isinstance(net.rnn, ModularRNN)
    lo, hi = 3, 6  # hidden_size=9 -> three 3-unit modules
    # dense within module
    assert torch.all(net.rnn.cell.hh_mask[:lo, :lo] == 1.0)
    assert torch.all(net.rnn.cell.hh_mask[lo:hi, lo:hi] == 1.0)
    assert torch.all(net.rnn.cell.hh_mask[hi:, hi:] == 1.0)
    # zero between input and output modules
    assert torch.all(net.rnn.cell.hh_mask[:lo, hi:] == 0.0)
    assert torch.all(net.rnn.cell.hh_mask[hi:, :lo] == 0.0)
    # 10% (fixed in-degree) between adjacent modules
    assert torch.all(net.rnn.cell.hh_mask[:lo, lo:hi].sum(dim=1) == round(0.1 * lo))


def test_cnn_modular_rnn_dqn_step_shape_and_hidden_changes():
    torch.manual_seed(0)
    net = CNNModularRNNDQN(in_channels=4, n_actions=2, image_size=32, rnn_hidden_size=9)
    x = torch.randn(6, 4, 32, 32)
    h0 = net.init_hidden(6, torch.device("cpu"))
    q, h1 = net.step(x, h0)
    assert q.shape == (6, 2)
    assert h1.shape == (6, 9)
    assert not torch.equal(h0, h1)  # hidden state should actually update given nonzero input


def test_select_action_modular_is_greedy_at_zero_epsilon():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = CNNModularRNNDQN(in_channels=1, n_actions=3, image_size=16, rnn_hidden_size=9)
    state = torch.randn(1, 1, 16, 16)
    hidden = net.init_hidden(1, device)
    with torch.no_grad():
        expected_q, expected_hidden = net.step(state, hidden)
    for steps_done in (0, 1, 100):
        action, next_hidden, eps = select_action_modular(
            state, hidden, net, n_actions=3, steps_done=steps_done, device=device, eps_start=0.0, eps_end=0.0
        )
        assert eps == 0.0
        assert torch.equal(action, expected_q.max(1).indices.view(1, 1))
        assert torch.equal(next_hidden, expected_hidden)


class _FixedModularQNet(nn.Module):
    """Like test_dqn_cartpole._FixedQNet, but with a step(x, h) interface: outputs a fixed
    (n_actions,) Q-vector regardless of state or hidden state, and passes hidden through
    unchanged, so a test can pin down exactly which action argmax/gather pick."""

    def __init__(self, q_values: list[float]):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(q_values, dtype=torch.float32))

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q.unsqueeze(0).expand(x.shape[0], -1), h


def test_optimize_model_modular_uses_double_dqn_target():
    # policy_net ranks action 1 highest; target_net ranks action 0 highest but disagrees on
    # action 1's value (2.0, vs its own top pick's 5.0) -- same setup as
    # test_cnn_rnn_dqn_cartpole.test_optimize_model_recurrent_uses_double_dqn_target, adapted
    # to CNNModularRNNDQN's step(x, h) interface.
    device = torch.device("cpu")
    policy_net = _FixedModularQNet([0.0, 1.0])  # argmax -> action 1
    target_net = _FixedModularQNet([5.0, 2.0])  # argmax -> action 0; value at action 1 is 2.0

    hidden = torch.zeros(1, 4)
    state = torch.zeros(1, 1, 4, 4)
    action = torch.tensor([[0]], dtype=torch.long)
    next_hidden = torch.zeros(1, 4)
    next_state = torch.zeros(1, 1, 4, 4)
    reward = torch.tensor([1.0])

    memory = ModularReplayMemory(capacity=1)
    memory.push(hidden, state, action, next_hidden, next_state, reward)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)

    loss = optimize_model_modular(memory, policy_net, target_net, optimizer, device, batch_size=1, gamma=0.99)

    state_action_value = torch.tensor([[0.0]])
    double_dqn_target = torch.tensor([[1.0 + 0.99 * 2.0]])
    vanilla_target = torch.tensor([[1.0 + 0.99 * 5.0]])

    expected_loss = nn.SmoothL1Loss()(state_action_value, double_dqn_target).item()
    wrong_loss = nn.SmoothL1Loss()(state_action_value, vanilla_target).item()

    assert abs(loss - expected_loss) < 1e-4
    assert abs(loss - wrong_loss) > 1e-2


def test_soft_update_modular_interpolates_towards_policy_net():
    policy_net = CNNModularRNNDQN(1, 2, image_size=16, rnn_hidden_size=9)
    target_net = CNNModularRNNDQN(1, 2, image_size=16, rnn_hidden_size=9)
    with torch.no_grad():
        for p in policy_net.parameters():
            p.fill_(1.0)
        for p in target_net.parameters():
            p.fill_(0.0)

    soft_update_modular(policy_net, target_net, tau=0.1)
    for p in target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1))

    soft_update_modular(policy_net, target_net, tau=0.1)
    for p in target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.19))


def test_evaluate_greedy_modular_runs_and_leaves_net_in_train_mode():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = CNNModularRNNDQN(4, 2, image_size=32, rnn_hidden_size=9)
    net.train()
    durations = evaluate_greedy_modular(net, device, image_size=32, frame_stack=4, num_episodes=3)
    assert len(durations) == 3
    assert all(d >= 1 for d in durations)
    assert net.training  # must restore train() mode, since optimize_model_modular requires it


def test_train_dqn_modular_smoke_runs_on_cartpole():
    # Small image size / few episodes per CLAUDE.md's "use small size of hidden units in
    # simple test" -- checks the full loop (frame rendering, hidden-state threading, replay,
    # optimize_model_modular, soft target update) runs end-to-end without error.
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    env = gym.make("CartPole-v1")
    try:
        num_episodes = 5
        durations, history = train_dqn_modular(
            env,
            device,
            num_episodes=num_episodes,
            image_size=32,
            rnn_hidden_size=9,
            batch_size=16,
            memory_capacity=500,
            eps_decay=200,
            verbose=False,
        )
    finally:
        env.close()

    assert len(durations) == num_episodes
    assert all(d >= 1 for d in durations)
    assert len(history) == num_episodes
    for i, entry in enumerate(history):
        assert entry["episode"] == i + 1
        assert entry["reward"] == durations[i]
        assert entry["loss"] is None or entry["loss"] >= 0.0


def test_train_dqn_modular_with_n_step_runs():
    for n_step in (1, 3):
        torch.manual_seed(0)
        random.seed(0)
        device = get_device()
        env = gym.make("CartPole-v1")
        try:
            durations, history = train_dqn_modular(
                env,
                device,
                num_episodes=5,
                image_size=32,
                rnn_hidden_size=9,
                batch_size=16,
                memory_capacity=500,
                eps_decay=200,
                verbose=False,
                n_step=n_step,
            )
        finally:
            env.close()
        assert len(durations) == 5
        assert len(history) == 5


def test_train_dqn_modular_stops_early_when_solved():
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    env = gym.make("CartPole-v1")
    try:
        durations, history = train_dqn_modular(
            env,
            device,
            num_episodes=20,
            image_size=32,
            rnn_hidden_size=9,
            batch_size=16,
            memory_capacity=500,
            eps_decay=200,
            verbose=False,
            eval_every=1,
            eval_episodes=2,
            solved_mean_reward=0,
        )
    finally:
        env.close()

    assert len(durations) == 1
    assert len(history) == 1
    assert history[0]["eval_mean_reward"] is not None
    assert history[0]["eval_mean_reward"] > 0
