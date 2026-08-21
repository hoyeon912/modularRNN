"""CNN-RNN-DQN on LunarLander-v3: identical to scripts/test_cnn_dqn_lunarlander.py's CNN-DQN
(same conv front-end reading LunarLander's native rendered frames, Double DQN target, n-step
returns, decoupled greedy evaluation, best-checkpoint saving) except the head's last hidden
layer is an RNNCell whose hidden state persists across timesteps within an episode instead of
a stateless Linear layer -- a "stored-state" DRQN (Hausknecht & Stone, 2015), matching
scripts/test_cnn_rnn_dqn_cartpole.py's architecture.

Kept in its own file, mirroring scripts/test_cnn_rnn_dqn_cartpole.py's separation from
test_dqn_cartpole.py: this file's model has a persistent hidden state to thread through
training and evaluation (`optimize_model_recurrent`, `evaluate_greedy_recurrent`,
`train_dqn_recurrent`), which scripts/test_cnn_dqn_lunarlander.py's plain CNN-DQN does not.
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
from scripts.test_cnn_dqn_lunarlander import frame_to_tensor, render_frame, stack_frames

ENV_ID = "LunarLander-v3"
N_ACTIONS = 4
SOLVED_MEAN_REWARD = 200.0  # Gymnasium's own "solved" bar, averaged over 100 episodes

RecurrentTransition = namedtuple(
    "RecurrentTransition", ("hidden", "state", "action", "next_hidden", "next_state", "reward")
)


class RecurrentReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(RecurrentTransition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class CNNRNNDQN(nn.Module):
    """Same conv front-end as scripts/test_cnn_dqn_lunarlander.DQN, but the last hidden layer
    is an RNNCell (`rnn_hidden_size` units, 128 by default) whose hidden state persists across
    timesteps within an episode -- see scripts/test_cnn_rnn_dqn_cartpole.CNNRNNDQN for the
    full rationale, identical here."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        image_size: int = 64,
        hidden_sizes: tuple[int, ...] = (),
        rnn_hidden_size: int = 128,
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

        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_cell = nn.RNNCell(in_dim, rnn_hidden_size)
        self.output = nn.Linear(rnn_hidden_size, n_actions)

    def init_hidden(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch_size, self.rnn_hidden_size, device=device, dtype=dtype)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.pre(self.conv(x).flatten(1))
        h_next = self.rnn_cell(features, h)
        q = self.output(h_next)
        return q, h_next


def select_action_recurrent(
    state: torch.Tensor,
    hidden: torch.Tensor,
    policy_net: CNNRNNDQN,
    n_actions: int,
    steps_done: int,
    device: torch.device,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: float = 1000,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """The hidden-state transition depends only on the observation, not on which action gets
    taken, so it's computed unconditionally before the epsilon-greedy branch."""
    with torch.no_grad():
        q, next_hidden = policy_net.step(state, hidden)

    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
    if random.random() > eps_threshold:
        action = q.max(1).indices.view(1, 1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    return action, next_hidden, eps_threshold


def optimize_model_recurrent(
    memory: RecurrentReplayMemory,
    policy_net: CNNRNNDQN,
    target_net: CNNRNNDQN,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int = 128,
    gamma: float = 0.99,
) -> float | None:
    """Double-Q target (van Hasselt et al., 2016) -- see
    scripts/test_dqn_cartpole.py's module docstring for the overestimation-bias rationale."""
    if len(memory) < batch_size:
        return None
    transitions = memory.sample(batch_size)
    batch = RecurrentTransition(*zip(*transitions))

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


def soft_update_recurrent(policy_net: CNNRNNDQN, target_net: CNNRNNDQN, tau: float) -> None:
    policy_state = policy_net.state_dict()
    target_state = target_net.state_dict()
    for key in policy_state:
        target_state[key] = policy_state[key] * tau + target_state[key] * (1 - tau)
    target_net.load_state_dict(target_state)


def evaluate_greedy_recurrent(
    policy_net: CNNRNNDQN,
    device: torch.device,
    image_size: int,
    frame_stack: int,
    num_episodes: int = 5,
    max_steps: int = 1000,
) -> list[float]:
    """Separate deterministic (argmax, no epsilon) rollout -- see
    scripts/test_dqn_cartpole.py's evaluate_greedy for why "solved" must be judged this way."""
    policy_net.eval()
    env = gym.make(ENV_ID, render_mode="rgb_array")
    totals = []
    with torch.no_grad():
        for _ in range(num_episodes):
            env.reset()
            frames = deque([render_frame(env, image_size)] * frame_stack, maxlen=frame_stack)
            state = frame_to_tensor(stack_frames(frames), device)
            hidden = policy_net.init_hidden(1, device)
            total = 0.0
            for t in count():
                q, hidden = policy_net.step(state, hidden)
                action = q.max(1).indices.view(1, 1)
                _, reward, terminated, truncated, _ = env.step(action.item())
                total += reward
                if terminated or truncated or t + 1 >= max_steps:
                    break
                frames.append(render_frame(env, image_size))
                state = frame_to_tensor(stack_frames(frames), device)
            totals.append(total)
    env.close()
    policy_net.train()
    return totals


def train_dqn_recurrent(
    env: gym.Env,
    device: torch.device,
    num_episodes: int,
    image_size: int = 64,
    hidden_sizes: tuple[int, ...] = (),
    rnn_hidden_size: int = 128,
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
    save_best_path: str | None = None,
    max_steps: int = 1000,
) -> tuple[list[float], list[dict]]:
    """CNNRNNDQN counterpart of scripts/test_cnn_dqn_lunarlander.train_dqn -- same algorithm
    (Double DQN, n-step returns, LR decay, decoupled greedy evaluation, best-checkpoint
    saving), but threads hidden state through every step (stored-state DRQN) instead of
    treating each frame stack as an independent Markov state. See
    scripts/test_cnn_rnn_dqn_cartpole.train_dqn_recurrent's docstring for the shared n_step/
    save_best_path semantics, identical here."""
    n_actions = env.action_space.n
    in_channels = frame_stack
    gamma_n = gamma**n_step

    policy_net = CNNRNNDQN(
        in_channels,
        n_actions,
        image_size=image_size,
        hidden_sizes=hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        conv_channels=conv_channels,
    ).to(device)
    target_net = CNNRNNDQN(
        in_channels,
        n_actions,
        image_size=image_size,
        hidden_sizes=hidden_sizes,
        rnn_hidden_size=rnn_hidden_size,
        conv_channels=conv_channels,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = RecurrentReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_mean_reward = float("-inf")
    episode_rewards = []
    history = []
    for ep in range(num_episodes):
        env.reset()
        frames = deque([render_frame(env, image_size)] * frame_stack, maxlen=frame_stack)
        state = frame_to_tensor(stack_frames(frames), device)
        hidden = policy_net.init_hidden(1, device)
        episode_losses = []
        eps = eps_start
        episode_reward = 0.0
        n_step_buffer = deque()  # (hidden, state, action, reward), oldest first
        for t in count():
            action, next_hidden, eps = select_action_recurrent(
                state, hidden, policy_net, n_actions, steps_done, device, eps_start, eps_end, eps_decay
            )
            steps_done += 1
            _, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            truncated = truncated or (t + 1 >= max_steps)

            if terminated:
                next_state = None
            else:
                frames.append(render_frame(env, image_size))
                next_state = frame_to_tensor(stack_frames(frames), device)

            n_step_buffer.append((hidden, state, action, reward))
            if len(n_step_buffer) >= n_step:
                h0, s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][2]
                n_step_return = sum(gamma**i * r for i, (_, _, _, r) in enumerate(n_step_buffer))
                memory.push(h0, s0, a0, next_hidden, next_state, torch.tensor([n_step_return], device=device))
                n_step_buffer.popleft()

            state, hidden = next_state, next_hidden

            loss = optimize_model_recurrent(memory, policy_net, target_net, optimizer, device, batch_size, gamma_n)
            if loss is not None:
                episode_losses.append(loss)
            soft_update_recurrent(policy_net, target_net, tau)

            if terminated or truncated:
                final_next_state = None if terminated else next_state
                final_next_hidden = next_hidden
                while n_step_buffer:
                    h0, s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1], n_step_buffer[0][2]
                    n_step_return = sum(gamma**i * r for i, (_, _, _, r) in enumerate(n_step_buffer))
                    memory.push(h0, s0, a0, final_next_hidden, final_next_state, torch.tensor([n_step_return], device=device))
                    n_step_buffer.popleft()
                break

        episode_rewards.append(episode_reward)
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None

        if lr_decay_every is not None and (ep + 1) % lr_decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor

        eval_mean_reward = None
        if eval_every is not None and (ep + 1) % eval_every == 0:
            eval_rewards = evaluate_greedy_recurrent(
                policy_net, device, image_size, frame_stack, num_episodes=eval_episodes, max_steps=max_steps
            )
            eval_mean_reward = sum(eval_rewards) / len(eval_rewards)
            if save_best_path is not None and eval_mean_reward > best_eval_mean_reward:
                torch.save(policy_net.state_dict(), save_best_path)
            best_eval_mean_reward = max(best_eval_mean_reward, eval_mean_reward)

        history.append(
            {
                "episode": ep + 1,
                "reward": episode_rewards[-1],
                "loss": avg_loss,
                "epsilon": eps,
                "eval_mean_reward": eval_mean_reward,
            }
        )
        if verbose:
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "n/a"
            eval_str = f" eval_mean_reward={eval_mean_reward:.1f}" if eval_mean_reward is not None else ""
            print(
                f"episode {ep + 1}/{num_episodes} reward {episode_rewards[-1]:.1f} loss {loss_str} eps {eps:.3f}{eval_str}",
                flush=True,
            )
        if results_path is not None:
            with open(results_path, "w") as f:
                json.dump(history, f, indent=2)

        if solved_mean_reward is not None and eval_mean_reward is not None and eval_mean_reward > solved_mean_reward:
            if verbose:
                print(f"solved: greedy eval mean reward {eval_mean_reward:.1f} > {solved_mean_reward}", flush=True)
            break

    return episode_rewards, history


# --- unit tests ---


def test_recurrent_replay_memory_push_and_sample():
    memory = RecurrentReplayMemory(capacity=5)
    for i in range(3):
        memory.push(f"h{i}", f"state{i}", f"action{i}", f"nh{i}", f"next{i}", f"reward{i}")
    assert len(memory) == 3
    sample = memory.sample(2)
    assert len(sample) == 2
    assert all(isinstance(t, RecurrentTransition) for t in sample)


def test_recurrent_replay_memory_respects_capacity():
    memory = RecurrentReplayMemory(capacity=3)
    for i in range(5):
        memory.push(i, i, i, i, i, i)
    assert len(memory) == 3
    assert [t.state for t in memory.memory] == [2, 3, 4]


def test_cnn_rnn_dqn_step_shape_and_hidden_changes():
    torch.manual_seed(0)
    net = CNNRNNDQN(in_channels=2, n_actions=N_ACTIONS, image_size=32, rnn_hidden_size=16)
    x = torch.randn(6, 2, 32, 32)
    h0 = net.init_hidden(6, torch.device("cpu"))
    q, h1 = net.step(x, h0)
    assert q.shape == (6, N_ACTIONS)
    assert h1.shape == (6, 16)
    assert not torch.equal(h0, h1)


def test_select_action_recurrent_is_greedy_at_zero_epsilon():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = CNNRNNDQN(in_channels=1, n_actions=N_ACTIONS, image_size=16, rnn_hidden_size=8)
    state = torch.randn(1, 1, 16, 16)
    hidden = net.init_hidden(1, device)
    with torch.no_grad():
        expected_q, expected_hidden = net.step(state, hidden)
    for steps_done in (0, 1, 100):
        action, next_hidden, eps = select_action_recurrent(
            state, hidden, net, n_actions=N_ACTIONS, steps_done=steps_done, device=device, eps_start=0.0, eps_end=0.0
        )
        assert eps == 0.0
        assert torch.equal(action, expected_q.max(1).indices.view(1, 1))
        assert torch.equal(next_hidden, expected_hidden)


class _FixedRecurrentQNet(nn.Module):
    """Like scripts/test_cnn_dqn_lunarlander._FixedQNet, but with a step(x, h) interface:
    outputs a fixed (n_actions,) Q-vector regardless of state or hidden state, and passes
    hidden through unchanged, so a test can pin down exactly which action argmax/gather pick."""

    def __init__(self, q_values: list[float]):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(q_values, dtype=torch.float32))

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q.unsqueeze(0).expand(x.shape[0], -1), h


def test_optimize_model_recurrent_uses_double_dqn_target():
    device = torch.device("cpu")
    policy_net = _FixedRecurrentQNet([0.0, 1.0])  # argmax -> action 1
    target_net = _FixedRecurrentQNet([5.0, 2.0])  # argmax -> action 0; value at action 1 is 2.0

    hidden = torch.zeros(1, 4)
    state = torch.zeros(1, 1, 4, 4)
    action = torch.tensor([[0]], dtype=torch.long)
    next_hidden = torch.zeros(1, 4)
    next_state = torch.zeros(1, 1, 4, 4)
    reward = torch.tensor([1.0])

    memory = RecurrentReplayMemory(capacity=1)
    memory.push(hidden, state, action, next_hidden, next_state, reward)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)

    loss = optimize_model_recurrent(memory, policy_net, target_net, optimizer, device, batch_size=1, gamma=0.99)

    state_action_value = torch.tensor([[0.0]])
    double_dqn_target = torch.tensor([[1.0 + 0.99 * 2.0]])
    vanilla_target = torch.tensor([[1.0 + 0.99 * 5.0]])

    expected_loss = nn.SmoothL1Loss()(state_action_value, double_dqn_target).item()
    wrong_loss = nn.SmoothL1Loss()(state_action_value, vanilla_target).item()

    assert abs(loss - expected_loss) < 1e-4
    assert abs(loss - wrong_loss) > 1e-2


def test_soft_update_recurrent_interpolates_towards_policy_net():
    policy_net = CNNRNNDQN(1, N_ACTIONS, image_size=16, rnn_hidden_size=8)
    target_net = CNNRNNDQN(1, N_ACTIONS, image_size=16, rnn_hidden_size=8)
    with torch.no_grad():
        for p in policy_net.parameters():
            p.fill_(1.0)
        for p in target_net.parameters():
            p.fill_(0.0)

    soft_update_recurrent(policy_net, target_net, tau=0.1)
    for p in target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1))

    soft_update_recurrent(policy_net, target_net, tau=0.1)
    for p in target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.19))


def test_evaluate_greedy_recurrent_runs_and_leaves_net_in_train_mode():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = CNNRNNDQN(2, N_ACTIONS, image_size=24, rnn_hidden_size=16)
    net.train()
    rewards = evaluate_greedy_recurrent(net, device, image_size=24, frame_stack=2, num_episodes=1, max_steps=30)
    assert len(rewards) == 1
    assert net.training


def test_train_dqn_recurrent_smoke_runs_on_lunarlander():
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        num_episodes = 2
        rewards, history = train_dqn_recurrent(
            env,
            device,
            num_episodes=num_episodes,
            image_size=24,
            rnn_hidden_size=16,
            frame_stack=2,
            batch_size=16,
            memory_capacity=200,
            eps_decay=200,
            verbose=False,
            max_steps=40,
        )
    finally:
        env.close()

    assert len(rewards) == num_episodes
    assert len(history) == num_episodes
    for i, entry in enumerate(history):
        assert entry["episode"] == i + 1
        assert entry["reward"] == rewards[i]
        assert entry["loss"] is None or entry["loss"] >= 0.0


def test_train_dqn_recurrent_with_n_step_runs():
    for n_step in (1, 3):
        torch.manual_seed(0)
        random.seed(0)
        device = torch.device("cpu")
        env = gym.make(ENV_ID, render_mode="rgb_array")
        try:
            rewards, history = train_dqn_recurrent(
                env,
                device,
                num_episodes=2,
                image_size=24,
                rnn_hidden_size=16,
                frame_stack=2,
                batch_size=16,
                memory_capacity=200,
                eps_decay=200,
                verbose=False,
                n_step=n_step,
                max_steps=40,
            )
        finally:
            env.close()
        assert len(rewards) == 2
        assert len(history) == 2


def test_train_dqn_recurrent_stops_early_when_solved():
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        rewards, history = train_dqn_recurrent(
            env,
            device,
            num_episodes=6,
            image_size=24,
            rnn_hidden_size=16,
            frame_stack=2,
            batch_size=16,
            memory_capacity=200,
            eps_decay=200,
            verbose=False,
            eval_every=1,
            eval_episodes=1,
            solved_mean_reward=-1e9,
            max_steps=30,
        )
    finally:
        env.close()

    assert len(rewards) < 6
    assert history[-1]["eval_mean_reward"] is not None


def test_train_dqn_recurrent_saves_best_checkpoint(tmp_path):
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    checkpoint_path = str(tmp_path / "best.pt")
    try:
        train_dqn_recurrent(
            env,
            device,
            num_episodes=2,
            image_size=24,
            rnn_hidden_size=16,
            frame_stack=2,
            batch_size=16,
            memory_capacity=200,
            eps_decay=200,
            verbose=False,
            eval_every=1,
            eval_episodes=1,
            max_steps=30,
            save_best_path=checkpoint_path,
        )
    finally:
        env.close()

    assert Path(checkpoint_path).exists()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net = CNNRNNDQN(2, N_ACTIONS, image_size=24, rnn_hidden_size=16)
    net.load_state_dict(state_dict)  # raises if shapes/keys don't match a real checkpoint
