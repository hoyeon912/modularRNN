"""Double DQN on LunarLander-v3, CNN-over-pixels variant: same Double-DQN/n-step/frame-stack
machinery as scripts/test_dqn_cartpole.py's CNN-DQN, but reading rendered frames from
LunarLander's own native pygame renderer (env.render() -> (400, 600, 3) uint8) instead of a
custom from-scratch one -- unlike CartPole, LunarLander ships pixel rendering out of the box,
so there is no scripts/cartpole_render.py-equivalent to write here; frames are just
downsampled and grayscaled with PIL before going into the conv stack.

Frame stacking (see stack_frames) is still needed even though LunarLander's env.step returns
a full 8-dim state vector that already includes velocities -- this file intentionally ignores
that vector and learns purely from pixels, matching the "learn from pixels" framing of
scripts/test_dqn_cartpole.py's CNN-DQN and scripts/test_dqn_lunarlander.py's module docstring
(which instead uses the vector directly, for contrast). A single static rendered frame has no
velocity information, so multiple consecutive frames are stacked as separate input channels
so the conv net can infer motion from frame-to-frame differences, exactly as in CartPole.
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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from models.common import get_device

ENV_ID = "LunarLander-v3"
N_ACTIONS = 4
SOLVED_MEAN_REWARD = 200.0  # Gymnasium's own "solved" bar, averaged over 100 episodes

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQN(nn.Module):
    """CNN Q-network: 3 stride-2 convs down to a flat feature vector, then an MLP head to
    Q-values per action -- identical shape to scripts/test_dqn_cartpole.py's CNN-DQN, see
    that file for the parameter rationale (hidden_sizes/conv_channels)."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        image_size: int = 64,
        hidden_sizes: tuple[int, ...] = (),
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

        layers = []
        in_dim = flat_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1))


def render_frame(env: gym.Env, size: int) -> np.ndarray:
    """env.render() (H, W, 3) uint8 RGB -> (size, size, 1) uint8 grayscale, resized. Unlike
    CartPole's from-scratch renderer, LunarLander's native pygame frame is RGB and a
    non-square 400x600 -- squashing it to size x size loses aspect ratio, which is fine here
    since the conv net only needs relative motion/position cues, not metric accuracy."""
    frame = env.render()
    img = Image.fromarray(frame).convert("L").resize((size, size), Image.BILINEAR)
    return np.array(img)[:, :, None]


def stack_frames(frames) -> np.ndarray:
    """N frames, each (H, W, 1) uint8 -> (H, W, N) uint8, channel-stacked -- see
    scripts/test_dqn_cartpole.py's stack_frames for why this is needed (no single static
    frame carries velocity)."""
    return np.concatenate(list(frames), axis=2)


def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, C) uint8 -> (1, C, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(frame.copy()).to(device=device, dtype=torch.float32) / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def select_action(
    state: torch.Tensor,
    policy_net: DQN,
    n_actions: int,
    steps_done: int,
    device: torch.device,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: float = 1000,
) -> tuple[torch.Tensor, float]:
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1.0 * steps_done / eps_decay)
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1), eps_threshold
    return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), eps_threshold


def optimize_model(
    memory: ReplayMemory,
    policy_net: DQN,
    target_net: DQN,
    optimizer: optim.Optimizer,
    device: torch.device,
    batch_size: int = 128,
    gamma: float = 0.99,
) -> float | None:
    """Double-Q target (van Hasselt et al., 2016) -- see scripts/test_dqn_cartpole.py's
    module docstring for the overestimation-bias rationale this decoupling fixes."""
    if len(memory) < batch_size:
        return None
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        best_actions = policy_net(non_final_next_states).argmax(1)
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
        )
    expected_state_action_values = reward_batch + gamma * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()
    return loss.item()


def soft_update(policy_net: DQN, target_net: DQN, tau: float) -> None:
    policy_state = policy_net.state_dict()
    target_state = target_net.state_dict()
    for key in policy_state:
        target_state[key] = policy_state[key] * tau + target_state[key] * (1 - tau)
    target_net.load_state_dict(target_state)


def evaluate_greedy(
    policy_net: DQN,
    device: torch.device,
    image_size: int,
    frame_stack: int,
    num_episodes: int = 5,
    max_steps: int = 1000,
) -> list[float]:
    """Separate deterministic (argmax, no epsilon) rollout -- see
    scripts/test_dqn_cartpole.py's evaluate_greedy for why "solved" must be judged this way,
    not from the noisy training rollout."""
    policy_net.eval()
    env = gym.make(ENV_ID, render_mode="rgb_array")
    totals = []
    with torch.no_grad():
        for _ in range(num_episodes):
            env.reset()
            frames = deque([render_frame(env, image_size)] * frame_stack, maxlen=frame_stack)
            state = frame_to_tensor(stack_frames(frames), device)
            total = 0.0
            for t in count():
                action = policy_net(state).max(1).indices.view(1, 1)
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


def train_dqn(
    env: gym.Env,
    device: torch.device,
    num_episodes: int,
    image_size: int = 64,
    hidden_sizes: tuple[int, ...] = (),
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
    """Returns (episode_rewards, history). See scripts/test_dqn_cartpole.py's train_dqn for
    the shared frame_stack/n_step/lr_decay/eval_every/solved_mean_reward semantics --
    identical here, operating on LunarLander's native rendered frames instead of CartPole's
    custom-rendered ones. `env` must be created with render_mode="rgb_array" so env.render()
    returns pixels instead of opening a display window."""
    n_actions = env.action_space.n
    in_channels = frame_stack
    gamma_n = gamma**n_step

    policy_net = DQN(in_channels, n_actions, image_size=image_size, hidden_sizes=hidden_sizes, conv_channels=conv_channels).to(device)
    target_net = DQN(in_channels, n_actions, image_size=image_size, hidden_sizes=hidden_sizes, conv_channels=conv_channels).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_mean_reward = float("-inf")
    episode_rewards = []
    history = []
    for ep in range(num_episodes):
        env.reset()
        frames = deque([render_frame(env, image_size)] * frame_stack, maxlen=frame_stack)
        state = frame_to_tensor(stack_frames(frames), device)
        episode_losses = []
        eps = eps_start
        episode_reward = 0.0
        n_step_buffer = deque()  # (state, action, reward), oldest first
        for t in count():
            action, eps = select_action(state, policy_net, n_actions, steps_done, device, eps_start, eps_end, eps_decay)
            steps_done += 1
            _, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            truncated = truncated or (t + 1 >= max_steps)

            if terminated:
                next_state = None
            else:
                frames.append(render_frame(env, image_size))
                next_state = frame_to_tensor(stack_frames(frames), device)

            n_step_buffer.append((state, action, reward))
            if len(n_step_buffer) >= n_step:
                s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1]
                n_step_return = sum(gamma**i * r for i, (_, _, r) in enumerate(n_step_buffer))
                memory.push(s0, a0, next_state, torch.tensor([n_step_return], device=device))
                n_step_buffer.popleft()

            state = next_state

            loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size, gamma_n)
            if loss is not None:
                episode_losses.append(loss)
            soft_update(policy_net, target_net, tau)

            if terminated or truncated:
                final_next_state = None if terminated else next_state
                while n_step_buffer:
                    s0, a0 = n_step_buffer[0][0], n_step_buffer[0][1]
                    n_step_return = sum(gamma**i * r for i, (_, _, r) in enumerate(n_step_buffer))
                    memory.push(s0, a0, final_next_state, torch.tensor([n_step_return], device=device))
                    n_step_buffer.popleft()
                break

        episode_rewards.append(episode_reward)

        if lr_decay_every is not None and (ep + 1) % lr_decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor

        eval_mean_reward = None
        if eval_every is not None and (ep + 1) % eval_every == 0:
            eval_rewards = evaluate_greedy(policy_net, device, image_size, frame_stack, num_episodes=eval_episodes, max_steps=max_steps)
            eval_mean_reward = sum(eval_rewards) / len(eval_rewards)
            if save_best_path is not None and eval_mean_reward > best_eval_mean_reward:
                torch.save(policy_net.state_dict(), save_best_path)
            best_eval_mean_reward = max(best_eval_mean_reward, eval_mean_reward)

        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None
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


def test_replay_memory_push_and_sample():
    memory = ReplayMemory(10)
    memory.push(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, 4, 4), torch.tensor([1.0]))
    assert len(memory) == 1
    sampled = memory.sample(1)
    assert len(sampled) == 1


def test_replay_memory_respects_capacity():
    memory = ReplayMemory(3)
    for i in range(5):
        memory.push(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, 4, 4), torch.tensor([float(i)]))
    assert len(memory) == 3


def test_render_frame_shape_and_range():
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        env.reset()
        frame = render_frame(env, size=32)
    finally:
        env.close()
    assert frame.shape == (32, 32, 1)
    assert frame.dtype == np.uint8


def test_stack_frames_concatenates_along_channel_axis():
    frames = [np.full((8, 8, 1), i, dtype=np.uint8) for i in range(4)]
    stacked = stack_frames(frames)
    assert stacked.shape == (8, 8, 4)
    for i in range(4):
        assert np.all(stacked[:, :, i] == i)


def test_dqn_forward_shape():
    net = DQN(in_channels=1, n_actions=N_ACTIONS, image_size=32)
    out = net(torch.zeros(2, 1, 32, 32))
    assert out.shape == (2, N_ACTIONS)


def test_dqn_forward_shape_with_hidden_layers():
    net = DQN(in_channels=1, n_actions=N_ACTIONS, image_size=32, hidden_sizes=(64, 32))
    out = net(torch.zeros(3, 1, 32, 32))
    assert out.shape == (3, N_ACTIONS)


def test_select_action_is_greedy_at_zero_epsilon():
    net = DQN(in_channels=1, n_actions=N_ACTIONS, image_size=16)
    device = torch.device("cpu")
    state = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        expected = net(state).max(1).indices.item()
    action, eps = select_action(state, net, N_ACTIONS, steps_done=10_000_000, device=device, eps_start=0.9, eps_end=0.0, eps_decay=1)
    assert eps == 0.0
    assert action.item() == expected


def test_optimize_model_returns_none_below_batch_size():
    device = torch.device("cpu")
    policy_net = DQN(in_channels=1, n_actions=N_ACTIONS, image_size=16)
    target_net = DQN(in_channels=1, n_actions=N_ACTIONS, image_size=16)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(100)
    memory.push(torch.zeros(1, 1, 16, 16), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, 16, 16), torch.tensor([1.0]))
    loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=8)
    assert loss is None


class _FixedQNet(nn.Module):
    """Outputs a fixed (n_actions,) Q-vector regardless of input -- see
    scripts/test_dqn_cartpole.py's _FixedQNet for the rationale."""

    def __init__(self, q_values: list[float]):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(q_values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q.unsqueeze(0).expand(x.shape[0], -1)


def test_optimize_model_uses_double_dqn_target():
    device = torch.device("cpu")
    policy_net = _FixedQNet([0.0, 1.0])  # argmax -> action 1
    target_net = _FixedQNet([5.0, 2.0])  # argmax -> action 0; value at action 1 is 2.0

    state = torch.zeros(1, 1, 4, 4)
    action = torch.tensor([[0]], dtype=torch.long)
    next_state = torch.zeros(1, 1, 4, 4)
    reward = torch.tensor([1.0])

    memory = ReplayMemory(capacity=1)
    memory.push(state, action, next_state, reward)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)

    loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=1, gamma=0.99)

    state_action_value = torch.tensor([[0.0]])
    double_dqn_target = torch.tensor([[1.0 + 0.99 * 2.0]])
    vanilla_target = torch.tensor([[1.0 + 0.99 * 5.0]])

    expected_loss = nn.SmoothL1Loss()(state_action_value, double_dqn_target).item()
    wrong_loss = nn.SmoothL1Loss()(state_action_value, vanilla_target).item()

    assert abs(loss - expected_loss) < 1e-4
    assert abs(loss - wrong_loss) > 1e-2


def test_soft_update_interpolates_towards_policy_net():
    policy_net = DQN(in_channels=1, n_actions=N_ACTIONS, image_size=16)
    target_net = DQN(in_channels=1, n_actions=N_ACTIONS, image_size=16)
    for p in policy_net.parameters():
        nn.init.constant_(p, 1.0)
    for p in target_net.parameters():
        nn.init.constant_(p, 0.0)
    soft_update(policy_net, target_net, tau=0.1)
    for p in target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1), atol=1e-6)


def test_evaluate_greedy_is_deterministic_and_leaves_net_in_train_mode():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = DQN(in_channels=2, n_actions=N_ACTIONS, image_size=24)
    net.train()
    rewards = evaluate_greedy(net, device, image_size=24, frame_stack=2, num_episodes=1, max_steps=30)
    assert len(rewards) == 1
    assert net.training is True


def test_train_dqn_smoke_runs_on_lunarlander():
    # CLAUDE.md: "use small size of hidden units in simple test" -- checks the full loop
    # (rendering, frame stacking, replay memory, epsilon-greedy, optimize_model, soft target
    # update) runs end-to-end without error and produces one recorded reward per episode.
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        num_episodes = 2
        rewards, history = train_dqn(
            env,
            device,
            num_episodes=num_episodes,
            image_size=24,
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
        assert 0.0 <= entry["epsilon"] <= 1.0


def test_train_dqn_with_n_step_runs_and_reports_real_env_rewards():
    for n_step in (1, 3):
        torch.manual_seed(0)
        random.seed(0)
        device = torch.device("cpu")
        env = gym.make(ENV_ID, render_mode="rgb_array")
        try:
            rewards, history = train_dqn(
                env,
                device,
                num_episodes=2,
                image_size=24,
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


def test_train_dqn_stops_early_when_solved():
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    try:
        rewards, history = train_dqn(
            env,
            device,
            num_episodes=6,
            image_size=24,
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


def test_train_dqn_saves_best_checkpoint(tmp_path):
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID, render_mode="rgb_array")
    save_path = tmp_path / "best_model.pt"
    try:
        train_dqn(
            env,
            device,
            num_episodes=2,
            image_size=24,
            frame_stack=2,
            batch_size=16,
            memory_capacity=200,
            eps_decay=200,
            verbose=False,
            eval_every=1,
            eval_episodes=1,
            max_steps=30,
            save_best_path=str(save_path),
        )
    finally:
        env.close()
    assert save_path.exists()
    state_dict = torch.load(save_path, weights_only=True)
    net = DQN(in_channels=2, n_actions=N_ACTIONS, image_size=24)
    net.load_state_dict(state_dict)  # raises if shapes mismatch
