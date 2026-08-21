"""Double DQN on CartPole-v1, following the PyTorch "Reinforcement Learning (DQN) Tutorial"
(https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html): a replay memory
of individual transitions, epsilon-greedy action selection with exponential decay, a target
network updated by Polyak/soft averaging (TAU) after every optimizer step, and a Huber
(SmoothL1) TD loss with gradient clipping.

Double-Q target (van Hasselt et al., 2016): the next-state action is chosen by `policy_net`
(argmax), but its value is read from `target_net`. Plain `target_net(...).max(1)` combines
selection and evaluation in the same (lagged, still-noisy) network, which systematically
overestimates Q-values -- observed empirically on this exact model as TD loss climbing ~25x
over a 500-episode run while reward stayed flat. Decoupling selection (online) from
evaluation (target) removes that positive bias. This project's scripts/train_cartpole.py's
build_td_objective hit the same failure mode and documents the identical fix.

The Q-network is a CNN over rendered CartPole frames, matching the tutorial's original
screen-based variant (learn from pixels, not the 4-dim physics state). Frames come from
scripts/cartpole_render.render_cartpole_frame -- this project's from-scratch renderer -- and
the conv stack mirrors models/cnn_modular_rnn.py's CNNEncoder (3 stride-2 convs) for
consistency with the rest of the codebase. This intentionally still does not use
ModularRNN: no recurrent core, no hidden state -- one rendered frame is fed straight to the
CNN and treated as a Markov state, same as the tutorial's stacked-frame CNN DQN.
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

from models.common import get_device
from scripts.cartpole_render import render_cartpole_frame

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
    """CNN Q-network: 3 stride-2 convs (matching CNNEncoder's kernel schedule) down to a
    flat feature vector, then an MLP head to Q-values per action. `hidden_sizes` sets the
    head's depth/width -- () gives the original single-Linear head straight to n_actions;
    (128,) or (256, 128) etc. insert ReLU-separated hidden layers before it. `conv_channels`
    sets the 3 conv layers' output channel counts -- (16, 32, 32) matches CNNEncoder's
    default; wider values give the feature extractor itself more capacity."""

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


def frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, C) uint8 -> (1, C, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(frame.copy()).to(device=device, dtype=torch.float32) / 255.0
    return t.permute(2, 0, 1).unsqueeze(0)


def stack_frames(frames) -> np.ndarray:
    """N frames, each (H, W, 1) uint8 -> (H, W, N) uint8, channel-stacked so the CNN can see
    motion. A single rendered frame has no velocity information -- x_dot/theta_dot are
    physically invisible in one static image -- so a policy conditioned on one frame can
    only react to the pole's current tilt, not how fast it's falling. Stacking recent frames
    as separate input channels (classic DQN's fix for the same problem on Atari) lets the
    conv net infer motion from frame-to-frame differences."""
    return np.concatenate(list(frames), axis=2)


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
    max_steps: int = 500,
) -> list[int]:
    """Runs `num_episodes` purely-greedy (argmax, no epsilon exploration) episodes and
    returns their durations/rewards. This must be a *separate* rollout from training: with
    eps_end=0.05 applied at every single step of the training rollout, the probability of
    500 consecutive non-random actions is 0.95**500 ~= 7e-12 regardless of how good the
    learned policy actually is -- so judging "is this solved" from the noisy training-time
    reward is mathematically almost impossible to trigger even for a converged policy.
    Matches the pattern already used by scripts/train_cartpole.py's evaluate_reward/
    rollout_episode (model.eval(), argmax, no epsilon) for the same reason."""
    policy_net.eval()
    env = gym.make("CartPole-v1")
    durations = []
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            frames = deque([render_cartpole_frame(obs, size=image_size)] * frame_stack, maxlen=frame_stack)
            state = frame_to_tensor(stack_frames(frames), device)
            for t in count():
                action = policy_net(state).max(1).indices.view(1, 1)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                if terminated or truncated or t + 1 >= max_steps:
                    durations.append(t + 1)
                    break
                frames.append(render_cartpole_frame(obs, size=image_size))
                state = frame_to_tensor(stack_frames(frames), device)
    env.close()
    policy_net.train()
    return durations


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
) -> tuple[list[int], list[dict]]:
    """Returns (episode_durations, history). `history` has one dict per episode with
    `episode`, `reward`, `loss` (mean over that episode's optimize_model calls, None if the
    replay buffer never reached batch_size during it), and `epsilon` (value at the episode's
    final step) -- enough to see reward/loss/epsilon trends across training. Printed to
    stdout every episode when `verbose`, and written to `results_path` as JSON after every
    episode (not just at the end) when given, so a killed/crashed run still leaves a log.

    `frame_stack` rendered frames are stacked as input channels (see stack_frames) so the
    network can infer velocity from motion across frames, not just instantaneous position --
    with frame_stack=1 the state is a single static frame, which has no velocity information
    at all.

    When `eval_every` is set, every `eval_every` episodes a separate greedy evaluation (see
    evaluate_greedy) of `eval_episodes` episodes runs, and its mean reward is recorded in
    that episode's history entry as `eval_mean_reward` (None on non-eval episodes). If
    `solved_mean_reward` is also set, training stops as soon as an eval's mean reward
    exceeds it -- deliberately decoupled from the noisy training rollout's reward (see
    evaluate_greedy's docstring for why checking the training rollout directly can't work).

    When `lr_decay_every` is set, the optimizer's learning rate is multiplied by
    `lr_decay_factor` every `lr_decay_every` episodes -- a constant lr can let training
    plateau at a "good enough but noisy" policy indefinitely; decaying it lets late training
    fine-tune toward the precision a sustained 500-step balance needs instead of continuing
    to take steps sized for early, coarse learning.

    `n_step` replaces each pushed transition's 1-step reward with a discounted sum of the
    next `n_step` rewards, bootstrapping from the state `n_step` steps later instead of the
    very next one -- with n_step=1 (the default) this reduces to plain 1-step TD, identical
    to prior behavior. 1-step TD only propagates a terminal outcome one step back per
    optimizer update, so a reward/penalty far from where a mistake was made takes many
    updates to reach the state that caused it; n-step returns carry it back n steps per
    update instead, which is the standard fix for slow credit assignment in DQN. At episode
    end, any remaining sub-n_step window is flushed as its own (shorter) return -- since its
    bootstrap state is masked out by optimize_model whenever the episode ended via genuine
    termination (only truncation-at-max-steps keeps a live bootstrap target), the exact
    discount exponent used for those few flushed transitions barely matters in practice."""
    n_actions = env.action_space.n
    in_channels = frame_stack
    gamma_n = gamma**n_step

    policy_net = DQN(in_channels, n_actions, image_size=image_size, hidden_sizes=hidden_sizes, conv_channels=conv_channels).to(device)
    target_net = DQN(in_channels, n_actions, image_size=image_size, hidden_sizes=hidden_sizes, conv_channels=conv_channels).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    episode_durations = []
    history = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        frames = deque([render_cartpole_frame(obs, size=image_size)] * frame_stack, maxlen=frame_stack)
        state = frame_to_tensor(stack_frames(frames), device)
        episode_losses = []
        eps = eps_start
        n_step_buffer = deque()  # (state, action, reward), oldest first
        for t in count():
            action, eps = select_action(state, policy_net, n_actions, steps_done, device, eps_start, eps_end, eps_decay)
            steps_done += 1
            obs, reward, terminated, truncated, _ = env.step(action.item())

            if terminated:
                next_state = None
            else:
                frames.append(render_cartpole_frame(obs, size=image_size))
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
                episode_reward = t + 1
                break

        episode_durations.append(episode_reward)
        avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else None

        if lr_decay_every is not None and (ep + 1) % lr_decay_every == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_decay_factor

        eval_mean_reward = None
        if eval_every is not None and (ep + 1) % eval_every == 0:
            eval_durations = evaluate_greedy(policy_net, device, image_size, frame_stack, num_episodes=eval_episodes)
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


def test_replay_memory_push_and_sample():
    memory = ReplayMemory(capacity=5)
    for i in range(3):
        memory.push(f"state{i}", f"action{i}", f"next{i}", f"reward{i}")
    assert len(memory) == 3
    sample = memory.sample(2)
    assert len(sample) == 2
    assert all(isinstance(t, Transition) for t in sample)


def test_replay_memory_respects_capacity():
    memory = ReplayMemory(capacity=3)
    for i in range(5):
        memory.push(i, i, i, i)
    assert len(memory) == 3
    assert [t.state for t in memory.memory] == [2, 3, 4]


def test_frame_to_tensor_shape_and_range():
    frame = render_cartpole_frame((0.0, 0.0, 0.0, 0.0), size=32)
    t = frame_to_tensor(frame, torch.device("cpu"))
    assert t.shape == (1, 1, 32, 32)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_stack_frames_concatenates_along_channel_axis():
    frames = [render_cartpole_frame((x, 0.0, 0.0, 0.0), size=32) for x in (-1.0, 0.0, 1.0)]
    stacked = stack_frames(frames)
    assert stacked.shape == (32, 32, 3)
    # each channel should reproduce its source frame exactly, in order
    for i, frame in enumerate(frames):
        assert (stacked[:, :, i] == frame[:, :, 0]).all()

    t = frame_to_tensor(stacked, torch.device("cpu"))
    assert t.shape == (1, 3, 32, 32)


def test_dqn_forward_shape():
    torch.manual_seed(0)
    net = DQN(in_channels=1, n_actions=2, image_size=32)
    x = torch.randn(6, 1, 32, 32)
    q = net(x)
    assert q.shape == (6, 2)


def test_dqn_forward_shape_with_hidden_layers():
    torch.manual_seed(0)
    net = DQN(in_channels=1, n_actions=2, image_size=32, hidden_sizes=(64, 32))
    x = torch.randn(6, 1, 32, 32)
    q = net(x)
    assert q.shape == (6, 2)
    # head should be [Linear, ReLU, Linear, ReLU, Linear]: two hidden layers, each ReLU-separated
    assert len(net.head) == 5
    assert isinstance(net.head[0], nn.Linear) and net.head[0].out_features == 64
    assert isinstance(net.head[2], nn.Linear) and net.head[2].out_features == 32
    assert isinstance(net.head[4], nn.Linear) and net.head[4].out_features == 2


def test_select_action_is_greedy_at_zero_epsilon():
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = DQN(in_channels=1, n_actions=3, image_size=16)
    state = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        expected = net(state).max(1).indices.view(1, 1)
    for steps_done in (0, 1, 100):
        action, eps = select_action(state, net, n_actions=3, steps_done=steps_done, device=device, eps_start=0.0, eps_end=0.0)
        assert eps == 0.0
        assert torch.equal(action, expected)


def test_select_action_epsilon_decays_towards_eps_end():
    device = torch.device("cpu")
    net = DQN(in_channels=1, n_actions=2, image_size=16)
    state = torch.randn(1, 1, 16, 16)
    _, eps_early = select_action(state, net, n_actions=2, steps_done=0, device=device, eps_start=0.9, eps_end=0.05, eps_decay=1000)
    _, eps_late = select_action(state, net, n_actions=2, steps_done=100000, device=device, eps_start=0.9, eps_end=0.05, eps_decay=1000)
    assert eps_early == 0.9
    assert eps_late < eps_early
    assert abs(eps_late - 0.05) < 1e-3


def test_optimize_model_returns_none_below_batch_size():
    device = torch.device("cpu")
    policy_net = DQN(1, 2, image_size=16)
    target_net = DQN(1, 2, image_size=16)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(capacity=10)
    memory.push(torch.randn(1, 1, 16, 16), torch.zeros(1, 1, dtype=torch.long), torch.randn(1, 1, 16, 16), torch.tensor([1.0]))
    loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=8)
    assert loss is None


def test_optimize_model_reduces_loss_over_several_steps():
    torch.manual_seed(0)
    device = torch.device("cpu")
    policy_net = DQN(1, 2, image_size=16)
    target_net = DQN(1, 2, image_size=16)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-2)

    memory = ReplayMemory(capacity=200)
    for i in range(64):
        state = torch.randn(1, 1, 16, 16)
        action = torch.tensor([[i % 2]], dtype=torch.long)
        next_state = torch.randn(1, 1, 16, 16)
        reward = torch.tensor([1.0])
        memory.push(state, action, next_state, reward)

    losses = []
    for _ in range(20):
        loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=32)
        losses.append(loss)

    assert all(loss is not None for loss in losses)
    assert losses[-1] < losses[0]


class _FixedQNet(nn.Module):
    """Outputs a fixed (n_actions,) Q-vector regardless of input -- lets a test pin down
    exactly which action argmax/gather pick, instead of depending on a real CNN's output."""

    def __init__(self, q_values: list[float]):
        super().__init__()
        self.q = nn.Parameter(torch.tensor(q_values, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q.unsqueeze(0).expand(x.shape[0], -1)


def test_optimize_model_uses_double_dqn_target():
    # policy_net ranks action 1 highest; target_net ranks action 0 highest but disagrees on
    # action 1's value (2.0, vs its own top pick's 5.0). Vanilla target_net.max(1) would
    # bootstrap from target_net's own argmax (5.0); Double DQN must instead bootstrap from
    # target_net's value at policy_net's argmax (2.0) -- these give clearly different losses.
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

    state_action_value = torch.tensor([[0.0]])  # policy_net's Q for action 0
    double_dqn_target = torch.tensor([[1.0 + 0.99 * 2.0]])  # reward + gamma * target_net's value at policy's argmax
    vanilla_target = torch.tensor([[1.0 + 0.99 * 5.0]])  # what target_net.max(1) alone would have used

    expected_loss = nn.SmoothL1Loss()(state_action_value, double_dqn_target).item()
    wrong_loss = nn.SmoothL1Loss()(state_action_value, vanilla_target).item()

    assert abs(loss - expected_loss) < 1e-4
    assert abs(loss - wrong_loss) > 1e-2


def test_soft_update_interpolates_towards_policy_net():
    policy_net = DQN(1, 2, image_size=16)
    target_net = DQN(1, 2, image_size=16)
    with torch.no_grad():
        for p in policy_net.parameters():
            p.fill_(1.0)
        for p in target_net.parameters():
            p.fill_(0.0)

    soft_update(policy_net, target_net, tau=0.1)
    for p in target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1))

    soft_update(policy_net, target_net, tau=0.1)
    for p in target_net.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.19))


def test_evaluate_greedy_is_deterministic_and_leaves_net_in_train_mode():
    # Same net, same eps=0 policy every call -> identical durations across repeated calls
    # (CartPole-v1's dynamics are deterministic given the same actions from the same seed
    # state distribution isn't fixed here, so instead check the *mechanism* directly: eval
    # uses argmax with no epsilon, so the action sequence for a fixed net depends only on
    # the (randomly reset) start state, not on run-to-run RNG draws for exploration).
    torch.manual_seed(0)
    device = torch.device("cpu")
    net = DQN(4, 2, image_size=32, hidden_sizes=(64,))
    net.train()
    durations = evaluate_greedy(net, device, image_size=32, frame_stack=4, num_episodes=3)
    assert len(durations) == 3
    assert all(d >= 1 for d in durations)
    assert net.training  # must restore train() mode, since optimize_model requires it


def test_train_dqn_smoke_runs_on_cartpole():
    # Small image size / few episodes per CLAUDE.md's "use small size of hidden units in
    # simple test" -- this checks the full tutorial loop (frame rendering, replay memory,
    # epsilon-greedy collection, optimize_model, soft target update) runs end-to-end without
    # error and produces one recorded duration per episode; it is not the 500-episode/
    # 5-streak acceptance bar from CLAUDE.md's Reinforcement learning test section, which
    # belongs to a full training run, not a fast unit test.
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    env = gym.make("CartPole-v1")
    try:
        num_episodes = 5
        durations, history = train_dqn(
            env,
            device,
            num_episodes=num_episodes,
            image_size=32,
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
        assert 0.0 <= entry["epsilon"] <= 1.0
    # batch_size=16 with a fresh replay buffer means the first few episodes' single-digit
    # step counts can't fill a batch yet -- loss should show up by the last episode.
    assert history[-1]["loss"] is not None


def test_train_dqn_with_n_step_runs_and_reports_real_env_rewards():
    # n_step only changes the TD target used to fit Q-values; episode_durations still
    # reflect actual environment reward (1 per surviving step), unaffected by n_step choice.
    # This smoke-tests the sliding-window n-step construction (including the end-of-episode
    # flush of shorter sub-n_step windows) runs end-to-end without error at both n_step=1
    # (must stay backward-compatible) and n_step=3.
    for n_step in (1, 3):
        torch.manual_seed(0)
        random.seed(0)
        device = get_device()
        env = gym.make("CartPole-v1")
        try:
            durations, history = train_dqn(
                env,
                device,
                num_episodes=5,
                image_size=32,
                batch_size=16,
                memory_capacity=500,
                eps_decay=200,
                verbose=False,
                n_step=n_step,
            )
        finally:
            env.close()
        assert len(durations) == 5
        assert all(d >= 1 for d in durations)
        assert len(history) == 5


def test_train_dqn_stops_early_when_solved():
    # solved_mean_reward=0 triggers on the very first eval (every episode always survives
    # at least 1 step, so mean reward > 0 always), with eval_every=1 -- confirms the
    # early-stop path fires off evaluate_greedy's result, not the noisy training reward,
    # without needing the model to actually learn anything.
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    env = gym.make("CartPole-v1")
    try:
        durations, history = train_dqn(
            env,
            device,
            num_episodes=20,
            image_size=32,
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
