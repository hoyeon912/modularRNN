"""Double DQN on LunarLander-v3, vector-state variant: a plain MLP reading the environment's
raw 8-dim observation (x, y, vx, vy, angle, angular_velocity, left_leg_contact,
right_leg_contact) directly -- no rendering, no CNN. Same algorithmic machinery proven on
CartPole in scripts/test_dqn_cartpole.py: Double-Q target, n-step returns, a Polyak-updated
target network, and a decoupled greedy evaluation used to judge "solved" instead of the
noisy training rollout (see evaluate_greedy's docstring for why that distinction matters).

LunarLander differs from CartPole in ways that change tuning, not the algorithm:
  - 4 discrete actions (do nothing, fire left/main/right engine) instead of 2.
  - The observation already includes both position and velocity terms, so there's no
    CartPole-style "one static frame has no velocity" problem here -- no frame stacking or
    recurrence is needed just to make the state Markov.
  - Reward is far denser and larger-scale (shaped: distance/speed/angle/leg-contact/fuel
    terms each step, plus +100 for a safe landing or -100 for a crash), and the standard
    "solved" bar (per Gymnasium's own definition) is an average reward >= 200 over 100
    consecutive episodes, not a fixed per-episode cap like CartPole's 500.
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

ENV_ID = "LunarLander-v3"
OBS_DIM = 8
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
    """MLP Q-network: obs_dim -> hidden_sizes (ReLU-separated) -> n_actions. hidden_sizes=()
    would give a single Linear straight from obs to Q-values; LunarLander's task is more
    involved than CartPole's, so the default here is a real hidden stack, not empty."""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = N_ACTIONS, hidden_sizes: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)


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
    """Double-Q target (van Hasselt et al., 2016): policy_net picks the next action, target_net
    scores it -- see scripts/test_dqn_cartpole.py's module docstring for why plain
    target_net(...).max(1) causes Q-value overestimation that this decoupling fixes."""
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


def evaluate_greedy(policy_net: DQN, device: torch.device, num_episodes: int = 5, max_steps: int = 1000) -> list[float]:
    """Separate deterministic (argmax, no epsilon) rollout, decoupled from the noisy training
    rollout -- same reasoning as scripts/test_dqn_cartpole.py's evaluate_greedy: with a
    persistent epsilon floor, the training-time reward can't reliably reflect policy quality.
    Returns per-episode total reward (LunarLander's reward is continuous-valued, unlike
    CartPole's integer step count, so this returns floats)."""
    policy_net.eval()
    env = gym.make(ENV_ID)
    totals = []
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            state = obs_to_tensor(obs, device)
            total = 0.0
            for t in count():
                action = policy_net(state).max(1).indices.view(1, 1)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                total += reward
                if terminated or truncated or t + 1 >= max_steps:
                    break
                state = obs_to_tensor(obs, device)
            totals.append(total)
    env.close()
    policy_net.train()
    return totals


def train_dqn(
    env: gym.Env,
    device: torch.device,
    num_episodes: int,
    hidden_sizes: tuple[int, ...] = (128, 128),
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
    the shared n_step/lr_decay/eval_every/solved_mean_reward/save_best_path semantics --
    identical here, just operating on raw observation vectors instead of rendered frames."""
    n_actions = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    gamma_n = gamma**n_step

    policy_net = DQN(obs_dim, n_actions, hidden_sizes=hidden_sizes).to(device)
    target_net = DQN(obs_dim, n_actions, hidden_sizes=hidden_sizes).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    memory = ReplayMemory(memory_capacity)

    steps_done = 0
    best_eval_mean_reward = float("-inf")
    episode_rewards = []
    history = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = obs_to_tensor(obs, device)
        episode_losses = []
        eps = eps_start
        episode_reward = 0.0
        n_step_buffer = deque()  # (state, action, reward), oldest first
        for t in count():
            action, eps = select_action(state, policy_net, n_actions, steps_done, device, eps_start, eps_end, eps_decay)
            steps_done += 1
            obs, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward
            truncated = truncated or (t + 1 >= max_steps)

            next_state = None if terminated else obs_to_tensor(obs, device)

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
            eval_rewards = evaluate_greedy(policy_net, device, num_episodes=eval_episodes, max_steps=max_steps)
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
    memory.push(torch.zeros(1, OBS_DIM), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, OBS_DIM), torch.tensor([1.0]))
    assert len(memory) == 1
    sampled = memory.sample(1)
    assert len(sampled) == 1


def test_replay_memory_respects_capacity():
    memory = ReplayMemory(3)
    for i in range(5):
        memory.push(torch.zeros(1, OBS_DIM), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, OBS_DIM), torch.tensor([float(i)]))
    assert len(memory) == 3


def test_dqn_forward_shape():
    net = DQN(OBS_DIM, N_ACTIONS, hidden_sizes=())
    out = net(torch.zeros(2, OBS_DIM))
    assert out.shape == (2, N_ACTIONS)


def test_dqn_forward_shape_with_hidden_layers():
    net = DQN(OBS_DIM, N_ACTIONS, hidden_sizes=(16, 16))
    out = net(torch.zeros(3, OBS_DIM))
    assert out.shape == (3, N_ACTIONS)


def test_select_action_is_greedy_at_zero_epsilon():
    net = DQN(OBS_DIM, N_ACTIONS)
    device = torch.device("cpu")
    state = torch.randn(1, OBS_DIM)
    with torch.no_grad():
        expected = net(state).max(1).indices.item()
    action, eps = select_action(state, net, N_ACTIONS, steps_done=10_000_000, device=device, eps_start=0.9, eps_end=0.0, eps_decay=1)
    assert eps == 0.0
    assert action.item() == expected


def test_select_action_epsilon_decays_towards_eps_end():
    net = DQN(OBS_DIM, N_ACTIONS)
    device = torch.device("cpu")
    state = torch.randn(1, OBS_DIM)
    _, eps_early = select_action(state, net, N_ACTIONS, steps_done=0, device=device, eps_start=0.9, eps_end=0.05, eps_decay=1000)
    _, eps_late = select_action(state, net, N_ACTIONS, steps_done=1_000_000, device=device, eps_start=0.9, eps_end=0.05, eps_decay=1000)
    assert eps_early > eps_late
    assert abs(eps_late - 0.05) < 1e-6


def test_optimize_model_returns_none_below_batch_size():
    device = torch.device("cpu")
    policy_net = DQN(OBS_DIM, N_ACTIONS)
    target_net = DQN(OBS_DIM, N_ACTIONS)
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-3)
    memory = ReplayMemory(100)
    memory.push(torch.zeros(1, OBS_DIM), torch.zeros(1, 1, dtype=torch.long), torch.ones(1, OBS_DIM), torch.tensor([1.0]))
    loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=8)
    assert loss is None


def test_optimize_model_reduces_loss_over_several_steps():
    torch.manual_seed(0)
    device = torch.device("cpu")
    policy_net = DQN(OBS_DIM, N_ACTIONS, hidden_sizes=(32,))
    target_net = DQN(OBS_DIM, N_ACTIONS, hidden_sizes=(32,))
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=1e-2)
    memory = ReplayMemory(200)
    for _ in range(64):
        memory.push(
            torch.randn(1, OBS_DIM),
            torch.randint(0, N_ACTIONS, (1, 1)),
            torch.randn(1, OBS_DIM),
            torch.tensor([random.random()]),
        )
    losses = []
    for _ in range(20):
        loss = optimize_model(memory, policy_net, target_net, optimizer, device, batch_size=32)
        losses.append(loss)
    assert all(l is not None for l in losses)
    assert sum(losses[-5:]) / 5 < sum(losses[:5]) / 5


class _FixedQNet(nn.Module):
    """Outputs a fixed (n_actions,) Q-vector regardless of input -- lets a test pin down
    exactly which action argmax/gather pick, instead of depending on a real MLP's output."""

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

    state = torch.zeros(1, OBS_DIM)
    action = torch.tensor([[0]], dtype=torch.long)
    next_state = torch.zeros(1, OBS_DIM)
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
    policy_net = DQN(OBS_DIM, N_ACTIONS)
    target_net = DQN(OBS_DIM, N_ACTIONS)
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
    net = DQN(OBS_DIM, N_ACTIONS)
    net.train()
    rewards = evaluate_greedy(net, device, num_episodes=2, max_steps=50)
    assert len(rewards) == 2
    assert net.training is True


def test_train_dqn_smoke_runs_on_lunarlander():
    # CLAUDE.md: "use small size of hidden units in simple test" -- this checks the full loop
    # (replay memory, epsilon-greedy collection, optimize_model, soft target update) runs
    # end-to-end without error and produces one recorded reward per episode; it is not the
    # 500-episode acceptance bar from CLAUDE.md's Reinforcement learning test section, which
    # belongs to a full training run, not a fast unit test.
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID)
    try:
        num_episodes = 3
        rewards, history = train_dqn(
            env,
            device,
            num_episodes=num_episodes,
            hidden_sizes=(16,),
            batch_size=16,
            memory_capacity=500,
            eps_decay=200,
            verbose=False,
            max_steps=100,
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
        env = gym.make(ENV_ID)
        try:
            rewards, history = train_dqn(
                env,
                device,
                num_episodes=3,
                hidden_sizes=(16,),
                batch_size=16,
                memory_capacity=500,
                eps_decay=200,
                verbose=False,
                n_step=n_step,
                max_steps=100,
            )
        finally:
            env.close()
        assert len(rewards) == 3
        assert len(history) == 3


def test_train_dqn_stops_early_when_solved():
    # solved_mean_reward=-1e9 triggers on the very first eval regardless of what the policy
    # actually does -- confirms the early-stop path fires off evaluate_greedy's result, not
    # the noisy training reward, without needing the model to actually learn anything.
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID)
    try:
        rewards, history = train_dqn(
            env,
            device,
            num_episodes=10,
            hidden_sizes=(16,),
            batch_size=16,
            memory_capacity=500,
            eps_decay=200,
            verbose=False,
            eval_every=1,
            eval_episodes=1,
            solved_mean_reward=-1e9,
            max_steps=50,
        )
    finally:
        env.close()
    assert len(rewards) < 10
    assert history[-1]["eval_mean_reward"] is not None


def test_train_dqn_saves_best_checkpoint(tmp_path):
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make(ENV_ID)
    save_path = tmp_path / "best_model.pt"
    try:
        train_dqn(
            env,
            device,
            num_episodes=3,
            hidden_sizes=(16,),
            batch_size=16,
            memory_capacity=500,
            eps_decay=200,
            verbose=False,
            eval_every=1,
            eval_episodes=1,
            max_steps=50,
            save_best_path=str(save_path),
        )
    finally:
        env.close()
    assert save_path.exists()
    state_dict = torch.load(save_path, weights_only=True)
    net = DQN(OBS_DIM, N_ACTIONS, hidden_sizes=(16,))
    net.load_state_dict(state_dict)  # raises if shapes mismatch
