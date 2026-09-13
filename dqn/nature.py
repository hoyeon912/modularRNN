"""PyTorch implementation of the Nature (2015) DQN learning algorithm.

The CNN is shared with model.CNNDQN. Atari dependencies are only needed by
train_nature.py. All schedules count agent decisions, not raw emulator frames.
"""

import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .model import CNNDQN


class NatureDQN(CNNDQN):
    """4 x 84 x 84 -> conv(32,64,64) -> FC(512) -> action Q-values.

    Accepts batched uint8 pixels (normalized internally) or floating point
    observations already normalized to [0, 1]. No softmax on the output.
    """

    def __init__(self, n_actions):
        if n_actions < 1:
            raise ValueError("n_actions must be positive")
        super().__init__(n_actions)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (4, 84, 84):
            raise ValueError("expected observations shaped (batch, 4, 84, 84)")
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
        return super().forward(x)


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 0.00025
    batch_size: int = 32
    replay_capacity: int = 1_000_000
    learning_starts: int = 50_000
    train_frequency: int = 4
    target_update_interval: int = 10_000
    exploration_steps: int = 1_000_000
    epsilon_final: float = 0.1

    def __post_init__(self):
        for name in (
            "batch_size",
            "replay_capacity",
            "train_frequency",
            "target_update_interval",
            "exploration_steps",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be >= batch_size")
        if self.learning_starts < 0 or self.learning_rate <= 0:
            raise ValueError("invalid learning_starts or learning_rate")
        if not 0 <= self.gamma <= 1 or not 0 <= self.epsilon_final <= 1:
            raise ValueError("gamma and epsilon_final must be in [0, 1]")

    def epsilon(self, step):
        fraction = min(
            1.0, max(0, step - self.learning_starts) / self.exploration_steps
        )
        return 1.0 + fraction * (self.epsilon_final - 1.0)


class NatureRMSprop(torch.optim.Optimizer):
    """Centered RMSProp with epsilon *inside* sqrt, as in DQN's Lua learner.

    m <- .95*m + .05*g; v <- .95*v + .05*g*g
    parameter <- parameter - lr*g/sqrt(v - m*m + .01)
    There is no additional momentum buffer or weight decay.
    """

    def __init__(self, params, lr=0.00025, decay=0.95, eps=0.01):
        if lr <= 0 or not 0 <= decay < 1 or eps <= 0:
            raise ValueError("invalid RMSProp hyperparameters")
        super().__init__(params, {"lr": lr, "decay": decay, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            decay = group["decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("NatureRMSprop requires dense gradients")
                state = self.state[parameter]
                if not state:
                    state["mean"] = torch.zeros_like(parameter)
                    state["square"] = torch.zeros_like(parameter)
                mean, square = state["mean"], state["square"]
                mean.lerp_(gradient, 1 - decay)
                square.mul_(decay).addcmul_(gradient, gradient, value=1 - decay)
                variance = (square - mean.square()).clamp_min_(0)
                denominator = variance.add_(group["eps"]).sqrt_()
                parameter.addcdiv_(gradient, denominator, value=-group["lr"])
        return loss


class ReplayMemory:
    """Uniform replay with shared, owned uint8 frames and a list ring buffer.

    start_episode establishes a zero-padded history. append stores one new
    frame; adjacent transitions share immutable frame objects. Sampling
    materializes full stacks only for the minibatch. Evicting a transition
    cannot invalidate history still referenced by another transition.
    """

    def __init__(self, capacity, seed=0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.history = None
        self.rng = random.Random(seed)

    @staticmethod
    def _copy_frame(frame):
        frame = np.asarray(frame)
        if frame.shape != (84, 84) or frame.dtype != np.uint8:
            raise ValueError("frames must be uint8 arrays shaped (84, 84)")
        frame = frame.copy()
        frame.flags.writeable = False
        return frame

    def start_episode(self, frame):
        frame = self._copy_frame(frame)
        padding = np.zeros_like(frame)
        padding.flags.writeable = False
        self.history = (padding, padding, padding, frame)

    @property
    def state(self):
        if self.history is None:
            raise RuntimeError("call start_episode before accessing state")
        return np.stack(self.history)

    def append(self, action, reward, next_frame, terminal):
        if self.history is None:
            raise RuntimeError("call start_episode before append")
        next_history = self.history[1:] + (self._copy_frame(next_frame),)
        transition = (
            self.history,
            int(action),
            float(np.clip(reward, -1, 1)),
            next_history,
            bool(terminal),
        )
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
        self.history = next_history

    def sample(self, batch_size):
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be in [1, len(memory)]")
        states, actions, rewards, next_states, terminals = zip(
            *self.rng.sample(self.memory, batch_size)
        )
        return (
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.stack(next_states)),
            torch.tensor(terminals, dtype=torch.bool),
        )

    def __len__(self):
        return len(self.memory)


def td_targets(rewards, terminals, next_q_values, gamma):
    """Vanilla DQN target: r + gamma * (1-terminal) * max_a Q_target(s', a)."""
    bootstrap = next_q_values.max(dim=1).values
    return rewards + gamma * bootstrap.masked_fill(terminals, 0.0)


class DQNAgent:
    def __init__(self, n_actions, config=None, device="cpu", seed=0):
        self.config = config or DQNConfig()
        self.device = torch.device(device)
        self.n_actions = int(n_actions)
        self.rng = random.Random(seed)
        self.online = NatureDQN(n_actions).to(self.device)
        self.target = deepcopy(self.online).eval().requires_grad_(False)
        self.optimizer = NatureRMSprop(
            self.online.parameters(), lr=self.config.learning_rate
        )

    @torch.no_grad()
    def act(self, state, epsilon=0.0):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if self.rng.random() < epsilon:
            return self.rng.randrange(self.n_actions)
        state = torch.as_tensor(state, device=self.device).unsqueeze(0)
        values = self.online(state)[0]
        best = (values == values.max()).nonzero().flatten().tolist()
        return self.rng.choice(best)

    def update(self, batch):
        states, actions, rewards, next_states, terminals = (
            tensor.to(self.device) for tensor in batch
        )
        values = self.online(states).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            targets = td_targets(
                rewards, terminals, self.target(next_states), self.config.gamma
            )
        # Sum preserves the original Lua backward's unaveraged minibatch
        # gradients. Huber(delta=1) clips each TD derivative to [-1, 1].
        losses = nn.functional.smooth_l1_loss(values, targets, reduction="none")
        self.optimizer.zero_grad(set_to_none=True)
        losses.sum().backward()
        self.optimizer.step()
        return losses.detach().mean().item()

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
