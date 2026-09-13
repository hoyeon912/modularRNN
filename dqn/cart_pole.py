import math
from typing import cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import (
    AddRenderObservation,
    FrameStackObservation,
    GrayscaleObservation,
    ResizeObservation,
    TransformObservation,
)
from model import CNNDQN, DQN, ReplayBuffer, Transition
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 32
UPDATE_FREQ = 4
HIDDEN_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 25_000
TAU = 0.005
LR = 0.00025
LEARNING_STARTS = 10_000

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# device = "cpu"

env = gym.make("CartPole-v1", render_mode="rgb_array")
env = AddRenderObservation(env, render_only=True)
env = ResizeObservation(env, (84, 84))
env = GrayscaleObservation(env, keep_dim=False)
env = FrameStackObservation(env, stack_size=4)
env = TransformObservation(
    env,
    func=lambda obs: obs.astype(np.float32) / 255.0,
    observation_space=Box(
        low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
    ),
)

# obs_dim = cast(Box, env.observation_space).shape[0]
action_dim = cast(Discrete, env.action_space).n

# policy_net = DQN(obs_dim, action_dim, HIDDEN_SIZE).to(device)
# target_net = DQN(obs_dim, action_dim, HIDDEN_SIZE).to(device)

policy_net = CNNDQN(action_dim).to(device)
target_net = CNNDQN(action_dim).to(device)

target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(
    policy_net.parameters(), 
    lr=LR,
    )
memory = ReplayBuffer(100_000)

writer = SummaryWriter(log_dir="runs/cnndqn-hardcopy")


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # print('Optimizing model...')
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = (
            target_net(non_final_next_states).max(1).values
        )
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


obs, info = env.reset()
total_reward = 0
n_episodes = 1
for step in range(1_000_000):
    state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    elapsed = max(0, step - LEARNING_STARTS)
    remaining = max(0.0, 1.0-elapsed / EPS_DECAY)
    if torch.rand(1).item() < EPS_END + (EPS_START - EPS_END) * remaining:
        action = torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )
    else:
        action = torch.argmax(policy_net(state)).view(-1, 1)

    # print(action.item())
    obs, reward, terminated, truncated, info = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    total_reward += reward.item()
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        next_state = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    memory.push(state, action, next_state, reward)

    if step > LEARNING_STARTS and step % UPDATE_FREQ == 0:
        loss = optimize_model()
        if loss:
            writer.add_scalar("train/loss", loss, step)

    if done:
        obs, info = env.reset()
        print(f"Episode {n_episodes}, Total reward: {total_reward}")
        writer.add_scalar("train/reward", total_reward, n_episodes)
        total_reward = 0
        n_episodes += 1

    if step % 10_000 == 0:
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        target_net.load_state_dict(policy_net_state_dict)

env.close()
writer.close()
