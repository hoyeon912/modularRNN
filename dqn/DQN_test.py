from typing import cast

import gymnasium as gym
import torch
from gymnasium.spaces import Box, Discrete
from model import DQN, ReplayBuffer
from torch import optim

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

env = gym.make("CartPole-v1", render_mode="human")

obs_dim = cast(Box, env.observation_space).shape[0]
action_dim = cast(Discrete, env.action_space).n

BATCH_SIZE = 128
HIDDEN_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

policy_net = DQN(obs_dim, action_dim, HIDDEN_SIZE)
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

memory = ReplayBuffer(10000)

obs, info = env.reset()
total_reward = 0

for step in range(100000):
    state = torch.tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        values = policy_net(state)

    action = torch.argmax(values).item()
    next_obs, reward, terminated, truncated, info = env.step(action)
    reward = torch.tensor([reward], device=device)
    if terminated:
        next_obs = None
    # else:
    # nex

    total_reward += reward
    obs = next_obs

    memory.push((obs, action, reward, next_obs))

    if terminated or truncated:
        obs, info = env.reset()
        print("Total reward:", total_reward)
        total_reward = 0

env.close()
