"""REINFORCE (policy-gradient) training of CNN-ModularRNN on CartPole-v1, as the REINFORCE
counterpart to scripts/test_cnn_modular_rnn_dqn_cartpole.py's TD/Double-DQN training.

Motivation: scripts/test_cnn_modular_rnn_dqn_cartpole.py's train_dqn_modular, searched
exhaustively across n_step/tau/lr/lr-decay/recurrent_gain (see
results/cnn_modular_rnn_dqn_search and results/cnn_modular_rnn_dqn_round2), never learned
CartPole at all -- flat eval reward for the full 5000-episode budget regardless of
hyperparameters. Meanwhile results/modular_rnn_sparsity_search showed that the *same*
restricted modular connectivity, trained via REINFORCE on raw (non-pixel) state, clearly
learns (steadily climbing reward, still improving at the 300-update cutoff) at every
near_module_sparsity value tested including CLAUDE.md's spec value of 0.1. That isolates the
axis: it's not the architecture that blocks TD-style training, it's TD/bootstrap-style
credit assignment specifically. This script tests whether REINFORCE also rescues the
*pixel*-input case (CNN + ModularRNN together), which is what CLAUDE.md's CNN-ModularRNN-DQN
goal actually needs to work.

Reuses models.cnn_modular_rnn.CNNModularRNN directly (CNN encoder + ModularRNN core, with
both a step(frame_t, h) incremental interface for rollout and a batched forward() -- only
the former is used here, since REINFORCE collects one episode at a time) rather than
redefining another CNN+ModularRNN network, and reuses scripts/train_cartpole.py's
compute_returns/reinforce_update (both architecture-agnostic: they operate on already-
collected log_probs/entropies/rewards, not on raw states) instead of reimplementing REINFORCE
bookkeeping.
"""

import json
import random
import sys
from collections import deque
from itertools import count
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch
import torch.optim as optim
from torch.distributions import Categorical

from models.cnn_modular_rnn import CNNModularRNN
from models.common import get_device
from scripts.cartpole_render import render_cartpole_frame
from scripts.test_dqn_cartpole import frame_to_tensor, stack_frames
from scripts.train_cartpole import reinforce_update


def collect_episode_stochastic_modular(
    model: CNNModularRNN,
    env: gym.Env,
    device: torch.device,
    image_size: int,
    frame_stack: int,
    max_steps: int = 500,
):
    """Rolls out one episode sampling actions from Categorical(logits=model.step(...)),
    carrying hidden state incrementally (same O(T) rationale as
    test_cnn_modular_rnn_dqn_cartpole's rollouts). Returns (log_probs, entropies, rewards)
    in the exact shape scripts.train_cartpole.reinforce_update expects."""
    model.train()
    obs, _ = env.reset()
    frames = deque([render_cartpole_frame(obs, size=image_size)] * frame_stack, maxlen=frame_stack)
    state = frame_to_tensor(stack_frames(frames), device)
    h = model.init_hidden(1, device)

    log_probs, entropies, rewards = [], [], []
    for _ in range(max_steps):
        logits, h = model.step(state, h)
        dist = Categorical(logits=logits[0])
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        entropies.append(dist.entropy())

        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards.append(reward)
        if terminated or truncated:
            break
        frames.append(render_cartpole_frame(obs, size=image_size))
        state = frame_to_tensor(stack_frames(frames), device)

    return log_probs, entropies, rewards


def evaluate_greedy_reinforce_modular(
    model: CNNModularRNN,
    device: torch.device,
    image_size: int,
    frame_stack: int,
    num_episodes: int = 5,
    max_steps: int = 500,
) -> list[int]:
    """Deterministic (argmax over policy logits, no sampling) rollout, decoupled from the
    noisy stochastic-policy training rollout -- same rationale as
    test_dqn_cartpole.evaluate_greedy's docstring."""
    model.eval()
    env = gym.make("CartPole-v1")
    durations = []
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, _ = env.reset()
            frames = deque([render_cartpole_frame(obs, size=image_size)] * frame_stack, maxlen=frame_stack)
            state = frame_to_tensor(stack_frames(frames), device)
            h = model.init_hidden(1, device)
            for t in count():
                logits, h = model.step(state, h)
                action = logits.argmax(dim=1)
                obs, reward, terminated, truncated, _ = env.step(action.item())
                if terminated or truncated or t + 1 >= max_steps:
                    durations.append(t + 1)
                    break
                frames.append(render_cartpole_frame(obs, size=image_size))
                state = frame_to_tensor(stack_frames(frames), device)
    env.close()
    model.train()
    return durations


def train_reinforce_modular(
    env: gym.Env,
    device: torch.device,
    num_updates: int,
    episodes_per_update: int = 8,
    image_size: int = 64,
    rnn_hidden_size: int = 300,
    near_module_sparsity: float = 0.1,
    cnn_feature_dim: int = 32,
    frame_stack: int = 4,
    gamma: float = 0.99,
    lr: float = 1e-3,
    entropy_coef: float = 0.01,
    results_path: str | None = None,
    verbose: bool = True,
    eval_every: int | None = None,
    eval_episodes: int = 5,
    solved_mean_reward: float | None = None,
    pretrained_encoder_path: str | None = None,
    freeze_encoder: bool = False,
) -> tuple[list[float], list[dict]]:
    """REINFORCE training loop for CNNModularRNN, mirroring
    test_cnn_modular_rnn_dqn_cartpole.train_dqn_modular's structure (per-update logging,
    incremental results_path writes, decoupled greedy evaluation, early stop on
    solved_mean_reward) but with REINFORCE's episode-batch policy-gradient update in place of
    replay-buffer TD.

    `pretrained_encoder_path`, if given, loads a CNNEncoder state_dict (see
    scripts/pretrain_cnn_encoder_cartpole.py) into model.encoder before training -- its
    cnn_feature_dim/frame_stack/image_size must match this call's, or loading raises.
    `freeze_encoder` then stops those weights from being fine-tuned (requires_grad=False,
    filtered out of the optimizer), isolating whether a pretrained-but-frozen perception
    front-end helps REINFORCE learn faster/more stably through the restricted ModularRNN
    core, versus only warm-starting it and still letting REINFORCE's gradient reach it."""
    n_actions = env.action_space.n
    model = CNNModularRNN(
        in_channels=frame_stack,
        hidden_size=rnn_hidden_size,
        output_size=n_actions,
        cnn_feature_dim=cnn_feature_dim,
        output_mode="last",
        near_module_sparsity=near_module_sparsity,
        image_size=image_size,
    ).to(device)
    if pretrained_encoder_path is not None:
        model.encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device, weights_only=True))
        if freeze_encoder:
            for p in model.encoder.parameters():
                p.requires_grad = False
    optimizer = optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

    training_rewards = []
    history = []
    for update in range(num_updates):
        batch = [
            collect_episode_stochastic_modular(model, env, device, image_size, frame_stack)
            for _ in range(episodes_per_update)
        ]
        loss = reinforce_update(model, optimizer, batch, gamma=gamma, entropy_coef=entropy_coef)
        batch_reward = sum(sum(rewards) for _, _, rewards in batch) / len(batch)
        training_rewards.append(batch_reward)

        eval_mean_reward = None
        if eval_every is not None and (update + 1) % eval_every == 0:
            eval_durations = evaluate_greedy_reinforce_modular(model, device, image_size, frame_stack, num_episodes=eval_episodes)
            eval_mean_reward = sum(eval_durations) / len(eval_durations)

        history.append(
            {
                "update": update + 1,
                "loss": loss,
                "reward": batch_reward,
                "eval_mean_reward": eval_mean_reward,
            }
        )
        if verbose:
            eval_str = f" eval_mean_reward={eval_mean_reward:.1f}" if eval_mean_reward is not None else ""
            print(f"update {update + 1}/{num_updates} loss {loss:.4f} reward {batch_reward:.1f}{eval_str}", flush=True)
        if results_path is not None:
            with open(results_path, "w") as f:
                json.dump(history, f, indent=2)

        if solved_mean_reward is not None and eval_mean_reward is not None and eval_mean_reward > solved_mean_reward:
            if verbose:
                print(f"solved: greedy eval mean reward {eval_mean_reward:.1f} > {solved_mean_reward}", flush=True)
            break

    return training_rewards, history


# --- unit tests ---


def test_collect_episode_stochastic_modular_returns_matching_lengths():
    torch.manual_seed(0)
    random.seed(0)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    model = CNNModularRNN(in_channels=4, hidden_size=9, output_size=2, image_size=32, near_module_sparsity=0.1)
    try:
        log_probs, entropies, rewards = collect_episode_stochastic_modular(model, env, device, image_size=32, frame_stack=4, max_steps=10)
    finally:
        env.close()
    assert len(log_probs) == len(entropies) == len(rewards)
    assert len(rewards) >= 1
    assert all(r == 1.0 for r in rewards)


def test_evaluate_greedy_reinforce_modular_runs_and_leaves_model_in_train_mode():
    torch.manual_seed(0)
    device = torch.device("cpu")
    model = CNNModularRNN(in_channels=4, hidden_size=9, output_size=2, image_size=32, near_module_sparsity=0.1)
    model.train()
    durations = evaluate_greedy_reinforce_modular(model, device, image_size=32, frame_stack=4, num_episodes=3)
    assert len(durations) == 3
    assert all(d >= 1 for d in durations)
    assert model.training


def test_train_reinforce_modular_smoke_runs_on_cartpole():
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    env = gym.make("CartPole-v1")
    try:
        num_updates = 3
        rewards, history = train_reinforce_modular(
            env,
            device,
            num_updates=num_updates,
            episodes_per_update=2,
            image_size=32,
            rnn_hidden_size=9,
            verbose=False,
        )
    finally:
        env.close()

    assert len(rewards) == num_updates
    assert len(history) == num_updates
    for i, entry in enumerate(history):
        assert entry["update"] == i + 1
        assert entry["reward"] == rewards[i]
        assert entry["loss"] is not None


def test_train_reinforce_modular_stops_early_when_solved():
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    env = gym.make("CartPole-v1")
    try:
        rewards, history = train_reinforce_modular(
            env,
            device,
            num_updates=10,
            episodes_per_update=2,
            image_size=32,
            rnn_hidden_size=9,
            verbose=False,
            eval_every=1,
            eval_episodes=2,
            solved_mean_reward=0,
        )
    finally:
        env.close()

    assert len(rewards) == 1
    assert len(history) == 1
    assert history[0]["eval_mean_reward"] is not None
    assert history[0]["eval_mean_reward"] > 0
