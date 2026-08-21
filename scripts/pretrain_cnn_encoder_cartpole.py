"""Supervised pretraining for models.cnn_modular_rnn.CNNEncoder on CartPole-v1: regress the
encoder's flat feature vector (via a throwaway Linear head) onto the true 4-dim physics state
(x, x_dot, theta, theta_dot) from a stacked-frame render, so a downstream REINFORCE run (see
scripts/test_cnn_modular_rnn_reinforce_cartpole.py) only has to learn the control policy
through the restricted ModularRNN core, not raw pixel perception, from scratch.

Motivation: scripts/search_cnn_modular_rnn_reinforce_longbudget.py showed CNN-ModularRNN via
REINFORCE *does* eventually learn given enough budget (rnn_hidden_size=600 broke out of a
flat ~25 reward plateau around update ~3500/8000, reaching up to ~320/500) but is slow to
start and noisy once it does. A pretrained encoder is the natural next lever: if perception
is solved upfront, REINFORCE's comparatively weak, high-variance gradient signal only has to
shape the policy, not simultaneously bootstrap a conv stack from random init.

Training states are collected by rolling out *random*-action episodes (not env.reset()'s
default narrow init distribution, which only covers |x|,|theta| <~ 0.05) so the regression
dataset actually covers the range of states a partially-trained policy will encounter,
including large excursions toward the failure boundary.

Saves only the encoder's state_dict (not the throwaway regression head) to --output, for
scripts/test_cnn_modular_rnn_reinforce_cartpole.py's train_reinforce_modular to load via its
pretrained_encoder_path option.
"""

import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from models.cnn_modular_rnn import CNNEncoder
from models.common import get_device
from scripts.cartpole_render import render_cartpole_frame
from scripts.test_dqn_cartpole import frame_to_tensor, stack_frames


def collect_dataset(num_samples: int, image_size: int, frame_stack: int, device: torch.device):
    """Rolls out random-action episodes, recording (stacked-frame tensor, true 4-dim state)
    pairs at every step, until num_samples pairs are collected. Random actions (not the
    narrow default reset distribution) push the cart/pole through a much wider range of
    states, including the large excursions a partially-trained policy will actually see."""
    env = gym.make("CartPole-v1")
    frames_x, targets_y = [], []
    while len(frames_x) < num_samples:
        obs, _ = env.reset()
        frames = deque([render_cartpole_frame(obs, size=image_size)] * frame_stack, maxlen=frame_stack)
        for _ in range(500):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            frames.append(render_cartpole_frame(obs, size=image_size))
            frames_x.append(stack_frames(frames))
            targets_y.append(np.asarray(obs, dtype=np.float32))
            if terminated or truncated or len(frames_x) >= num_samples:
                break
    env.close()

    x = torch.from_numpy(np.stack(frames_x)).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    y = torch.from_numpy(np.stack(targets_y)).to(device=device, dtype=torch.float32)
    return x, y


def pretrain(
    num_samples: int = 20000,
    image_size: int = 64,
    frame_stack: int = 4,
    cnn_feature_dim: int = 32,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    output: str = "cnn_encoder_pretrained.pt",
) -> None:
    torch.manual_seed(seed)
    device = get_device()
    print(f"device={device}", flush=True)

    print(f"collecting {num_samples} (frame, state) pairs from random-action rollouts...", flush=True)
    x, y = collect_dataset(num_samples, image_size, frame_stack, device)
    # Per-dimension normalization: x/x_dot/theta/theta_dot live on very different scales
    # (position in [-2.4, 2.4], angle in [-0.21, 0.21] rad, velocities effectively unbounded
    # but usually O(1)) -- an unnormalized MSE loss would be dominated by whichever dimension
    # has the largest raw magnitude, learning that one well and the others poorly.
    y_mean, y_std = y.mean(dim=0), y.std(dim=0).clamp_min(1e-6)
    y_norm = (y - y_mean) / y_std

    encoder = CNNEncoder(frame_stack, cnn_feature_dim, image_size=image_size).to(device)
    head = nn.Linear(cnn_feature_dim, 4).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.MSELoss()

    n = x.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred = head(encoder(x[idx]))
            loss = criterion(pred, y_norm[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch + 1}/{epochs} mse {epoch_loss / n_batches:.5f}", flush=True)

    torch.save(encoder.state_dict(), output)
    print(f"saved pretrained encoder to {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20000)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--cnn-feature-dim", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="cnn_encoder_pretrained.pt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pretrain(
        num_samples=args.num_samples,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        cnn_feature_dim=args.cnn_feature_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        output=args.output,
    )


# --- unit tests ---


def test_collect_dataset_shapes():
    device = torch.device("cpu")
    x, y = collect_dataset(num_samples=20, image_size=16, frame_stack=2, device=device)
    assert x.shape[0] >= 20
    assert x.shape[1:] == (2, 16, 16)
    assert y.shape == (x.shape[0], 4)
    assert x.min() >= 0.0 and x.max() <= 1.0


def test_pretrain_smoke_runs_and_saves_encoder(tmp_path):
    output = tmp_path / "encoder.pt"
    pretrain(
        num_samples=32,
        image_size=16,
        frame_stack=2,
        cnn_feature_dim=8,
        epochs=1,
        batch_size=16,
        output=str(output),
    )
    assert output.exists()

    state_dict = torch.load(output, weights_only=True)
    encoder = CNNEncoder(2, 8, image_size=16)
    encoder.load_state_dict(state_dict)  # raises if shapes/keys mismatch
