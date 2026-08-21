"""Worker for scripts/search_cnn_modular_rnn_dqn_cartpole.py: trains one CNN-ModularRNN-DQN
hyperparameter combination on CartPole-v1, on whichever single GPU the orchestrator has
exposed via CUDA_VISIBLE_DEVICES before this process starts. Writes hyperparameters.json
once up front and results.json after every episode (train_dqn_modular's own behavior with
results_path set) under --results-dir, matching the layout used by earlier searches (see
results/cnn_rnn_dqn_search). Runs verbose so every episode's reward/loss/epsilon/eval line is
also captured in the orchestrator-redirected train.log.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import torch

from models.common import get_device
from scripts.test_cnn_modular_rnn_dqn_cartpole import train_dqn_modular


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "hyperparameters.json").write_text(json.dumps(config, indent=2))

    seed = config["seed"]
    torch.manual_seed(seed)
    random.seed(seed)

    device = get_device()
    print(f"device={device}", flush=True)
    env = gym.make("CartPole-v1")
    train_kwargs = {k: v for k, v in config.items() if k not in ("seed", "results_dir", "name")}
    try:
        train_dqn_modular(
            env,
            device,
            results_path=str(results_dir / "results.json"),
            verbose=True,
            **train_kwargs,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
