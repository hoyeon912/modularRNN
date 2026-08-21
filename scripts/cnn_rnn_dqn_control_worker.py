"""Control-arm worker for scripts/search_cnn_modular_rnn_dqn_round2.py: trains one
*unrestricted* CNN-RNN-DQN (scripts/test_cnn_rnn_dqn_cartpole.train_dqn_recurrent, plain
nn.RNNCell, no modular masking) on CartPole-v1, on whichever single GPU the orchestrator has
exposed via CUDA_VISIBLE_DEVICES. Otherwise identical to
scripts/cnn_modular_rnn_dqn_search_worker.py -- same config-file/output-layout contract --
so its results are directly comparable to the modular arm's, isolating whether restricted
modular connectivity itself is why CNN-ModularRNN-DQN failed to learn CartPole in round 1
(see results/cnn_modular_rnn_dqn_search/summary.json), independent of any hyperparameter
choice.
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
from scripts.test_cnn_rnn_dqn_cartpole import train_dqn_recurrent


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
        train_dqn_recurrent(
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
