"""Worker for scripts/search_modular_rnn_sparsity_cartpole.py: trains ModularRNN directly on
CartPole-v1's raw 4-dim state (no CNN, no pixel rendering -- cheapest possible test of the
recurrent core itself) via scripts/train_cartpole.py's Adam/REINFORCE loop, at one
near_module_sparsity value, on whichever single GPU the orchestrator has exposed via
CUDA_VISIBLE_DEVICES. Diagnostic only: near_module_sparsity is fixed at 0.1 by CLAUDE.md's
spec, this sweeps above it purely to find where (if anywhere) learning becomes possible, per
results/cnn_modular_rnn_dqn_round2/summary.json's finding that neither pixel-CNN-DQN nor
raw-state ModularRNN learns at all at the specified 10% near-module connectivity.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.common import get_device
from models.modular_rnn import ModularRNN
from scripts.train_cartpole import train_adam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "hyperparameters.json").write_text(json.dumps(config, indent=2))

    torch.manual_seed(config["seed"])

    device = get_device()
    print(f"device={device}", flush=True)
    model = ModularRNN(
        input_size=4,
        hidden_size=config["hidden_size"],
        output_size=2,
        output_mode="all",
        near_module_sparsity=config["near_module_sparsity"],
    ).to(device)

    train_adam(
        model,
        device,
        num_updates=config["num_updates"],
        episodes_per_update=config["episodes_per_update"],
        results_path=str(results_dir / "results.json"),
        model_path=str(results_dir / "model.pt"),
        lr=config["lr"],
    )


if __name__ == "__main__":
    main()
