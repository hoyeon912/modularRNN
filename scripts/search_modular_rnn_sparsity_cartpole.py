"""Diagnostic sweep of ModularRNN's near_module_sparsity on CartPole-v1's raw 4-dim state
(no CNN -- cheapest possible test of the recurrent core itself), one value per GPU, all run
simultaneously via scripts/modular_rnn_sparsity_search_worker.py.

CLAUDE.md fixes near_module_sparsity at 0.1 (10%); results/cnn_modular_rnn_dqn_round2's
findings show that at 0.1, neither pixel-CNN-DQN nor raw-state ModularRNN (existing prior-
session results in results/modular_rnn_cartpole and results/modular_rnn_hf/cartpole) learns
CartPole at all, while the same raw-state/Adam-REINFORCE code path with an *unrestricted*
recurrent core (SimpleRNN, BidirectionalRNN -- see results/simple_rnn_cartpole,
results/bidirectional_rnn_cartpole) solves it within 18-97 updates. A verified-clean gradient
check (every permitted block, including the 10%-sparse near-module blocks, gets a non-trivial
gradient) rules out a masking/backward-hook bug as the explanation.

This sweep is purely diagnostic, not a deployable config search: it sweeps
near_module_sparsity from 0.1 up to 1.0 (fully dense near-module blocks -- input<->output
stays hardcoded at zero regardless, per _build_hh_mask, so this isolates exactly the
near-module link density's effect) to find where -- if anywhere -- learning becomes possible
at all, i.e. how far the spec's 10% is from a value that would let a modular RNN learn a
simple control task via this training recipe.

Writes results/modular_rnn_sparsity_search/<name>/{hyperparameters.json, results.json,
train.log, model.pt} per config and a combined summary.json once every run finishes.

Launched fully detached (nohup ... & disown) per CLAUDE.md's Requirement section.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "modular_rnn_sparsity_search"
WORKER = ROOT / "scripts" / "modular_rnn_sparsity_search_worker.py"
SOLVED_THRESHOLD = 495.0

SHARED = dict(
    hidden_size=300,
    num_updates=300,
    episodes_per_update=8,
    lr=1e-3,
    seed=0,
)

SPARSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def config_name(sparsity: float) -> str:
    return f"sparsity{sparsity:.1f}"


def launch(sparsity: float, gpu_index: int) -> subprocess.Popen:
    name = config_name(sparsity)
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)

    full_config = {**SHARED, "near_module_sparsity": sparsity, "results_dir": str(results_dir)}
    config_path = results_dir / "config.json"
    config_path.write_text(json.dumps(full_config, indent=2))

    log_file = open(results_dir / "train.log", "w")
    worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
    print(f"launching {name} on GPU {gpu_index}", flush=True)
    return subprocess.Popen(
        [sys.executable, str(WORKER), "--config", str(config_path)],
        cwd=str(ROOT),
        env=worker_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def summarize() -> list[dict]:
    summary = []
    for sparsity in SPARSITIES:
        name = config_name(sparsity)
        results_path = RESULTS_ROOT / name / "results.json"
        if not results_path.exists():
            summary.append({"name": name, "near_module_sparsity": sparsity, "status": "missing"})
            continue

        history = json.loads(results_path.read_text())
        rewards = [h["reward"] for h in history]
        solved = [h for h in history if h["reward"] >= SOLVED_THRESHOLD]
        solved_at = solved[0]["update"] if solved else None

        trend = None
        if len(rewards) >= 4:
            q = len(rewards) // 4
            trend = sum(rewards[-q:]) / q - sum(rewards[:q]) / q

        summary.append(
            {
                "name": name,
                "near_module_sparsity": sparsity,
                "solved_at_update": solved_at,
                "best_reward": max(rewards) if rewards else None,
                "final_reward": rewards[-1] if rewards else None,
                "trend_last_quarter_minus_first_quarter": trend,
                "trained_updates": history[-1]["update"] if history else 0,
            }
        )

    summary.sort(key=lambda r: (r.get("solved_at_update") is None, r.get("solved_at_update") or float("inf"), r["near_module_sparsity"]))
    return summary


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    assert len(SPARSITIES) <= 8, "one config per GPU"

    procs = [launch(sparsity, i) for i, sparsity in enumerate(SPARSITIES)]
    for p in procs:
        p.wait()

    summary = summarize()
    (RESULTS_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
