"""Round 2 of the CNN-ModularRNN-DQN CartPole search, split across all 8 GPUs into two arms
run simultaneously:

  GPUs 0-3 (`modular` arm): CNNModularRNNDQN with `recurrent_gain` swept at {1.4, 1.0, 0.8,
    0.5} (everything else fixed at round 1's best hyperparameters: n_step=7, tau=0.005,
    lr=1e-4). Round 1 (results/cnn_modular_rnn_dqn_search/summary.json) found every config's
    eval reward flat around 15-30/500 for the full 5000-episode budget regardless of
    n_step/tau/lr -- ModularRNN's recurrent_gain defaults to 1.4, deliberately pushing the
    recurrent weight's spectral radius past 1 (see models/common.py's
    scaled_recurrent_init_ docstring) for BPTT-trained sequence classification, which is
    untested for a Q-network whose hidden state is threaded step-by-step through online TD
    updates. This arm checks whether that expansive-dynamics prior is what's blocking
    learning.

  GPUs 4-7 (`control` arm): the *unrestricted* CNN-RNN-DQN (plain nn.RNNCell, no modular
    masking; scripts/test_cnn_rnn_dqn_cartpole.train_dqn_recurrent) at the same
    rnn_hidden_size=300 and an equivalent n_step/tau/lr spread, under the identical
    budget/seed as round 1. This isolates whether 5000 episodes is simply too few for this
    pixel-rendered architecture in general, versus the modular connectivity itself being the
    bottleneck: if the control arm also stays flat, round 1's negative result says nothing
    modular-specific; if it solves (or clearly trends upward) while the modular arm doesn't,
    the 10%-sparse near-module connectivity is the limiting factor.

Both arms write results/cnn_modular_rnn_dqn_round2/<name>/{hyperparameters.json,
results.json, train.log} (results.json updated after every episode) and a combined
results/cnn_modular_rnn_dqn_round2/summary.json once every run finishes.

Launched fully detached (nohup ... & disown) per CLAUDE.md's Requirement section so the
search survives an SSH disconnect.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "cnn_modular_rnn_dqn_round2"
SOLVED_THRESHOLD = 495.0

SHARED = dict(
    image_size=64,
    hidden_sizes=[],
    rnn_hidden_size=300,
    conv_channels=[16, 32, 32],
    frame_stack=4,
    gamma=0.99,
    eps_start=0.9,
    eps_end=0.05,
    memory_capacity=10000,
    num_episodes=5000,
    eval_every=25,
    eval_episodes=5,
    solved_mean_reward=None,
    seed=0,
    batch_size=128,
    lr=1e-4,
    lr_decay_every=None,
    lr_decay_factor=0.5,
)

MODULAR_ONLY = dict(near_module_sparsity=0.1)

CONFIGS = [
    # modular arm: recurrent_gain sweep at round 1's best hyperparameters (n_step=7, tau=0.005)
    dict(name="modular_gain1.4_baseline", kind="modular", n_step=7, tau=0.005, recurrent_gain=1.4, input_gain=1.0),
    dict(name="modular_gain1.0", kind="modular", n_step=7, tau=0.005, recurrent_gain=1.0, input_gain=1.0),
    dict(name="modular_gain0.8", kind="modular", n_step=7, tau=0.005, recurrent_gain=0.8, input_gain=1.0),
    dict(name="modular_gain0.5", kind="modular", n_step=7, tau=0.005, recurrent_gain=0.5, input_gain=1.0),
    # control arm: unrestricted CNN-RNN-DQN, same hidden size/budget/seed, mirroring round 1's
    # best-performing hyperparameter spread
    dict(name="control_n7_tau005", kind="control", n_step=7, tau=0.005),
    dict(name="control_n3_tau001", kind="control", n_step=3, tau=0.001),
    dict(name="control_n1_tau005", kind="control", n_step=1, tau=0.005),
    dict(name="control_n3_tau005", kind="control", n_step=3, tau=0.005),
]

WORKER = {
    "modular": ROOT / "scripts" / "cnn_modular_rnn_dqn_search_worker.py",
    "control": ROOT / "scripts" / "cnn_rnn_dqn_control_worker.py",
}


def launch(config: dict, gpu_index: int) -> subprocess.Popen:
    name = config["name"]
    kind = config["kind"]
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)

    extra = {k: v for k, v in config.items() if k not in ("name", "kind")}
    full_config = {**SHARED, **(MODULAR_ONLY if kind == "modular" else {}), **extra, "results_dir": str(results_dir)}
    config_path = results_dir / "config.json"
    config_path.write_text(json.dumps(full_config, indent=2))

    log_file = open(results_dir / "train.log", "w")
    worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
    print(f"launching {name} ({kind}) on GPU {gpu_index}", flush=True)
    return subprocess.Popen(
        [sys.executable, str(WORKER[kind]), "--config", str(config_path)],
        cwd=str(ROOT),
        env=worker_env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def summarize() -> list[dict]:
    summary = []
    for config in CONFIGS:
        name = config["name"]
        results_path = RESULTS_ROOT / name / "results.json"
        if not results_path.exists():
            summary.append({"name": name, "kind": config["kind"], "status": "missing"})
            continue

        history = json.loads(results_path.read_text())
        evals = [h for h in history if h["eval_mean_reward"] is not None]
        solved_evals = [h for h in evals if h["eval_mean_reward"] >= SOLVED_THRESHOLD]
        solved_at = solved_evals[0]["episode"] if solved_evals else None

        post_solve = [h for h in evals if solved_at is not None and h["episode"] >= solved_at]
        post_solve_rate = (
            sum(1 for h in post_solve if h["eval_mean_reward"] >= SOLVED_THRESHOLD) / len(post_solve)
            if post_solve
            else None
        )

        best_eval = max((h["eval_mean_reward"] for h in evals), default=None)
        # crude trend signal: mean of the last quarter of evals minus mean of the first
        # quarter, so a flat/noisy run (round 1's pattern) reads near zero while genuine
        # improvement reads clearly positive.
        trend = None
        if len(evals) >= 4:
            q = len(evals) // 4
            first_q = sum(h["eval_mean_reward"] for h in evals[:q]) / q
            last_q = sum(h["eval_mean_reward"] for h in evals[-q:]) / q
            trend = last_q - first_q

        summary.append(
            {
                "name": name,
                "kind": config["kind"],
                "solved_at_episode": solved_at,
                "post_solve_solved_rate": post_solve_rate,
                "best_eval_mean_reward": best_eval,
                "final_eval_mean_reward": evals[-1]["eval_mean_reward"] if evals else None,
                "trend_last_quarter_minus_first_quarter": trend,
                "trained_episodes": history[-1]["episode"] if history else 0,
            }
        )

    summary.sort(
        key=lambda r: (
            r.get("kind", ""),
            r.get("solved_at_episode") is None,
            -(r.get("post_solve_solved_rate") or 0.0),
            r.get("solved_at_episode") or float("inf"),
            -(r.get("best_eval_mean_reward") or float("-inf")),
        )
    )
    return summary


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    assert len(CONFIGS) <= 8, "one config per GPU"

    procs = [launch(config, i) for i, config in enumerate(CONFIGS)]
    for p in procs:
        p.wait()

    summary = summarize()
    (RESULTS_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
