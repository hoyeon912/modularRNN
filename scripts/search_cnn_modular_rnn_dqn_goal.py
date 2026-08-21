"""Goal: find an actual working (i.e. it learns/solves) hyperparameter combination for
CNN-ModularRNN-DQN on CartPole-v1, per the project's CLAUDE.md RL bar. Splits all 8 GPUs into
two arms run simultaneously, each sweeping rnn_hidden_size (divisible by 3, per ModularRNN's
constraint) at {150, 300, 600, 900} -- "resize the number of RNN units" as a built-in lever
rather than a sequential fallback, since parallel GPUs make trying it upfront free:

  GPUs 0-3 (`td` arm): scripts/test_cnn_modular_rnn_dqn_cartpole.train_dqn_modular
    (Double-DQN/TD, CNNModularRNNDQN) at round 1/2's best hyperparameters (n_step=7,
    tau=0.005, lr=1e-4, recurrent_gain=1.4 default -- round 2 ruled out gain as the cause).
    Rounds 1-2 (results/cnn_modular_rnn_dqn_search, results/cnn_modular_rnn_dqn_round2) found
    every TD config flat/non-learning at rnn_hidden_size=300 across 5000 episodes regardless
    of n_step/tau/lr/lr-decay/gain; this arm checks whether hidden size specifically was the
    missing lever.

  GPUs 4-7 (`reinforce` arm): scripts/test_cnn_modular_rnn_reinforce_cartpole.
    train_reinforce_modular (REINFORCE/policy-gradient, CNNModularRNN). results/
    modular_rnn_sparsity_search showed REINFORCE clearly learns CartPole from *raw* state at
    every near_module_sparsity tested including CLAUDE.md's spec 0.1 -- steadily improving,
    still climbing at the 300-update cutoff. This arm checks whether that also holds once a
    CNN encoder is in front of the same restricted ModularRNN core, which is what CLAUDE.md's
    CNN-ModularRNN-DQN goal actually needs.

Both arms use solved_mean_reward=495 (early-stops the moment a config solves, unlike round 1
which deliberately disabled it to study stability -- here the goal is just finding *a* working
combination, so stopping early saves GPU time once one is found).

Writes results/cnn_modular_rnn_dqn_goal_search/<name>/{hyperparameters.json, results.json,
train.log} per config (results.json updated after every episode/update) and a combined
summary.json ranking once every run finishes.

Launched fully detached (nohup ... & disown) per CLAUDE.md's Requirement section.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "cnn_modular_rnn_dqn_goal_search"
SOLVED_THRESHOLD = 495.0

HIDDEN_SIZES = [150, 300, 600, 900]

TD_SHARED = dict(
    image_size=64,
    hidden_sizes=[],
    near_module_sparsity=0.1,
    recurrent_gain=1.4,
    input_gain=1.0,
    conv_channels=[16, 32, 32],
    frame_stack=4,
    gamma=0.99,
    eps_start=0.9,
    eps_end=0.05,
    eps_decay=1000,
    tau=0.005,
    lr=1e-4,
    batch_size=128,
    memory_capacity=10000,
    num_episodes=5000,
    eval_every=25,
    eval_episodes=5,
    solved_mean_reward=SOLVED_THRESHOLD,
    lr_decay_every=None,
    lr_decay_factor=0.5,
    n_step=7,
    seed=0,
)

REINFORCE_SHARED = dict(
    image_size=64,
    near_module_sparsity=0.1,
    cnn_feature_dim=32,
    frame_stack=4,
    gamma=0.99,
    lr=1e-3,
    entropy_coef=0.01,
    num_updates=500,
    episodes_per_update=8,
    eval_every=25,
    eval_episodes=5,
    solved_mean_reward=SOLVED_THRESHOLD,
    seed=0,
)

CONFIGS = [
    *[dict(name=f"td_h{h}", kind="td", rnn_hidden_size=h) for h in HIDDEN_SIZES],
    *[dict(name=f"reinforce_h{h}", kind="reinforce", rnn_hidden_size=h) for h in HIDDEN_SIZES],
]

WORKER = {
    "td": ROOT / "scripts" / "cnn_modular_rnn_dqn_search_worker.py",
    "reinforce": ROOT / "scripts" / "cnn_modular_rnn_reinforce_search_worker.py",
}
SHARED_BY_KIND = {"td": TD_SHARED, "reinforce": REINFORCE_SHARED}


def launch(config: dict, gpu_index: int) -> subprocess.Popen:
    name = config["name"]
    kind = config["kind"]
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)

    extra = {k: v for k, v in config.items() if k not in ("name", "kind")}
    full_config = {**SHARED_BY_KIND[kind], **extra, "results_dir": str(results_dir)}
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
        evals = [h for h in history if h.get("eval_mean_reward") is not None]
        solved_evals = [h for h in evals if h["eval_mean_reward"] >= SOLVED_THRESHOLD]
        step_key = "episode" if config["kind"] == "td" else "update"
        solved_at = solved_evals[0][step_key] if solved_evals else None
        best_eval = max((h["eval_mean_reward"] for h in evals), default=None)

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
                "rnn_hidden_size": config["rnn_hidden_size"],
                "solved": solved_at is not None,
                f"solved_at_{step_key}": solved_at,
                "best_eval_mean_reward": best_eval,
                "final_eval_mean_reward": evals[-1]["eval_mean_reward"] if evals else None,
                "trend_last_quarter_minus_first_quarter": trend,
                "trained_steps": history[-1].get(step_key) if history else 0,
            }
        )

    summary.sort(
        key=lambda r: (
            not r.get("solved", False),
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
