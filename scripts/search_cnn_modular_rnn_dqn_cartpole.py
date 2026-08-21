"""Hyperparameter search for CNN-ModularRNN-DQN (scripts/test_cnn_modular_rnn_dqn_cartpole.py)
on CartPole-v1: one config pinned to each available GPU via CUDA_VISIBLE_DEVICES, all run in
parallel as separate processes (scripts/cnn_modular_rnn_dqn_search_worker.py). Picks the
combination that solves *fastest* (fewest episodes to a 5-episode greedy eval mean reward
above SOLVED_THRESHOLD -- CLAUDE.md's RL bar is "gain maximum reward ... before 500 episodes
for 5 continuous episodes", and eval_episodes=5 makes each eval check exactly that 5-episode
window) while staying *stable* (it keeps solving eval windows for the rest of training,
instead of a one-off spike). `solved_mean_reward` is left unset (None) so training does NOT
early-stop the moment it first solves -- that's the only way to observe whether a config
stays solved afterward, which is exactly the stability signal this search is for.

num_episodes=5000 (not the 500-episode acceptance bar itself): an earlier 500-episode run of
this same grid found every config still near-random (eval reward 9-25 out of 500) at episode
500, matching this codebase's prior CNN-RNN-DQN searches (results/cnn_rnn_dqn_search), which
needed budgets up to 100000 episodes for the pixel-rendered CNN+recurrent architecture to
solve CartPole -- 500 episodes is the final acceptance bar for a *chosen* config, not enough
runway to differentiate hyperparameters during search. If no config crosses SOLVED_THRESHOLD
within the budget, summarize() falls back to ranking by best eval reward reached, so the
search still surfaces a "least bad" combination.

rnn_hidden_size is fixed at 300 (CLAUDE.md's "concrete test" hidden-unit budget, divisible by
3) and near_module_sparsity at 0.1 (the fixed 10% near-module connectivity spec) for every
config -- those are architecture requirements, not something to search over. Only training
hyperparameters (lr, tau, n_step, eps_decay, lr decay) vary across configs.

Writes results/cnn_modular_rnn_dqn_search/<name>/{hyperparameters.json, results.json,
train.log} per config (results.json updated after every episode by train_dqn_modular) and a
results/cnn_modular_rnn_dqn_search/summary.json ranking once every config finishes.

Meant to be launched fully detached (nohup ... & disown) per CLAUDE.md's Requirement section,
so the search survives an SSH disconnect.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "cnn_modular_rnn_dqn_search"
SOLVED_THRESHOLD = 495.0

COMMON = dict(
    image_size=64,
    hidden_sizes=[],
    rnn_hidden_size=300,
    near_module_sparsity=0.1,
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
)

CONFIGS = [
    dict(name="baseline_n3_tau005_lr1e-4", lr=1e-4, n_step=3, tau=0.005, eps_decay=1000, batch_size=128, lr_decay_every=None, lr_decay_factor=0.5),
    dict(name="n1_tau005_lr1e-4", lr=1e-4, n_step=1, tau=0.005, eps_decay=1000, batch_size=128, lr_decay_every=None, lr_decay_factor=0.5),
    dict(name="n5_tau005_lr1e-4", lr=1e-4, n_step=5, tau=0.005, eps_decay=1000, batch_size=128, lr_decay_every=None, lr_decay_factor=0.5),
    dict(name="n7_tau005_lr1e-4", lr=1e-4, n_step=7, tau=0.005, eps_decay=1000, batch_size=128, lr_decay_every=None, lr_decay_factor=0.5),
    dict(name="n3_tau005_lr5e-5", lr=5e-5, n_step=3, tau=0.005, eps_decay=1000, batch_size=128, lr_decay_every=None, lr_decay_factor=0.5),
    dict(name="n3_tau001_lr1e-4", lr=1e-4, n_step=3, tau=0.001, eps_decay=1000, batch_size=128, lr_decay_every=None, lr_decay_factor=0.5),
    dict(name="n3_tau01_lr1e-4", lr=1e-4, n_step=3, tau=0.01, eps_decay=1000, batch_size=128, lr_decay_every=None, lr_decay_factor=0.5),
    dict(name="n3_tau005_lr1e-4_decay150", lr=1e-4, n_step=3, tau=0.005, eps_decay=1000, batch_size=128, lr_decay_every=150, lr_decay_factor=0.5),
]


def launch(config: dict, gpu_index: int) -> subprocess.Popen:
    name = config["name"]
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)

    full_config = {**COMMON, **{k: v for k, v in config.items() if k != "name"}, "results_dir": str(results_dir)}
    config_path = results_dir / "config.json"
    config_path.write_text(json.dumps(full_config, indent=2))

    log_file = open(results_dir / "train.log", "w")
    worker_env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_index))
    print(f"launching {name} on GPU {gpu_index}", flush=True)
    return subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "cnn_modular_rnn_dqn_search_worker.py"), "--config", str(config_path)],
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
            summary.append({"name": name, "status": "missing"})
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
        if post_solve:
            mean_ = sum(h["eval_mean_reward"] for h in post_solve) / len(post_solve)
            post_solve_std = (sum((h["eval_mean_reward"] - mean_) ** 2 for h in post_solve) / len(post_solve)) ** 0.5
        else:
            post_solve_std = None

        best_eval = max((h["eval_mean_reward"] for h in evals), default=None)

        summary.append(
            {
                "name": name,
                "solved_at_episode": solved_at,
                "post_solve_solved_rate": post_solve_rate,
                "post_solve_eval_std": post_solve_std,
                "best_eval_mean_reward": best_eval,
                "final_eval_mean_reward": evals[-1]["eval_mean_reward"] if evals else None,
                "trained_episodes": history[-1]["episode"] if history else 0,
            }
        )

    # Primary ranking: solved fastest, then stayed solved most consistently. Configs that
    # never crossed SOLVED_THRESHOLD sort after every config that did (regardless of how high
    # their best eval reward got) and are ranked among themselves by that best reward, so the
    # summary still surfaces a "least bad" combination if nothing formally solved.
    summary.sort(
        key=lambda r: (
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
