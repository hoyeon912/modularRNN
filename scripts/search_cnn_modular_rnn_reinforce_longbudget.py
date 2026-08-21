"""Follow-up to scripts/search_cnn_modular_rnn_dqn_goal.py's REINFORCE arm, per user request
("try giving REINFORCE+CNN a much longer budget"): the goal search's REINFORCE+CNN configs
(500 updates, all 4 hidden sizes) stayed completely flat around reward 22-25 the entire run --
in sharp contrast to results/modular_rnn_sparsity_search's *raw-state* ModularRNN, which under
the same REINFORCE recipe climbed steadily from ~22 to ~76+ over just 300 updates and was
still rising. Timing from that run (~0.23s/update on a single GPU) makes a much larger budget
cheap, so this reruns the same 4 hidden sizes at num_updates=8000 (16x the goal search's
budget) to see whether the CNN+ModularRNN combination eventually breaks out of the flat
plateau given enough updates, or whether it's stuck regardless of budget (which would point
toward a different fix, e.g. separate learning rates or a pretrained encoder).

Writes results/cnn_modular_rnn_reinforce_longbudget/<name>/{hyperparameters.json,
results.json, train.log} per config and a combined summary.json once every run finishes.

Launched fully detached (nohup ... & disown) per CLAUDE.md's Requirement section.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "cnn_modular_rnn_reinforce_longbudget"
WORKER = ROOT / "scripts" / "cnn_modular_rnn_reinforce_search_worker.py"
SOLVED_THRESHOLD = 495.0

SHARED = dict(
    image_size=64,
    near_module_sparsity=0.1,
    cnn_feature_dim=32,
    frame_stack=4,
    gamma=0.99,
    lr=1e-3,
    entropy_coef=0.01,
    num_updates=8000,
    episodes_per_update=8,
    eval_every=50,
    eval_episodes=5,
    solved_mean_reward=SOLVED_THRESHOLD,
    seed=0,
)

HIDDEN_SIZES = [150, 300, 600, 900]


def config_name(h: int) -> str:
    return f"reinforce_longbudget_h{h}"


def launch(hidden_size: int, gpu_index: int) -> subprocess.Popen:
    name = config_name(hidden_size)
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)

    full_config = {**SHARED, "rnn_hidden_size": hidden_size, "results_dir": str(results_dir)}
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
    for hidden_size in HIDDEN_SIZES:
        name = config_name(hidden_size)
        results_path = RESULTS_ROOT / name / "results.json"
        if not results_path.exists():
            summary.append({"name": name, "rnn_hidden_size": hidden_size, "status": "missing"})
            continue

        history = json.loads(results_path.read_text())
        evals = [h for h in history if h.get("eval_mean_reward") is not None]
        solved_evals = [h for h in evals if h["eval_mean_reward"] >= SOLVED_THRESHOLD]
        solved_at = solved_evals[0]["update"] if solved_evals else None
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
                "rnn_hidden_size": hidden_size,
                "solved": solved_at is not None,
                "solved_at_update": solved_at,
                "best_eval_mean_reward": best_eval,
                "final_eval_mean_reward": evals[-1]["eval_mean_reward"] if evals else None,
                "trend_last_quarter_minus_first_quarter": trend,
                "trained_updates": history[-1]["update"] if history else 0,
            }
        )

    summary.sort(key=lambda r: (not r.get("solved", False), -(r.get("best_eval_mean_reward") or float("-inf"))))
    return summary


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    assert len(HIDDEN_SIZES) <= 8, "one config per GPU"

    procs = [launch(h, i) for i, h in enumerate(HIDDEN_SIZES)]
    for p in procs:
        p.wait()

    summary = summarize()
    (RESULTS_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
