"""Try 3 of the CNN-ModularRNN-DQN goal search (user-directed follow-up after try 1): does a
CNN encoder pretrained to predict cart/pole state from pixels (scripts/
pretrain_cnn_encoder_cartpole.py) help REINFORCE learn faster/more stably than random init?

results/cnn_modular_rnn_reinforce_longbudget's try-1 findings (8000 updates, random-init
encoder): rnn_hidden_size=600 broke out of a flat ~25 reward plateau around update ~3500,
reaching up to ~320/500, but stayed noisy (oscillating 80-250) afterward rather than
converging; rnn_hidden_size=300 showed a smaller, similarly-late, similarly-noisy breakout.
This sweep reruns both hidden sizes at the same 8000-update budget, each with the pretrained
encoder either frozen (isolates whether solved perception alone helps) or fine-tuned
(warm-started but still REINFORCE-trainable), for a direct 4-way comparison against try-1's
random-init numbers at the same hidden sizes and budget.

Writes results/cnn_modular_rnn_reinforce_pretrained/<name>/{hyperparameters.json,
results.json, train.log} per config and a combined summary.json once every run finishes.

Launched fully detached (nohup ... & disown) per CLAUDE.md's Requirement section.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "results" / "cnn_modular_rnn_reinforce_pretrained"
WORKER = ROOT / "scripts" / "cnn_modular_rnn_reinforce_search_worker.py"
ENCODER_PATH = ROOT / "results" / "cnn_modular_rnn_pretrained_encoder" / "encoder.pt"
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
    pretrained_encoder_path=str(ENCODER_PATH),
)

CONFIGS = [
    dict(name="pretrained_frozen_h300", rnn_hidden_size=300, freeze_encoder=True),
    dict(name="pretrained_finetune_h300", rnn_hidden_size=300, freeze_encoder=False),
    dict(name="pretrained_frozen_h600", rnn_hidden_size=600, freeze_encoder=True),
    dict(name="pretrained_finetune_h600", rnn_hidden_size=600, freeze_encoder=False),
]


def launch(config: dict, gpu_index: int) -> subprocess.Popen:
    name = config["name"]
    results_dir = RESULTS_ROOT / name
    results_dir.mkdir(parents=True, exist_ok=True)

    extra = {k: v for k, v in config.items() if k != "name"}
    full_config = {**SHARED, **extra, "results_dir": str(results_dir)}
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
    for config in CONFIGS:
        name = config["name"]
        results_path = RESULTS_ROOT / name / "results.json"
        if not results_path.exists():
            summary.append({"name": name, "status": "missing"})
            continue

        history = json.loads(results_path.read_text())
        evals = [h for h in history if h.get("eval_mean_reward") is not None]
        solved_evals = [h for h in evals if h["eval_mean_reward"] >= SOLVED_THRESHOLD]
        solved_at = solved_evals[0]["update"] if solved_evals else None
        best_eval = max((h["eval_mean_reward"] for h in evals), default=None)

        trend = None
        first_breakout_update = None
        if len(evals) >= 4:
            q = len(evals) // 4
            first_q = sum(h["eval_mean_reward"] for h in evals[:q]) / q
            last_q = sum(h["eval_mean_reward"] for h in evals[-q:]) / q
            trend = last_q - first_q
            # first eval update where reward clearly leaves the noise floor (>100), a rough
            # proxy for "when did it break out of the flat plateau" comparable across configs
            breakout = next((h for h in evals if h["eval_mean_reward"] > 100), None)
            first_breakout_update = breakout["update"] if breakout else None

        summary.append(
            {
                "name": name,
                "rnn_hidden_size": config["rnn_hidden_size"],
                "freeze_encoder": config["freeze_encoder"],
                "solved": solved_at is not None,
                "solved_at_update": solved_at,
                "best_eval_mean_reward": best_eval,
                "final_eval_mean_reward": evals[-1]["eval_mean_reward"] if evals else None,
                "trend_last_quarter_minus_first_quarter": trend,
                "first_breakout_update_over_100": first_breakout_update,
                "trained_updates": history[-1]["update"] if history else 0,
            }
        )

    summary.sort(key=lambda r: (not r.get("solved", False), -(r.get("best_eval_mean_reward") or float("-inf"))))
    return summary


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    assert ENCODER_PATH.exists(), f"pretrained encoder not found at {ENCODER_PATH}"
    assert len(CONFIGS) <= 8, "one config per GPU"

    procs = [launch(config, i) for i, config in enumerate(CONFIGS)]
    for p in procs:
        p.wait()

    summary = summarize()
    (RESULTS_ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
