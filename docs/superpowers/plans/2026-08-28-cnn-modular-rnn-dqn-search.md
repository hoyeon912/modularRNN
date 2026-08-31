# CNN-ModularRNN-DQN CartPole Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, deploy, and start a recoverable two-server staged search for the highest-ranked CNN-ModularRNN-DQN CartPole configuration.

**Architecture:** Keep deterministic training/checkpointing in the existing CNN-ModularRNN-DQN training module, put scoring and staged configuration generation in focused pure-Python modules, and run one queue scheduler in tmux on each GPU server. A workstation coordinator assigns immutable queue snapshots to DL3 and DL6, reconstructs state from result artifacts, and advances the search only after each stage is complete.

**Tech Stack:** Python 3, PyTorch, Gymnasium CartPole-v1, pytest, JSON, SSH, tmux, systemd user lingering

**Spec:** `docs/superpowers/specs/2026-08-28-cnn-modular-rnn-dqn-search-design.md`

## Global Constraints

- Search training seeds are exactly `0, 1, 2`.
- Initial validation training seeds are exactly `0, 1, 2, 3, 4, 5, 6, 7`.
- Search evaluation seeds are `2_000_000..2_000_009`; validation evaluation seeds are `2_000_000..2_000_019`.
- Evaluate every 25 training episodes; an evaluation succeeds at mean reward `>= 495`; a seed solves at five consecutive successful evaluations.
- Every scored run executes all 10,000 training episodes; first-solve speed and the 500-episode threshold do not affect ranking.
- Rank by solved-seed count, mean maximum success streak, mean total successful evaluations, then mean final-20-evaluation reward.
- Run one GPU worker on each of NeuRLab-DL3 and NeuRLab-DL6 inside tmux.
- A retryable worker failure is retried at most two times.
- Do not launch a long run until unit tests, GPU smoke tests, checkpoint resume, matching environments, lingering, tmux disconnect survival, free disk, and GPU ownership are verified.
- Do not terminate unknown GPU processes or overwrite pre-existing remote results.

## File Structure

- Modify `scripts/test_cnn_modular_rnn_dqn_cartpole.py`: deterministic reset/evaluation seeds, resumable training state, atomic artifacts, and best checkpoints.
- Modify `scripts/cnn_modular_rnn_dqn_search_worker.py`: validate a run config, capture environment metadata, resume safely, and report terminal state.
- Create `scripts/dqn_search_metrics.py`: per-seed metrics and configuration ranking only.
- Create `scripts/dqn_search_manifest.py`: canonical stage definitions, stable IDs, validation manifests, and queue partitioning only.
- Create `scripts/cnn_modular_rnn_dqn_scheduler.py`: execute one immutable server queue, retry workers, and atomically record queue state.
- Create `scripts/cnn_modular_rnn_dqn_coordinator.py`: inspect both servers, assign pending runs, aggregate completed stages, and emit the next-stage manifest.
- Create `tests/test_dqn_search_metrics.py`: pure scoring tests.
- Create `tests/test_dqn_search_manifest.py`: search-space and stable-ID tests.
- Create `tests/test_cnn_modular_rnn_dqn_resume.py`: deterministic evaluation, atomic writes, and interrupted/resumed training tests.
- Create `tests/test_cnn_modular_rnn_dqn_scheduler.py`: state-machine, retry, idempotency, and coordinator reconstruction tests.
- Create `requirements-search.txt`: exact direct dependency versions used by the two search environments.

---

### Task 1: Pure Success Metrics and Ranking

**Files:**
- Create: `scripts/dqn_search_metrics.py`
- Create: `tests/test_dqn_search_metrics.py`

**Interfaces:**
- Consumes: result history entries shaped as `{"episode": int, "eval_mean_reward": float | None}`.
- Produces: `summarize_seed(history: list[dict], threshold: float = 495.0, final_window: int = 20) -> dict` and `rank_configs(rows: list[dict]) -> list[dict]`.

- [ ] **Step 1: Write failing metric tests**

```python
from scripts.dqn_search_metrics import rank_configs, summarize_seed


def test_summarize_seed_tracks_longest_streak_and_total_successes():
    rewards = [500, 496, 494, 500, 500, 500, 500, 500]
    history = [
        {"episode": 25 * (i + 1), "eval_mean_reward": reward}
        for i, reward in enumerate(rewards)
    ]
    summary = summarize_seed(history)
    assert summary["max_consecutive_successes"] == 5
    assert summary["total_successful_evals"] == 7
    assert summary["solved"] is True
    assert summary["first_solved_episode"] == 200


def test_rank_configs_uses_approved_lexicographic_order():
    rows = [
        {"config_id": "more_streak", "seed_summaries": [
            {"solved": True, "max_consecutive_successes": 8,
             "total_successful_evals": 9, "final_window_mean_reward": 496.0}
        ]},
        {"config_id": "more_total_only", "seed_summaries": [
            {"solved": True, "max_consecutive_successes": 7,
             "total_successful_evals": 20, "final_window_mean_reward": 500.0}
        ]},
    ]
    assert [row["config_id"] for row in rank_configs(rows)] == [
        "more_streak", "more_total_only"
    ]
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `.venv/bin/pytest tests/test_dqn_search_metrics.py -v`

Expected: FAIL because `scripts.dqn_search_metrics` does not exist.

- [ ] **Step 3: Implement the metric functions**

```python
def summarize_seed(history, threshold=495.0, final_window=20):
    evaluations = [row for row in history if row.get("eval_mean_reward") is not None]
    streak = maximum = total = 0
    first_solved_episode = None
    for row in evaluations:
        if row["eval_mean_reward"] >= threshold:
            total += 1
            streak += 1
            maximum = max(maximum, streak)
            if streak == 5 and first_solved_episode is None:
                first_solved_episode = row["episode"]
        else:
            streak = 0
    tail = evaluations[-final_window:]
    return {
        "solved": maximum >= 5,
        "max_consecutive_successes": maximum,
        "total_successful_evals": total,
        "final_window_mean_reward": (
            sum(row["eval_mean_reward"] for row in tail) / len(tail)
            if tail else None
        ),
        "first_solved_episode": first_solved_episode,
    }
```

Implement `rank_configs` by adding aggregate fields and sorting descending on the four approved fields. Reject missing seed summaries and `None` final-window values with `ValueError`; never treat incomplete runs as zero scores.

- [ ] **Step 4: Add edge-case tests and run them**

Add tests for no evaluations, a four-success streak, two separated streaks, exact threshold 495, incomplete seed summaries, and tie preservation by `config_id` as the final deterministic display key.

Run: `.venv/bin/pytest tests/test_dqn_search_metrics.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the metric unit**

```bash
git add scripts/dqn_search_metrics.py tests/test_dqn_search_metrics.py
git commit -m "feat: add modular DQN search ranking metrics"
```

### Task 2: Canonical Staged Manifests

**Files:**
- Create: `scripts/dqn_search_manifest.py`
- Create: `tests/test_dqn_search_manifest.py`

**Interfaces:**
- Consumes: `build_stage(stage: str, inherited: dict)`, where inherited contains the preceding winner's hyperparameters.
- Produces: `stable_config_id(stage: str, params: dict) -> str`, `expand_runs(configs: list[dict], seeds: tuple[int, ...]) -> list[dict]`, `partition_runs(runs: list[dict], hosts: tuple[str, ...]) -> dict[str, list[dict]]`, and `build_validation(winner: dict) -> list[dict]`.

- [ ] **Step 1: Write failing manifest tests**

```python
from scripts.dqn_search_manifest import BASELINE, build_stage, expand_runs


def test_connectivity_stage_matches_approved_values():
    configs = build_stage("connectivity", BASELINE)
    assert [c["near_module_sparsity"] for c in configs] == [
        0.1, 0.25, 0.5, 0.75, 1.0
    ]
    assert all(c["rnn_hidden_size"] == 150 for c in configs)


def test_expand_runs_uses_three_search_seeds_without_duplicates():
    runs = expand_runs(build_stage("baseline", BASELINE), (0, 1, 2))
    assert [run["training_seed"] for run in runs] == [0, 1, 2]
    assert len({run["run_id"] for run in runs}) == 3
```

- [ ] **Step 2: Run tests and verify import failure**

Run: `.venv/bin/pytest tests/test_dqn_search_manifest.py -v`

Expected: FAIL because the manifest module does not exist.

- [ ] **Step 3: Implement immutable defaults and stable IDs**

Define `BASELINE` with all approved model and DQN values plus `num_episodes=10000`, `eval_every=25`, `eval_seeds=range(2_000_000, 2_000_010)`, `solved_mean_reward=None`, and `checkpoint_every=250`. Generate IDs from canonical sorted JSON and a 12-character SHA-256 suffix; include a readable stage prefix.

Implement exact stage values from the spec. For one-variable-at-a-time substages, require the caller to pass the preceding winner and vary only the named field. `build_validation` must set training seeds `0..7` and evaluation seeds `2_000_000..2_000_019`.

- [ ] **Step 4: Test every stage, validation, and partition stability**

Assert exact values and counts for baseline, connectivity, hidden size, recurrent gain, input gain, learning rate, n-step, and tau. Assert stable IDs do not depend on dict insertion order. Assert round-robin partitioning assigns each run exactly once and produces the same output on repeated calls.

Run: `.venv/bin/pytest tests/test_dqn_search_manifest.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the manifest unit**

```bash
git add scripts/dqn_search_manifest.py tests/test_dqn_search_manifest.py
git commit -m "feat: define staged modular DQN search manifests"
```

### Task 3: Atomic Artifacts and Serializable Training State

**Files:**
- Modify: `scripts/test_cnn_modular_rnn_dqn_cartpole.py`
- Create: `tests/test_cnn_modular_rnn_dqn_resume.py`

**Interfaces:**
- Produces: `atomic_write_json(path: Path, value: object) -> None`, `atomic_torch_save(path: Path, value: object) -> None`, `ModularReplayMemory.state_dict() -> dict`, and `ModularReplayMemory.load_state_dict(state: dict, device: torch.device) -> None`.
- Consumed later by the training loop and worker.

- [ ] **Step 1: Write failing atomic-write and replay round-trip tests**

```python
def test_atomic_write_json_replaces_complete_document(tmp_path):
    path = tmp_path / "results.json"
    atomic_write_json(path, [{"episode": 1}])
    assert json.loads(path.read_text()) == [{"episode": 1}]
    assert not list(tmp_path.glob("*.tmp"))


def test_replay_memory_state_round_trip_to_cpu():
    memory = ModularReplayMemory(3)
    tensor = torch.ones(1, 2)
    memory.push(tensor, tensor, torch.tensor([[0]]), tensor, tensor, tensor[:, 0])
    restored = ModularReplayMemory(3)
    restored.load_state_dict(memory.state_dict(), torch.device("cpu"))
    assert len(restored) == 1
    assert torch.equal(restored.memory[0].state, tensor)
```

- [ ] **Step 2: Run the focused tests and verify failures**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py -v`

Expected: FAIL because the new functions and methods do not exist.

- [ ] **Step 3: Implement atomic replacement and replay serialization**

Write temporary files in the target directory, flush and `os.fsync`, then call `os.replace`. For torch artifacts, serialize a recursively CPU-converted object so checkpoints do not retain a source GPU device. `load_state_dict` reconstructs each `ModularTransition` and moves tensors to the requested device.

- [ ] **Step 4: Test interrupted writes and device-independent reload**

Monkeypatch `os.replace` to raise and assert the original artifact remains readable. Assert capacity and transition order survive replay round-trip.

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py -v`

Expected: PASS.

- [ ] **Step 5: Commit atomic artifacts**

```bash
git add scripts/test_cnn_modular_rnn_dqn_cartpole.py tests/test_cnn_modular_rnn_dqn_resume.py
git commit -m "feat: add atomic modular DQN training artifacts"
```

### Task 4: Deterministic Evaluation and Episode Seeding

**Files:**
- Modify: `scripts/test_cnn_modular_rnn_dqn_cartpole.py`
- Modify: `tests/test_cnn_modular_rnn_dqn_resume.py`

**Interfaces:**
- Changes `evaluate_greedy_modular(..., episode_seeds: list[int] | tuple[int, ...] | None = None) -> list[int]`.
- Adds `training_episode_seed(training_seed: int, episode_index: int) -> int`.
- Extends `train_dqn_modular(..., training_seed: int = 0, evaluation_seeds: tuple[int, ...] | None = None)`.

- [ ] **Step 1: Write failing deterministic-seed tests**

Use a fixed tiny network and monkeypatch `gym.make` with a recording environment. Assert evaluation resets receive the exact supplied seeds, and assert training episode 0/1 for seed 2 receives `1_040_000` and `1_040_001`.

- [ ] **Step 2: Run tests and verify the missing arguments**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py -k 'seed or evaluation' -v`

Expected: FAIL with unexpected keyword or missing helper errors.

- [ ] **Step 3: Pass explicit seeds to every reset**

Use `env.reset(seed=seed)` in greedy evaluation and `env.reset(seed=training_episode_seed(training_seed, ep))` in training. When `episode_seeds` is `None`, retain the legacy `num_episodes` behavior for existing tests; when supplied, its length defines the evaluation episode count.

- [ ] **Step 4: Run modular DQN tests**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py scripts/test_cnn_modular_rnn_dqn_cartpole.py -v`

Expected: PASS.

- [ ] **Step 5: Commit deterministic resets**

```bash
git add scripts/test_cnn_modular_rnn_dqn_cartpole.py tests/test_cnn_modular_rnn_dqn_resume.py
git commit -m "feat: make modular DQN evaluation deterministic"
```

### Task 5: Complete Checkpoint and Resume

**Files:**
- Modify: `scripts/test_cnn_modular_rnn_dqn_cartpole.py`
- Modify: `tests/test_cnn_modular_rnn_dqn_resume.py`

**Interfaces:**
- Extends `train_dqn_modular` with `checkpoint_path: str | None`, `resume_from: str | None`, `best_path: str | None`, `checkpoint_every: int = 250`, and test-only `stop_after_episode: int | None = None`.
- Checkpoint schema version is integer `1` and includes all state listed in the spec.

- [ ] **Step 1: Write a failing uninterrupted-versus-resumed test**

Run a tiny CPU configuration for four episodes. Run the same seed for two episodes with `stop_after_episode=2`, then resume to episode four. Assert episode numbers, rewards, epsilon/global step, model parameters, optimizer state, replay length, and history equal the uninterrupted run.

- [ ] **Step 2: Run the resume test and verify failure**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py -k resume -v`

Expected: FAIL because checkpoint arguments are absent.

- [ ] **Step 3: Implement versioned complete checkpoints**

At an episode boundary save schema version, config fingerprint, next episode index, policy/target/optimizer states, replay state, steps done, histories, best evaluation value, Python RNG state, PyTorch CPU RNG state, and CUDA RNG states when available. Validate schema and config fingerprint before restore. Save `best_path` whenever evaluation mean strictly improves. Never use `solved_mean_reward` for these 10,000-episode runs.

- [ ] **Step 4: Add rejection and rotation tests**

Test invalid schema, mismatched config fingerprint, corrupt checkpoint, best-checkpoint update, and latest-checkpoint replacement. Measure a replay checkpoint in the smoke configuration and log size/write seconds.

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py scripts/test_cnn_modular_rnn_dqn_cartpole.py -v`

Expected: PASS.

- [ ] **Step 5: Commit resumable training**

```bash
git add scripts/test_cnn_modular_rnn_dqn_cartpole.py tests/test_cnn_modular_rnn_dqn_resume.py
git commit -m "feat: resume modular DQN training from checkpoints"
```

### Task 6: Harden the Single-Run Worker

**Files:**
- Modify: `scripts/cnn_modular_rnn_dqn_search_worker.py`
- Modify: `tests/test_cnn_modular_rnn_dqn_resume.py`

**Interfaces:**
- CLI remains `python scripts/cnn_modular_rnn_dqn_search_worker.py --config PATH`.
- Produces `config.json`, `environment.json`, `results.json`, `summary.json`, `latest.pt`, `best.pt`, and `status.json` in `results_dir`.

- [ ] **Step 1: Write failing worker validation tests**

Call `main` through a subprocess with a two-episode CPU config. Assert unknown keys fail before training, config `results_dir` matches the CLI artifact directory, terminal status is `completed`, environment metadata contains required versions, and rerunning resumes rather than truncates.

- [ ] **Step 2: Run tests and observe missing artifacts**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py -k worker -v`

Expected: FAIL because metadata, summaries, and status artifacts are absent.

- [ ] **Step 3: Implement strict config parsing and terminal reporting**

Set Python and PyTorch seeds, enable deterministic algorithms where supported, record deterministic warnings, pass explicit evaluation seeds and checkpoint paths, summarize with `summarize_seed`, and atomically write `status.json`. On exceptions, write `{"state": "failed", "error_type": ..., "message": ...}` and re-raise for the scheduler.

- [ ] **Step 4: Run focused and regression tests**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_resume.py scripts/test_cnn_modular_rnn_dqn_cartpole.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the worker**

```bash
git add scripts/cnn_modular_rnn_dqn_search_worker.py tests/test_cnn_modular_rnn_dqn_resume.py
git commit -m "feat: harden modular DQN search worker"
```

### Task 7: Server Queue Scheduler

**Files:**
- Create: `scripts/cnn_modular_rnn_dqn_scheduler.py`
- Create: `tests/test_cnn_modular_rnn_dqn_scheduler.py`

**Interfaces:**
- CLI: `python scripts/cnn_modular_rnn_dqn_scheduler.py --queue PATH --results-root PATH --gpu 0`.
- Produces atomic queue state with `pending`, `running`, `completed`, `retryable_failure`, or `failed` per run and at most two retries.

- [ ] **Step 1: Write failing scheduler state tests**

Use a fake worker command that exits 0, exits 1 twice then succeeds, and always exits 1. Assert exact state transitions, attempt counts `1`, `3`, and `3`, one-at-a-time execution, and no rerun of an already completed result directory.

- [ ] **Step 2: Run tests and verify import failure**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_scheduler.py -v`

Expected: FAIL because the scheduler does not exist.

- [ ] **Step 3: Implement the scheduler**

Validate that queue entries have unique run IDs and paths below `results_root`. Set `CUDA_VISIBLE_DEVICES=0`, stream each worker's stdout/stderr to `train.log`, atomically persist state before and after every attempt, and use existing valid `completed` status artifacts as the source of idempotency.

- [ ] **Step 4: Test SIGTERM and restart behavior**

Terminate the scheduler while its fake worker runs, restart it, and assert the interrupted run becomes retryable and then completes without duplicating a completed run.

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_scheduler.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the scheduler**

```bash
git add scripts/cnn_modular_rnn_dqn_scheduler.py tests/test_cnn_modular_rnn_dqn_scheduler.py
git commit -m "feat: add resumable modular DQN queue scheduler"
```

### Task 8: Two-Server Coordinator and Stage Advancement

**Files:**
- Create: `scripts/cnn_modular_rnn_dqn_coordinator.py`
- Modify: `tests/test_cnn_modular_rnn_dqn_scheduler.py`

**Interfaces:**
- CLI operations: `prepare-stage`, `status`, `reconstruct`, `rank-stage`, and `prepare-validation`.
- Uses SSH host aliases `NeuRLab-DL3` and `NeuRLab-DL6`; never kills remote processes.

- [ ] **Step 1: Write failing coordinator tests with fake remote directories**

Assert `prepare-stage` assigns every pending run once, preserves running/completed assignments, and gives newly unassigned work to an idle host. Assert `reconstruct` derives the same authoritative state twice from remote status artifacts. Assert stage advancement rejects incomplete or failed seed sets.

- [ ] **Step 2: Run tests and verify failure**

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_scheduler.py -k coordinator -v`

Expected: FAIL because coordinator functions are absent.

- [ ] **Step 3: Implement coordinator operations**

Keep remote transport behind a `RemoteHost` interface with `read_json`, `write_json_atomic`, `exists`, and `list_run_statuses`. Provide filesystem and SSH implementations so unit tests need no network. Aggregate only `completed` artifacts, use `rank_configs`, and require an explicit `--winner-config-id` when producing the next one-variable stage or local-interaction stage.

- [ ] **Step 4: Add dry-run command tests**

Assert every mutating coordinator operation supports `--dry-run` and prints exact destination hosts, queue paths, and run IDs without writes. Assert `status` reports tmux, scheduler PID, worker PID, GPU, latest metric episode, and artifact age separately.

Run: `.venv/bin/pytest tests/test_cnn_modular_rnn_dqn_scheduler.py -v`

Expected: PASS.

- [ ] **Step 5: Commit the coordinator**

```bash
git add scripts/cnn_modular_rnn_dqn_coordinator.py tests/test_cnn_modular_rnn_dqn_scheduler.py
git commit -m "feat: coordinate modular DQN search across servers"
```

### Task 9: Pin the Search Environment and Run the Full Local Gate

**Files:**
- Create: `requirements-search.txt`

**Interfaces:**
- Produces the common direct dependency set installed on both servers.

- [ ] **Step 1: Record compatible direct versions**

Create:

```text
torch==2.13.0
torchvision==0.28.0
gymnasium==1.3.0
numpy==2.5.1
pytest==9.1.1
matplotlib==3.11.1
pygame-ce==2.5.6
```

Before committing, verify these exact versions exist for both remote Python/CUDA environments. If a remote Python version has no matching wheel, install a common supported Python version first rather than silently changing the file on one host.

- [ ] **Step 2: Run the complete local test suite**

Run: `.venv/bin/pytest -q`

Expected: all tests PASS with no newly introduced warnings treated as errors.

- [ ] **Step 3: Run static artifact checks**

Run: `git diff --check && .venv/bin/python -m compileall -q scripts models tests`

Expected: exit 0.

- [ ] **Step 4: Commit environment pins**

```bash
git add requirements-search.txt
git commit -m "build: pin modular DQN search dependencies"
```

### Task 10: Deploy Identical Code and Environments to DL3 and DL6

**Files:**
- No repository files changed.

**Interfaces:**
- Consumes the tested implementation commit and `requirements-search.txt`.
- Produces matching remote commits, `.venv` environments, and metadata snapshots.

- [ ] **Step 1: Verify remote worktrees are safe to update**

Run on each host:

```bash
ssh HOST 'cd /home/neurlab/gitroot/modularRNN && git status --short && git rev-parse HEAD'
```

Expected: clean status. Stop and ask the user if either host has uncommitted work; do not overwrite it.

- [ ] **Step 2: Transfer the exact tested commit**

Push the local branch only after confirming the intended remote, then on each server use `git fetch` and fast-forward to that exact commit. Verify with `git rev-parse HEAD`. Do not use `git reset --hard`.

- [ ] **Step 3: Create matching virtual environments**

On each host run:

```bash
cd /home/neurlab/gitroot/modularRNN
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-search.txt
.venv/bin/python -m pip check
```

Expected: installation and `pip check` succeed. Network/package installation requires the normal approval path if the sandbox blocks it.

- [ ] **Step 4: Verify versions and GPU access**

Run on each host:

```bash
.venv/bin/python -c 'import gymnasium, torch; print(torch.__version__, torch.version.cuda, gymnasium.__version__); print(torch.cuda.get_device_name(0))'
```

Expected: the pinned package versions, RTX 3060 on DL3, and RTX 5070 Ti on DL6.

### Task 11: GPU, Checkpoint, and Disconnect-Survival Smoke Gate

**Files:**
- No repository files changed; smoke artifacts go under a new timestamped `results/cnn_modular_rnn_dqn_smoke/` directory.

**Interfaces:**
- Produces evidence that both servers can train, checkpoint, resume, and survive SSH disconnect.

- [ ] **Step 1: Confirm GPU ownership and disk immediately before testing**

Run `nvidia-smi` and `df -h /home/neurlab/gitroot/modularRNN` on both hosts. Stop if an unknown compute process uses GPU 0 or free disk is below 100 GB.

- [ ] **Step 2: Enable and verify lingering**

On each host run:

```bash
sudo -n loginctl enable-linger neurlab
loginctl show-user neurlab -p Linger --value
```

Expected: `yes`. If passwordless authorization is unavailable, pause for the user to run the enable command; do not claim tmux resilience while it remains `no`.

- [ ] **Step 3: Run a short same-seed GPU smoke test on each host**

Generate isolated two-episode configs with hidden size 75, image size 32, small replay/batch sizes, evaluation every episode, and checkpoint every episode. Launch via the real worker and assert `status.json=completed`, CUDA is used, metrics are finite, and all required artifacts exist.

- [ ] **Step 4: Verify checkpoint resume**

Run a four-episode config with the test stop hook at episode two, resume it through the real worker, and confirm history continues through episode four without duplicate episode numbers.

- [ ] **Step 5: Measure checkpoint cost**

Record `latest.pt` size and write latency with a full 10,000-transition replay buffer. If one checkpoint exceeds 5 GB or a write exceeds 60 seconds, increase `checkpoint_every` from 250 to 500 in the manifest, rerun its tests, and commit that single documented adjustment.

- [ ] **Step 6: Verify tmux survival across SSH disconnect**

On each host create `tmux new-session -d -s modular_dqn_smoke` running a five-minute heartbeat worker. End the SSH connection, wait at least one minute, reconnect, and assert the session, heartbeat PID, and advancing heartbeat file remain. Remove only the named smoke session afterward.

- [ ] **Step 7: Compare same-seed smoke outcomes**

Report both histories and numerical differences. Require identical control flow, episode counts, and finite metrics; document GPU-dependent numerical differences rather than requiring bitwise equality.

### Task 12: Launch and Monitor the Staged Search

**Files:**
- No repository files changed; all outputs are new result artifacts.

**Interfaces:**
- Produces two named tmux sessions, live queue/status artifacts, stage rankings, and final validation results.

- [ ] **Step 1: Prepare and inspect the baseline queue**

Run coordinator `prepare-stage baseline --dry-run`, inspect the three run IDs and both assignments, then run without `--dry-run`. Confirm destination result roots do not already exist; if they do, reconstruct and validate them instead of overwriting.

- [ ] **Step 2: Launch named tmux scheduler sessions**

On each host launch exactly one detached session named `cnn_modular_dqn_search` whose command activates `.venv`, runs the server scheduler on GPU 0, and appends scheduler output to a timestamped log. Immediately verify `tmux has-session`, scheduler PID, child worker PID, `nvidia-smi`, and advancing results.

- [ ] **Step 3: Monitor without conflating liveness signals**

At each checkpoint report separately: tmux alive, scheduler alive, worker alive, expected GPU in use, latest completed episode, artifact age, retry count, and disk free. A live tmux session with stale metrics is not healthy.

- [ ] **Step 4: Complete each approved stage in order**

For baseline, connectivity, hidden size, recurrent gain, input gain, learning rate, n-step, and tau: wait for all three seeds of every config; reject incomplete rankings; generate the aggregate ranking; record the winner and exact tie-break fields; then prepare the next stage from that winner. Never select on first-solve episode.

- [ ] **Step 5: Select at most six local-interaction configurations**

Use the top two observed values of major parameters and write the six selected combinations plus rationale to the stage manifest before seeing their results. Run all three seeds and rank with the same rules.

- [ ] **Step 6: Run initial eight-seed validation**

Prepare validation for the search winner with training seeds `0..7`, 20 fixed evaluation seeds, 25-episode evaluation frequency, and 10,000 episodes. If resources permit, prepare the runner-up before seeing validation outcomes. Keep search and validation aggregates separate.

- [ ] **Step 7: Produce the final evidence report**

Report bounded search space, environment metadata, every failed/retried run, per-seed and aggregate rankings, final-validation results, winning config, best checkpoint paths, exact reproduction command, and remaining uncertainty. Do not call the result a global optimum.
