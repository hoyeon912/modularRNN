# CNN-ModularRNN-DQN CartPole Search Design

## Goal

Find the best CNN-ModularRNN-DQN configuration within an explicitly bounded,
staged search space on CartPole-v1. Run the search on NeuRLab-DL3 and
NeuRLab-DL6, keep every long-running process inside `tmux`, and make runs
recoverable after an SSH disconnect or worker failure.

"Best" does not mean a global optimum outside this search space. It means the
highest-ranked configuration under the success and ranking rules below.

## Success and Ranking Rules

Training runs use fixed seeds. Search configurations use training seeds
`0, 1, 2`; the initial final validation uses eight fixed training seeds
`0, 1, 2, 3, 4, 5, 6, 7`. More validation seeds may be appended later, but the
first winner decision uses exactly these eight so that the decision rule is
fixed before results are observed.

During search, run a greedy evaluation every 25 training episodes using fixed
evaluation seeds `2_000_000` through `2_000_009`. During final validation, use
evaluation seeds `2_000_000` through `2_000_019` at each evaluation point. An
evaluation point succeeds when its mean reward is at least 495. A training seed
is solved when at least five successful evaluation points occur consecutively.

Every run continues through 10,000 training episodes even after it is solved.
Whether a configuration solves within the first 500 episodes and the episode of
its first solution are not ranking inputs.

For each training seed, calculate:

- `max_consecutive_successes`: longest consecutive run of successful evaluation
  points;
- `total_successful_evals`: count of successful evaluation points across all
  10,000 episodes;
- `final_window_mean_reward`: mean reward over the final 20 evaluation points;
- `solved`: whether `max_consecutive_successes >= 5`.

Rank configurations lexicographically by:

1. number of solved training seeds;
2. mean `max_consecutive_successes` across training seeds;
3. mean `total_successful_evals` across training seeds;
4. mean `final_window_mean_reward` across training seeds.

`total_successful_evals` indirectly favors configurations that become good
earlier, but it is retained because repeated success is an explicit objective.
The episode of first success is recorded for diagnosis only.

## Compute Environment

Use one worker per server because each selected server has one GPU:

- NeuRLab-DL3: NVIDIA RTX 3060, 12 GB;
- NeuRLab-DL6: NVIDIA RTX 5070 Ti, 16 GB.

Both repositories were at commit `ee8069e` on `main` when the design was
prepared. Neither server had a `.venv`. Both had systemd user lingering disabled.
DL3 had `tmux`; DL6 received `tmux` 3.0a before this design was approved.

Create the same `.venv` and install the same pinned dependencies on both
servers. Record the Git commit, Python, PyTorch, CUDA, Gymnasium, GPU, and OS
versions with every run. Different GPU generations may prevent bitwise-identical
results even with the same seeds; comparisons rely on multi-seed aggregates,
not bitwise equality.

Enable and verify systemd user lingering before launching the search. A tmux
session alone is not considered sufficient because prior project runs were
killed when the final login session closed with lingering disabled. Verify
survival by disconnecting SSH, reconnecting, and checking both the tmux session
and its worker process.

## Execution Architecture

Each server runs one persistent tmux session containing a scheduler. A scheduler
claims one pending run at a time, launches its worker on GPU 0, and records state
transitions. A run is uniquely identified by search stage, configuration ID, and
training seed. Results live under that identifier so concurrent workers cannot
overwrite one another.

The two servers use the same experiment manifest but do not need a shared
database. A coordinator maintains the authoritative manifest, records explicit
server assignments, and atomically copies each server's queue snapshot to that
server. A server scheduler only claims work from its own snapshot. While the
coordinator is active, it gives newly available work to whichever server finishes
first; if the coordinator is interrupted, both server schedulers safely finish
their already assigned queues and then wait. Reconstructing coordinator state
from server result directories must be idempotent.

The run state machine is:

`pending -> running -> completed | retryable_failure | failed`

A scheduler may retry a retryable failure up to two times from the latest
checkpoint. Repeated CUDA OOMs, invalid metrics, or deterministic worker errors
become `failed` and retain their logs. No failure is silently converted into a
poor score.

## Search Strategy

The search is sequential by stage. Each stage uses three fixed training seeds
per configuration and 10,000 episodes per seed. Except where a stage varies a
parameter, it inherits the winning values from the preceding stage. Duplicate
baseline configurations are not rerun when their artifacts are already valid.

### Stage 0: Baseline Reproduction

- RNN hidden size: 150
- near-module sparsity: 0.1
- recurrent gain: 1.4
- input gain: 1.0
- learning rate: 1e-4
- n-step return: 7
- target update tau: 0.005

### Stage 1: Connectivity

Sweep near-module sparsity over `0.1, 0.25, 0.5, 0.75, 1.0`.

Near-module sparsity 1.0 still does not create direct input-module to
output-module recurrent links, so it remains a ModularRNN rather than becoming
the unrestricted RNN control.

### Stage 2: Size and Dynamics

Run one-variable-at-a-time sweeps, carrying each substage winner forward:

- RNN hidden size: `75, 150, 300, 600`;
- recurrent gain: `0.7, 1.0, 1.2, 1.4`;
- input gain: `0.5, 1.0, 1.5`.

### Stage 3: DQN Optimization

Continue one-variable-at-a-time sweeps:

- learning rate: `2e-5, 5e-5, 1e-4, 2e-4`;
- n-step return: `1, 3, 5, 7, 10`;
- target update tau: `0.001, 0.003, 0.005, 0.01`.

### Stage 4: Local Interaction Check

Construct at most six configurations from the top two values of major
parameters. Select combinations deliberately to cover plausible interactions
rather than running the full Cartesian product. Evaluate all selected
configurations with the same three training seeds and ranking rules.

The full staged search is expected to contain about 25-30 unique configurations,
or 75-90 long training runs. Wall-clock duration is unknown until a benchmark is
run on both servers.

## Final Validation

Validate the highest-ranked search candidate first with fixed training seeds
`0` through `7`. Use 20 fixed greedy evaluation episodes at every 25-episode
evaluation point. Run every seed for all 10,000 episodes and apply the same
ranking metrics. Additional fixed seeds may be appended after this initial
decision, but their results must be reported as a separate extended validation.

If compute permits, validate the runner-up with the same seed set before
declaring a winner. This guards against selecting a configuration because of
three favorable search seeds. Clearly distinguish search metrics from validation
metrics in the final report.

## Reproducibility and Checkpointing

Seed Python, PyTorch, the environment, replay sampling, and evaluation explicitly.
Use the training seed directly for model initialization and replay sampling.
Derive a training-episode environment reset seed as
`1_000_000 + training_seed * 20_000 + episode_index`. Use the common fixed
evaluation seed ranges defined above for every configuration.

Write metrics atomically through a temporary file followed by replacement. Keep
only the latest resumable checkpoint and the best evaluation checkpoint for each
run. A resumable checkpoint contains:

- policy and target model states;
- optimizer state;
- replay buffer;
- epsilon/global step and current episode;
- Python and PyTorch RNG states;
- accumulated metrics and consecutive-success state.

Checkpoint only at episode boundaries. Rotate the previous latest checkpoint
after the replacement is verified. Measure checkpoint size and write latency in
the smoke test; reduce checkpoint frequency if replay serialization materially
stalls training, but do not remove resumability.

## Validation Before Long Runs

Before launching the 10,000-episode search:

1. run local unit tests for deterministic evaluation seeds, metric aggregation,
   atomic writes, scheduler state transitions, and checkpoint resume;
2. create and verify matching virtual environments on DL3 and DL6;
3. run a short GPU smoke test on both servers;
4. interrupt and resume a smoke run, confirming that its metrics and counters
   continue correctly;
5. compare a short same-seed run across servers within a documented numerical
   tolerance rather than requiring bitwise identity;
6. enable lingering, launch a tmux smoke worker, disconnect SSH, reconnect, and
   verify that the tmux session and worker survived;
7. confirm result paths, free disk space, GPU ownership, and log freshness.

Only after all checks pass may the staged search begin.

## Monitoring and Outputs

Each run produces a manifest/config snapshot, environment metadata, append-safe
log, atomic metrics file, latest checkpoint, and best checkpoint. Each stage
produces an aggregate ranking with per-seed values and the exact tie-break path.

Monitoring reports must distinguish:

- tmux session alive;
- scheduler alive;
- worker alive and using the expected GPU;
- metrics file advancing;
- run completed or failed.

The final report includes the bounded search space, all failed or retried runs,
search ranking, final validation ranking, and the exact command and checkpoint
needed to reproduce the winning configuration.
