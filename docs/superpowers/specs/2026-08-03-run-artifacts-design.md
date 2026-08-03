# Run artifacts: figure, log, and RNN unit-activity saving

Date: 2026-08-03
Status: approved

## Motivation

`RNN/`, `biRNN/`, `modRNN/`, and `hfRNN/` each train a model on MNIST and CartPole
(`test_mnist.py`, `test_cartpole.py`). Today each run saves a metrics-history JSON and a
model checkpoint (`save_results`), shows a live matplotlib window (`live_plot.py`) that is
never persisted, and reports progress via bare `print()`. There is no record of what the
hidden units were actually doing during training, and no on-disk log or figure to look back
at after a run finishes.

This adds, to every one of the 8 training scripts (4 model variants × {mnist, cartpole}):

1. A persisted training-curve figure.
2. A text log file alongside console output.
3. Periodic snapshots of RNN hidden-unit activity (raw tensor + heatmap) captured during
   training on a fixed representative input.

All three land in a per-run directory: `results/{env_name}/{run_timestamp}/`.

## Scope

All 4 model directories, both training scripts each = 8 training runs. Test-suite files
(`test_model.py`, `test_hf_optimizer.py`) get a small number of new unit tests for the new
model/helper behavior; they are not otherwise in scope.

## Architecture

The repository already duplicates code across the four model directories rather than
sharing a package — `live_plot.py` is byte-identical in all four, each directory is a
self-contained, independently-runnable variant. This feature follows that existing
convention: a new module is written once and copied verbatim into each of the four
directories, the same way `live_plot.py` already is. Introducing a shared package (e.g. a
top-level `common/` importable via `sys.path` manipulation) was considered and rejected —
it's a bigger structural change than this feature needs, breaks the "each directory stands
alone" property the repo currently has, and isn't required to satisfy the request.

## Components

### 1. `model.py` (all 4 dirs) — expose the hidden-state trajectory

Add `return_hidden: bool = False` to each model's `forward()`. When `True`, returns
`(output, hidden_states)` instead of just `output`, where `hidden_states` is the raw
per-timestep hidden trajectory *before* output projection:

- `modRNN`, `hfRNN` (`ModularRNN`, unidirectional): shape `(batch, seq_len, hidden_size)`.
- `RNN`, `biRNN` (`SimpleRNN`, `BidirectionalRNN`): shape
  `(batch, seq_len, hidden_size*2)`, forward and backward states concatenated along the
  last dim (same convention the bidirectional `output_mode="all"` path already uses
  internally).

Default `False` preserves every existing call site and test unchanged — this is purely
additive.

### 2. New `run_artifacts.py` (new file, duplicated into each of the 4 dirs)

Three functions:

- `make_run_dir(env_name: str, root: str = "results") -> Path`
  Creates `{root}/{env_name}/{YYYYMMDD_HHMMSS}/` and a `activity/` subdirectory inside it,
  timestamped at call time (i.e. run start), and returns the run directory path.

- `setup_logger(name: str, log_path: Path) -> logging.Logger`
  Returns a logger with two handlers: a `StreamHandler` (console, same output users see
  today) and a `FileHandler` pointed at `log_path`. Format includes a timestamp and level.
  In addition to the messages already printed today (per-epoch/update loss/accuracy/reward),
  logs run start with device and model hyperparameters (`input_size`, `hidden_size`,
  `output_size`, and any model-specific kwargs already available in each script's `main()`).

- `save_activity_snapshot(hidden_states: torch.Tensor, activity_dir: Path, step_label: str, module_bounds: list[int] | None = None) -> None`
  Takes `hidden_states` for a **single sample** (shape `(seq_len, H)`, batch dim already
  indexed/squeezed by the caller), saves the raw tensor to
  `activity_dir/{step_label}.pt` (`torch.save`), and renders a heatmap (`imshow`, units on
  the y-axis, timestep on the x-axis, diverging colormap centered at 0 since activity is
  `tanh`-bounded) to `activity_dir/{step_label}.png`. If `module_bounds` is given (a list of
  row indices), draws horizontal separator lines at those rows — used to mark the
  input/intermediate/output module boundaries for `modRNN`/`hfRNN`, and the forward/backward
  split for `RNN`/`biRNN`. Closes the figure after saving to avoid unbounded memory growth
  across many snapshots in one run.

### 3. `live_plot.py` (all 4 dirs) — persist the curve

Add `LiveTrainingPlot.save(path) -> None`: calls `self.fig.savefig(path)`; no-ops if
plotting was disabled (no GUI backend), matching the class's existing self-disabling
behavior.

### 4. `test_mnist.py` / `test_cartpole.py` (all 4 dirs) — wire it together

`main()`:
- Calls `run_dir = make_run_dir("mnist")` (or `"cartpole"`).
- Calls `logger = setup_logger(__name__, run_dir / "train.log")`, logs device + model
  hyperparameters.
- Passes `run_dir` into `train(...)` instead of today's separate `results_path`/
  `model_path` arguments.

`train()`:
- Derives `results.json`, `model.pt` paths from `run_dir` (replaces the current
  `results_path: str = "mnist_results.json"` / `model_path: str = "mnist_model.pt"`
  defaults — this is the one small breaking change to `train()`'s signature; acceptable
  since it's only called from each file's own `main()` and its own tests, both of which are
  updated together).
- Replaces `print(...)` progress lines with `logger.info(...)`.
- Each epoch (MNIST) / update (CartPole), runs one fixed representative input through
  `model(x, return_hidden=True)` and calls `save_activity_snapshot`:
  - **MNIST**: the first image of the (unshuffled) test set, fixed for the whole run so
    snapshots are comparable across epochs. Selected once before the training loop starts.
  - **CartPole**: `evaluate_reward` averages over multiple episodes and doesn't expose any
    single trajectory, so it can't be reused directly. Extend `rollout_episode` with an
    optional `return_states: bool = False` that, when set, returns `(total_reward, states)`
    instead of just `total_reward`. Each update, call it once more
    (`_, states = rollout_episode(model, env, device, return_states=True)`) to get one
    deterministic-policy trajectory, then a single extra `model(x, return_hidden=True)` call
    over that full trajectory to get the hidden states for the snapshot. This adds one extra
    episode of rollout cost per update on top of `evaluate_reward`'s existing 3.
  - Step label: `epoch_{n:02d}` / `update_{n:02d}`.
  - `module_bounds`: passed for `modRNN`/`hfRNN` (from each model's `hidden_size // 3`
    boundaries) and for `RNN`/`biRNN` (the `hidden_size` midpoint splitting forward/backward).
- On completion, in the existing `finally` block alongside `save_results`, calls
  `live_plot.save(run_dir / "curve.png")`.

## Data layout

```
results/
  mnist/
    20260803_143000/
      results.json
      model.pt
      train.log
      curve.png
      activity/
        epoch_01.pt  epoch_01.png
        epoch_02.pt  epoch_02.png
        ...
  cartpole/
    20260803_150210/
      results.json
      model.pt
      train.log
      curve.png
      activity/
        update_01.pt  update_01.png
        ...
```

`results/` is created relative to each script's working directory (i.e. inside
`modRNN/results/...` when run from `modRNN/`), consistent with how `mnist_results.json` is
written to cwd today. Not added to `.gitignore` scope changes in this design — if the user
wants `results/` git-ignored, that's a follow-up, not blocking this feature.

## Error handling

- `save_activity_snapshot` runs inside the same best-effort spirit as `LiveTrainingPlot`:
  if matplotlib has no GUI/Agg backend available, figure saving should still work (`savefig`
  doesn't require a GUI backend, unlike the interactive `plt.ion()` path in
  `LiveTrainingPlot`), so no special fallback is needed there. No new failure modes are
  introduced beyond what plain file I/O already has (disk full, permissions) — these are not
  handled specially, matching how `save_results` doesn't handle them today.
- `make_run_dir` uses `Path.mkdir(parents=True)`; if a run directory for the same timestamp
  already exists (two runs started in the same second), this raises — acceptable, matches
  the low-stakes nature of these smoke-test scripts.

## Testing

Per model directory, add to `test_model.py`:
- `forward(x, return_hidden=True)` returns a tuple whose second element has the expected
  shape (`hidden_size` or `hidden_size*2` depending on directionality).
- Existing `forward(x)` (no `return_hidden`) behavior is unchanged (regression check).

New `test_run_artifacts.py` (or appended to `test_model.py` if small enough) per directory:
- `make_run_dir` creates the expected nested directory structure with a timestamp-shaped
  name.
- `setup_logger` writes to both console and the given file path.
- `save_activity_snapshot` produces both the `.pt` and `.png` file, and the `.pt` tensor
  round-trips (`torch.load` gives back the same shape/values).

`test_mnist.py`/`test_cartpole.py` themselves remain full training smoke tests (already
slow, run manually) — no new automated assertions added there beyond ensuring `main()`
still runs end to end; this is exercised manually per the project's existing convention
(these scripts aren't part of the fast unit-test suite).

## Non-goals

- No shared package / import restructuring across the 4 directories.
- No change to `.gitignore` for `results/`.
- No retention/cleanup policy for old run directories.
- No change to what `output_mode="last"`/`"all"` return for the non-hidden path — only a
  new opt-in second return value is added.
