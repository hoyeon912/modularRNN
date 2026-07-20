# Live Training Progress GUI — Design

## Purpose

All four existing test scripts (`RNN/test_mnist.py`, `RNN/test_cartpole.py`,
`biRNN/test_mnist.py`, `biRNN/test_cartpole.py`) currently only print per-epoch
loss to the console, with accuracy reported once at the end. This design adds a
live-updating GUI window (matplotlib, interactive mode) so training progress —
loss and held-out accuracy, per epoch — is visible in real time while a script runs.

## Architecture

### `live_plot.py` (duplicated into `RNN/` and `biRNN/`, not shared)

A small module defining `LiveTrainingPlot`:

- `__init__(title: str)`: turns on `plt.ion()`, creates a figure with two subplots
  (loss on the left, accuracy on the right, y-axis fixed to `[0, 1]`), each starting
  with an empty line.
- `update(epoch: int, loss: float, accuracy: float)`: appends the new point to both
  lines, rescales the loss axis, redraws the canvas (`fig.canvas.draw()` +
  `flush_events()`), and does a short `plt.pause()` so the window actually repaints
  before training continues.
- **Fallback, not a crash risk:** construction is wrapped in `try/except`. If
  matplotlib can't initialize a GUI backend (no display available), `LiveTrainingPlot`
  sets an internal `enabled = False` flag, prints one warning, and `update()` becomes
  a no-op. The live plot is a bonus on top of training — it must never be able to
  break the accuracy assertions the test scripts already enforce.

Each folder gets its own copy of this file, consistent with `RNN/` and `biRNN/`'s
existing pattern of staying independently runnable (mirrors the earlier decision to
duplicate rather than share `heuristic_policy.py`... except that one *is* actually
imported cross-folder today — this one is duplicated instead because both copies
are meant to stay editable independently per-folder, not because of a hard technical
constraint).

### Training loop change (all four scripts)

`train()` currently loops over epochs computing only loss, with a separate
`evaluate()` call after training finishes. It changes to:

- Take `test_loader` and an optional `live_plot` as parameters.
- After each epoch's training pass, call `evaluate(model, test_loader, device)`
  immediately (per-epoch, not just at the end).
- Call `live_plot.update(epoch + 1, avg_loss, accuracy)` if a `live_plot` was passed.
- Return the **final** epoch's accuracy.

`main()` then uses `train()`'s return value directly for the threshold assertion,
removing the old redundant final `evaluate()` call.

## Cost tradeoff (accepted)

Evaluating every epoch instead of once means extra forward passes throughout
training. For `biRNN/test_cartpole.py` — already the slowest script (~28 minutes,
per-timestep Python-loop recurrence over ~500-step episodes) — this adds
meaningfully more wall-clock time, since each epoch now includes a full pass over
the 50 held-out episodes too. Accepted as a known cost of this feature.

## Dependency

Add `matplotlib` to `RNN/requirements.txt` (the single venv both `RNN/` and
`biRNN/` share — no new `requirements.txt` for `biRNN/`).

## Verification limits

Claude can verify the scripts still run to completion and still pass their
accuracy thresholds, and that `LiveTrainingPlot` doesn't throw. Claude cannot
visually confirm a GUI window actually renders correctly on the user's desktop —
that needs a manual check on the user's own machine the first time each script runs.

## Scope / non-goals

- No new shared/common module across `RNN/`/`biRNN/` — duplication is intentional
  here (see above).
- No persistent window-holds-open-after-training behavior — the goal is watching
  progress *during* the run; the window closing when the script exits is fine.
- No change to hyperparameters, thresholds, or model code — this only touches the
  training loop's reporting and adds the plotting helper.
