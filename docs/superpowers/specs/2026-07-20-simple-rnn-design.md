# Simple RNN (non-modular baseline) — Design

## Purpose

The project's core spec (`CLAUDE.md`) describes a modular bidirectional RNN whose hidden
layer is split into input/intermediate/output sub-modules with restricted connectivity.
This design covers a **plain, non-modular baseline RNN** in a new `RNN/` folder, to serve
as a point of comparison: same general style (bidirectional, GPU-accelerated, tested on
MNIST and CartPole), but with an ordinary fully-connected hidden layer and no module
structure or connectivity restrictions.

This is a from-scratch build — the repo currently contains only `.venv/` and `CLAUDE.md`.

## Architecture

`RNN/model.py` defines a single `SimpleRNN(nn.Module)`:

- `nn.Linear(input_size, hidden_size)` — input projection
- `nn.RNN(hidden_size, hidden_size, bidirectional=True, batch_first=True)` — vanilla
  Elman cell (tanh), densely connected, no sub-modules
- `nn.Linear(hidden_size * 2, output_size)` — output projection

`output_mode` controls how the output projection is applied:
- `"last"` — concatenate the final forward/backward hidden states into one vector,
  apply the output layer once → one prediction per sequence (used for MNIST).
- `"all"` — apply the output layer at every timestep's hidden state → one prediction
  per timestep (used for CartPole).

Device selection is automatic: `cuda` → `mps` (Apple Silicon) → `cpu`.

## Tests

### `RNN/test_mnist.py` (simple test)

Sequential MNIST: each 28×28 image is fed as a 28-step sequence of 28-pixel rows.
`SimpleRNN` runs bidirectionally over the full image, `output_mode="last"`, classifies
digit 0–9. Reports test-set accuracy.

### `RNN/test_cartpole.py` (hard test)

Bidirectional RNNs need the full sequence upfront, which conflicts with CartPole's
normal one-step-at-a-time control loop. Per user decision, this is reconciled by
reframing the test as **offline behavior cloning** rather than live control:

1. Collect episodes from `gymnasium`'s `CartPole-v1` using a small hand-written
   PD-style heuristic controller (based on pole angle + angular velocity) — not a
   random policy, since random actions fail almost immediately and produce sequences
   too short to be useful.
2. Each episode yields a sequence of 4-d states and the heuristic's action (0/1) at
   each step.
3. Train `SimpleRNN` (`output_mode="all"`) to predict the heuristic's action at every
   timestep from the full bidirectional state sequence.
4. Report per-timestep action-prediction accuracy on held-out episodes.

This exercises sequence modeling on continuous-valued, dynamics-driven data (harder
than MNIST) while staying consistent with the bidirectional-everywhere decision.

## Environment

`.venv` currently has no ML packages installed (Python 3.14.6, stdlib only). This
design adds `torch`, `torchvision` (MNIST dataset), and `gymnasium` to `.venv`.

Risk: Python 3.14 is very new; if official `torch` wheels aren't available for it,
this will be surfaced explicitly rather than worked around silently (e.g. via a
downgraded interpreter), since that decision belongs to the user.

## Scope / non-goals

- No CLI argument framework, no config files, no checkpointing/model saving.
- No attempt to reproduce the modular RNN's connectivity — this is intentionally the
  "no modulation" baseline.
- Hyperparameters (hidden size, epochs, learning rate, episode count) are hardcoded
  with sensible defaults directly in each test script.
- Scripts are run directly: `python RNN/test_mnist.py`, `python RNN/test_cartpole.py`.
