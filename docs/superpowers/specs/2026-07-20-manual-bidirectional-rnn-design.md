# Manual Bidirectional RNN (biRNN) — Design

## Purpose

`RNN/model.py`'s `SimpleRNN` gets bidirectionality "for free" via PyTorch's
`nn.RNN(bidirectional=True)`. The project's eventual modular RNN (per `CLAUDE.md`)
needs custom, restricted connectivity that `nn.RNN` cannot express, so its
bidirectional recurrence will have to be implemented by hand.

This design covers `biRNN/` — a hand-rolled bidirectional vanilla RNN, built as the
scaffold the modular RNN will extend. It is still a **dense, unrestricted** RNN (no
module structure yet); the only difference from `RNN/`'s `SimpleRNN` is *how* the
bidirectional recurrence is computed (explicit forward/backward loops over
`nn.RNNCell`, instead of a single `nn.RNN(bidirectional=True)` call).

## Architecture

`biRNN/model.py` defines `BidirectionalRNN`, matching `SimpleRNN`'s public interface:

- `__init__(input_size, hidden_size, output_size, output_mode="last")`
- `forward(x)` where `x` is `(batch, seq_len, input_size)`

Internals:
- `nn.Linear(input_size, hidden_size)` — input projection (same as `SimpleRNN`)
- `self.fwd_cell = nn.RNNCell(hidden_size, hidden_size)` and
  `self.bwd_cell = nn.RNNCell(hidden_size, hidden_size)` — independent weights,
  no sharing between directions
- Forward pass: loop `t = 0..T-1`, `h_fwd = fwd_cell(x_t, h_fwd)`, collecting each
  step's hidden state
- Backward pass: loop `t = T-1..0`, `h_bwd = bwd_cell(x_t, h_bwd)`, collecting each
  step's hidden state (re-aligned to forward time order for concatenation)
- `output_mode="all"`: concatenate `[h_fwd_t, h_bwd_t]` at every `t` →
  `nn.Linear(hidden_size*2, output_size)` applied per-timestep →
  `(batch, seq_len, output_size)`
- `output_mode="last"`: concatenate the forward cell's final state (t=T-1) with the
  backward cell's final state (t=0, since that's where the backward pass
  terminates) → one output projection → `(batch, output_size)`
- `get_device()` — duplicated from `RNN/model.py` (small enough not to warrant a
  cross-folder import; each folder stays runnable standalone)

## Correctness verification

`nn.RNNCell` and `nn.RNN`'s per-layer parameters have identical shapes
(`weight_ih_l0` ↔ `RNNCell.weight_ih`, `weight_hh_l0` ↔ `RNNCell.weight_hh`, same
for biases, and the `_reverse` variants for the backward direction). `biRNN/test_model.py`
exploits this for a parity test: build a `RNN.model.SimpleRNN`, copy its weights
into a `BidirectionalRNN`'s input projection / cells / output projection, run
identical random input through both, and assert `torch.allclose` on the output for
both `output_mode="last"` and `output_mode="all"`. This proves the manual loop
reproduces `nn.RNN(bidirectional=True)`'s exact math before anything gets built on
top of it.

## Tests

`biRNN/test_mnist.py` and `biRNN/test_cartpole.py` mirror `RNN/`'s scripts exactly
(same data pipeline, same >90% accuracy thresholds) with `BidirectionalRNN` in
place of `SimpleRNN`. `test_cartpole.py` imports `heuristic_action` from
`RNN/heuristic_policy.py` via a `sys.path` insert rather than duplicating it. No new
`requirements.txt` — the existing `.venv` already has everything needed.

## Scope / non-goals

- No modular connectivity restrictions yet — this is still the dense/vanilla case,
  just with a hand-rolled recurrence instead of `nn.RNN`.
- No shared base class between `RNN/model.py` and `biRNN/model.py` — each folder
  stays self-contained and independently runnable, consistent with the existing
  `RNN/` folder's pattern.
- Hyperparameters and thresholds match `RNN/`'s existing scripts exactly, so results
  are directly comparable.
