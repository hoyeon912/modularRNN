# Modular Bidirectional RNN (modRNN) — Design

## Purpose

`biRNN/model.py`'s `BidirectionalRNN` is a hand-rolled bidirectional RNN, built explicitly
as scaffolding for restricted hidden-layer connectivity that `nn.RNN`/`nn.RNNCell` cannot
express. This design implements that restriction: the actual modular RNN described in
`CLAUDE.md`.

Per `CLAUDE.md`, the hidden layer is broken into 3 modules — `input`, `intermediate`,
`output` — with:
- dense links within the same module (input↔input, intermediate↔intermediate, output↔output)
- sparse links between adjacent modules (input↔intermediate, intermediate↔output)
- no links between the two extreme modules (input↔output)

## Architecture

`modRNN/model.py` defines `ModularBidirectionalRNN`, matching `BidirectionalRNN`'s public
interface plus one new constructor argument:

- `__init__(input_size, hidden_size, output_size, output_mode="last", near_module_sparsity=0.1)`
- `forward(x)` where `x` is `(batch, seq_len, input_size)`, same output shapes as `biRNN`
  per `output_mode`.
- `hidden_size` must be evenly divisible by 3; the constructor raises `ValueError` otherwise.

### Module layout

Hidden units are split into 3 contiguous, equal-size blocks along the `hidden_size`
dimension: `input` = `[0, h/3)`, `intermediate` = `[h/3, 2h/3)`, `output` = `[2h/3, h)`,
where `h = hidden_size`. This block layout is shared by both directions.

### Masking mechanism

A new `ModularRNNCell(nn.Module)` replaces `nn.RNNCell`, since restricted connectivity
requires per-entry masking that `nn.RNNCell` doesn't expose. It holds its own
`weight_ih`, `weight_hh` (each `(hidden_size, hidden_size)`), `bias_ih`, `bias_hh`
parameters — same shapes as `nn.RNNCell(hidden_size, hidden_size)` — plus two fixed,
non-trainable buffers applied by elementwise multiply immediately before each matmul:

- `ih_mask` — masks rows of `weight_ih`. Only rows belonging to the `input` module are 1;
  `intermediate`/`output` rows are 0. This is what restricts external input to affecting
  only the input module's pre-activation directly — other modules can only be reached
  through recurrent propagation via `weight_hh`.
- `hh_mask` — masks blocks of `weight_hh`:
  - same-module blocks (3 diagonal blocks, including self-loops): all 1s (dense)
  - adjacent-module blocks (input↔intermediate, intermediate↔output — 4 off-diagonal
    blocks total, since the relation isn't required to be symmetric): independent
    Bernoulli(`near_module_sparsity`) draws per entry
  - input↔output blocks (2 off-diagonal blocks): all 0s

Forward math per cell (matching `nn.RNNCell`'s default `tanh` nonlinearity exactly):

```
h' = tanh((weight_ih * ih_mask) @ z + bias_ih + (weight_hh * hh_mask) @ h + bias_hh)
```

Both `fwd_cell` and `bwd_cell` are independent `ModularRNNCell` instances. Each direction
gets its own independently-sampled `hh_mask` (the two directions' sparse near-module
connectivity patterns need not match). Masks are fixed at construction time (registered
as buffers, not parameters) — training only ever updates unmasked weight entries in
effect, since masked entries always multiply to zero in the forward pass regardless of
their value.

### I/O boundary

- `input_proj: nn.Linear(input_size, hidden_size)` — unchanged from `biRNN`, feeds the
  same `z` into both cells. The input-module restriction happens entirely inside
  `ih_mask`, not by shrinking `input_proj`'s output width.
- `output_proj: nn.Linear(hidden_size*2, output_size)` — restricted via a fixed
  `output_mask` buffer of shape `(hidden_size*2,)`, broadcast across `output_proj.weight`'s
  rows, that zeroes every column except the `output` module's slice in both the forward
  half (`[0, h)`) and backward half (`[h, 2h)`) of the concatenated vector. Applied the
  same way as the cell masks: `F.linear(combined, output_proj.weight * output_mask, output_proj.bias)`.
- `output_mode="last"` and `output_mode="all"` behave exactly as in `biRNN`: `"last"`
  concatenates the forward cell's final state (t=T-1) with the backward cell's final
  state (t=0); `"all"` concatenates and projects at every timestep.
- `get_device()` — duplicated from `biRNN/model.py`, same as that folder duplicated it
  from `RNN/model.py`.

## Correctness verification

Unlike `biRNN` vs `RNN` (which are numerically identical by design), `modRNN` cannot have
a parity test against `biRNN` — the input↔output block is always zero regardless of
`near_module_sparsity`, so the two can never compute the same function. Instead,
`modRNN/test_model.py` verifies structure directly:

- After construction, `fwd_cell.weight_hh.data * (1 - fwd_cell.hh_mask)` is all zero for
  the input↔output blocks (and likewise for `bwd_cell`) — i.e. the *masked* weight is
  provably zero in forbidden blocks, not just zero-by-luck.
- The `hh_mask` density in each adjacent-module block is close to `near_module_sparsity`
  (checked over a large-enough hidden size / seeded RNG for a stable assertion).
- `ih_mask` is 1 for exactly the input-module rows and 0 elsewhere.
- `output_mask` is 1 for exactly the output-module columns (both halves) and 0 elsewhere.
- Shape tests for both `output_mode`s, default `output_mode`, invalid `output_mode`
  raising `ValueError`, invalid `hidden_size` (not divisible by 3) raising `ValueError`,
  and `get_device()` — copied from `biRNN/test_model.py`'s equivalents.

## Tests

`modRNN/test_mnist.py` and `modRNN/test_cartpole.py` mirror `biRNN/`'s scripts exactly
(same data pipeline, same hyperparameters, same `>90%` accuracy threshold) with
`ModularBidirectionalRNN` in place of `BidirectionalRNN`. `test_cartpole.py` imports
`heuristic_action` from `RNN/heuristic_policy.py` via the same `sys.path` insert pattern
`biRNN/test_cartpole.py` already uses. No new `requirements.txt`.

## Scope / non-goals

- No shared base class between `RNN/`, `biRNN/`, and `modRNN/` — each folder stays
  self-contained and independently runnable, consistent with the existing pattern.
- `near_module_sparsity` only controls the two adjacent-module blocks; same-module
  blocks are always fully dense and input↔output blocks are always fully zero — these
  are architectural invariants, not tunable.
- Masks are fixed at construction (no dynamic pruning/growing of connectivity during
  training).
- Hyperparameters and thresholds match `biRNN/`'s existing scripts exactly, for direct
  comparability; if `>90%` doesn't hold at the same `hidden_size`, the fix is to increase
  `hidden_size` (kept divisible by 3), not to lower the threshold.
