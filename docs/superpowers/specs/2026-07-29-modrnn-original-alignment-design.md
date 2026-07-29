# modRNN Original-Reference Alignment — Design

## Purpose

`modRNN/model.py` implements the modular connectivity pattern from `CLAUDE.md`, but its
connectivity-generation, weight-initialization, input-embedding, and mask-enforcement
choices were made independently of `hfopt-matlab`'s original MATLAB "modular RNN"
reference (`examples/pathologicals/matlab/train_pathological_cases_modular.m` +
`utils/make_subpools.m`, describing Michaels, Schaffelhofer, Agudelo-Toro & Scherberger
2020, PNAS). This design brings those four aspects in line with the original, while
leaving the deliberate departures already required by `CLAUDE.md` untouched
(bidirectionality, GPU execution, PyTorch).

Explicitly out of scope (unchanged from the current implementation, not part of this
request): equal-thirds module sizing (`hidden_size % 3 == 0`), fully-dense same-module
blocks, bias initialization, and the training algorithm (Adam/REINFORCE) — the latter is
being addressed separately as `hebbRNN`, a new sibling model trained with a from-scratch
PyTorch Hessian-Free/CG optimizer (own design doc).

## Architecture

All four changes live inside `ModularRNNCell` and `ModularBidirectionalRNN` in
`modRNN/model.py`. The guiding principle, taken directly from `make_subpools.m`: each
connection type (module-pair, or input/output routing) is its own independently-generated
block, with its own connection count and its own gain.

### 1. Near-module connectivity generation

`_build_hh_mask`'s two adjacent-module blocks (input↔intermediate, intermediate↔output)
currently draw each entry as an independent `Bernoulli(near_module_sparsity)` — row
in-degree fluctuates around, but isn't exactly, `near_module_sparsity * source_size`.

Replace with fixed in-degree per row, matching `make_subpools.m`'s
`rpidxs = sort(randperm(psize_f)(1:this_c))` loop: for each row (post-synaptic unit) in
the target module, sample exactly `c = round(near_module_sparsity * source_module_size)`
column indices from the source module via `torch.randperm`, and set those to 1. `c` is
computed per directed block (e.g. input←intermediate uses intermediate's size;
intermediate←input uses input's size), so the four off-diagonal blocks generally get
different fixed in-degrees when module sizes differ (not the case here since modules are
equal thirds, but the implementation shouldn't assume that).

Same-module (diagonal) blocks are unchanged: fully dense (`mask = 1`), since this request
only concerns *near*-module (off-diagonal) generation.

### 2. Weight initialization

Currently: a single `nn.init.uniform_(weight, -bound, bound)` over the whole
`(hidden_size, hidden_size)` tensor, `bound = hidden_size ** -0.5`, applied identically to
`weight_ih`, `weight_hh`, and (via `nn.Linear`'s default) `output_proj.weight`.

Replace with block-wise Gaussian init, mirroring
`A(tidx, rpidxs) = randn(1, this_c) * this_g / sqrt(this_c)`: each block is initialized
independently as `randn(...) * gain / sqrt(c)`, where `c` is that block's actual
per-row connection count and `gain` is a per-connection-type constant:

| weight | blocks | `c` per block | `gain` | new constructor param, default |
|---|---|---|---|---|
| `weight_hh` (diagonal) | same-module | `module_size` (dense) | `recurrent_gain` | `recurrent_gain=1.4` (= original `g_rec`) |
| `weight_hh` (off-diagonal) | adjacent-module | fixed in-degree from §1 | `recurrent_gain` | same param as above |
| `weight_ih` | input→input-module | `1` per input dim (each raw input dim is its own size-1 pool in the original, so no `sqrt` averaging across dims) | `input_gain` | `input_gain=1.0` |
| `output_proj.weight` | output-module→output | `output_module_size` (dense) | `output_gain` | `output_gain=1.0` |

Because `weight_ih`'s per-dimension `c=1`, its input-module rows reduce to
`randn(third, input_size) * input_gain` — dense, with no `1/sqrt(input_size)` term (this
is a deliberate property of the original, not an oversight: the original models each raw
input channel as an unrelated size-1 pool rather than jointly fan-in-normalizing across
the whole input vector).

Bias initialization is unchanged (kept as the current `U(±hidden_size**-0.5)`) — not part
of this request.

### 3. Input embedding

Currently: `input_proj = nn.Linear(input_size, hidden_size)` densely projects raw input
to `hidden_size` before entering the cell; `ih_mask` (shape `(hidden_size, hidden_size)`)
then restricts which *rows* (hidden units) read from that projection.

Replace: delete `input_proj` entirely. `ModularRNNCell.weight_ih` becomes shape
`(hidden_size, input_size)` directly (no intermediate embedding stage), matching the
original's `n_Wru_v` (`V x N`, masked). `_build_ih_mask` takes `input_size` as the column
count instead of `hidden_size`; only input-module rows are 1, all `input_size` columns
dense within those rows (matches `RUCGraph(1,:) = 1`, i.e. every raw input dimension
connects to every input-module unit).

`ModularRNNCell.step`/`forward` and `ModularBidirectionalRNN.forward` are updated so the
raw `x[:, t, :]` (shape `(batch, input_size)`) is passed directly to `fwd_cell`/`bwd_cell`
— the `x = self.input_proj(x)` line is removed.

### 4. Mask enforcement

Currently: `masked_weights()` computes `weight * mask` fresh every forward call. This
already zeroes the *gradient* at masked positions via the chain rule, but the underlying
`nn.Parameter.data` at those positions holds arbitrary (never-corrected) values from
initialization — it's only ever masked *implicitly*, at forward-compute time.

Mirror the original's `modMask` more literally — "the sparsity pattern is locked; absent
synapses are always exactly 0", enforced structurally rather than incidentally:

- After block-wise init (§2) and mask construction (§1/§3), zero the raw parameter data
  outside the mask: `weight_ih.data *= ih_mask`, `weight_hh.data *= hh_mask`.
- Register a backward hook on each masked parameter,
  `param.register_hook(lambda g, m=mask: g * m)`, so no optimizer step (including
  weight-decay variants that bypass `.grad`, e.g. decoupled AdamW-style decay acting
  directly on `.data`... note: decay acting on an already-exactly-zero value stays zero,
  so this is only a concern for the gradient path, which the hook covers) can move a
  masked entry away from zero.
- `masked_weights()`'s explicit `weight * mask` forward multiply is kept as a cheap
  defense-in-depth (now a no-op in the steady state, since the raw data is already
  masked), rather than removed.

`output_mask`'s enforcement (currently forward-only, on `output_proj.weight`) gets the
same treatment for consistency: zero `output_proj.weight.data` outside `output_mask` after
init, plus the same gradient hook.

## Correctness verification

Extends `modRNN/test_model.py`'s existing structural checks:

- Near-module block row in-degree is *exactly* `round(near_module_sparsity * source_size)`
  for every row (not just close-to, as the old Bernoulli test tolerance allowed).
- `weight_ih`/`weight_hh`/`output_proj.weight` raw `.data` (not just the masked product)
  is exactly zero outside their respective masks, immediately after construction.
- After one optimizer step on a tiny synthetic batch, masked-out entries in
  `weight_ih.data`/`weight_hh.data`/`output_proj.weight.data` are still exactly zero
  (regression test for the gradient-hook enforcement).
- Empirical weight scale sanity check: for a large `hidden_size`, the sample std of
  initialized entries in a given block is close to `gain / sqrt(c)` for that block.
- Shape test: `weight_ih` is `(hidden_size, input_size)`; `ModularBidirectionalRNN` no
  longer has an `input_proj` attribute.

## Tests

`modRNN/test_mnist.py` and `modRNN/test_cartpole.py` are unaffected in structure (same
`ModularBidirectionalRNN` public interface: `__init__`/`forward` signatures unchanged
apart from the new `recurrent_gain`/`input_gain`/`output_gain` constructor kwargs, all
defaulted). Re-run both to confirm the `>90%`/solve thresholds still hold with the new
init/connectivity scheme — if not, the fix is tuning `recurrent_gain` (analogous to how
the original tunes `g_rec` per task), not lowering the threshold.

## Scope / non-goals

- Module sizing stays equal-thirds only (`hidden_size % 3 == 0`); the original's unequal
  34/33/33 split is not adopted.
- Same-module blocks stay fully dense (not the original's fixed-in-degree-15
  approximation) — only *near*-module generation changes, per the request.
- Bias initialization is untouched.
- Training algorithm (Adam / REINFORCE) is untouched here — HF/CG is `hebbRNN`'s concern,
  covered by a separate design doc.
