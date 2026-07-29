# hebbRNN — Hessian-Free/CG-trained Modular RNN — Design

## Purpose

`modRNN/` (after the original-alignment work in
`2026-07-29-modrnn-original-alignment-design.md`) matches the original `hfopt-matlab`
reference's connectivity generation, weight initialization, input embedding, and mask
enforcement — but still trains with first-order Adam/REINFORCE, unlike the original's
Hessian-Free (HF) second-order optimizer (`optimizer/hfopt2.m`, CG sub-solver in
`optimizer/conjgrad_2.m`).

`hebbRNN` is a new, self-contained sibling folder: architecturally identical to the
aligned `modRNN` model, trained with a from-scratch PyTorch port of HF/CG that runs on
GPU (unlike both the original MATLAB code and the existing NumPy port in
`hfopt-matlab/python/optimizer/hfopt2.py`, both CPU-only). "Follow the original" here
means the *algorithm* (Gauss-Newton-CG second-order optimization, Levenberg-Marquardt
damping, CG backtracking) — not a line-for-line port of `hfopt2.m`'s MATLAB-specific
plumbing (parallel pool data distribution, cell-array trial format, etc.), which doesn't
carry over to a GPU/PyTorch/tensor-batch setting.

## Architecture

`hebbRNN/` is independently runnable, no shared base class with `modRNN/`/`biRNN/`/`RNN/`
(existing repo convention).

### Files

- **`model.py`** — copy of the aligned `modRNN/model.py` verbatim (`ModularRNNCell`,
  `ModularBidirectionalRNN`, `get_device`, all four original-alignment changes included).
  Training algorithm is the only axis that differs from `modRNN`.
- **`hf_optimizer.py`** (new) — the HF/CG optimizer, detailed below.
- **`live_plot.py`** — copy of `modRNN/live_plot.py`, unchanged.
- **`test_model.py`** — copy of `modRNN/test_model.py`'s structural/mask tests (same
  architecture, same invariants).
- **`test_hf_optimizer.py`** (new) — unit tests for the Hv-product and CG solver in
  isolation from the RNN.
- **`test_mnist.py`** — same data pipeline/task as `modRNN/test_mnist.py`, `HFOptimizer`
  in place of `torch.optim.Adam`.
- **`test_cartpole.py`** — same environment/policy structure as `modRNN/test_cartpole.py`,
  `HFOptimizer` driving a REINFORCE surrogate loss instead of Adam.

### `hf_optimizer.py` components

**1. `gauss_newton_hvp(forward_fn, params, v, curvature_fn)`**

Computes `H @ v` where `H` is the Gauss-Newton approximation to the Hessian of the network
output w.r.t. `params`, via the Pearlmutter R-op trick implemented as two nested
`torch.autograd.grad` calls (double-backward), never materializing a full Jacobian or
Hessian:

```
z = forward_fn(params)                     # forward pass, e.g. logits
dummy = torch.zeros_like(z, requires_grad=True)
r  = grad(z, params, grad_outputs=dummy, create_graph=True)   # function of dummy: J^T·dummy
Jv = grad(r, dummy, grad_outputs=v, create_graph=True)        # J·v
Hv_out = curvature_fn(z, Jv)               # H_L applied to Jv (analytic, see below)
Hv = grad(z, params, grad_outputs=Hv_out, retain_graph=False) # J^T·(H_L·Jv)
```

One forward pass + three backward passes per call, all as GPU tensor ops (no `.item()`
inside the hot path).

`curvature_fn` is loss-type-specific and supplied by the caller:
- **`"mse"`** (unused by the two shipped tasks, included for completeness/testing since
  MSE's Gauss-Newton curvature equals its true Hessian exactly — this is what
  `test_hf_optimizer.py` checks against a brute-force Hessian): `curvature_fn(z, u) = u`.
- **`"categorical"`** (used by both MNIST classification and CartPole's policy output —
  same softmax/categorical Fisher curvature in both cases): per-example
  `curvature_fn(z, u) = p * u - p * (p · u)` where `p = softmax(z)`, i.e. the standard
  softmax GN-Hessian/Fisher-vector product, batched over the leading dimension.

**2. `conjugate_gradient(matvec, b, x0, max_iter, min_iter, tol, checkpoint_every, eval_fn)`**

Solves `(H + λI) x = b` (the damped system; `matvec` already includes the `+ λI` term) via
standard CG. Mirrors `conjgrad_2.m`'s two safety mechanisms:
- **Early stop**: relative residual `‖b - Ax_i‖ / ‖b‖ < tol`, bounded by `[min_iter,
  max_iter]`.
- **Backtracking**: every `checkpoint_every` iterations (and at the final iteration),
  `eval_fn(x_i)` runs one real forward pass to get the actual objective at
  `θ + x_i` — since CG on an approximate/damped quadratic can overshoot past the point
  where the true (non-quadratic) objective stops improving, the iterate with the lowest
  真 objective across all checkpoints is returned, not necessarily the last one.

**3. `class HFOptimizer`**

```python
HFOptimizer(model, curvature="categorical", initial_damping=1.0,
            cg_max_iter=60, cg_min_iter=10, cg_tol=1e-4, checkpoint_every=5)
```

`step(objective_fn)` — `objective_fn(model) -> (loss, z)` (a closure the caller supplies,
since MNIST's cross-entropy-of-logits and CartPole's REINFORCE surrogate compute their
scalar loss differently, but both hand back `z` = the categorical logits the curvature is
computed against):

1. Forward pass, compute `loss` and `grad = ∂loss/∂θ` (standard `torch.autograd.grad`).
2. Run `conjugate_gradient` with `matvec(v) = gauss_newton_hvp(..., v, curvature_fn) + λv`,
   `b = -grad`, `eval_fn(x) = objective_fn(model, θ + x)`.
3. Apply the best backtracked CG iterate to `θ`.
4. **Damping update (Levenberg-Marquardt)**: `ρ = (loss_before - loss_after) /
   (-grad·x - 0.5 x·(Hx))` (actual improvement over CG's quadratic-model-predicted
   improvement). If `ρ > 0.75`, `λ *= 2/3`; if `ρ < 0.25`, `λ *= 1.5`; `λ` clamped to
   `[1e-8, 1e4]` — standard Martens (2010) HF damping rule.

Returns a diagnostics dict (`loss_before`, `loss_after`, `cg_iters`, `damping`) so
`test_mnist.py`/`test_cartpole.py` can log/plot it the same way the Adam-based scripts log
`loss`/`accuracy`.

### CartPole / REINFORCE integration

`test_cartpole.py` collects a batch of trajectories under the current policy exactly as
`modRNN/test_cartpole.py` already does, then builds:
- `loss = -mean(log π(a_t|s_t) · advantage_t)` (the REINFORCE surrogate) as the scalar
  `objective_fn` return value.
- `z` = the policy's action logits at the visited states, for `gauss_newton_hvp`'s
  `curvature="categorical"` Fisher-vector product — this makes `HFOptimizer.step` compute
  a *natural policy gradient* direction (CG-approximated), a standard and well-founded way
  to combine policy gradients with second-order/Fisher-curvature optimization (same
  family as TRPO's Fisher-vector product, just solved with CG instead of TRPO's own
  conjugate-gradient-plus-line-search — so this reuses, not reinvents, that combination).

This is why `curvature_fn` only needs `"mse"`/`"categorical"`, not a third RL-specific
mode: the curvature is about the *output distribution* (categorical over classes or
actions), and REINFORCE only changes how the outer scalar `loss`/`grad` is built from that
distribution.

## Correctness verification

- `gauss_newton_hvp(..., curvature="mse")` on a small 2-layer MLP matches
  `torch.autograd.functional.hessian(...) @ v` exactly (MSE's GN curvature is the true
  Hessian) within float tolerance.
- `gauss_newton_hvp(..., curvature="categorical")` is positive semi-definite: `v @ Hv >= 0`
  for random `v`, checked over many trials (Fisher matrices are always PSD; a sign error
  in the softmax curvature formula would show up here).
- `conjugate_gradient` recovers the exact solution of a small hand-built SPD system with a
  known answer (e.g. solve `Ax=b` for a random SPD `A`, compare to `torch.linalg.solve`).
- `HFOptimizer.step()` on a tiny fixed MNIST-shaped batch: `loss_after <= loss_before` (HF
  with backtracking should never *increase* the true objective — if it does, damping/CG
  bookkeeping has a bug).
- `test_model.py` (copied from `modRNN`) still passes unchanged — confirms the copied
  architecture didn't drift.

## Tests

`test_mnist.py`/`test_cartpole.py` reuse the existing `>90%` accuracy / solve thresholds
from `modRNN`'s scripts where applicable; if HF/CG needs different hyperparameters
(`cg_max_iter`, `initial_damping`) to reach them, tune those rather than lowering the
threshold, same policy as `modRNN`'s existing scripts.

## Scope / non-goals

- Per-CG-iteration Hv minibatch resampling (`S` in the original) is not implemented — both
  the gradient and every CG matvec use the same fixed batch passed to `step()`. The
  `matvec` signature is written so this could be added later without an API change.
- No `matlabpool`/`parfor`-style data-parallel worker distribution — PyTorch's own
  GPU-batched tensor ops replace it.
- Fixed-point / linear-stability analysis tools (`find_*_fp.m` equivalents) are not
  ported.
- No shared code between `modRNN/hf_optimizer` (doesn't exist) and `hebbRNN/hf_optimizer`
  — the optimizer is specific to this folder, consistent with each folder being
  self-contained.
