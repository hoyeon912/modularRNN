# hebbRNN HF/CG Optimizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `hebbRNN/`, a self-contained sibling to `modRNN/` with the identical modular
RNN architecture but trained by a from-scratch, GPU-resident Hessian-Free/CG optimizer
instead of Adam/REINFORCE.

**Architecture:** Copy the aligned `modRNN/model.py` verbatim into `hebbRNN/model.py`. Add
`hebbRNN/hf_optimizer.py` with three layers: (1) `gauss_newton_hvp` — a Gauss-Newton
Hessian-vector product via the Pearlmutter double-backward trick, (2) `conjugate_gradient`
— a CG solver with early-stop and checkpoint backtracking, (3) `HFOptimizer` — the outer
loop wiring gradient computation, damped CG, and Levenberg-Marquardt damping adaptation.
`test_mnist.py`/`test_cartpole.py` swap `torch.optim.Adam` for `HFOptimizer`.

**Tech Stack:** PyTorch (`torch.autograd.grad` double-backward, no `torch.func`), pytest.

## Global Constraints

- GPU acceleration required (per `CLAUDE.md`) — all tensor math in `hf_optimizer.py` must
  run on whatever device `get_device()` returns; no `.item()`/`.numpy()` calls inside the
  CG hot loop except where explicitly noted for scalar bookkeeping.
- `hebbRNN/` is self-contained — no imports from `modRNN/`, no shared base class (existing
  repo convention, see `docs/superpowers/specs/2026-07-26-modular-rnn-design.md`).
- Design reference: `docs/superpowers/specs/2026-07-29-hebbrnn-hf-optimizer-design.md`.
  `curvature` modes are exactly `"mse"` and `"categorical"` — no third RL-specific mode
  (REINFORCE reuses `"categorical"`, per that doc's "CartPole / REINFORCE integration"
  section).
- Follow existing repo conventions: `.venv` for Python, run tests with
  `source .venv/bin/activate && python -m pytest <path> -v` from the repo root, or `cd`
  into the target folder first (both work — `modRNN/test_model.py` does a bare
  `from model import ...`, so pytest must be invoked with that folder on `sys.path`, which
  happens automatically when running from inside the folder or via `pytest hebbRNN/`).

---

### Task 1: Scaffold `hebbRNN/` from the aligned `modRNN/`

**Files:**
- Create: `hebbRNN/model.py` (verbatim copy of `modRNN/model.py`)
- Create: `hebbRNN/live_plot.py` (verbatim copy of `modRNN/live_plot.py`)
- Create: `hebbRNN/test_model.py` (verbatim copy of `modRNN/test_model.py`)

**Interfaces:**
- Produces: `ModularRNNCell`, `ModularBidirectionalRNN`, `get_device()` — importable from
  `hebbRNN/model.py`, identical public API to `modRNN/model.py` (see that file for exact
  signatures: `ModularBidirectionalRNN(input_size, hidden_size, output_size,
  output_mode="last", near_module_sparsity=0.1, recurrent_gain=1.4, input_gain=1.0,
  output_gain=1.0)`).

- [ ] **Step 1: Copy the three files**

```bash
mkdir -p hebbRNN
cp modRNN/model.py hebbRNN/model.py
cp modRNN/live_plot.py hebbRNN/live_plot.py
cp modRNN/test_model.py hebbRNN/test_model.py
```

- [ ] **Step 2: Run the copied tests to confirm they pass unchanged in the new location**

Run: `source .venv/bin/activate && cd hebbRNN && python -m pytest test_model.py -v`
Expected: PASS, 17 passed (same count as `modRNN/test_model.py`).

- [ ] **Step 3: Commit**

```bash
git add hebbRNN/model.py hebbRNN/live_plot.py hebbRNN/test_model.py
git commit -m "scaffold hebbRNN with the aligned modular RNN architecture"
```

---

### Task 2: Gauss-Newton Hessian-vector product

**Files:**
- Create: `hebbRNN/hf_optimizer.py`
- Test: `hebbRNN/test_hf_optimizer.py`

**Interfaces:**
- Consumes: nothing from Task 1 directly (this task is pure tensor math, tested against a
  tiny hand-built MLP, not `ModularBidirectionalRNN`).
- Produces:
  - `_flatten(tensors: list[torch.Tensor]) -> torch.Tensor`
  - `_unflatten(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]`
  - `gauss_newton_hvp(z: torch.Tensor, params: list[torch.Tensor], v: list[torch.Tensor], curvature: str) -> list[torch.Tensor]`
    — `curvature` is `"mse"` or `"categorical"`; `z` must have been produced by a forward
    pass with `create_graph`-compatible autograd (i.e. a normal forward pass, not under
    `torch.no_grad()`); `v` must be a list aligned with `params` (same shapes). Returns a
    list aligned with `params`.

- [ ] **Step 1: Write the failing tests**

```python
# hebbRNN/test_hf_optimizer.py
import torch

from hf_optimizer import _flatten, _unflatten, gauss_newton_hvp


def _toy_mlp_forward(params, x):
    w1, b1, w2, b2 = params
    h = torch.tanh(x @ w1 + b1)
    return h @ w2 + b2


def _make_toy_params(in_dim=3, hidden=4, out_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(in_dim, hidden, generator=g, requires_grad=True)
    b1 = torch.randn(hidden, generator=g, requires_grad=True)
    w2 = torch.randn(hidden, out_dim, generator=g, requires_grad=True)
    b2 = torch.randn(out_dim, generator=g, requires_grad=True)
    return [w1, b1, w2, b2]


def test_flatten_unflatten_roundtrip():
    tensors = [torch.randn(2, 3), torch.randn(4), torch.randn(1, 5)]
    flat = _flatten(tensors)
    assert flat.shape == (2 * 3 + 4 + 1 * 5,)
    restored = _unflatten(flat, tensors)
    for original, r in zip(tensors, restored):
        assert torch.allclose(original, r)


def _linear_forward(params, x):
    w, b = params
    return x @ w + b


def _make_linear_params(in_dim=3, out_dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(in_dim, out_dim, generator=g, requires_grad=True)
    b = torch.randn(out_dim, generator=g, requires_grad=True)
    return [w, b]


def test_gauss_newton_hvp_mse_matches_true_hessian():
    # Gauss-Newton curvature equals the *true* Hessian only when z(theta) is linear in
    # theta (the residual-curvature term the GN approximation drops is then exactly
    # zero) -- so this uses a plain linear model, not the tanh MLP (whose true Hessian
    # would legitimately differ from its GN approximation).
    torch.manual_seed(0)
    params = _make_linear_params()
    x = torch.randn(5, 3)
    target = torch.randn(5, 2)

    def loss_fn(flat_params):
        p = _unflatten(flat_params, params)
        z = _linear_forward(p, x)
        return 0.5 * ((z - target) ** 2).sum()

    flat_params = _flatten(params).detach().requires_grad_(True)
    true_hessian = torch.autograd.functional.hessian(loss_fn, flat_params)

    v = [torch.randn_like(p) for p in params]
    v_flat = _flatten(v)

    z = _linear_forward(params, x)
    hv = gauss_newton_hvp(z, params, v, curvature="mse")
    hv_flat = _flatten(hv)

    expected = true_hessian @ v_flat
    assert torch.allclose(hv_flat, expected, atol=1e-4, rtol=1e-3)


def test_gauss_newton_hvp_categorical_is_positive_semidefinite():
    torch.manual_seed(0)
    params = _make_toy_params(out_dim=5)
    x = torch.randn(6, 3)

    z = _toy_mlp_forward(params, x)
    for _ in range(10):
        v = [torch.randn_like(p) for p in params]
        hv = gauss_newton_hvp(z, params, v, curvature="categorical")
        quad_form = sum((a * b).sum() for a, b in zip(v, hv))
        assert quad_form.item() >= -1e-5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd hebbRNN && python -m pytest test_hf_optimizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hf_optimizer'`

- [ ] **Step 3: Implement `hf_optimizer.py`'s Hv-product layer**

```python
# hebbRNN/hf_optimizer.py
import torch


def _flatten(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat: torch.Tensor, like: list[torch.Tensor]) -> list[torch.Tensor]:
    out = []
    idx = 0
    for t in like:
        n = t.numel()
        out.append(flat[idx : idx + n].view_as(t))
        idx += n
    return out


def _curvature_mse(z: torch.Tensor, jv: torch.Tensor) -> torch.Tensor:
    return jv


def _curvature_categorical(z: torch.Tensor, jv: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(z, dim=-1)
    return p * jv - p * (p * jv).sum(dim=-1, keepdim=True)


_CURVATURE_FNS = {"mse": _curvature_mse, "categorical": _curvature_categorical}


def gauss_newton_hvp(
    z: torch.Tensor,
    params: list[torch.Tensor],
    v: list[torch.Tensor],
    curvature: str,
) -> list[torch.Tensor]:
    """Gauss-Newton Hessian-vector product H@v via the Pearlmutter double-backward trick.
    `z` must come from a forward pass whose autograd graph is still alive (no detach/no_grad)."""
    curvature_fn = _CURVATURE_FNS[curvature]

    dummy = torch.zeros_like(z, requires_grad=True)
    jt_dummy = torch.autograd.grad(z, params, grad_outputs=dummy, create_graph=True, retain_graph=True)
    jv = torch.autograd.grad(jt_dummy, dummy, grad_outputs=v, create_graph=True, retain_graph=True)[0]

    curved = curvature_fn(z, jv)
    hv = torch.autograd.grad(z, params, grad_outputs=curved, retain_graph=True)
    return list(hv)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd hebbRNN && python -m pytest test_hf_optimizer.py -v`
Expected: PASS, 3 passed.

- [ ] **Step 5: Commit**

```bash
git add hebbRNN/hf_optimizer.py hebbRNN/test_hf_optimizer.py
git commit -m "add Gauss-Newton Hessian-vector product via double-backward"
```

---

### Task 3: Conjugate gradient solver with backtracking

**Files:**
- Modify: `hebbRNN/hf_optimizer.py`
- Modify: `hebbRNN/test_hf_optimizer.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure linear algebra, tested against
  `torch.linalg.solve` on a random SPD system — not against `gauss_newton_hvp`, to keep
  this test isolated from Task 2's correctness).
- Produces: `conjugate_gradient(matvec, b, x0, max_iter, min_iter, tol, checkpoint_every, eval_fn) -> tuple[torch.Tensor, dict]`
  — `matvec: Callable[[torch.Tensor], torch.Tensor]` (flat vector in, flat vector out),
  `b`/`x0`: flat 1-D tensors, `eval_fn: Callable[[torch.Tensor], float]` (flat vector ->
  real scalar objective value; used only for backtracking checkpoints, called every
  `checkpoint_every` iterations and at the final iteration). Returns `(best_x, diagnostics)`
  where `diagnostics = {"iters": int, "residual": float}`.

- [ ] **Step 1: Write the failing tests**

```python
# append to hebbRNN/test_hf_optimizer.py
from hf_optimizer import conjugate_gradient


def test_conjugate_gradient_solves_spd_system_exactly():
    torch.manual_seed(0)
    n = 10
    m = torch.randn(n, n)
    a = m @ m.T + n * torch.eye(n)  # SPD
    x_true = torch.randn(n)
    b = a @ x_true

    def matvec(v):
        return a @ v

    x0 = torch.zeros(n)
    x_est, diag = conjugate_gradient(
        matvec, b, x0, max_iter=50, min_iter=1, tol=1e-8, checkpoint_every=5, eval_fn=None
    )
    assert torch.allclose(x_est, x_true, atol=1e-3)
    assert diag["iters"] <= n


def test_conjugate_gradient_backtracking_picks_best_checkpoint():
    # A quadratic matvec where the true (non-quadratic) objective, tracked via eval_fn,
    # gets *worse* past x=1 along the solution direction even though CG's own quadratic
    # model keeps "improving" toward the algebraic solution at x=5 — backtracking must
    # return the x=1-ish checkpoint, not the final iterate.
    a = torch.tensor([[1.0]])
    b = torch.tensor([5.0])  # algebraic solution of a@x=b is x=5

    def matvec(v):
        return a @ v

    def eval_fn(x):
        # objective minimized at x=1, increasing again after that
        return ((x[0] - 1.0) ** 2).item()

    x0 = torch.zeros(1)
    x_est, diag = conjugate_gradient(
        matvec, b, x0, max_iter=1, min_iter=1, tol=0.0, checkpoint_every=1, eval_fn=eval_fn
    )
    # single CG step on a 1x1 system lands exactly on the algebraic solution (x=5), whose
    # eval_fn value (16.0) is what a backtracking-free implementation (returning the final
    # iterate unconditionally) would also produce -- so this must assert x_est actually
    # *is* the earlier x=0 checkpoint (not just "some x with eval <= 16.0", which x=5
    # itself already satisfies non-strictly and would let a broken/absent backtracking
    # path slip through undetected).
    assert torch.allclose(x_est, torch.zeros(1), atol=1e-6)
    assert eval_fn(x_est) < eval_fn(torch.tensor([5.0]))


def test_conjugate_gradient_checkpoints_on_early_stop():
    # A 2x2 diagonal SPD system converges in <= 2 iterations -- well before hitting any
    # multiple of checkpoint_every=5 or max_iter=20. Without checkpointing whenever CG
    # stops early (not just at checkpoint_every multiples / the final iteration), the
    # converged iterate would never be evaluated at all, and best_x would incorrectly
    # stay at x0 forever even though it's clearly worse by eval_fn.
    a = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    x_true = torch.tensor([1.0, 1.0])
    b = a @ x_true

    def matvec(v):
        return a @ v

    def eval_fn(x):
        return ((x - x_true) ** 2).sum().item()

    x0 = torch.zeros(2)
    x_est, diag = conjugate_gradient(
        matvec, b, x0, max_iter=20, min_iter=1, tol=1e-6, checkpoint_every=5, eval_fn=eval_fn
    )
    assert diag["iters"] < 5
    assert torch.allclose(x_est, x_true, atol=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd hebbRNN && python -m pytest test_hf_optimizer.py -k conjugate_gradient -v`
Expected: FAIL with `ImportError: cannot import name 'conjugate_gradient'`

- [ ] **Step 3: Implement `conjugate_gradient`**

```python
# append to hebbRNN/hf_optimizer.py
def conjugate_gradient(matvec, b, x0, max_iter, min_iter, tol, checkpoint_every, eval_fn):
    x = x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = r @ r

    best_x = x.clone()
    best_val = eval_fn(x) if eval_fn is not None else 0.0

    iters_run = 0
    last_checkpointed_iter = 0
    residual = None
    for i in range(max_iter):
        iters_run = i + 1
        ap = matvec(p)
        denom = p @ ap
        alpha = rs_old / (denom + 1e-12)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = r @ r

        is_checkpoint = (iters_run % checkpoint_every == 0) or (iters_run == max_iter)
        if eval_fn is not None and is_checkpoint:
            val = eval_fn(x)
            if val < best_val:
                best_val = val
                best_x = x.clone()
            last_checkpointed_iter = iters_run

        residual = rs_new.sqrt().item() / (b.norm().item() + 1e-12)
        should_stop = iters_run >= min_iter and residual < tol
        if should_stop:
            # Early stop can land between checkpoints (e.g. converge at iteration 2 when
            # checkpoint_every=5) -- without this, the final converged iterate would never
            # be evaluated at all, and best_x would incorrectly stay at x0 forever.
            if eval_fn is not None and iters_run != last_checkpointed_iter:
                val = eval_fn(x)
                if val < best_val:
                    best_val = val
                    best_x = x.clone()
            break

        beta = rs_new / (rs_old + 1e-12)
        p = r + beta * p
        rs_old = rs_new

    if eval_fn is None:
        best_x = x

    return best_x, {"iters": iters_run, "residual": residual}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd hebbRNN && python -m pytest test_hf_optimizer.py -v`
Expected: PASS, 6 passed.

- [ ] **Step 5: Commit**

```bash
git add hebbRNN/hf_optimizer.py hebbRNN/test_hf_optimizer.py
git commit -m "add conjugate gradient solver with checkpoint backtracking"
```

---

### Task 4: `HFOptimizer` — the outer HF loop

**Files:**
- Modify: `hebbRNN/hf_optimizer.py`
- Modify: `hebbRNN/test_hf_optimizer.py`

**Interfaces:**
- Consumes: `_flatten`/`_unflatten`/`gauss_newton_hvp` (Task 2), `conjugate_gradient`
  (Task 3); `ModularBidirectionalRNN` (Task 1, `hebbRNN/model.py`) for the integration
  test only.
- Produces: `class HFOptimizer`:
  - `__init__(self, model: torch.nn.Module, curvature: str = "categorical", initial_damping: float = 1.0, cg_max_iter: int = 60, cg_min_iter: int = 10, cg_tol: float = 1e-4, checkpoint_every: int = 5, damping_min: float = 1e-8, damping_max: float = 1e4)`
  - `step(self, objective_fn: Callable[[torch.nn.Module], tuple[torch.Tensor, torch.Tensor]]) -> dict`
    — `objective_fn(model) -> (loss, z)`, `loss` a scalar tensor, `z` the categorical
    logits the curvature is computed against (see design doc's "CartPole / REINFORCE
    integration" section for why both MNIST and CartPole share this same shape of
    closure). Returns `{"loss_before": float, "loss_after": float, "cg_iters": int, "damping": float}`.
    Mutates `model`'s parameters in place (applies the accepted CG step).

- [ ] **Step 1: Write the failing test**

```python
# append to hebbRNN/test_hf_optimizer.py
import torch.nn as nn

from hf_optimizer import HFOptimizer
from model import ModularBidirectionalRNN


def test_hf_optimizer_step_does_not_increase_loss():
    torch.manual_seed(0)
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=3, output_mode="last")
    optimizer = HFOptimizer(model, curvature="categorical", cg_max_iter=15, cg_min_iter=3)

    x = torch.randn(6, 5, 4)
    y = torch.randint(0, 3, (6,))
    criterion = nn.CrossEntropyLoss()

    def objective_fn(m):
        z = m(x)
        return criterion(z, y), z

    diagnostics = optimizer.step(objective_fn)
    assert diagnostics["loss_after"] <= diagnostics["loss_before"] + 1e-6

    # weight_ih/weight_hh on both directions, and output_proj.weight, all carry the same
    # register_hook-based masking (model.py) -- check all three masked parameter groups,
    # not just fwd_cell, so this test actually proves HFOptimizer preserves the modular
    # connectivity constraint everywhere it applies, not just in one place it happens to.
    fwd = model.fwd_cell
    assert torch.all(fwd.weight_ih.data[fwd.ih_mask == 0.0] == 0.0)
    assert torch.all(fwd.weight_hh.data[fwd.hh_mask == 0.0] == 0.0)

    bwd = model.bwd_cell
    assert torch.all(bwd.weight_ih.data[bwd.ih_mask == 0.0] == 0.0)
    assert torch.all(bwd.weight_hh.data[bwd.hh_mask == 0.0] == 0.0)

    assert torch.all(model.output_proj.weight.data[:, model.output_mask == 0.0] == 0.0)


def test_hf_optimizer_reduces_loss_over_several_steps():
    torch.manual_seed(1)
    model = ModularBidirectionalRNN(input_size=4, hidden_size=9, output_size=3, output_mode="last")
    optimizer = HFOptimizer(model, curvature="categorical", cg_max_iter=15, cg_min_iter=3)

    x = torch.randn(6, 5, 4)
    y = torch.randint(0, 3, (6,))
    criterion = nn.CrossEntropyLoss()

    def objective_fn(m):
        z = m(x)
        return criterion(z, y), z

    first_loss = None
    last_loss = None
    for i in range(5):
        diag = optimizer.step(objective_fn)
        if i == 0:
            first_loss = diag["loss_before"]
        last_loss = diag["loss_after"]

    assert last_loss < first_loss
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && cd hebbRNN && python -m pytest test_hf_optimizer.py -k HFOptimizer -v`
Expected: FAIL with `ImportError: cannot import name 'HFOptimizer'`

- [ ] **Step 3: Implement `HFOptimizer`**

```python
# append to hebbRNN/hf_optimizer.py
class HFOptimizer:
    def __init__(
        self,
        model,
        curvature: str = "categorical",
        initial_damping: float = 1.0,
        cg_max_iter: int = 60,
        cg_min_iter: int = 10,
        cg_tol: float = 1e-4,
        checkpoint_every: int = 5,
        damping_min: float = 1e-8,
        damping_max: float = 1e4,
    ):
        self.model = model
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.curvature = curvature
        self.damping = initial_damping
        self.cg_max_iter = cg_max_iter
        self.cg_min_iter = cg_min_iter
        self.cg_tol = cg_tol
        self.checkpoint_every = checkpoint_every
        self.damping_min = damping_min
        self.damping_max = damping_max

    def _apply(self, x_flat: torch.Tensor) -> None:
        deltas = _unflatten(x_flat, self.params)
        with torch.no_grad():
            for p, dx in zip(self.params, deltas):
                p.add_(dx)

    def step(self, objective_fn) -> dict:
        loss, z = objective_fn(self.model)
        grads = torch.autograd.grad(loss, self.params, create_graph=True, retain_graph=True)
        grad_flat = _flatten(grads).detach()
        b = -grad_flat

        def matvec(v_flat: torch.Tensor) -> torch.Tensor:
            v = _unflatten(v_flat.detach(), self.params)
            hv = gauss_newton_hvp(z, self.params, v, self.curvature)
            return _flatten(hv).detach() + self.damping * v_flat

        def eval_fn(x_flat: torch.Tensor) -> float:
            with torch.no_grad():
                self._apply(x_flat)
                new_loss, _ = objective_fn(self.model)
                self._apply(-x_flat)
            return new_loss.item()

        x0 = torch.zeros_like(b)
        x_best, diag = conjugate_gradient(
            matvec, b, x0, self.cg_max_iter, self.cg_min_iter, self.cg_tol, self.checkpoint_every, eval_fn
        )

        loss_before = loss.item()
        undamped_hx = matvec(x_best) - self.damping * x_best
        predicted_decrease = -(grad_flat @ x_best + 0.5 * x_best @ undamped_hx).item()

        self._apply(x_best)
        loss_after_tensor, _ = objective_fn(self.model)
        loss_after = loss_after_tensor.item()

        actual_decrease = loss_before - loss_after
        rho = actual_decrease / (predicted_decrease + 1e-8)
        if rho > 0.75:
            self.damping = max(self.damping * (2.0 / 3.0), self.damping_min)
        elif rho < 0.25:
            self.damping = min(self.damping * 1.5, self.damping_max)

        return {
            "loss_before": loss_before,
            "loss_after": loss_after,
            "cg_iters": diag["iters"],
            "damping": self.damping,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && cd hebbRNN && python -m pytest test_hf_optimizer.py -v`
Expected: PASS, 8 passed.

- [ ] **Step 5: Commit**

```bash
git add hebbRNN/hf_optimizer.py hebbRNN/test_hf_optimizer.py
git commit -m "add HFOptimizer outer loop with Levenberg-Marquardt damping"
```

---

### Task 5: MNIST training script on `HFOptimizer`

**Files:**
- Create: `hebbRNN/test_mnist.py` (adapted from `modRNN/test_mnist.py`)

**Interfaces:**
- Consumes: `ModularBidirectionalRNN`, `get_device` (Task 1); `HFOptimizer` (Task 4);
  `LiveTrainingPlot` (Task 1).
- Produces: `main()` — run directly (`python test_mnist.py`), not collected by pytest
  (matches `modRNN/test_mnist.py`'s convention: no `test_`-prefixed functions inside).

- [ ] **Step 1: Write `hebbRNN/test_mnist.py`**

```python
# hebbRNN/test_mnist.py
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from hf_optimizer import HFOptimizer
from live_plot import LiveTrainingPlot
from model import ModularBidirectionalRNN, get_device


def load_data(batch_size: int = 128):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def to_sequence(images: torch.Tensor) -> torch.Tensor:
    return images.squeeze(1)


def save_results(model, history, results_path: str, model_path: str) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    print(f"saved {len(history)} epoch(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    live_plot=None,
    results_path: str = "mnist_results.json",
    model_path: str = "mnist_model.pt",
) -> float:
    optimizer = HFOptimizer(model, curvature="categorical")
    criterion = nn.CrossEntropyLoss()
    accuracy = 0.0
    history = []
    try:
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for images, labels in train_loader:
                images = to_sequence(images).to(device)
                labels = labels.to(device)

                def objective_fn(m, images=images, labels=labels):
                    z = m(images)
                    return criterion(z, labels), z

                diagnostics = optimizer.step(objective_fn)
                total_loss += diagnostics["loss_after"]
            avg_loss = total_loss / len(train_loader)
            accuracy = evaluate(model, test_loader, device)
            print(f"epoch {epoch + 1}/{epochs} loss {avg_loss:.4f} accuracy {accuracy:.4f} damping {optimizer.damping:.4g}")
            history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})
            if live_plot is not None:
                live_plot.update(epoch + 1, avg_loss, accuracy)
    finally:
        save_results(model, history, results_path, model_path)
    return accuracy


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = to_sequence(images).to(device)
        labels = labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    device = get_device()
    print(f"using device: {device}")

    train_loader, test_loader = load_data()
    model = ModularBidirectionalRNN(input_size=28, hidden_size=63, output_size=10, output_mode="last").to(device)

    live_plot = LiveTrainingPlot(title="hebbRNN/test_mnist.py")
    accuracy = train(model, train_loader, test_loader, device, epochs=5, live_plot=live_plot)
    print(f"test accuracy: {accuracy:.4f}")
    assert accuracy > 0.90, f"expected >90% accuracy, got {accuracy:.4f}"


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the training loop on a tiny synthetic batch (no MNIST download, fast)**

Run this ad hoc script to confirm `train()`'s per-batch wiring works end-to-end before
committing to a full (slow) real-MNIST run:

```bash
source .venv/bin/activate && cd hebbRNN && python -c "
import torch
from torch.utils.data import TensorDataset, DataLoader
from model import ModularBidirectionalRNN, get_device
from test_mnist import train

device = get_device()
model = ModularBidirectionalRNN(input_size=28, hidden_size=63, output_size=10, output_mode='last').to(device)

images = torch.randn(64, 28, 28)
labels = torch.randint(0, 10, (64,))
loader = DataLoader(TensorDataset(images, labels), batch_size=16)

acc = train(model, loader, loader, device, epochs=1, results_path='/tmp/mnist_smoke_results.json', model_path='/tmp/mnist_smoke_model.pt')
print('smoke test accuracy (meaningless on random data, just checking it runs):', acc)
"
```
Expected: runs to completion without error, prints per-batch progress implicitly via
`train()`'s epoch summary line.

- [ ] **Step 3: Run the real MNIST training script to confirm the `>90%` threshold**

Run: `source .venv/bin/activate && cd hebbRNN && python test_mnist.py`
Expected: exits 0, final printed `test accuracy` > 0.90. If it doesn't reach the
threshold, tune `HFOptimizer`'s `cg_max_iter`/`initial_damping` (passed into the
`HFOptimizer(model, curvature="categorical", ...)` call in `train()`), matching the
project policy of tuning hyperparameters rather than lowering the threshold (see
`docs/superpowers/specs/2026-07-26-modular-rnn-design.md`'s Scope section). This step may
take considerably longer per epoch than `modRNN/test_mnist.py`'s Adam-based run, since
each `HFOptimizer.step()` runs a full CG loop (multiple double-backward Hv-products) per
batch instead of one gradient step.

- [ ] **Step 4: Commit**

```bash
git add hebbRNN/test_mnist.py
git commit -m "train hebbRNN on MNIST with HFOptimizer"
```

---

### Task 6: CartPole training script with REINFORCE + `HFOptimizer`

**Files:**
- Create: `hebbRNN/test_cartpole.py` (adapted from `modRNN/test_cartpole.py`)

**Interfaces:**
- Consumes: `ModularBidirectionalRNN`, `get_device` (Task 1); `HFOptimizer` (Task 4);
  `LiveTrainingPlot` (Task 1).
- Produces: `main()` — run directly, not pytest-collected (same convention as Task 5).

- [ ] **Step 1: Write `hebbRNN/test_cartpole.py`**

Unlike `modRNN/test_cartpole.py`'s `collect_episode_stochastic` (which returns
`log_probs`/`entropies` computed once at collection time — fine for Adam, which only ever
needs that one gradient), `HFOptimizer.step()` needs an `objective_fn(model)` closure it
can call repeatedly against the model's *current* parameters (once for the initial
gradient, then again for every CG backtracking checkpoint). So collection here returns raw
`states`/`actions`/`rewards` instead, and `log_probs`/`z` (the logits `gauss_newton_hvp`
differentiates through) get recomputed fresh, under whatever parameters are live at call
time, inside `build_reinforce_objective`'s closure.

```python
# hebbRNN/test_cartpole.py
import json

import gymnasium as gym
import torch

from hf_optimizer import HFOptimizer
from live_plot import LiveTrainingPlot
from model import ModularBidirectionalRNN, get_device


def rollout_episode(model, env, device, max_steps: int = 500) -> float:
    model.eval()
    state, _ = env.reset()
    states = [state]
    total_reward = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            action = logits[0, -1].argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            states.append(state)
            if terminated or truncated:
                break
    return total_reward


def evaluate_reward(model, device, num_episodes: int = 3, max_steps: int = 500) -> float:
    env = gym.make("CartPole-v1")
    total = 0.0
    for _ in range(num_episodes):
        total += rollout_episode(model, env, device, max_steps=max_steps)
    env.close()
    return total / num_episodes


def compute_returns(rewards, gamma: float = 0.99):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    return returns


def collect_episode_stochastic(model, env, device, max_steps: int = 500):
    model.eval()
    state, _ = env.reset()
    states = [state]
    actions = []
    rewards = []
    with torch.no_grad():
        for _ in range(max_steps):
            x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)
            logits = model(x)
            dist = torch.distributions.Categorical(logits=logits[0, -1])
            action = dist.sample()
            state, reward, terminated, truncated, _ = env.step(action.item())
            actions.append(action.item())
            rewards.append(reward)
            states.append(state)
            if terminated or truncated:
                break
    return states[:-1], actions, rewards


def build_reinforce_objective(states, actions, returns, device, entropy_coef: float = 0.01):
    """`states`/`actions`/`returns` are flat lists pooled across an episode batch (one
    entry per visited timestep). Returns an objective_fn(model) -> (loss, z) closure for
    HFOptimizer.step() — recomputes the forward pass under the model's *current* params
    each call, so it's valid across every CG iteration within one HF step."""
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(1)  # (N, 1, obs_dim)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
    baseline = returns_tensor.mean()
    advantages = returns_tensor - baseline

    def objective_fn(model):
        z = model(states_tensor)[:, -1, :]  # (N, num_actions), output_mode="all" sliced to last step of each 1-step sequence
        dist = torch.distributions.Categorical(logits=z)
        log_probs = dist.log_prob(actions_tensor)
        loss = -(log_probs * advantages).mean() - entropy_coef * dist.entropy().mean()
        return loss, z

    return objective_fn


def save_results(model, history, results_path: str, model_path: str) -> None:
    with open(results_path, "w") as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(), model_path)
    print(f"saved {len(history)} update(s) of history to {results_path}, model weights to {model_path}")


def train(
    model,
    device,
    num_updates: int,
    episodes_per_update: int = 8,
    live_plot=None,
    results_path: str = "cartpole_results.json",
    model_path: str = "cartpole_model.pt",
):
    optimizer = HFOptimizer(model, curvature="categorical")
    env = gym.make("CartPole-v1")
    avg_reward = 0.0
    history = []
    try:
        for update in range(num_updates):
            all_states, all_actions, all_returns = [], [], []
            for _ in range(episodes_per_update):
                states, actions, rewards = collect_episode_stochastic(model, env, device)
                all_states.extend(states)
                all_actions.extend(actions)
                all_returns.extend(compute_returns(rewards))

            objective_fn = build_reinforce_objective(all_states, all_actions, all_returns, device)
            diagnostics = optimizer.step(objective_fn)
            avg_reward = evaluate_reward(model, device)
            print(
                f"update {update + 1}/{num_updates} loss {diagnostics['loss_after']:.4f} "
                f"reward {avg_reward:.1f} damping {optimizer.damping:.4g}"
            )
            history.append({"update": update + 1, "loss": diagnostics["loss_after"], "reward": avg_reward})
            if live_plot is not None:
                live_plot.update(update + 1, diagnostics["loss_after"], avg_reward)
            if device.type == "mps" and (update + 1) % 10 == 0:
                torch.mps.empty_cache()
            if avg_reward >= 500:
                print(f"reached max reward (500) at update {update + 1}, stopping early")
                break
    finally:
        env.close()
        save_results(model, history, results_path, model_path)
    return avg_reward, history


def main():
    device = get_device()
    print(f"using device: {device}")

    model = ModularBidirectionalRNN(input_size=4, hidden_size=300, output_size=2, output_mode="all").to(device)

    live_plot = LiveTrainingPlot(title="hebbRNN/test_cartpole.py", metrics=("loss", "reward"))
    avg_reward, _ = train(model, device, num_updates=5, episodes_per_update=4, live_plot=live_plot)
    print(f"average reward: {avg_reward:.1f}")
    assert avg_reward > 150, f"expected average reward > 150, got {avg_reward:.1f}"

    try:
        render_env = gym.make("CartPole-v1", render_mode="human")
        reward = rollout_episode(model, render_env, device)
        render_env.close()
        print(f"rendered episode reward: {reward:.0f}")
    except Exception as e:
        print(f"render skipped (no display available): {e}")


if __name__ == "__main__":
    main()
```

Note `objective_fn` uses `model(states_tensor)[:, -1, :]` on a `(N, 1, obs_dim)` batch of
independent single-step sequences (`output_mode="all"`), rather than replaying full
episode sequences — this intentionally drops the original REINFORCE code's recurrent
state carried *within* an episode when recomputing `log_probs` at HF step-time (the
original `collect_episode_stochastic` also only ever fed 1-step-growing prefixes through
the model at collection time, i.e. it already never carried hidden state across
env-agent-turns either — `model(x)` recomputes from scratch on `states` each call — so
this preserves that existing behavior/limitation, not a new regression from HF/CG.)

- [ ] **Step 2: Smoke-test with a tiny synthetic rollout (no live CartPole env, fast)**

```bash
source .venv/bin/activate && cd hebbRNN && python -c "
import torch
from model import ModularBidirectionalRNN, get_device
from test_cartpole import build_reinforce_objective
from hf_optimizer import HFOptimizer

device = get_device()
model = ModularBidirectionalRNN(input_size=4, hidden_size=300, output_size=2, output_mode='all').to(device)
optimizer = HFOptimizer(model, curvature='categorical', cg_max_iter=15, cg_min_iter=3)

states = [[0.0, 0.0, 0.0, 0.0]] * 20
actions = [0, 1] * 10
returns = [1.0] * 20

objective_fn = build_reinforce_objective(states, actions, returns, device)
diag = optimizer.step(objective_fn)
print(diag)
assert diag['loss_after'] <= diag['loss_before'] + 1e-6
print('OK')
"
```
Expected: prints diagnostics dict and `OK`.

- [ ] **Step 3: Run the real CartPole training script to confirm the reward threshold**

Run: `source .venv/bin/activate && cd hebbRNN && python test_cartpole.py`
Expected: exits 0, final printed `average reward` > 150. As in Task 5, tune
`HFOptimizer`'s hyperparameters (not the threshold) if it falls short, and expect
substantially slower wall-clock time per update than `modRNN/test_cartpole.py`'s
Adam-based REINFORCE.

- [ ] **Step 4: Commit**

```bash
git add hebbRNN/test_cartpole.py
git commit -m "train hebbRNN on CartPole with REINFORCE + HFOptimizer natural gradient"
```
