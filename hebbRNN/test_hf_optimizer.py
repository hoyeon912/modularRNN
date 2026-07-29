import torch

from hf_optimizer import _flatten, _unflatten, gauss_newton_hvp, conjugate_gradient


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

    fwd = model.fwd_cell
    assert torch.all(fwd.weight_ih.data[fwd.ih_mask == 0.0] == 0.0)
    assert torch.all(fwd.weight_hh.data[fwd.hh_mask == 0.0] == 0.0)


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
