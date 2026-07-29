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
