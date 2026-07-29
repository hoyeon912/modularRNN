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
