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
