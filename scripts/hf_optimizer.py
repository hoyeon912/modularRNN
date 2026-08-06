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

    def _probe(self, x_flat: torch.Tensor, objective_fn):
        """Evaluate objective_fn at theta + x without an in-place write to `self.params`.
        `eval_fn` runs mid-CG-loop while `matvec` still needs to backward through the
        `z` graph from the original forward pass; an in-place p.add_/sub_ bumps each
        leaf's version counter even after being reverted back to its original value,
        which invalidates that retained graph for every later CG iteration. Swapping
        `p.data` to a new tensor (and back) instead leaves the original tensor object
        the graph saved untouched, so the retained graph stays valid."""
        deltas = _unflatten(x_flat, self.params)
        originals = [p.data for p in self.params]
        with torch.no_grad():
            for p, dx in zip(self.params, deltas):
                p.data = p.data + dx
            result = objective_fn(self.model)
            for p, orig in zip(self.params, originals):
                p.data = orig
        return result

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
            new_loss, _ = self._probe(x_flat, objective_fn)
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
