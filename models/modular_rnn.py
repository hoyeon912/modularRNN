import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import get_device

__all__ = ["ModularRNN", "ModularRNNCell", "get_device"]


def _module_slices(hidden_size: int) -> tuple[slice, slice, slice]:
    third = hidden_size // 3
    return slice(0, third), slice(third, 2 * third), slice(2 * third, hidden_size)


def _build_ih_mask(hidden_size: int, input_size: int) -> torch.Tensor:
    input_sl, _, _ = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size, input_size)
    mask[input_sl, :] = 1.0
    return mask


def _sample_fixed_indegree(rows: int, source_size: int, sparsity: float) -> torch.Tensor:
    """Each row gets exactly round(sparsity * source_size) connections (make_subpools.m's
    per-row randperm sampling), instead of an independent Bernoulli draw per entry."""
    block = torch.zeros(rows, source_size)
    c = round(sparsity * source_size)
    if c <= 0:
        return block
    for r in range(rows):
        idx = torch.randperm(source_size)[:c]
        block[r, idx] = 1.0
    return block


def _build_hh_mask(hidden_size: int, near_module_sparsity: float) -> torch.Tensor:
    input_sl, inter_sl, output_sl = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size, hidden_size)

    for sl in (input_sl, inter_sl, output_sl):
        mask[sl, sl] = 1.0

    for row_sl, col_sl in (
        (input_sl, inter_sl),
        (inter_sl, input_sl),
        (inter_sl, output_sl),
        (output_sl, inter_sl),
    ):
        rows = row_sl.stop - row_sl.start
        source_size = col_sl.stop - col_sl.start
        mask[row_sl, col_sl] = _sample_fixed_indegree(rows, source_size, near_module_sparsity)

    return mask


def _build_output_mask(hidden_size: int) -> torch.Tensor:
    _, _, output_sl = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size)
    mask[output_sl] = 1.0
    return mask


def _init_hh_weight(hidden_size: int, near_module_sparsity: float, gain: float) -> torch.Tensor:
    """Block-wise randn(...) * gain / sqrt(c), c = each block's own connection count —
    matches make_subpools.m's A(tidx,rpidxs) = randn(1,this_c) * this_g / sqrt(this_c).
    Blocks are filled densely; _build_hh_mask decides which entries actually survive, so
    which specific entries get zeroed doesn't affect the surviving entries' distribution."""
    input_sl, inter_sl, output_sl = _module_slices(hidden_size)
    weight = torch.zeros(hidden_size, hidden_size)

    for sl in (input_sl, inter_sl, output_sl):
        n = sl.stop - sl.start
        weight[sl, sl] = torch.randn(n, n) * gain / (n**0.5)

    for row_sl, col_sl in (
        (input_sl, inter_sl),
        (inter_sl, input_sl),
        (inter_sl, output_sl),
        (output_sl, inter_sl),
    ):
        rows = row_sl.stop - row_sl.start
        source_size = col_sl.stop - col_sl.start
        c = round(near_module_sparsity * source_size)
        if c > 0:
            weight[row_sl, col_sl] = torch.randn(rows, source_size) * gain / (c**0.5)

    return weight


def _init_ih_weight(hidden_size: int, input_size: int, gain: float) -> torch.Tensor:
    """Each raw input dimension is its own size-1 pool (make_subpools.m treats every input
    channel as an independent size-1 pool), so there's no sqrt(input_size) averaging term —
    just randn(...) * gain over the dense input-module block."""
    input_sl, _, _ = _module_slices(hidden_size)
    n = input_sl.stop - input_sl.start
    weight = torch.zeros(hidden_size, input_size)
    weight[input_sl, :] = torch.randn(n, input_size) * gain
    return weight


def _init_output_weight(hidden_size: int, output_size: int, gain: float) -> torch.Tensor:
    """Output module -> output, dense. c is the output module's own fan-in size,
    matching make_subpools.m's ZRCGraph variance-preserving initialization."""
    _, _, output_sl = _module_slices(hidden_size)
    n = output_sl.stop - output_sl.start
    weight = torch.zeros(output_size, hidden_size)
    weight[:, output_sl.start : output_sl.stop] = torch.randn(output_size, n) * gain / (n**0.5)
    return weight


class ModularRNNCell(nn.Module):
    """Hand-rolled RNNCell restricted to modular connectivity: dense within module, sparse near, none far."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        near_module_sparsity: float,
        recurrent_gain: float = 1.4,
        input_gain: float = 1.0,
    ):
        super().__init__()
        self.weight_ih = nn.Parameter(_init_ih_weight(hidden_size, input_size, input_gain))
        self.weight_hh = nn.Parameter(_init_hh_weight(hidden_size, near_module_sparsity, recurrent_gain))
        bound = hidden_size**-0.5
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        nn.init.uniform_(self.bias_ih, -bound, bound)
        nn.init.uniform_(self.bias_hh, -bound, bound)

        self.register_buffer("ih_mask", _build_ih_mask(hidden_size, input_size))
        self.register_buffer("hh_mask", _build_hh_mask(hidden_size, near_module_sparsity))

        # Lock the sparsity pattern (mirrors modMask: absent synapses are always exactly
        # 0, not just masked-out at forward time) — zero the raw data once, then block any
        # gradient from ever moving a masked entry away from zero.
        with torch.no_grad():
            self.weight_ih.mul_(self.ih_mask)
            self.weight_hh.mul_(self.hh_mask)
        # Look up self.ih_mask/self.hh_mask at call time (not captured at construction) so
        # the hook still matches after .to(device) swaps the buffer tensors in place.
        self.weight_ih.register_hook(lambda grad: grad * self.ih_mask)
        self.weight_hh.register_hook(lambda grad: grad * self.hh_mask)

    def masked_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weight_ih * self.ih_mask, self.weight_hh * self.hh_mask

    def step(self, x: torch.Tensor, h: torch.Tensor, masked_ih: torch.Tensor, masked_hh: torch.Tensor) -> torch.Tensor:
        return torch.tanh(F.linear(x, masked_ih, self.bias_ih) + F.linear(h, masked_hh, self.bias_hh))

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        masked_ih, masked_hh = self.masked_weights()
        return self.step(x, h, masked_ih, masked_hh)


class ModularRNN(nn.Module):
    """RNN whose hidden layer is split into input/intermediate/output modules with restricted connectivity."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        output_mode: str = "last",
        near_module_sparsity: float = 0.1,
        recurrent_gain: float = 1.4,
        input_gain: float = 1.0,
        output_gain: float = 1.0,
    ):
        super().__init__()
        if output_mode not in ("last", "all"):
            raise ValueError(f"output_mode must be 'last' or 'all', got {output_mode!r}")
        if hidden_size % 3 != 0:
            raise ValueError(f"hidden_size must be divisible by 3, got {hidden_size}")
        self.output_mode = output_mode
        self.hidden_size = hidden_size

        self.cell = ModularRNNCell(input_size, hidden_size, near_module_sparsity, recurrent_gain, input_gain)

        self.output_proj = nn.Linear(hidden_size, output_size)
        self.register_buffer("output_mask", _build_output_mask(hidden_size))
        with torch.no_grad():
            self.output_proj.weight.copy_(_init_output_weight(hidden_size, output_size, output_gain))
            self.output_proj.weight.mul_(self.output_mask)
        self.output_proj.weight.register_hook(lambda grad: grad * self.output_mask)

    def init_hidden(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One recurrent step: x_t (batch, input_size), h (batch, hidden_size) ->
        (q_t, h_next). Equivalent to calling forward() on the sequence ending at x_t and
        taking its last timestep, but without redoing every earlier timestep -- lets
        rollouts carry hidden state forward instead of replaying the whole growing history
        on every env step (which is O(T^2) and, with an expensive per-frame encoder in
        front, prohibitively slow for long episodes)."""
        ih, hh = self.cell.masked_weights()
        h_next = self.cell.step(x_t, h, ih, hh)
        masked_weight = self.output_proj.weight * self.output_mask
        q_t = F.linear(h_next, masked_weight, self.output_proj.bias)
        return q_t, h_next

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        ih, hh = self.cell.masked_weights()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        states = []
        for t in range(seq_len):
            h = self.cell.step(x[:, t, :], h, ih, hh)
            states.append(h)

        outputs = torch.stack(states, dim=1)
        masked_weight = self.output_proj.weight * self.output_mask

        if self.output_mode == "all":
            result = F.linear(outputs, masked_weight, self.output_proj.bias)
        else:
            result = F.linear(states[-1], masked_weight, self.output_proj.bias)

        if return_hidden:
            return result, outputs
        return result
