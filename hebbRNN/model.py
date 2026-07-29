import torch
import torch.nn as nn
import torch.nn.functional as F


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    mask = torch.zeros(hidden_size * 2)
    mask[output_sl] = 1.0
    mask[hidden_size + output_sl.start : hidden_size + output_sl.stop] = 1.0
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
    """Output module -> output, dense both directions. c is the combined fwd+bwd fan-in
    (2 * output-module size) rather than each half's own size, since bidirectionality
    (a CLAUDE.md requirement absent from the original single-direction reference) doubles
    the number of live incoming connections per output unit relative to make_subpools.m's
    ZRCGraph; using the combined count keeps output pre-activation variance ~= gain**2,
    matching the original's variance-preserving intent instead of doubling it."""
    _, _, output_sl = _module_slices(hidden_size)
    n = output_sl.stop - output_sl.start
    c = 2 * n
    weight = torch.zeros(output_size, hidden_size * 2)
    weight[:, output_sl.start : output_sl.stop] = torch.randn(output_size, n) * gain / (c**0.5)
    weight[:, hidden_size + output_sl.start : hidden_size + output_sl.stop] = (
        torch.randn(output_size, n) * gain / (c**0.5)
    )
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


class ModularBidirectionalRNN(nn.Module):
    """Bidirectional RNN whose hidden layer is split into input/intermediate/output modules with restricted connectivity."""

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

        self.fwd_cell = ModularRNNCell(input_size, hidden_size, near_module_sparsity, recurrent_gain, input_gain)
        self.bwd_cell = ModularRNNCell(input_size, hidden_size, near_module_sparsity, recurrent_gain, input_gain)

        self.output_proj = nn.Linear(hidden_size * 2, output_size)
        self.register_buffer("output_mask", _build_output_mask(hidden_size))
        with torch.no_grad():
            self.output_proj.weight.copy_(_init_output_weight(hidden_size, output_size, output_gain))
            self.output_proj.weight.mul_(self.output_mask)
        self.output_proj.weight.register_hook(lambda grad: grad * self.output_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        fwd_ih, fwd_hh = self.fwd_cell.masked_weights()
        h_fwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        fwd_states = []
        for t in range(seq_len):
            h_fwd = self.fwd_cell.step(x[:, t, :], h_fwd, fwd_ih, fwd_hh)
            fwd_states.append(h_fwd)

        bwd_ih, bwd_hh = self.bwd_cell.masked_weights()
        h_bwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        bwd_states = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_bwd = self.bwd_cell.step(x[:, t, :], h_bwd, bwd_ih, bwd_hh)
            bwd_states[t] = h_bwd

        masked_weight = self.output_proj.weight * self.output_mask

        if self.output_mode == "all":
            combined = torch.stack(
                [torch.cat([fwd_states[t], bwd_states[t]], dim=1) for t in range(seq_len)],
                dim=1,
            )
            return F.linear(combined, masked_weight, self.output_proj.bias)

        combined = torch.cat([fwd_states[-1], bwd_states[0]], dim=1)
        return F.linear(combined, masked_weight, self.output_proj.bias)
