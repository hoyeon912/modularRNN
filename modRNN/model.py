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


def _build_ih_mask(hidden_size: int) -> torch.Tensor:
    input_sl, _, _ = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size, hidden_size)
    mask[input_sl, :] = 1.0
    return mask


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
        cols = col_sl.stop - col_sl.start
        mask[row_sl, col_sl] = (torch.rand(rows, cols) < near_module_sparsity).float()

    return mask


def _build_output_mask(hidden_size: int) -> torch.Tensor:
    _, _, output_sl = _module_slices(hidden_size)
    mask = torch.zeros(hidden_size * 2)
    mask[output_sl] = 1.0
    mask[hidden_size + output_sl.start : hidden_size + output_sl.stop] = 1.0
    return mask


class ModularRNNCell(nn.Module):
    """Hand-rolled RNNCell restricted to modular connectivity: dense within module, sparse near, none far."""

    def __init__(self, hidden_size: int, near_module_sparsity: float):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        bound = hidden_size ** -0.5
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        nn.init.uniform_(self.bias_ih, -bound, bound)
        nn.init.uniform_(self.bias_hh, -bound, bound)

        self.register_buffer("ih_mask", _build_ih_mask(hidden_size))
        self.register_buffer("hh_mask", _build_hh_mask(hidden_size, near_module_sparsity))

    def masked_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weight_ih * self.ih_mask, self.weight_hh * self.hh_mask

    def step(self, z: torch.Tensor, h: torch.Tensor, masked_ih: torch.Tensor, masked_hh: torch.Tensor) -> torch.Tensor:
        return torch.tanh(F.linear(z, masked_ih, self.bias_ih) + F.linear(h, masked_hh, self.bias_hh))

    def forward(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        masked_ih, masked_hh = self.masked_weights()
        return self.step(z, h, masked_ih, masked_hh)


class ModularBidirectionalRNN(nn.Module):
    """Bidirectional RNN whose hidden layer is split into input/intermediate/output modules with restricted connectivity."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        output_mode: str = "last",
        near_module_sparsity: float = 0.1,
    ):
        super().__init__()
        if output_mode not in ("last", "all"):
            raise ValueError(f"output_mode must be 'last' or 'all', got {output_mode!r}")
        if hidden_size % 3 != 0:
            raise ValueError(f"hidden_size must be divisible by 3, got {hidden_size}")
        self.output_mode = output_mode
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.fwd_cell = ModularRNNCell(hidden_size, near_module_sparsity)
        self.bwd_cell = ModularRNNCell(hidden_size, near_module_sparsity)
        self.output_proj = nn.Linear(hidden_size * 2, output_size)
        self.register_buffer("output_mask", _build_output_mask(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
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
