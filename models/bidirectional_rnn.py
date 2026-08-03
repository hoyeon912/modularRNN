import torch
import torch.nn as nn

from models.common import get_device

__all__ = ["BidirectionalRNN", "get_device"]


class BidirectionalRNN(nn.Module):
    """Hand-rolled bidirectional Elman RNN: explicit forward/backward loops over nn.RNNCell."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_mode: str = "last"):
        super().__init__()
        if output_mode not in ("last", "all"):
            raise ValueError(f"output_mode must be 'last' or 'all', got {output_mode!r}")
        self.output_mode = output_mode
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.fwd_cell = nn.RNNCell(hidden_size, hidden_size)
        self.bwd_cell = nn.RNNCell(hidden_size, hidden_size)
        self.output_proj = nn.Linear(hidden_size * 2, output_size)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        batch_size, seq_len, _ = x.shape

        h_fwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        fwd_states = []
        for t in range(seq_len):
            h_fwd = self.fwd_cell(x[:, t, :], h_fwd)
            fwd_states.append(h_fwd)

        h_bwd = torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
        bwd_states = [None] * seq_len
        for t in reversed(range(seq_len)):
            h_bwd = self.bwd_cell(x[:, t, :], h_bwd)
            bwd_states[t] = h_bwd

        outputs = torch.stack(
            [torch.cat([fwd_states[t], bwd_states[t]], dim=1) for t in range(seq_len)],
            dim=1,
        )

        if self.output_mode == "all":
            result = self.output_proj(outputs)
        else:
            combined = torch.cat([fwd_states[-1], bwd_states[0]], dim=1)
            result = self.output_proj(combined)

        if return_hidden:
            return result, outputs
        return result
