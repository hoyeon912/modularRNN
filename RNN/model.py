import torch
import torch.nn as nn


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SimpleRNN(nn.Module):
    """Plain bidirectional Elman RNN baseline: dense hidden layer, no module structure."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_mode: str = "last"):
        super().__init__()
        if output_mode not in ("last", "all"):
            raise ValueError(f"output_mode must be 'last' or 'all', got {output_mode!r}")
        self.output_mode = output_mode

        self.input_proj = nn.Linear(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        outputs, hidden = self.rnn(x)
        # outputs: (batch, seq_len, hidden_size*2)
        # hidden: (2, batch, hidden_size) -- num_layers=1, bidirectional=True -> [forward, backward]

        if self.output_mode == "all":
            return self.output_proj(outputs)

        forward_last = hidden[0]
        backward_last = hidden[1]
        combined = torch.cat([forward_last, backward_last], dim=1)
        return self.output_proj(combined)
