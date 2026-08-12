import torch


def scaled_recurrent_init_(weight_hh: torch.Tensor, gain: float = 1.4) -> None:
    """Overwrite weight_hh in place with randn(H,H) * gain / sqrt(H).

    PyTorch's default RNN init (uniform(-1/sqrt(H), 1/sqrt(H))) has spectral
    radius well under 1, so the recurrent dynamics contract to a trivial,
    state-independent fixed point under TD training. gain > 1 pushes the
    spectral radius past 1 (matching ModularRNN's recurrent_gain), keeping
    the dynamics in a non-collapsing regime.
    """
    hidden_size = weight_hh.shape[0]
    with torch.no_grad():
        weight_hh.copy_(torch.randn(hidden_size, hidden_size) * gain / (hidden_size**0.5))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
