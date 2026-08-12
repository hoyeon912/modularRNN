import torch
import torch.nn as nn

from models.modular_rnn import ModularRNN

__all__ = ["CNNModularRNN"]


class CNNEncoder(nn.Module):
    """Shared-weight per-frame encoder: 3 stride-2 convs down to a flat feature vector.
    Applied identically at every timestep (weights don't depend on t), matching how a
    single camera would be processed frame-by-frame before the recurrent core."""

    def __init__(self, in_channels: int, feature_dim: int, image_size: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            flat_dim = self.conv(dummy).flatten(1).shape[1]
        self.proj = nn.Linear(flat_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x).flatten(1))


class CNNModularRNN(nn.Module):
    """CNN front-end (per-CLAUDE.md: "input can be ... images (M, N, C)") feeding a
    ModularRNN core -- the CNN only changes how each timestep's raw input is encoded into
    a feature vector; the recurrent module topology (dense-within/sparse-near/no-far) and
    output head are exactly ModularRNN's, reused unmodified."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        output_size: int,
        cnn_feature_dim: int = 32,
        output_mode: str = "all",
        near_module_sparsity: float = 0.1,
        image_size: int = 64,
    ):
        super().__init__()
        self.encoder = CNNEncoder(in_channels, cnn_feature_dim, image_size=image_size)
        self.rnn = ModularRNN(
            input_size=cnn_feature_dim,
            hidden_size=hidden_size,
            output_size=output_size,
            output_mode=output_mode,
            near_module_sparsity=near_module_sparsity,
        )

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, c, h, w = x.shape
        frames = x.reshape(batch_size * seq_len, c, h, w)
        features = self.encoder(frames).reshape(batch_size, seq_len, -1)
        return self.rnn(features, return_hidden=return_hidden)

    def init_hidden(self, batch_size: int, device, dtype=torch.float32) -> torch.Tensor:
        return self.rnn.init_hidden(batch_size, device, dtype)

    def step(self, frame_t: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """frame_t: (batch, C, H, W) single frame -> (q_t, h_next), see ModularRNN.step."""
        feat = self.encoder(frame_t)
        return self.rnn.step(feat, h)
