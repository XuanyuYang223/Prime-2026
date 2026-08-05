from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        groups = min(8, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.time = nn.Linear(time_dim, channels * 2)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time(temb).chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return x + self.conv2(F.silu(h))


class ConditionalVideoFlow(nn.Module):
    """Compact 3D U-Net vector field conditioned on glyph and position maps."""

    def __init__(self, base_channels: int = 32, time_dim: int = 64, condition_channels: int = 2):
        super().__init__()
        self.condition_channels = condition_channels
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.stem = nn.Conv3d(1 + condition_channels, base_channels, 3, padding=1)
        self.enc = ResBlock(base_channels, time_dim)
        self.down = nn.Conv3d(base_channels, base_channels * 2, 4, stride=(1, 2, 2), padding=1)
        self.mid = ResBlock(base_channels * 2, time_dim)
        self.up = nn.ConvTranspose3d(base_channels * 2, base_channels, 4, stride=(1, 2, 2), padding=1)
        self.dec = ResBlock(base_channels, time_dim)
        self.out = nn.Conv3d(base_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, glyph: torch.Tensor, positions: torch.Tensor,
                aligned_glyph: torch.Tensor | None = None) -> torch.Tensor:
        temb = self.time_mlp(t.reshape(-1, 1))
        conditions = (glyph, positions) if aligned_glyph is None else (glyph, positions, aligned_glyph)
        actual_channels = sum(condition.shape[1] for condition in conditions)
        if actual_channels != self.condition_channels:
            raise ValueError(f"model expects {self.condition_channels} condition channels, got {actual_channels}")
        h0 = self.enc(self.stem(torch.cat((x, *conditions), dim=1)), temb)
        h = self.mid(self.down(h0), temb)
        h = self.up(h)
        h = self.dec(h + h0, temb)
        return self.out(h)
