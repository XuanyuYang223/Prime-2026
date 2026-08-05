from __future__ import annotations

import torch
from torch.nn import functional as F


def flow_matching_loss(model, target: torch.Tensor, glyph: torch.Tensor, positions: torch.Tensor,
                       aligned_glyph: torch.Tensor | None = None, foreground_weight: float = 0.0) -> torch.Tensor:
    """Conditional flow matching on the straight optimal-transport path."""
    noise = torch.randn_like(target)
    t = torch.rand(target.shape[0], device=target.device)
    view_t = t[:, None, None, None, None]
    x_t = (1 - view_t) * noise + view_t * target
    target_velocity = target - noise
    prediction = model(x_t, t, glyph, positions, aligned_glyph)
    error = (prediction - target_velocity).square()
    if foreground_weight <= 0:
        return error.mean()
    # Target is [-1,1]. Antialiased glyph intensity becomes a soft foreground mask.
    foreground = ((target + 1) / 2).clamp(0, 1)
    weights = 1 + foreground_weight * foreground
    return (error * weights).mean() / weights.mean()


@torch.no_grad()
def euler_sample(model, glyph: torch.Tensor, positions: torch.Tensor, steps: int = 40, seed: int | None = None,
                 aligned_glyph: torch.Tensor | None = None, layout_guidance: float = 0.0) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    if seed is not None:
        generator = torch.Generator(device=glyph.device).manual_seed(seed)
        x = torch.randn((glyph.shape[0], 1, *glyph.shape[2:]), device=glyph.device, generator=generator)
    else:
        x = torch.randn((glyph.shape[0], 1, *glyph.shape[2:]), device=glyph.device, dtype=glyph.dtype)
    dt = 1.0 / steps
    if layout_guidance > 0 and aligned_glyph is None:
        raise ValueError("layout_guidance requires aligned_glyph")
    layout_target = aligned_glyph * 2 - 1 if aligned_glyph is not None else None
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        velocity = model(x, t, glyph, positions, aligned_glyph)
        if layout_guidance > 0:
            remaining = max(dt, 1 - i / steps)
            velocity = velocity + layout_guidance * (layout_target - x) / remaining
        x = x + dt * velocity
    return x.clamp(-1, 1)


def align_glyph_to_positions(glyph: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Translate each canonical glyph frame to the argmax of its position heatmap."""
    batch, channels, frames, height, width = glyph.shape
    flat_positions = positions[:, 0].reshape(batch, frames, -1).argmax(dim=-1)
    target_y = (flat_positions // width).float()
    target_x = (flat_positions % width).float()
    dx = target_x - (width - 1) / 2
    dy = target_y - (height - 1) / 2
    theta = torch.zeros(batch * frames, 2, 3, device=glyph.device, dtype=glyph.dtype)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = (-2 * dx / max(1, width - 1)).reshape(-1)
    theta[:, 1, 2] = (-2 * dy / max(1, height - 1)).reshape(-1)
    images = glyph.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    grid = F.affine_grid(theta, images.shape, align_corners=True)
    aligned = F.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return aligned.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
