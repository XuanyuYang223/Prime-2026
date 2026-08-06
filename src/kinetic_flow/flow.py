from __future__ import annotations

import torch


def flow_matching_loss(model, target: torch.Tensor, glyph: torch.Tensor, positions: torch.Tensor,
                       foreground_weight: float = 0.0) -> torch.Tensor:
    """Conditional flow matching on the straight optimal-transport path."""
    noise = torch.randn_like(target)
    t = torch.rand(target.shape[0], device=target.device)
    view_t = t[:, None, None, None, None]
    x_t = (1 - view_t) * noise + view_t * target
    target_velocity = target - noise
    prediction = model(x_t, t, glyph, positions)
    error = (prediction - target_velocity).square()
    if foreground_weight <= 0:
        return error.mean()
    # Target is [-1,1]. Antialiased glyph intensity becomes a soft foreground mask.
    foreground = ((target + 1) / 2).clamp(0, 1)
    weights = 1 + foreground_weight * foreground
    return (error * weights).mean() / weights.mean()


@torch.no_grad()
def euler_sample(model, glyph: torch.Tensor, positions: torch.Tensor, steps: int = 40,
                 seed: int | None = None) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    if seed is not None:
        generator = torch.Generator(device=glyph.device).manual_seed(seed)
        x = torch.randn((glyph.shape[0], 1, *glyph.shape[2:]), device=glyph.device, generator=generator)
    else:
        x = torch.randn((glyph.shape[0], 1, *glyph.shape[2:]), device=glyph.device, dtype=glyph.dtype)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((x.shape[0],), i / steps, device=x.device)
        velocity = model(x, t, glyph, positions)
        x = x + dt * velocity
    return x.clamp(-1, 1)
