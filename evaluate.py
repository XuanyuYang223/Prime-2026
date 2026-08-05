from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kinetic_flow.data import KineticTypographyDataset, VideoSpec
from kinetic_flow.flow import align_glyph_to_positions, euler_sample
from kinetic_flow.model import ConditionalVideoFlow


def main():
    p = argparse.ArgumentParser(description="Evaluate readability on a fixed synthetic sentence set")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--data-seed", type=int, default=9000)
    p.add_argument("--noise-seed", type=int, default=123)
    p.add_argument("--ode-steps", type=int, default=60)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--thresholds", type=float, nargs="+", default=[-0.6, -0.5, -0.4])
    p.add_argument("--layout-guidance", type=float, default=0.0)
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    spec = VideoSpec(**checkpoint["spec"])
    data_mode = checkpoint.get("data_mode", "sentence")
    dataset = KineticTypographyDataset(args.samples, spec, seed=args.data_seed,
                                        mode="character" if data_mode == "character" else "sentence")
    batch = next(iter(DataLoader(dataset, batch_size=args.samples)))
    target = batch["video"]
    glyph = batch["glyph"].to(args.device)
    positions = batch["positions"].to(args.device)
    condition_channels = checkpoint.get("condition_channels", 2)
    model = ConditionalVideoFlow(checkpoint["base_channels"], condition_channels=condition_channels).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if checkpoint.get("aligned_glyph", False):
        aligned = batch["aligned"].to(args.device) if data_mode == "character" else align_glyph_to_positions(glyph, positions)
    else:
        aligned = None
    generated = euler_sample(model, glyph, positions, args.ode_steps, args.noise_seed, aligned, args.layout_guidance).cpu()
    foreground = target > -0.5

    print(f"checkpoint: {args.checkpoint}")
    print(f"samples: {args.samples}")
    print(f"mse: {torch.mean((generated - target) ** 2).item():.6f}")
    print(f"foreground_mean: {generated[foreground].mean().item():.6f}")
    best = (-1.0, None)
    for threshold in args.thresholds:
        predicted = generated > threshold
        intersection = (foreground & predicted).flatten(1).sum(1).float()
        dice = (2 * intersection / (foreground.flatten(1).sum(1) + predicted.flatten(1).sum(1))).mean().item()
        print(f"dice@{threshold:g}: {dice:.6f}")
        if dice > best[0]:
            best = (dice, threshold)
    print(f"best_threshold: {best[1]:g}")
    print(f"best_dice: {best[0]:.6f}")


if __name__ == "__main__":
    main()
