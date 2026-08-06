from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from kinetic_flow.data import VideoSpec, build_character_sequence_condition
from kinetic_flow.flow import euler_sample
from kinetic_flow.io import save_gif
from kinetic_flow.model import ConditionalVideoFlow


def output_name(text: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "sample"
    return Path("outputs") / f"{slug}.gif"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one kinetic-typography GIF")
    parser.add_argument("--text", default="MAKE TEXT MOVE")
    parser.add_argument("--motion", choices=("horizontal", "vertical", "bounce", "circle"), default="bounce")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ode-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    if checkpoint.get("data_mode") != "character":
        raise ValueError("demo.py expects a character-level checkpoint")

    spec = VideoSpec(**checkpoint["spec"])
    model = ConditionalVideoFlow(
        checkpoint["base_channels"],
        condition_channels=checkpoint["condition_channels"],
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    words = tuple(args.text.split())
    if not words:
        raise ValueError("text must contain at least one word")
    _, glyphs, positions, layouts = build_character_sequence_condition(words, args.motion, spec)
    glyphs = glyphs[None].to(args.device)
    positions = positions[None].to(args.device)
    aligned = layouts[None].to(args.device) if checkpoint.get("aligned_glyph", False) else None
    video = euler_sample(model, glyphs, positions, args.ode_steps, args.seed, aligned)

    output = args.output or output_name(args.text)
    save_gif(video[0], output, scale=4, threshold=-0.5, resample="lanczos")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()

