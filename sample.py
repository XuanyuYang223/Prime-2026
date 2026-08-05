from __future__ import annotations

import argparse
from pathlib import Path

import torch

from kinetic_flow.data import VideoSpec, build_character_sequence_condition, build_sequence_condition, build_video_condition
from kinetic_flow.flow import align_glyph_to_positions, euler_sample
from kinetic_flow.io import save_gif
from kinetic_flow.model import ConditionalVideoFlow


def main():
    p = argparse.ArgumentParser(description="Generate a kinetic typography GIF with Euler ODE integration")
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/model.pt"))
    p.add_argument("--text", default="FLOW")
    p.add_argument("--sentence", help='Display a changing sequence, e.g. "MAKE TEXT MOVE"')
    p.add_argument("--motion", choices=("horizontal", "vertical", "bounce", "circle"), default="bounce")
    p.add_argument("--ode-steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--output", type=Path, default=Path("outputs/sample.gif"))
    p.add_argument("--threshold", type=float, help="Optional crisp black/white cutoff in model range [-1,1]")
    p.add_argument("--scale", type=int, default=4, help="Integer GIF display scale; preserves aspect ratio")
    p.add_argument("--resample", choices=("nearest", "lanczos"), default="nearest")
    p.add_argument("--layout-guidance", type=float, default=0.0,
                   help="Euler guidance toward aligned glyph geometry; requires an aligned checkpoint")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    spec = VideoSpec(**checkpoint["spec"])
    condition_channels = checkpoint.get("condition_channels", 2)
    model = ConditionalVideoFlow(checkpoint["base_channels"], condition_channels=condition_channels).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    if checkpoint.get("data_mode") == "character":
        character_text = args.sentence or args.text
        _, glyph, positions, aligned_from_data = build_character_sequence_condition(tuple(character_text.split()), args.motion, spec)
    elif args.sentence:
        _, glyph, positions = build_sequence_condition(tuple(args.sentence.split()), args.motion, spec)
    else:
        _, glyph, positions = build_video_condition(args.text, args.motion, spec)
    glyph, positions = glyph[None].to(args.device), positions[None].to(args.device)
    if checkpoint.get("aligned_glyph", False):
        aligned = aligned_from_data[None].to(args.device) if checkpoint.get("data_mode") == "character" else align_glyph_to_positions(glyph, positions)
    else:
        aligned = None
    generated = euler_sample(model, glyph, positions, args.ode_steps, args.seed, aligned, args.layout_guidance)
    save_gif(generated[0], args.output, scale=args.scale, threshold=args.threshold, resample=args.resample)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
