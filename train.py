from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from kinetic_flow.data import KineticTypographyDataset, VideoSpec
from kinetic_flow.flow import align_glyph_to_positions, euler_sample, flow_matching_loss
from kinetic_flow.io import save_gif
from kinetic_flow.model import ConditionalVideoFlow


def parse_args():
    p = argparse.ArgumentParser(description="Train conditional flow matching for kinetic typography")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--glyph-size", type=int, default=18)
    p.add_argument("--transition", choices=("cut", "crossfade"), default="cut")
    p.add_argument("--data-mode", choices=("word", "phrase", "sequence", "sentence", "character"), default="word")
    p.add_argument("--max-chars", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--output", type=Path, default=Path("outputs/model.pt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", type=Path)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--aligned-glyph", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--foreground-weight", type=float, default=0.0,
                   help="Extra flow-loss weight on glyph pixels; useful for sparse text")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    spec = VideoSpec(frames=args.frames, size=args.size, width=args.width, glyph_size=args.glyph_size,
                     transition=args.transition, max_chars=args.max_chars)
    dataset = KineticTypographyDataset(length=max(args.steps * args.batch_size, 1024), spec=spec, seed=args.seed, mode=args.data_mode)
    loader = iter(DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=device.type == "cuda"))
    channels_per_condition = args.max_chars if args.data_mode == "character" else 1
    condition_channels = channels_per_condition * (3 if args.aligned_glyph else 2)
    model = ConditionalVideoFlow(args.base_channels, condition_channels=condition_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.lr
        start_step = checkpoint.get("step", 0)
        print(f"resumed {args.resume} at step {start_step}")
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def save_checkpoint(path: Path, step: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "spec": vars(spec), "base_channels": args.base_channels, "data_mode": args.data_mode,
                    "condition_channels": condition_channels, "aligned_glyph": args.aligned_glyph,
                    "foreground_weight": args.foreground_weight}, path)

    log_path = args.output.with_suffix(".csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if start_step == 0 or not log_path.exists():
        log_path.write_text("step,ema_loss\n", encoding="utf-8")

    model.train()
    running = 0.0
    report_every = max(1, min(100, args.steps // 20))
    for step in range(start_step + 1, args.steps + 1):
        batch = next(loader)
        target = batch["video"].to(device, non_blocking=True)
        glyph = batch["glyph"].to(device, non_blocking=True)
        positions = batch["positions"].to(device, non_blocking=True)
        if args.aligned_glyph:
            aligned = batch["aligned"].to(device, non_blocking=True) if args.data_mode == "character" else align_glyph_to_positions(glyph, positions)
        else:
            aligned = None
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss = flow_matching_loss(model, target, glyph, positions, aligned, args.foreground_weight)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running = 0.95 * running + 0.05 * loss.item() if step > 1 else loss.item()
        if step % report_every == 0 or step == args.steps:
            print(f"step {step:>6}/{args.steps}  loss {running:.4f}")
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{step},{running:.6f}\n")
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(args.output.with_name(f"{args.output.stem}_step_{step}.pt"), step)

    save_checkpoint(args.output, args.steps)
    model.eval()
    if args.aligned_glyph:
        sample_aligned = aligned[:1]
    else:
        sample_aligned = None
    sample = euler_sample(model, glyph[:1], positions[:1], steps=40, seed=args.seed, aligned_glyph=sample_aligned)
    preview = args.output.with_name("training_preview.gif")
    save_gif(sample[0], preview)
    print(f"checkpoint: {args.output}\npreview: {preview}")


if __name__ == "__main__":
    main()
