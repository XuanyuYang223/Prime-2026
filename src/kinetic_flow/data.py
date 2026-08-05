from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


WORDS = ("MOVE", "FLOW", "TYPE", "FILM", "ZOOM", "PLAY", "WAVE", "TEXT")
PHRASES = ("HELLO WORLD", "KEEP MOVING", "DREAM BIG", "START NOW", "BE BOLD", "STAY CURIOUS")
SEQUENCES = (
    ("MAKE", "TEXT", "MOVE"),
    ("WORDS", "IN", "MOTION"),
    ("CREATE", "THE", "FUTURE"),
    ("TELL", "YOUR", "STORY"),
    ("THINK", "BUILD", "SHARE"),
    ("READY", "SET", "FLOW"),
    ("STAY", "CURIOUS"),
    ("DREAM", "BIG"),
)
NOUNS = ("TEXT", "WORDS", "STORIES", "IDEAS", "DREAMS", "LIGHT", "COLOR", "MOTION", "MUSIC", "FUTURE", "WORLD", "DESIGN", "CHANGE", "MOMENTS", "ENERGY", "VISION")
VERBS = ("MAKE", "MOVE", "CREATE", "BUILD", "SHARE", "TELL", "START", "FIND", "FOLLOW", "IMAGINE", "EXPLORE", "CHOOSE", "SHAPE", "DREAM")
ADJECTIVES = ("BOLD", "BRIGHT", "BIG", "NEW", "TRUE", "ALIVE", "READY", "CURIOUS", "FREE", "STRONG", "QUICK", "WILD")
EXTRA_WORDS = ("WE", "YOU", "THE", "IN", "NOW", "KEEP", "STAY", "LET", "JUMP", "FOX", "LAZY", "ZERO", "ZONE")


@dataclass(frozen=True)
class VideoSpec:
    frames: int = 8
    size: int = 32
    width: int | None = None
    glyph_size: int = 18
    transition: str = "cut"
    max_chars: int = 8

    @property
    def frame_width(self) -> int:
        return self.size if self.width is None else self.width


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.ImageFont:
    # DejaVu ships with Pillow on the common Linux/Windows wheels.
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=256)
def render_glyph(text: str, spec: VideoSpec) -> np.ndarray:
    font = _font(spec.glyph_size)
    scratch = Image.new("L", (spec.frame_width * 3, spec.size), 0)
    draw = ImageDraw.Draw(scratch)
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.text(((scratch.width - width) / 2 - box[0], (scratch.height - height) / 2 - box[1]), text, 255, font=font)
    crop = scratch.getbbox()
    if crop is None:
        return np.zeros((1, 1), dtype=np.float32)
    glyph = scratch.crop(crop)
    max_w, max_h = spec.frame_width - 4, spec.size - 4
    scale = min(max_w / glyph.width, max_h / glyph.height, 1.0)
    glyph = glyph.resize((max(1, round(glyph.width * scale)), max(1, round(glyph.height * scale))), Image.Resampling.LANCZOS)
    return np.asarray(glyph, dtype=np.float32) / 255.0


def trajectory(kind: str, frames: int, height: int, width: int, glyph_shape: tuple[int, int]) -> np.ndarray:
    gh, gw = glyph_shape
    margin_x, margin_y = gw / 2 + 1, gh / 2 + 1
    u = np.linspace(0, 1, frames, dtype=np.float32)
    if kind == "horizontal":
        x = margin_x + u * max(0, width - 2 * margin_x)
        y = np.full_like(x, height / 2)
    elif kind == "vertical":
        y = margin_y + u * max(0, height - 2 * margin_y)
        x = np.full_like(y, width / 2)
    elif kind == "bounce":
        x = margin_x + u * max(0, width - 2 * margin_x)
        y = height / 2 + np.sin(u * 2 * math.pi) * max(1, (height - gh) / 4)
    elif kind == "circle":
        radius = max(1, min(width - gw, height - gh) / 4)
        x = width / 2 + radius * np.cos(u * 2 * math.pi)
        y = height / 2 + radius * np.sin(u * 2 * math.pi)
    else:
        raise ValueError(f"Unknown motion {kind!r}")
    return np.stack((x, y), axis=1)


def build_video_condition(text: str, motion: str, spec: VideoSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target video, canonical glyph condition, and position heatmaps."""
    glyph = render_glyph(text.upper(), spec)
    width = spec.frame_width
    points = trajectory(motion, spec.frames, spec.size, width, glyph.shape)
    video = np.zeros((spec.frames, spec.size, width), dtype=np.float32)
    positions = np.zeros_like(video)
    yy, xx = np.mgrid[: spec.size, : width]
    gh, gw = glyph.shape
    for i, (cx, cy) in enumerate(points):
        x0, y0 = round(cx - gw / 2), round(cy - gh / 2)
        x0, y0 = int(np.clip(x0, 0, width - gw)), int(np.clip(y0, 0, spec.size - gh))
        video[i, y0 : y0 + gh, x0 : x0 + gw] = np.maximum(video[i, y0 : y0 + gh, x0 : x0 + gw], glyph)
        positions[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))

    canonical = np.zeros((spec.size, width), dtype=np.float32)
    x0, y0 = (width - gw) // 2, (spec.size - gh) // 2
    canonical[y0 : y0 + gh, x0 : x0 + gw] = glyph
    glyph_video = np.repeat(canonical[None], spec.frames, axis=0)
    # [C,T,H,W]; target is in the standard diffusion/flow range.
    return (
        torch.from_numpy(video[None] * 2 - 1),
        torch.from_numpy(glyph_video[None]),
        torch.from_numpy(positions[None]),
    )


def build_sequence_condition(words: tuple[str, ...] | list[str], motion: str, spec: VideoSpec):
    """Create a clip whose displayed English word changes over time."""
    if not words:
        raise ValueError("words cannot be empty")
    glyphs = [render_glyph(word.upper(), spec) for word in words]
    max_shape = (max(g.shape[0] for g in glyphs), max(g.shape[1] for g in glyphs))
    width = spec.frame_width
    points = trajectory(motion, spec.frames, spec.size, width, max_shape)
    video = np.zeros((spec.frames, spec.size, width), dtype=np.float32)
    glyph_video = np.zeros_like(video)
    positions = np.zeros_like(video)
    yy, xx = np.mgrid[: spec.size, : width]

    if spec.transition not in {"cut", "crossfade"}:
        raise ValueError("transition must be cut or crossfade")

    def place(canvas: np.ndarray, glyph: np.ndarray, cx: float, cy: float, weight: float) -> None:
        gh, gw = glyph.shape
        x0, y0 = round(cx - gw / 2), round(cy - gh / 2)
        x0, y0 = int(np.clip(x0, 0, width - gw)), int(np.clip(y0, 0, spec.size - gh))
        canvas[y0 : y0 + gh, x0 : x0 + gw] += glyph * weight

    for i, (cx, cy) in enumerate(points):
        phase = (i + 0.5) * len(glyphs) / spec.frames
        word_index = min(len(glyphs) - 1, int(phase))
        local_phase = phase - word_index
        blend = 0.0
        if spec.transition == "crossfade" and word_index < len(glyphs) - 1 and local_phase > 0.5:
            blend = min(1.0, (local_phase - 0.5) * 2)
        place(video[i], glyphs[word_index], cx, cy, 1 - blend)
        place(glyph_video[i], glyphs[word_index], width / 2, spec.size / 2, 1 - blend)
        if blend > 0:
            place(video[i], glyphs[word_index + 1], cx, cy, blend)
            place(glyph_video[i], glyphs[word_index + 1], width / 2, spec.size / 2, blend)
        np.clip(video[i], 0, 1, out=video[i])
        np.clip(glyph_video[i], 0, 1, out=glyph_video[i])
        positions[i] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 2.0**2))

    return (
        torch.from_numpy(video[None] * 2 - 1),
        torch.from_numpy(glyph_video[None]),
        torch.from_numpy(positions[None]),
    )


def build_character_sequence_condition(words: tuple[str, ...] | list[str], motion: str, spec: VideoSpec):
    """Render word sequences with a separate glyph, heatmap, and layout channel per character."""
    if not words:
        raise ValueError("words cannot be empty")
    width, height, channels = spec.frame_width, spec.size, spec.max_chars
    prepared: list[tuple[list[np.ndarray], int]] = []
    word_shapes: list[tuple[int, int]] = []
    base_spacing = max(1, spec.glyph_size // 10)
    for word in words:
        spacing = base_spacing
        chars = [render_glyph(char, spec).copy() for char in word.upper()[:channels]]
        total_width = sum(g.shape[1] for g in chars) + spacing * max(0, len(chars) - 1)
        scale = min(1.0, (width - 4) / max(1, total_width))
        if scale < 1:
            chars = [
                np.asarray(
                    Image.fromarray((g * 255).astype(np.uint8)).resize(
                        (max(1, round(g.shape[1] * scale)), max(1, round(g.shape[0] * scale))),
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.float32,
                )
                / 255.0
                for g in chars
            ]
            spacing = max(1, round(spacing * scale))
        prepared.append((chars, spacing))
        word_shapes.append((max(g.shape[0] for g in chars), sum(g.shape[1] for g in chars) + spacing * max(0, len(chars) - 1)))

    max_shape = (max(s[0] for s in word_shapes), max(s[1] for s in word_shapes))
    centers = trajectory(motion, spec.frames, height, width, max_shape)
    video = np.zeros((spec.frames, height, width), dtype=np.float32)
    glyphs = np.zeros((channels, spec.frames, height, width), dtype=np.float32)
    positions = np.zeros_like(glyphs)
    layouts = np.zeros_like(glyphs)
    yy, xx = np.mgrid[:height, :width]

    def place(canvas: np.ndarray, glyph: np.ndarray, cx: float, cy: float) -> None:
        gh, gw = glyph.shape
        x0, y0 = int(round(cx - gw / 2)), int(round(cy - gh / 2))
        x0, y0 = int(np.clip(x0, 0, width - gw)), int(np.clip(y0, 0, height - gh))
        canvas[y0 : y0 + gh, x0 : x0 + gw] = np.maximum(canvas[y0 : y0 + gh, x0 : x0 + gw], glyph)

    for frame, (base_x, base_y) in enumerate(centers):
        word_index = min(len(prepared) - 1, frame * len(prepared) // spec.frames)
        chars, spacing = prepared[word_index]
        total_width = sum(g.shape[1] for g in chars) + spacing * max(0, len(chars) - 1)
        cursor = base_x - total_width / 2
        for char_index, glyph in enumerate(chars):
            gh, gw = glyph.shape
            char_x = cursor + gw / 2
            # A phase offset gives every character its own motion while preserving the word layout.
            phase = 2 * math.pi * frame / spec.frames + char_index * math.pi / 3
            char_y = base_y + math.sin(phase) * max(1.0, (height - max_shape[0]) / 12)
            place(video[frame], glyph, char_x, char_y)
            place(layouts[char_index, frame], glyph, char_x, char_y)
            place(glyphs[char_index, frame], glyph, width / 2, height / 2)
            positions[char_index, frame] = np.exp(-((xx - char_x) ** 2 + (yy - char_y) ** 2) / (2 * 2.0**2))
            cursor += gw + spacing

    return (
        torch.from_numpy(video[None] * 2 - 1),
        torch.from_numpy(glyphs),
        torch.from_numpy(positions),
        torch.from_numpy(layouts),
    )


class KineticTypographyDataset(Dataset):
    """Infinite-style deterministic synthetic dataset; no video files required."""

    motions = ("horizontal", "vertical", "bounce", "circle")

    def __init__(self, length: int = 10_000, spec: VideoSpec = VideoSpec(), seed: int = 0, mode: str = "word"):
        if mode not in {"word", "phrase", "sequence", "sentence", "character"}:
            raise ValueError("mode must be word, phrase, sequence, sentence, or character")
        self.length, self.spec, self.seed, self.mode = length, spec, seed, mode

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        rng = random.Random(self.seed + index)
        motion = rng.choice(self.motions)
        if self.mode in {"sequence", "sentence", "character"}:
            if self.mode in {"sentence", "character"}:
                template = rng.randrange(6)
                if template == 0:
                    words = (rng.choice(VERBS), rng.choice(NOUNS))
                elif template == 1:
                    words = (rng.choice(VERBS), "THE", rng.choice(NOUNS))
                elif template == 2:
                    words = ("WE", rng.choice(VERBS), rng.choice(NOUNS))
                elif template == 3:
                    words = ("KEEP", rng.choice(ADJECTIVES))
                elif template == 4:
                    words = ("STAY", rng.choice(ADJECTIVES))
                else:
                    vocabulary = NOUNS + VERBS + ADJECTIVES + EXTRA_WORDS
                    words = tuple(rng.sample(vocabulary, rng.choice((2, 3))))
            else:
                words = rng.choice(SEQUENCES)
            text = " ".join(words)
            if self.mode == "character":
                video, glyph, positions, aligned = build_character_sequence_condition(words, motion, self.spec)
            else:
                video, glyph, positions = build_sequence_condition(words, motion, self.spec)
        else:
            text = rng.choice(PHRASES if self.mode == "phrase" else WORDS)
            video, glyph, positions = build_video_condition(text, motion, self.spec)
        item = {"video": video, "glyph": glyph, "positions": positions, "text": text, "motion": motion}
        if self.mode == "character":
            item["aligned"] = aligned
        return item
