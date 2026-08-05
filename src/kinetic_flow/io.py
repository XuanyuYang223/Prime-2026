from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_gif(video: torch.Tensor, path: str | Path, fps: int = 8, scale: int = 4,
             threshold: float | None = None, resample: str = "nearest") -> None:
    """Save [1,T,H,W] or [T,H,W] tensor in [-1,1] as an animated GIF."""
    array = video.detach().float().cpu()
    if array.ndim == 4:
        array = array[0]
    if threshold is not None:
        array = (array > threshold).byte() * 255
    else:
        array = ((array.clamp(-1, 1) + 1) * 127.5).byte()
    array = array.numpy()
    height, width = array.shape[-2:]
    filters = {"nearest": Image.Resampling.NEAREST, "lanczos": Image.Resampling.LANCZOS}
    if resample not in filters:
        raise ValueError("resample must be nearest or lanczos")
    frames = [Image.fromarray(frame, mode="L").resize((width * scale, height * scale), filters[resample]) for frame in array]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps), loop=0)
