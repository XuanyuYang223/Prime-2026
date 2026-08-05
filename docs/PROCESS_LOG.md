# Kinetic Flow — Development Process Log

## Project Goal

Build a small conditional flow-matching model that generates short kinetic typography videos. The project is designed as a clear, reproducible demonstration rather than a production video generator.

## Design Decisions

### 1. Start with synthetic data

An external video dataset would introduce annotation and preprocessing work unrelated to the research question. Instead, Pillow renders known words onto blank frames. Because the renderer also generates the conditions, the true glyph and trajectory are available exactly.

### 2. Model the complete video tensor

Each clip has shape `[channel, time, height, width] = [1, 8, 32, 32]`. Gaussian noise has the same shape. The learned vector field therefore transforms a complete noisy clip into a complete typography clip.

### 3. Separate shape and position

The glyph condition is a centered canonical bitmap repeated across time. The position condition is a changing Gaussian heatmap. Keeping these separate makes the planned ablation experiment meaningful: either source of information can be removed independently.

### 4. Use a straight flow path

The interpolation `x_t = (1-t)x₀ + tx₁` produces the analytic target velocity `x₁-x₀`. This keeps the mathematical explanation and implementation small enough for a final-project presentation.

### 5. Use a compact 3D U-Net

3D convolutions allow the network to see neighboring frames. Spatial downsampling reduces computation, while preserving the temporal length avoids discarding short-term motion. The default network has 619,201 trainable parameters.

### 6. Use Euler integration

Euler integration makes every sampling step visible and easy to explain. It also supports a direct experiment on the tradeoff between ODE step count, output quality, and runtime.

## Implementation Record

### Data pipeline

Implemented in `src/kinetic_flow/data.py`:

- `VideoSpec` centralizes frame count, resolution, and font size.
- `render_glyph` rasterizes and fits a word into the canvas.
- `trajectory` produces horizontal, vertical, bounce, or circle centers.
- `build_video_condition` creates the clean clip, canonical glyph, and position maps.
- `KineticTypographyDataset` generates deterministic examples by index.

### Model

Implemented in `src/kinetic_flow/model.py`:

- concatenates noisy video, glyph, and position channels;
- injects continuous time through an MLP and FiLM modulation;
- uses 3D residual blocks;
- downsamples and upsamples only spatial dimensions;
- predicts a velocity tensor with the same shape as the input video.

### Flow matching and sampling

Implemented in `src/kinetic_flow/flow.py`:

- samples Gaussian source videos and uniform times;
- constructs points on the linear probability path;
- trains with mean-squared velocity error;
- integrates the learned ODE with a configurable number of Euler steps.

### User-facing scripts

- `train.py` trains the model, saves a checkpoint, and exports a preview GIF.
- `sample.py` loads a checkpoint and generates a requested word/motion GIF.
- `src/kinetic_flow/io.py` converts video tensors into animated GIFs.

## Problems Found During Verification

### Python launcher ambiguity

The default Windows `python.exe` launcher could not start in the execution environment. The actual interpreter at `C:\Users\yangx\AppData\Local\Python\bin\python.exe` worked and was used for verification.

### Unnecessary progress-bar dependency

The first end-to-end attempt stopped because `tqdm` was not installed. Since it was only used for display, the dependency was removed and training now uses simple built-in progress messages.

## Verification Record

Completed on August 4, 2026:

| Check | Result |
|---|---|
| Compile all Python sources | Passed |
| Data shape/range assertions | Passed |
| Direction of horizontal trajectory | Passed |
| Finite flow-matching loss | Passed |
| Backpropagation | Passed |
| Finite Euler output | Passed |
| One-step CPU training | Passed |
| Checkpoint creation | Passed |
| Preview GIF creation | Passed |
| Reload checkpoint and sample | Passed |

Smoke-test command:

```bash
python train.py --steps 1 --batch-size 1 --base-channels 8 \
  --frames 2 --size 16 --device cpu --output outputs/smoke.pt
```

Sampling check:

```bash
python sample.py --checkpoint outputs/smoke.pt --text FLOW \
  --motion bounce --ode-steps 2 --device cpu \
  --output outputs/smoke_sample.gif
```

The one-step output is only a software test. It must not be presented as a trained generative result.

## Next Experimental Run

1. Train the default network for 3,000 steps on GPU.
2. Save the loss curve and representative generated GIFs.
3. Generate full-condition and ablated samples from identical noise.
4. Evaluate character readability and trajectory error.
5. Compare Euler step counts while holding all other variables fixed.
6. Insert measured results into slides 19–23 of `PRIME_PRESENTATION.md`.
