# Conditional Flow Matching for Kinetic Typography Videos

Xuanyu Yang<br>
PRIME Final Project

> Slide-ready project record. Each horizontal rule begins a new slide. The structure follows the supplied PRIME reference presentation, while all content describes this project.

---

## 1. Motivation

Current video generators often fail to produce readable text.

- Semantic text embeddings describe what a word means.
- They do not precisely describe the shape of every character.
- Small geometric errors accumulate across video frames.
- The result may flicker, deform, or become unreadable.

**Idea:** give the model explicit glyph shapes and explicit frame-by-frame positions.

---

## 2. Research Question

**Can a small conditional flow-matching model generate readable moving text when glyph shape and spatial trajectory are provided explicitly?**

Supporting questions:

1. Can one vector field generate the full video tensor?
2. Does glyph conditioning preserve character shape?
3. Does position conditioning control motion across frames?
4. How sensitive is generation to the number of Euler steps?

---

## 3. Project Overview

Tools: Python, PyTorch, NumPy, and Pillow

1. Generate synthetic kinetic typography videos.
2. Represent each clip as one high-dimensional tensor.
3. Define a linear conditional probability path.
4. Train a conditional 3D velocity network.
5. Start from a Gaussian-noise video.
6. Solve the learned ODE using Euler steps.
7. Compare full conditioning with ablations.

---

## 4. Synthetic Dataset

No external video dataset is required.

- Resolution: `32 × 32` grayscale
- Frames per clip: `T = 8`
- Tensor shape: `1 × 8 × 32 × 32`
- Words: MOVE, FLOW, TYPE, FILM, ZOOM, PLAY, WAVE, TEXT
- Motions: horizontal, vertical, bounce, circle
- Samples are rendered online during training.

Advantages:

- The correct glyph and position are exactly known.
- Unlimited randomized training examples can be produced.
- Experiments are inexpensive and reproducible.

---

## 5. One Training Example

Each example contains three tensors:

| Tensor | Shape | Meaning |
|---|---:|---|
| Clean video, `x₁` | `1 × T × H × W` | Rendered moving word |
| Glyph, `g` | `1 × T × H × W` | Canonical centered word repeated over time |
| Positions, `p` | `1 × T × H × W` | One Gaussian center heatmap per frame |

The model does **not** receive the final rendered video as a condition. It must learn how to combine a canonical shape with a changing position.

---

## 6. Glyph Conditioning

The requested word is rasterized into a canonical bitmap.

- It explicitly contains character strokes and spacing.
- It remains constant across the clip.
- The bitmap is repeated along the time dimension.
- This supplies geometric information that a semantic embedding may omit.

Expected effect: improved spelling and character readability.

---

## 7. Position Conditioning

For center `(cₓ(t), cᵧ(t))`, frame `t` receives a Gaussian heatmap:

```text
p_t(x,y) = exp(-((x-cₓ(t))² + (y-cᵧ(t))²) / (2σ²))
```

The heatmap changes from frame to frame and describes the requested trajectory.

Expected effect: stable, controllable motion without asking the model to infer position from text semantics.

---

## 8. Source and Data Distributions

Define:

```text
x₀ ~ N(0, I)                  Gaussian-noise video
x₁ ~ q_data(x | glyph, path)  clean typography video
```

The generative process transforms the whole noisy video `x₀` into the whole clean video `x₁`.

Unlike frame-by-frame generation, all frames are modeled together, allowing 3D convolutions to learn temporal consistency.

---

## 9. Conditional Probability Path

Choose the straight interpolation:

```text
x_t = (1 - t)x₀ + tx₁,     t ∈ [0,1]
```

- At `t = 0`: the sample is Gaussian noise.
- At `t = 1`: the sample is a clean video.
- Between them: noise gradually becomes moving text.

This path gives a simple supervised target for flow matching.

---

## 10. Conditional Velocity

Differentiate the straight path:

```text
dx_t / dt = x₁ - x₀
```

Therefore the target conditional velocity is:

```text
u_t(x_t | x₁, x₀) = x₁ - x₀
```

For each batch, `x₀`, `x₁`, and `t` are sampled, so this velocity can be calculated directly. It has no explicit dependence on `t`, although the learned marginal vector field does.

---

## 11. Conditional Flow Matching

The network predicts:

```text
vθ(x_t, t, g, p)
```

Training loss:

```text
L_CFM = E ||vθ(x_t, t, g, p) - (x₁ - x₀)||²
```

For every batch:

1. Sample a clean video and its conditions.
2. Sample a Gaussian-noise video.
3. Sample `t ~ Uniform(0,1)`.
4. Construct `x_t`.
5. Predict and regress the velocity.

---

## 12. Neural Network Model

A compact 3D U-Net predicts one velocity for every video pixel.

```text
[x_t, glyph, positions]
        ↓ concatenate channels
3D convolution → residual block
        ↓ spatial downsample: 32×32 → 16×16
time-conditioned middle residual block
        ↑ spatial upsample: 16×16 → 32×32
skip connection → residual block → velocity
```

- Time is injected using a small MLP and FiLM scale/shift.
- Spatial dimensions are compressed.
- The temporal dimension is preserved.
- Default model size: **619,201 parameters**.

---

## 13. Why 3D Convolutions?

A normal 2D model processes one image at a time.

A 3D convolution observes:

- neighboring pixels within a frame;
- neighboring frames in time;
- how glyph edges move between frames.

This makes the output a video vector field rather than a collection of unrelated image vector fields.

---

## 14. Training Configuration

Default configuration:

| Setting | Value |
|---|---:|
| Training steps | 3,000 |
| Batch size | 16 |
| Learning rate | 0.0002 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Base channels | 32 |
| Video shape | `1 × 8 × 32 × 32` |

The data is synthesized online, so “epoch” is less meaningful than the total number of optimization steps.

---

## 15. ODE Sampling

After training, generate a video by solving:

```text
dx_t / dt = vθ(x_t, t, glyph, positions)
```

All randomness comes from the initial Gaussian video.

The glyph and position conditions remain fixed during one sampling trajectory.

---

## 16. Euler Solver

With `N` steps and `h = 1/N`:

```text
x_(k+1) = x_k + h · vθ(x_k, k/N, glyph, positions)
```

Sampling procedure:

1. Sample one Gaussian-noise video.
2. Predict its conditional velocity.
3. Move a small distance along the velocity.
4. Repeat `N` times.
5. Clamp the final tensor to the image range.
6. Convert the frames into an animated GIF.

---

## 17. Implementation Process

1. Created a deterministic synthetic dataset.
2. Implemented glyph rasterization and four trajectories.
3. Separated glyph shape from position information.
4. Implemented a time-conditioned 3D U-Net.
5. Implemented the conditional flow-matching loss.
6. Implemented Euler ODE sampling.
7. Added training, checkpointing, and GIF export.
8. Added shape, gradient, and end-to-end smoke tests.

---

## 18. Current Verification

The following pipeline checks have been completed:

- Source files compile successfully.
- Data tensors have the expected shapes and ranges.
- The horizontal position heatmap moves in the correct direction.
- Flow-matching loss is finite.
- Backpropagation succeeds.
- Euler sampling returns a finite video tensor.
- A one-step training smoke test saves a checkpoint.
- The checkpoint can be loaded by the separate sampling script.
- Training preview and requested sample GIF files are created.

**Important:** one optimization step validates the software pipeline; it is not evidence of generation quality.

---

## 19. Experiment 1 — Generated Videos

Use the same trained model to generate:

| Requested word | Motion |
|---|---|
| FLOW | horizontal |
| TYPE | vertical |
| MOVE | bounce |
| WAVE | circle |

Record for each sample:

- final animated GIF;
- selected frames at `t = 0`, `0.25`, `0.5`, `0.75`, `1`;
- whether all requested characters remain readable;
- whether motion follows the requested path.

> Result images should be inserted after the full 3,000-step run.

---

## 20. Experiment 2 — Conditioning Ablation

Generate from the same initial Gaussian noise under three settings:

1. Full glyph + position conditioning.
2. Position only: replace the glyph tensor with zeros.
3. Glyph only: replace the position tensor with zeros.

Hypotheses:

- Removing glyph information will reduce readability.
- Removing position information will reduce motion accuracy.
- Full conditioning should best satisfy both requirements.

This is the central experiment for the research question.

---

## 21. Experiment 3 — Euler Steps

Hold the trained model, condition, and initial noise constant.

Compare:

```text
N = 5, 10, 20, 40, 80
```

Measure:

- visual readability;
- difference from the `N = 160` reference output;
- sampling time;
- temporal roughness.

Expected tradeoff: more steps improve numerical accuracy but cost more inference time.

---

## 22. Experiment 4 — Motion Accuracy

For each frame:

1. Threshold the generated text mask.
2. Calculate its intensity-weighted center.
3. Compare it with the requested heatmap center.

```text
trajectory error = (1/T) Σ_t ||ĉ_t - c_t||₂
```

Report average trajectory error for each of the four motion types.

This converts “the motion looks correct” into a quantitative measurement.

---

## 23. Experiment 5 — Temporal Consistency

Measure average frame-to-frame change:

```text
temporal variation = (1/(T-1)) Σ_t mean(|x_(t+1) - x_t|)
```

Interpret carefully:

- Very high variation may indicate flicker or broken glyphs.
- Very low variation may indicate that the requested motion was ignored.
- Compare only samples with the same requested trajectory.

---

## 24. Reproducibility

Training:

```bash
python train.py --steps 3000 --batch-size 16
```

Sampling:

```bash
python sample.py --text FLOW --motion bounce --ode-steps 40 --seed 12
```

The checkpoint stores:

- learned model weights;
- video specification;
- base channel count.

A fixed seed controls the initial Gaussian video for fair comparisons.

---

## 25. Limitations

- Only `32 × 32` grayscale videos are modeled.
- The vocabulary is fixed and small.
- Training uses one bundled/default font style.
- Backgrounds are blank and motion paths are predefined.
- The model predicts all pixels directly, which scales poorly to high resolution.
- Readability is not yet measured with OCR.
- The Euler solver has finite discretization error.
- Current verification proves that the pipeline runs, not that 3,000-step training has converged.

---

## 26. Future Work

- Train with arbitrary strings and multiple fonts.
- Add color, scale, rotation, and opacity animation.
- Add natural video backgrounds.
- Replace raster glyphs with richer glyph or layout encoders.
- Use attention for variable-length text.
- Compare Euler with Heun or adaptive ODE solvers.
- Evaluate readability using OCR accuracy.
- Move to latent-space video flow matching for higher resolution.

---

## 27. Conclusion

- Built a complete conditional flow-matching video pipeline.
- Treated the entire clip as one high-dimensional tensor.
- Used explicit glyph information to represent character shape.
- Used position heatmaps to represent motion.
- Learned a time-dependent 3D velocity field.
- Generated videos by transforming Gaussian noise with Euler integration.
- Verified the full software path from data generation to GIF output.

The next milestone is the full training run and conditioning ablation study, which will test whether explicit shape and position information improves readable kinetic typography.
