# Kinetic Flow

A presentation-sized conditional flow-matching model that generates short kinetic typography videos. It learns a 3D velocity field that transports an entire Gaussian-noise video into readable moving text.

The key experiment is explicit conditioning. The model receives:

- a canonical bitmap of the requested word (glyph shape);
- one Gaussian center heatmap per frame (spatial trajectory);
- the noisy video and continuous flow time.

It does **not** receive the final rendered frames as conditioning. A compact 3D U-Net must learn to preserve the glyph while moving it through time.

## Method

For a clean video `x₁`, Gaussian video `x₀`, and random `t ~ U(0,1)`:

```text
xₜ = (1 - t)x₀ + tx₁
target velocity = x₁ - x₀
loss = ||vθ(xₜ, t, glyph, positions) - (x₁ - x₀)||²
```

At inference, Euler integration transforms fresh noise into a video:

```text
x ← Gaussian noise
x ← x + Δt · vθ(x, t, glyph, positions)
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

## Train and sample

Quick smoke run (checks the pipeline, but is too short for good images):

```bash
python train.py --steps 10 --batch-size 2 --base-channels 8
```

Useful GPU run:

```bash
python train.py --steps 3000 --batch-size 16
python sample.py --text FLOW --motion bounce --ode-steps 40
```

English sentences shown one word at a time:

```bash
python train.py --steps 10000 --batch-size 16 --base-channels 16 \
  --frames 8 --size 32 --width 64 --data-mode sentence \
  --output outputs/english_sentence/model.pt
python sample.py --checkpoint outputs/english_sentence/model.pt \
  --sentence "CREATE THE FUTURE" --motion bounce
```

For crisp monochrome text, preserve the video aspect ratio and threshold the gray prediction:

```bash
python sample.py --checkpoint outputs/english_sentence_aligned/model.pt \
  --sentence "HELLO NEW WORLD" --motion horizontal \
  --layout-guidance 0.1 --threshold -0.5 --scale 4 --resample lanczos \
  --output outputs/crisp.gif
```

`sentence` mode produces thousands of short combinations from verbs, nouns, and adjectives. The whole sentence is not squeezed into one tiny frame; its words are assigned to consecutive frame segments, and the glyph condition changes with them.

For smoother word changes, use 12 frames and cross-fade conditions:

```bash
python train.py --steps 10000 --batch-size 8 --base-channels 16 \
  --frames 12 --size 48 --width 96 --glyph-size 28 --data-mode sentence \
  --transition crossfade --aligned-glyph --foreground-weight 5 \
  --output outputs/english_sentence_hq/model.pt
```

`--aligned-glyph` adds a spatial layout channel computed only from the requested canonical glyph and position heatmap. It is useful when the shallow convolutional model cannot transport exact strokes from the center of a wide canvas to distant target positions.

Because most video pixels are black background, `--foreground-weight 5` prevents the MSE objective from under-emphasizing the relatively sparse character strokes.

Outputs are animated GIFs under `outputs/`. Training data is synthesized online from a small vocabulary and four trajectories, so no dataset download is needed.

### Best validated English-sentence configuration

The current best checkpoint was trained at 64×128 with a 42 px glyph. The recommended inference settings are small layout guidance, a mask threshold, and aspect-preserving Lanczos display scaling:

```bash
python sample.py --checkpoint outputs/english_sentence_128/model_best.pt \
  --sentence "MAKE YOUR STORY" --motion bounce --ode-steps 20 \
  --layout-guidance 0.1 --threshold -0.5 --scale 4 --resample lanczos \
  --output outputs/final.gif
```

Reproducible readability evaluation:

```bash
python evaluate.py --checkpoint outputs/english_sentence_128/model_best.pt \
  --samples 8 --layout-guidance 0.1 --ode-steps 20
```

### Character-level kinetic typography

The character model assigns each letter its own glyph channel, position heatmap, aligned layout, and phase-shifted trajectory. Padded channels support words up to eight characters:

```bash
python train.py --steps 10000 --batch-size 4 --base-channels 16 \
  --frames 12 --size 64 --width 128 --glyph-size 42 \
  --data-mode character --max-chars 8 --transition cut \
  --aligned-glyph --foreground-weight 5 \
  --output outputs/english_character_128/model.pt

python sample.py --checkpoint outputs/english_character_128/model_best.pt \
  --sentence "CREATE YOUR STORY" --motion circle --ode-steps 20 \
  --threshold -0.5 --scale 4 --resample lanczos \
  --output outputs/character_result.gif
```

Unlike word-level motion, the characters remain arranged as a readable word while receiving different vertical motion phases.

## Code map

- `src/kinetic_flow/data.py`: glyph rasterization, trajectories, video/condition construction
- `src/kinetic_flow/model.py`: time-conditioned 3D U-Net velocity field
- `src/kinetic_flow/flow.py`: conditional flow-matching loss and Euler solver
- `train.py`: optimization and checkpoint/preview creation
- `sample.py`: conditional video generation
- `evaluate.py`: fixed-seed MSE and thresholded glyph Dice evaluation

## Suggested presentation experiment

Train the full model, then ablate either the glyph channel or position channel at inference by replacing it with zeros. Compare (1) text readability, (2) trajectory error from the requested center, and (3) temporal consistency. This directly demonstrates why shape and position conditions are useful beyond a semantic text embedding.

This is intentionally a teaching prototype: 32×32 grayscale frames and a fixed synthetic vocabulary keep the full-video flow model inexpensive. Natural fonts, backgrounds, colors, arbitrary text, and attention-based conditioning are clear extensions rather than hidden complexity.

## Project documentation

- [`docs/PRIME_PRESENTATION.md`](docs/PRIME_PRESENTATION.md): slide-ready final-project narrative
- [`docs/PROCESS_LOG.md`](docs/PROCESS_LOG.md): implementation decisions and verification record
- [`docs/TRAINING_LOG.md`](docs/TRAINING_LOG.md): chronological English-sentence training record
