# Experiment notes

## Setup

All clips are rendered online with Pillow. Text and one of four trajectories come from a deterministic random stream based on the dataset seed and sample index. No external image or video dataset is used.

The main experiment uses 12 frames at 64 by 128 pixels, a glyph size of 42, up to eight characters, 16 base channels, a batch size of 4, a foreground weight of 5, and 20 Euler steps at evaluation time.

Each character has two condition channels:

```text
canonical glyph + position heatmap
```

For eight character slots, the model receives 16 condition channels. It never receives glyphs already moved into their target locations.

Evaluation uses eight fixed synthetic samples with data seed 9000 and noise seed 123. MSE is measured in the model's `[-1, 1]` range. Dice is the best overlap at thresholds `-0.6`, `-0.5`, and `-0.4`.

## Checkpoint sweep

| Training step | MSE | Dice |
| ---: | ---: | ---: |
| 1,000 | 0.037667 | 0.474000 |
| 2,000 | 0.032938 | 0.516422 |
| 3,000 | 0.030971 | 0.531150 |
| 4,000 | 0.033443 | 0.527077 |
| 5,000 | 0.031324 | 0.535185 |
| 6,000 | 0.031031 | 0.538019 |
| 7,000 | 0.031318 | 0.545025 |
| **8,000** | **0.030199** | **0.553023** |
| 9,000 | 0.030856 | 0.548857 |
| 10,000 | 0.030864 | 0.550446 |

The 8,000-step checkpoint is the main model. It learns approximate character positions and motion, but the generated letter shapes are still visibly distorted.

## Discarded aligned-layout experiment

An earlier version also supplied each character already translated into its target location. It reported MSE `0.000022` and Dice `0.995071`, but this was target leakage:

$$x_{\mathrm{target}} \approx 2\max_i(\mathrm{aligned\ layout}_i)-1.$$

Across 32 checked samples, the maximum pixel difference was only `1.49e-8`. The model could reconstruct the answer by merging the aligned channels, so that checkpoint and score are not used as the main result.

## Reproduction commands

Train and save checkpoints every 1,000 steps:

```bash
python train.py \
  --data-mode character --frames 12 --size 64 --width 128 \
  --glyph-size 42 --max-chars 8 --base-channels 16 --batch-size 4 \
  --foreground-weight 5 --steps 10000 --save-every 1000 \
  --output outputs/character/model.pt
```

Evaluate one checkpoint:

```bash
python evaluate.py \
  --checkpoint outputs/character/model_step_8000.pt \
  --samples 8 --data-seed 9000 --noise-seed 123 --ode-steps 20
```

Generate a custom sequence from the included checkpoint:

```bash
python demo.py --text "MAKE TEXT MOVE" --motion circle
```

Words longer than eight characters require a new character model trained with a larger `--max-chars` value.
