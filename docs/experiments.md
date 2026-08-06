# Experiment notes

## Setup

All training clips are rendered online with Pillow. The generator selects text and one of four trajectories from a deterministic random stream based on the dataset seed and sample index. No external image or video dataset is used.

The final character experiment uses 12 frames at 64 by 128 pixels, a glyph size of 42, up to eight characters, 16 base channels, a batch size of 4, 10,000 optimizer steps, a foreground weight of 5, and 20 Euler steps at evaluation time. The condition has separate glyph, position, and aligned-layout channels for each character.

Evaluation uses eight fixed synthetic samples. MSE is measured in the model's `[-1, 1]` range. Dice is the overlap between target foreground pixels and generated pixels after selecting the best of the tested thresholds.

## Main comparison

| Model | MSE | Dice |
| --- | ---: | ---: |
| Word-level baseline | 0.009147 | 0.785849 |
| Character-level model | **0.000022** | **0.995071** |

The word baseline has one glyph channel and one position channel for the current word. The character model has independent channels for every letter, so it receives much more exact spatial information. The comparison answers whether explicit character geometry can be preserved; it is not a claim about open-domain video quality.

## Word-level ablations

| Change | Dice |
| --- | ---: |
| 48x96 baseline | 0.746000 |
| 64x128, 16 base channels | 0.777538 |
| 20,000 training steps | 0.771100 |
| 96x192 resolution | 0.761575 |
| 24 base channels | 0.754837 |
| Layout guidance 0.1 | **0.785849** |

Increasing resolution, width, or training length did not improve Dice by itself. Mild layout guidance gave the best word-level score, while stronger guidance copied condition artifacts into the sample.

## Reproduction commands

Train the character model:

```bash
python train.py \
  --data-mode character --frames 12 --size 64 --width 128 \
  --glyph-size 42 --max-chars 8 --base-channels 16 --batch-size 4 \
  --aligned-glyph --foreground-weight 5 --steps 10000 \
  --output outputs/character/model.pt
```

Evaluate it:

```bash
python evaluate.py --checkpoint outputs/character/model.pt --samples 8 --ode-steps 20
```

Generate a custom sequence from the included checkpoint:

```bash
python demo.py --text "MAKE TEXT MOVE" --motion circle
```

Words longer than eight characters require either the word-level model or a new character model trained with a larger `--max-chars` value.
