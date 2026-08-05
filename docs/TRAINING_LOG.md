# 英文动态句子训练记录

记录日期：2026-08-04

## 训练目标

第一阶段不直接生成包含很多字符的完整句子画面，而是模拟短视频广告常见的逐词呈现：

```text
CREATE → THE → FUTURE
```

每个视频仍然是一个完整的高维 tensor。一个句子中的单词被分配到不同时间段，glyph condition 随帧改变，position condition 控制各帧中文字的中心位置。

这种设计的理由：

- 32×64 的画面不足以同时清楚显示长句；
- 逐词切换更接近 kinetic typography；
- 可以直接研究文字切换时的可读性和时间一致性；
- 后续可以自然扩展成淡入淡出、缩放和旋转转场。

## 环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| 显存 | 12 GB |
| Framework | PyTorch 2.12.1 + CUDA 13.0 |
| Mixed precision | FP16 autocast |
| ODE solver | Euler, 40 steps for evaluation |

## 阶段 1：固定英文短句

### 数据

最初使用 8 个固定脚本，例如：

```text
MAKE TEXT MOVE
WORDS IN MOTION
CREATE THE FUTURE
STAY CURIOUS
```

每个脚本随机搭配 horizontal、vertical、bounce 或 circle 运动。

### 配置

```text
frames          8
resolution      32 × 64
batch size      16
base channels   16
training steps  10,000
optimizer       AdamW
learning rate   0.0002
```

### 过程

1. 首先尝试 12 帧、48×96、base channels 24。
2. 显存只使用约 2.2 GB，但 3D 卷积计算速度不适合作为快速探索配置。
3. 改为 8 帧、32×64、base channels 16。
4. 发现数据端反复加载字体和渲染相同单词。
5. 为字体和 glyph rasterization 增加 LRU cache。
6. 优化后 20-step benchmark 约 3.6 秒，完整训练速度足够。

### 结果

```text
step 100      EMA loss ≈ 0.381
step 3,000    EMA loss ≈ 0.107
step 5,000    EMA loss ≈ 0.090
step 8,000    EMA loss ≈ 0.053
step 10,000   EMA loss ≈ 0.049
```

`MAKE → TEXT → MOVE` 的顺序和运动已经出现，已训练词的形状基本可辨认。

但是测试 `HELLO → NEW → WORLD` 时，未见过的词形会混淆。这说明模型部分记住了固定脚本，不能据此声称支持任意句子。

### 文件

- `outputs/english_sequence/model.pt`
- `outputs/english_sequence/make_text_move_10k.gif`
- `outputs/english_sequence/hello_new_world_unseen.gif`
- `outputs/english_sequence/contact_sheet.png`

## 阶段 2：随机英文句子组合

### 数据改进

增加 `sentence` 数据模式，不再只采样 8 个固定脚本。它从动词、名词、形容词和多种模板生成数千种组合，例如：

```text
VERB + NOUN
VERB + THE + NOUN
WE + VERB + NOUN
KEEP + ADJECTIVE
STAY + ADJECTIVE
```

词表还加入 `QUICK`、`JUMP`、`FOX`、`LAZY`、`ZERO` 等词，以覆盖更多英文字母形状。

### 命令

```bash
python train.py --steps 10000 --batch-size 16 --base-channels 16 \
  --frames 8 --size 32 --width 64 --data-mode sentence --device cuda \
  --output outputs/english_sentence/model.pt --save-every 1000
```

### 结果

```text
step 100      EMA loss ≈ 0.412
step 1,000    EMA loss ≈ 0.177
step 3,000    EMA loss ≈ 0.118
step 5,000    EMA loss ≈ 0.098
step 8,000    EMA loss ≈ 0.081
step 10,000   EMA loss ≈ 0.075
```

随机句子版的最终 loss 高于固定句子版，这是合理的：它必须学习更多词形和组合，而不是记忆 8 个脚本。

观察结果：

- `CREATE → THE → FUTURE` 能按时间段切换；
- 较短的词比长词清楚；
- 未在词表中的 `HELLO` 已能形成近似结构，但仍存在断笔和形状混淆；
- position condition 能控制整体运动；
- 8 帧中放置 3 个词时，每个词只有 2–3 帧，转场偏突然。

### 文件

- `outputs/english_sentence/model.pt`
- `outputs/english_sentence/create_the_future.gif`
- `outputs/english_sentence/hello_new_world.gif`
- `outputs/english_sentence/contact_sheet.png`
- 每 1,000 steps 的 checkpoint 位于同一目录。

## 当前结论

已经验证：

1. 一个 flow-matching 模型可以把整个 noise video tensor 转换成逐词变化的视频。
2. 逐帧 glyph condition 可以表达句子中的文字变化。
3. position heatmap 可以独立表达运动轨迹。
4. 增加训练文本多样性能够减少固定句子记忆问题。

尚未验证：

1. 任意英文句子都能保持正确拼写。
2. 高分辨率下长词能够稳定生成。
3. glyph 和 position condition 相比无条件模型的定量提升。

## 下一轮建议

优先顺序：

1. 将帧数提高到 12，让每个词至少保持 4 帧。
2. 加入单词之间的 cross-fade、scale 或 slide transition，而不是瞬间替换。
3. 使用多字体和随机字距，提高字形泛化。
4. 在 48×96 分辨率上进行较长训练。
5. 增加 glyph ablation、position ablation 和相同 seed 对照。
6. 使用 OCR 或字符模板匹配测量可读性。

## 后续自动记录

从下一次训练开始，`train.py` 会在 checkpoint 旁自动写入同名 CSV：

```text
outputs/english_sentence/model.csv
```

列为：

```text
step,ema_loss
```

断点续训时会继续追加，方便绘制 loss curve 并保留实验记录。

## 阶段 3：12 帧、Cross-fade 与空间对齐（完成）

实现了两项扩展：

- 帧数从 8 增加到 12，使三个词各自拥有约 4 帧；
- `VideoSpec.transition="crossfade"`，在时间段边界混合前后两个 glyph 与目标文字。

第一轮 48×96、5,000-step 训练的软件流程和 loss 均正常，最低 EMA loss 约为 `0.054`。多帧检查随后发现画布增大后仍使用默认 18 px 字体，导致文字只占画面很小一部分，不能公平评估高分辨率收益。

因此新增 `--glyph-size` 参数，并将正确的高分辨率配置设为：

```text
frames          12
resolution      48 × 96
glyph size      28
transition      crossfade
batch size      8
base channels   16
```

这一配置从已训练的 5,000-step 权重继续微调，使模型保留运动与转场知识，同时适应更大的 glyph。微调到 10,000 steps 后，loss 降到约 `0.0425`，但精确笔画在远离画布中心时仍不够稳定。

### 空间对齐问题

原因不是单纯的训练不足：canonical glyph 位于画布中央，而 horizontal/bounce 轨迹会把目标文字放到画布边缘。浅层 3D U-Net 主要使用局部卷积，难以把完整字形搬运几十个像素。

为此增加 aligned glyph layout：

1. 从 position heatmap 找到每帧目标中心；
2. 使用 differentiable grid sampling 将 canonical glyph 平移到该中心；
3. 将结果作为第三个条件通道；
4. 原始 glyph 与 position heatmap 仍分别保留。

aligned layout 完全由用户提供的 glyph 和 position 计算，不使用未知的生成结果。

### Aligned 模型配置

```bash
python train.py --steps 10000 --batch-size 8 --base-channels 16 \
  --frames 12 --size 48 --width 96 --glyph-size 28 \
  --data-mode sentence --transition crossfade --aligned-glyph \
  --device cuda --output outputs/english_sentence_aligned/model.pt
```

### Aligned 模型结果

```text
step 100      EMA loss ≈ 0.2740
step 1,000    EMA loss ≈ 0.0658
step 3,000    EMA loss ≈ 0.0392
step 5,000    EMA loss ≈ 0.0307
step 7,200    EMA loss ≈ 0.0204
step 9,100    EMA loss ≈ 0.0177
step 10,000   EMA loss ≈ 0.0195
```

观察：

- 3,000 steps 时已经比未对齐 10,000-step 模型更清楚；
- `CREATE → THE → FUTURE` 能保持拼写和 bounce 运动；
- 未在训练词表中的 `HELLO → NEW → WORLD` 也保持了正确的主要字母结构；
- cross-fade 帧同时包含前后两个词，稳定帧则保持单词清晰；
- 从 3,000 继续到 10,000 steps 后，笔画附近的残余噪点进一步减少。

推荐使用的最终文件：

- `outputs/english_sentence_aligned/model.pt`
- `outputs/english_sentence_aligned/model.csv`
- `outputs/english_sentence_aligned/create_the_future_10k.gif`
- `outputs/english_sentence_aligned/hello_new_world_10k.gif`
- `outputs/english_sentence_aligned/contact_sheet_10k.png`

这一阶段的关键结论是：提供 glyph 和 position 本身还不够，二者必须以模型容易利用的方式进行空间对齐。aligned glyph 显著提升了收敛速度和未见单词可读性。

## 阶段 4：模糊问题与前景加权

虽然 aligned 10k 模型的 loss 很低，但原始灰度 GIF 仍显得模糊。定量检查 `HELLO NEW WORLD`：

```text
background prediction mean   -0.983
glyph foreground mean         0.154
whole-video MSE               0.0166
```

背景接近理想值 `-1`，但字符笔画远低于理想白色 `+1`。因为绝大多数像素是黑背景，普通 MSE 会掩盖稀疏前景的错误。

增加 soft foreground-weighted flow loss：

```text
foreground = clamp((target + 1) / 2, 0, 1)
weight = 1 + 5 × foreground
loss = mean(weight × velocity_error²) / mean(weight)
```

从 aligned 10k checkpoint 微调到 12k 后：

```text
glyph foreground mean         0.208   (+35%)
background prediction mean   -0.971
whole-video MSE               0.0202
```

笔画更亮，但连续灰度 flow 输出仍保留抗锯齿和不确定边缘。对于当前黑底白字任务，将输出解释为文字 mask 并使用 `threshold=-0.5` 能产生清楚的硬边结果。

同时修复了 GIF 导出器：旧版本把所有视频强制 resize 到 256×256，使 2:1 画面被纵向拉伸；新版本使用整数倍放大并保持原始宽高比。

推荐的 crisp 采样：

```bash
python sample.py --checkpoint outputs/english_sentence_weighted/model.pt \
  --sentence "HELLO NEW WORLD" --motion horizontal --ode-steps 60 \
  --threshold -0.5 --scale 4 \
  --output outputs/english_sentence_weighted/hello_new_world_crisp.gif
```

文件：

- `outputs/english_sentence_weighted/model.pt`
- `outputs/english_sentence_weighted/model.csv`
- `outputs/english_sentence_weighted/hello_new_world_12k.gif`（原始灰度）
- `outputs/english_sentence_weighted/hello_new_world_crisp.gif`（二值清晰版）
- `outputs/english_sentence_weighted/create_the_future_crisp.gif`

研究结论需要区分两件事：模型已经正确生成字符结构与运动，但 48×96 连续灰度输出不是广告级文字渲染。若后续目标是平滑且高质量的边缘，应该提高原生分辨率，或让 flow 模型生成运动/样式参数并由矢量 glyph renderer 产生最终文字。

## 阶段 5：分辨率、容量与采样 Guidance 消融

### 64×128 基线

将原生分辨率提高到 64×128、glyph size 提高到 42 px，其他设置保持：

```text
frames              12
base channels       16
batch size          4
aligned glyph       yes
foreground weight   5
training steps      10,000
```

固定验证集（8 个随机句子、data seed 9000、noise seed 123、Euler 60 steps）：

```text
MSE                 0.010151
best threshold      -0.5
best Dice           0.777538
```

相比 48×96 模型：

```text
48×96 Dice          0.746 (单条 HELLO 测试)
64×128 Dice         0.836 (同一 HELLO 测试)
```

在随机 8 句上扫描 `-0.8` 到 `0.2`，`-0.5` 也是平均 Dice 最佳阈值，不是针对一条 GIF 人工挑选。

### 继续训练到 20k

10k 后将 learning rate 从 `2e-4` 降到 `1e-4` 并训练到 20k。结果轻微退化：

```text
10k Dice            0.777538
20k Dice            0.771100
10k MSE             0.010151
20k MSE             0.011310
```

因此 early stopping 选择 10k，并复制为：

```text
outputs/english_sentence_128/model_best.pt
```

训练脚本也修复了 resume 时 optimizer state 覆盖新 learning rate 的问题。

### 96×192 消融

训练 96×192、glyph size 56、base channels 16 到 10k：

```text
MSE                 0.005207
best Dice           0.761575
```

虽然总体灰度误差更低，但 threshold 后的细笔画覆盖更差，实际动画更容易出现断笔。因此“更高分辨率”没有自动成为更好的文字模型。

### Base channels 24 消融

在 64×128 上将 base channels 从 16 增加到 24，并训练 10k：

```text
base 16 Dice        0.777538
base 24 Dice        0.754837
```

更宽模型没有提升验证可读性，因此保留 base 16。

### Layout guidance sweep

Euler sampling 中加入：

```text
v_guided = v_model + s × (aligned_layout - x_t) / (1 - t)
```

固定验证结果：

```text
s = 0.00     Dice 0.777538
s = 0.10     Dice 0.782788
s = 0.25     Dice 0.761360
s = 0.50     Dice 0.706862
s = 1.00     Dice 0.649697
```

小 guidance 有帮助，过强 guidance 会因亚像素对齐误差破坏细笔画。推荐 `s = 0.1`。

### 当前最佳完整配置

```text
checkpoint          outputs/english_sentence_128/model_best.pt
native resolution   64 × 128
frames              12
glyph size          42
base channels       16
Euler steps         20 for crisp masks; 60 for smoother grayscale
layout guidance     0.1
threshold           -0.5
display resampling  Lanczos
```

最终示例：

- `outputs/english_sentence_128/final_create_the_future.gif`
- `outputs/english_sentence_128/final_make_your_story.gif`
- `outputs/english_sentence_128/final_hello_new_world.gif`

新增 `evaluate.py` 用固定数据 seed、noise seed、MSE 与 threshold Dice 比较 checkpoint，后续选择模型不再依赖单张视觉印象。

### Euler step sweep

使用最佳 checkpoint、layout guidance `0.1` 和相同固定验证集：

```text
20 steps      MSE 0.009147     Dice 0.785849
40 steps      MSE 0.009014     Dice 0.784333
60 steps      MSE 0.008966     Dice 0.782788
100 steps     MSE 0.008916     Dice 0.781252
```

更多 Euler steps 略微改善连续灰度 MSE，但 threshold 后的字形 Dice 略降。最终二值文字推荐 20 steps；需要研究连续灰度轨迹时可使用 60–100 steps。

## 阶段 6：Character-level Conditioning

旧版本为每个正在显示的单词提供一个整体 glyph 和一个中心轨迹。字符版本改为最多 8 个独立通道：

```text
character i
├─ canonical glyph channel i
├─ position heatmap channel i
└─ aligned layout channel i
```

因此 aligned 模型总共接收 `8 × 3 = 24` 个条件通道。长度不足 8 的单词使用全零 padding channels。

一个单词仍保持正常的水平排版，但每个字符的垂直位置加入不同相位：

```text
y_i(t) = y_base(t) + A sin(2πt/T + iπ/3)
```

这使 `M/A/K/E` 可以分别上下运动，而不是整个 `MAKE` 刚性平移。

### 配置

```text
resolution          64 × 128
frames              12
glyph size          42
max characters      8
base channels       16
condition channels  24
batch size          4
foreground weight   5
Euler steps         20
threshold           -0.5
```

### 固定验证结果

```text
3k checkpoint       MSE 0.000120     Dice 0.990236
5k checkpoint       MSE 0.000114     Dice 0.995504
10k checkpoint      MSE 0.000017     Dice 0.995737
```

10k 是固定验证集最佳，固化为：

```text
outputs/english_character_128/model_best.pt
```

### 最终示例

- `outputs/english_character_128/final_make_text_move.gif`
- `outputs/english_character_128/final_hello_new_world.gif`
- `outputs/english_character_128/final_create_your_story.gif`
- `outputs/english_character_128/contact_sheet.png`

与 word-level 最佳 Dice `0.785849` 相比，character-level 达到 `0.995737`。需要谨慎解释：字符模型获得了更精确的逐字符 aligned layout，因此任务本身也更容易；提升不能全部归因于网络结构。不过它确实实现了原研究目标中更严格的“character shape + character position”条件。
