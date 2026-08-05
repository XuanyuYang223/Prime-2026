import torch

from kinetic_flow.data import (
    KineticTypographyDataset,
    VideoSpec,
    build_character_sequence_condition,
    build_sequence_condition,
    build_video_condition,
)
from kinetic_flow.flow import align_glyph_to_positions, euler_sample, flow_matching_loss
from kinetic_flow.model import ConditionalVideoFlow


def test_data_shapes_and_motion():
    spec = VideoSpec(frames=4, size=16, glyph_size=8)
    video, glyph, positions = build_video_condition("GO", "horizontal", spec)
    assert video.shape == glyph.shape == positions.shape == (1, 4, 16, 16)
    assert video.min() >= -1 and video.max() <= 1
    centers = positions[0].flatten(1).argmax(1) % spec.size
    assert centers[-1] > centers[0]


def test_loss_and_euler_are_finite():
    spec = VideoSpec(frames=2, size=8, glyph_size=5)
    target, glyph, positions = build_video_condition("A", "bounce", spec)
    model = ConditionalVideoFlow(base_channels=8, time_dim=16)
    batch = [x.unsqueeze(0) for x in (target, glyph, positions)]
    loss = flow_matching_loss(model, *batch)
    loss.backward()
    assert torch.isfinite(loss)
    result = euler_sample(model, batch[1], batch[2], steps=2, seed=1)
    assert result.shape == batch[0].shape and torch.isfinite(result).all()


def test_changing_english_sequence_and_rectangular_frames():
    spec = VideoSpec(frames=6, size=16, width=32, glyph_size=8)
    video, glyph, positions = build_sequence_condition(("MAKE", "MOVE"), "bounce", spec)
    assert video.shape == glyph.shape == positions.shape == (1, 6, 16, 32)
    assert not torch.equal(glyph[:, 1], glyph[:, -1])


def test_sentence_mode_produces_multiple_words():
    item = KineticTypographyDataset(length=1, spec=VideoSpec(frames=6, size=16, width=32, glyph_size=8), mode="sentence")[0]
    assert len(item["text"].split()) >= 2


def test_crossfade_creates_a_blended_condition():
    cut = VideoSpec(frames=12, size=16, width=32, glyph_size=8, transition="cut")
    fade = VideoSpec(frames=12, size=16, width=32, glyph_size=8, transition="crossfade")
    _, cut_glyph, _ = build_sequence_condition(("ONE", "TWO", "THREE"), "vertical", cut)
    _, fade_glyph, _ = build_sequence_condition(("ONE", "TWO", "THREE"), "vertical", fade)
    assert not torch.equal(cut_glyph, fade_glyph)


def test_aligned_glyph_moves_toward_position_peak():
    spec = VideoSpec(frames=2, size=16, width=32, glyph_size=8)
    _, glyph, positions = build_video_condition("GO", "horizontal", spec)
    aligned = align_glyph_to_positions(glyph.unsqueeze(0), positions.unsqueeze(0))
    left_mass = aligned[0, 0, 0, :, :16].sum()
    right_mass = aligned[0, 0, 0, :, 16:].sum()
    assert left_mass > right_mass


def test_foreground_weighted_loss_is_finite():
    spec = VideoSpec(frames=2, size=8, glyph_size=5)
    target, glyph, positions = build_video_condition("A", "vertical", spec)
    model = ConditionalVideoFlow(base_channels=8, time_dim=16)
    batch = [x.unsqueeze(0) for x in (target, glyph, positions)]
    loss = flow_matching_loss(model, *batch, foreground_weight=5)
    assert torch.isfinite(loss)


def test_character_conditions_have_independent_channels_and_positions():
    spec = VideoSpec(frames=6, size=24, width=48, glyph_size=12, max_chars=8)
    video, glyphs, positions, layouts = build_character_sequence_condition(("MOVE", "TYPE"), "vertical", spec)
    assert video.shape == (1, 6, 24, 48)
    assert glyphs.shape == positions.shape == layouts.shape == (8, 6, 24, 48)
    assert glyphs[0].sum() > 0 and glyphs[3].sum() > 0 and glyphs[4].sum() == 0
    first_center = positions[0, 0].flatten().argmax()
    second_center = positions[1, 0].flatten().argmax()
    assert first_center != second_center
