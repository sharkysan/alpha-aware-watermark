from watermark_remover.presets import PRESETS, preset_values


def test_fast_preset_prioritizes_throughput() -> None:
    fast = PRESETS["Fast"]
    balanced = PRESETS["Balanced"]
    assert fast.resize_ratio < balanced.resize_ratio
    assert fast.neighbor_length < balanced.neighbor_length
    assert fast.motion_compensation is False
    assert fast.fp16 is True


def test_high_quality_preset_preserves_detail() -> None:
    quality = PRESETS["High Quality"]
    assert quality.resize_ratio == 1.0
    assert quality.motion_compensation is True
    assert quality.roi_padding >= PRESETS["Balanced"].roi_padding


def test_unknown_preset_falls_back_to_balanced() -> None:
    assert preset_values("unknown") == preset_values("Balanced")
