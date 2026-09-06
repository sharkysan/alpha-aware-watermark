from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessingPreset:
    temporal_radius: int
    chunk_size: int
    motion_compensation: bool
    neighbor_length: int
    ref_stride: int
    resize_ratio: float
    fp16: bool
    roi_padding: int


PRESETS: dict[str, ProcessingPreset] = {
    "Fast": ProcessingPreset(
        temporal_radius=1,
        chunk_size=32,
        motion_compensation=False,
        neighbor_length=5,
        ref_stride=15,
        resize_ratio=0.50,
        fp16=True,
        roi_padding=24,
    ),
    "Balanced": ProcessingPreset(
        temporal_radius=2,
        chunk_size=24,
        motion_compensation=True,
        neighbor_length=8,
        ref_stride=10,
        resize_ratio=0.75,
        fp16=True,
        roi_padding=32,
    ),
    "High Quality": ProcessingPreset(
        temporal_radius=3,
        chunk_size=24,
        motion_compensation=True,
        neighbor_length=10,
        ref_stride=10,
        resize_ratio=1.0,
        fp16=True,
        roi_padding=48,
    ),
}


def preset_values(name: str) -> tuple[int, int, bool, int, int, float, bool, int]:
    preset = PRESETS.get(name, PRESETS["Balanced"])
    return (
        preset.temporal_radius,
        preset.chunk_size,
        preset.motion_compensation,
        preset.neighbor_length,
        preset.ref_stride,
        preset.resize_ratio,
        preset.fp16,
        preset.roi_padding,
    )
