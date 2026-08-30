import cv2
import numpy as np

from watermark_remover.motion import align_to_reference, aligned_temporal_median


def _textured_frame() -> np.ndarray:
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(frame, (18, 20), (50, 55), (180, 80, 220), -1)
    cv2.circle(frame, (68, 64), 12, (30, 220, 100), -1)
    cv2.line(frame, (10, 80), (85, 15), (240, 240, 240), 3)
    return frame


def test_optical_flow_alignment_reduces_translation_error():
    reference = _textured_frame()
    transform = np.float32([[1, 0, 3], [0, 1, 2]])
    shifted = cv2.warpAffine(
        reference,
        transform,
        (reference.shape[1], reference.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )

    aligned = align_to_reference(reference, shifted)
    crop = np.s_[8:-8, 8:-8]
    raw_error = np.mean(np.abs(reference[crop].astype(float) - shifted[crop].astype(float)))
    aligned_error = np.mean(np.abs(reference[crop].astype(float) - aligned[crop].astype(float)))

    assert aligned_error < raw_error


def test_aligned_temporal_median_preserves_shape():
    reference = _textured_frame()
    result = aligned_temporal_median(reference, [reference.copy()])
    assert result.shape == reference.shape
