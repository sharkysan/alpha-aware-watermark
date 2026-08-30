import numpy as np

from watermark_remover.alpha import (
    blend_analytic,
    inverse_composite,
    residual_inpaint_mask,
    temporal_median,
)


def test_temporal_median_returns_middle_value():
    frames = [
        np.full((2,2,3), 10, dtype=np.uint8),
        np.full((2,2,3), 50, dtype=np.uint8),
        np.full((2,2,3), 90, dtype=np.uint8),
    ]
    result = temporal_median(frames)
    assert np.all(result == 50)


def test_inverse_composite_reconstructs_known_background():
    background = np.full((2,2,3), 80, dtype=np.uint8)
    foreground = np.ones((2,2,3), dtype=np.float32)
    alpha = np.full((2,2), 0.25, dtype=np.float32)

    observed = (
        alpha[...,None] * foreground * 255.0
        + (1-alpha[...,None]) * background
    ).astype(np.uint8)

    recovered, confidence = inverse_composite(observed, alpha, foreground)

    assert np.allclose(recovered, background, atol=2)
    assert np.all(confidence > 0)


def test_residual_mask_marks_high_alpha_and_low_confidence():
    alpha = np.array([[0.2, 0.8]], dtype=np.float32)
    confidence = np.array([[0.9, 0.9]], dtype=np.float32)
    support = np.array([[255, 255]], dtype=np.uint8)

    mask = residual_inpaint_mask(
        alpha,
        confidence,
        support,
        alpha_threshold=0.55,
        confidence_min=0.28,
        dilate=0,
    )

    assert mask[0,0] == 0
    assert mask[0,1] == 255


def test_blend_analytic_keeps_observed_when_confidence_zero():
    observed = np.full((1,1,3), 100, dtype=np.uint8)
    recovered = np.full((1,1,3), 200, dtype=np.uint8)
    alpha = np.array([[0.8]], dtype=np.float32)
    confidence = np.array([[0.0]], dtype=np.float32)

    out = blend_analytic(observed, recovered, alpha, confidence)
    assert np.array_equal(out, observed)
