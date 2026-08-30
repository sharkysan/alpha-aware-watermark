from __future__ import annotations

import cv2
import numpy as np


def temporal_median(frames: list[np.ndarray]) -> np.ndarray:
    if not frames:
        raise ValueError("At least one frame is required")
    arr = np.stack(frames, axis=0).astype(np.float32)
    return np.median(arr, axis=0)


def estimate_alpha_and_foreground(
    observed: np.ndarray,
    background: np.ndarray,
    support_mask: np.ndarray,
    alpha_floor: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate a soft alpha matte and watermark foreground colour.

    This is an underdetermined inverse problem. The implementation uses a
    robust residual-magnitude heuristic within the supplied support mask.
    """
    obs = observed.astype(np.float32) / 255.0
    bg = background.astype(np.float32) / 255.0
    residual = obs - bg

    magnitude = np.linalg.norm(residual, axis=2) / np.sqrt(3.0)
    support = support_mask > 0
    values = magnitude[support]

    if values.size:
        lo = float(np.percentile(values, 20))
        hi = float(np.percentile(values, 95))
    else:
        lo, hi = 0.0, 1.0

    denom = max(1e-6, hi - lo)
    alpha = np.clip((magnitude - lo) / denom, 0.0, 1.0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 1.2)
    alpha *= support_mask.astype(np.float32) / 255.0
    alpha[alpha < alpha_floor] = 0.0

    a = np.clip(alpha[..., None], 1e-3, 1.0)
    foreground = (obs - (1.0 - a) * bg) / a
    foreground = np.clip(foreground, 0.0, 1.0)

    return alpha, foreground


def inverse_composite(
    observed: np.ndarray,
    alpha: np.ndarray,
    foreground: np.ndarray,
    stable_alpha_max: float = 0.72,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover background B from O = aF + (1-a)B.

    Confidence decreases as alpha approaches the unstable/opaque range.
    """
    obs = observed.astype(np.float32) / 255.0
    a = np.clip(alpha[..., None], 0.0, 0.995)
    denom = np.maximum(1e-3, 1.0 - a)

    background = np.clip((obs - a * foreground) / denom, 0.0, 1.0)
    confidence = np.clip(
        (stable_alpha_max - alpha) / max(stable_alpha_max, 1e-6),
        0.0,
        1.0,
    )

    return (background * 255.0).astype(np.uint8), confidence


def residual_inpaint_mask(
    alpha: np.ndarray,
    confidence: np.ndarray,
    support_mask: np.ndarray,
    alpha_threshold: float,
    confidence_min: float,
    dilate: int,
) -> np.ndarray:
    residual = (
        ((alpha >= alpha_threshold) | (confidence < confidence_min))
        & (support_mask > 0)
    )
    mask = residual.astype(np.uint8) * 255

    if dilate > 0:
        kernel_size = dilate * 2 + 1
        kernel: np.ndarray = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def blend_analytic(
    observed: np.ndarray,
    recovered: np.ndarray,
    alpha: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    weight = (alpha * confidence)[..., None]
    out = (
        observed.astype(np.float32) * (1.0 - weight)
        + recovered.astype(np.float32) * weight
    )
    return np.clip(out, 0, 255).astype(np.uint8)
