from __future__ import annotations

import cv2
import numpy as np


def align_to_reference(
    reference: np.ndarray,
    neighbor: np.ndarray,
) -> np.ndarray:
    """Warp a neighboring frame into the reference frame using dense optical flow."""
    if reference.shape != neighbor.shape:
        raise ValueError("reference and neighbor must have the same shape")

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    neighbor_gray = cv2.cvtColor(neighbor, cv2.COLOR_BGR2GRAY)

    # Flow maps each reference pixel to the corresponding location in neighbor.
    flow = cv2.calcOpticalFlowFarneback(
        reference_gray,
        neighbor_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )

    height, width = reference_gray.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    map_x = grid_x + flow[..., 0]
    map_y = grid_y + flow[..., 1]

    return cv2.remap(
        neighbor,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def aligned_temporal_median(
    reference: np.ndarray,
    neighbors: list[np.ndarray],
) -> np.ndarray:
    """Build a motion-compensated temporal median around a reference frame."""
    aligned = [reference]
    for neighbor in neighbors:
        if neighbor is reference:
            continue
        aligned.append(align_to_reference(reference, neighbor))
    stack = np.stack(aligned, axis=0).astype(np.float32)
    return np.median(stack, axis=0)
