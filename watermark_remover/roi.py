from __future__ import annotations

from pathlib import Path

import cv2

from .models import Region


def find_mask_union_roi(
    masks_dir: Path,
    frame_width: int,
    frame_height: int,
    padding: int,
) -> Region | None:
    """Return the padded union of all non-zero residual mask pixels."""
    left = frame_width
    top = frame_height
    right = -1
    bottom = -1

    for mask_path in sorted(masks_dir.glob("*.png")):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read residual mask: {mask_path}")
        points = cv2.findNonZero(mask)
        if points is None:
            continue
        x, y, width, height = cv2.boundingRect(points)
        left = min(left, x)
        top = min(top, y)
        right = max(right, x + width)
        bottom = max(bottom, y + height)

    if right <= left or bottom <= top:
        return None

    return Region(
        x=max(0, left - padding),
        y=max(0, top - padding),
        width=min(frame_width, right + padding) - max(0, left - padding),
        height=min(frame_height, bottom + padding) - max(0, top - padding),
    ).clamp(frame_width, frame_height)


def crop_sequences(
    frames_dir: Path,
    masks_dir: Path,
    roi: Region,
    output_frames_dir: Path,
    output_masks_dir: Path,
) -> None:
    """Crop matching full-resolution frame and mask sequences to an ROI."""
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError("No analytic frames were found for ROI processing.")

    y2 = roi.y + roi.height
    x2 = roi.x + roi.width
    for frame_path in frame_paths:
        mask_path = masks_dir / frame_path.name
        frame = cv2.imread(str(frame_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Could not read analytic frame: {frame_path}")
        if mask is None:
            raise RuntimeError(f"Could not read residual mask: {mask_path}")
        cv2.imwrite(str(output_frames_dir / frame_path.name), frame[roi.y:y2, roi.x:x2])
        cv2.imwrite(str(output_masks_dir / frame_path.name), mask[roi.y:y2, roi.x:x2])


def composite_roi_video(
    base_frames_dir: Path,
    roi_video: Path,
    roi: Region,
    output_frames_dir: Path,
) -> int:
    """Composite ProPainter's ROI video back into full-resolution analytic frames."""
    output_frames_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(roi_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open ProPainter ROI video: {roi_video}")

    frame_paths = sorted(base_frames_dir.glob("*.png"))
    written = 0
    try:
        for frame_path in frame_paths:
            ok, roi_frame = capture.read()
            if not ok or roi_frame is None:
                raise RuntimeError(
                    "ProPainter ROI video contains fewer frames than the analytic sequence."
                )
            if roi_frame.shape[1] != roi.width or roi_frame.shape[0] != roi.height:
                roi_frame = cv2.resize(
                    roi_frame,
                    (roi.width, roi.height),
                    interpolation=cv2.INTER_LINEAR,
                )

            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Could not read analytic frame: {frame_path}")
            frame[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = roi_frame
            cv2.imwrite(str(output_frames_dir / f"{written:06d}.png"), frame)
            written += 1
    finally:
        capture.release()

    if written == 0:
        raise RuntimeError("No frames were composited from the ProPainter ROI output.")
    return written
