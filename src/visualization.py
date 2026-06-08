from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def _resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_thumbnail_grid(images: List[np.ndarray], max_width: int = 260, pad: int = 8) -> Optional[np.ndarray]:
    """Create a simple horizontal thumbnail collage for input images."""
    if not images:
        return None

    thumbs = []
    for image in images:
        scale = min(1.0, max_width / image.shape[1])
        width = int(round(image.shape[1] * scale))
        height = int(round(image.shape[0] * scale))
        thumbs.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))

    canvas_height = max(thumb.shape[0] for thumb in thumbs)
    canvas_width = sum(thumb.shape[1] for thumb in thumbs) + pad * (len(thumbs) - 1)
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)

    x_offset = 0
    for thumb in thumbs:
        y_offset = (canvas_height - thumb.shape[0]) // 2
        canvas[y_offset : y_offset + thumb.shape[0], x_offset : x_offset + thumb.shape[1]] = thumb
        x_offset += thumb.shape[1] + pad
    return canvas


def make_side_by_side(left: np.ndarray, right: np.ndarray, target_height: int = 420) -> np.ndarray:
    """Create side-by-side comparison image."""
    left_small = _resize_to_height(left, target_height)
    right_small = _resize_to_height(right, target_height)
    pad = 10
    height = max(left_small.shape[0], right_small.shape[0])
    width = left_small.shape[1] + right_small.shape[1] + pad
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[: left_small.shape[0], : left_small.shape[1]] = left_small
    x_offset = left_small.shape[1] + pad
    canvas[: right_small.shape[0], x_offset : x_offset + right_small.shape[1]] = right_small
    return canvas


def save_image(path: Path, image: Optional[np.ndarray]) -> None:
    """Save image if it exists."""
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
