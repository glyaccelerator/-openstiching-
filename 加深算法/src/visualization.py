from typing import List

import cv2
import numpy as np


def make_thumbnail_grid(images: List[np.ndarray], max_width: int = 240, pad: int = 8):
    if not images:
        return None
    thumbnails = []
    for image in images:
        scale = min(1.0, max_width / image.shape[1])
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        thumbnails.append(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA))

    canvas_h = max(thumb.shape[0] for thumb in thumbnails)
    canvas_w = sum(thumb.shape[1] for thumb in thumbnails) + pad * (len(thumbnails) - 1)
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    x_offset = 0
    for thumb in thumbnails:
        y_offset = (canvas_h - thumb.shape[0]) // 2
        canvas[y_offset : y_offset + thumb.shape[0], x_offset : x_offset + thumb.shape[1]] = thumb
        x_offset += thumb.shape[1] + pad
    return canvas
