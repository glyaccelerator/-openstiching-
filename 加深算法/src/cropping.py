from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class CropResult:
    original: np.ndarray
    cropped: np.ndarray
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    black_ratio_before: float
    black_ratio_after: float


def compute_non_black_mask(image: np.ndarray, threshold: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def black_area_ratio(image: np.ndarray, threshold: int = 8) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray <= threshold))


def auto_crop_black_border(image: np.ndarray) -> CropResult:
    mask = compute_non_black_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    before_ratio = black_area_ratio(image)
    if not contours:
        h, w = image.shape[:2]
        return CropResult(image, image, mask, (0, 0, w, h), before_ratio, before_ratio)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = image[y : y + h, x : x + w]
    return CropResult(image, cropped, mask, (x, y, w, h), before_ratio, black_area_ratio(cropped))
