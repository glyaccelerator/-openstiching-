from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    resize_width: int = 0
    gamma: float = 1.0
    clahe: bool = False

    def method_name(self) -> str:
        parts = []
        if self.resize_width and self.resize_width > 0:
            parts.append(f"resize_{self.resize_width}")
        if abs(self.gamma - 1.0) > 1e-6:
            parts.append(f"gamma_{self.gamma:g}")
        if self.clahe:
            parts.append("clahe")
        return "+".join(parts) if parts else "none"


def resize_by_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image while preserving aspect ratio."""
    if target_width <= 0 or image.shape[1] <= target_width:
        return image
    scale = target_width / image.shape[1]
    target_height = int(round(image.shape[0] * scale))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Normalize luminance channel to improve global brightness consistency."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction. gamma=1.0 keeps the image unchanged."""
    if gamma <= 0 or abs(gamma - 1.0) <= 1e-6:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array([(idx / 255.0) ** inv_gamma * 255 for idx in range(256)]).astype(
        "uint8"
    )
    return cv2.LUT(image, table)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE on luminance channel to enhance local contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge([l_channel, a_channel, b_channel])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_image(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Run the configured preprocessing pipeline for one image."""
    result = resize_by_width(image, config.resize_width)
    result = normalize_brightness(result)
    result = apply_gamma(result, config.gamma)
    if config.clahe:
        result = apply_clahe(result)
    return result


def preprocess_images(images: List[np.ndarray], config: PreprocessConfig) -> List[np.ndarray]:
    """Preprocess all images in a dataset."""
    return [preprocess_image(image, config) for image in images]
