from typing import List

import cv2
import numpy as np


def match_mean_std_luminance(images: List[np.ndarray]) -> List[np.ndarray]:
    """Simple exposure compensation in LAB luminance space."""
    if not images:
        return []

    reference_lab = cv2.cvtColor(images[0], cv2.COLOR_BGR2LAB)
    reference_l = reference_lab[:, :, 0].astype(np.float32)
    ref_mean, ref_std = cv2.meanStdDev(reference_l)
    ref_mean = float(ref_mean[0][0])
    ref_std = max(float(ref_std[0][0]), 1.0)

    compensated = []
    for image in images:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_float = l_channel.astype(np.float32)
        mean, std = cv2.meanStdDev(l_float)
        mean = float(mean[0][0])
        std = max(float(std[0][0]), 1.0)
        adjusted = (l_float - mean) * (ref_std / std) + ref_mean
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        compensated.append(
            cv2.cvtColor(cv2.merge([adjusted, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
        )
    return compensated
