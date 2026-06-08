import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from stitching import Stitcher

from .cropping import auto_crop_black_border
from .io_utils import save_image


@dataclass
class StitchRunResult:
    stitch_success: bool
    runtime_seconds: float
    panorama: Optional[np.ndarray]
    notes: str = ""


def stitch_with_feature_algorithm(
    images,
    feature_algo: str,
    output_path: Path,
    crop_enabled: bool = False,
) -> StitchRunResult:
    """Run OpenStitching with the selected detector."""
    start = time.perf_counter()
    if len(images) < 2:
        return StitchRunResult(False, 0.0, None, "Need at least two images.")

    try:
        stitcher = Stitcher(detector=feature_algo, nfeatures=4000)
        panorama = stitcher.stitch(images)
        if panorama is None or panorama.size == 0:
            runtime = time.perf_counter() - start
            return StitchRunResult(False, runtime, None, "OpenStitching returned empty result.")

        if crop_enabled:
            panorama = auto_crop_black_border(panorama).cropped
        save_image(output_path, panorama)
        runtime = time.perf_counter() - start
        return StitchRunResult(True, runtime, panorama)
    except Exception as exc:
        runtime = time.perf_counter() - start
        print(f"[WARN] {feature_algo} stitching failed: {exc}")
        return StitchRunResult(False, runtime, None, str(exc))
