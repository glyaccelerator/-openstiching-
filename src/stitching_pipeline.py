import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from stitching import Stitcher

from .cropping import CropResult, auto_crop_black_border, black_area_ratio
from .exposure import match_mean_std_luminance
from .io_utils import list_image_files, read_images
from .preprocessing import PreprocessConfig, preprocess_images
from .visualization import make_side_by_side, make_thumbnail_grid, save_image


@dataclass
class StitchExperimentResult:
    dataset_name: str
    image_count: int
    input_resolution: str
    preprocessing_method: str
    exposure_compensation_enabled: bool
    crop_enabled: bool
    stitch_success: bool
    runtime_seconds: float
    output_width: int
    output_height: int
    black_area_ratio_before_crop: float
    black_area_ratio_after_crop: float
    notes: str


def _stitch(images) -> Optional[np.ndarray]:
    """Run OpenStitching and return panorama, or None when stitching fails."""
    if len(images) < 2:
        return None
    try:
        stitcher = Stitcher()
        panorama = stitcher.stitch(images)
        if panorama is None or panorama.size == 0:
            return None
        return panorama
    except Exception as exc:
        print(f"[WARN] Stitching failed: {exc}")
        return None


def _resolution_text(images) -> str:
    if not images:
        return "none"
    first = images[0]
    return f"{first.shape[1]}x{first.shape[0]}"


def run_dataset(
    dataset_dir: Path,
    output_root: Path,
    preprocess_config: PreprocessConfig,
    exposure_enabled: bool = False,
    crop_enabled: bool = False,
) -> StitchExperimentResult:
    """Run the full experiment workflow for one image folder."""
    start_time = time.perf_counter()
    dataset_name = dataset_dir.name
    panorama_dir = output_root / "panoramas"
    vis_dir = output_root / "visualizations" / dataset_name
    panorama_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(dataset_dir)
    images, valid_paths = read_images(image_paths)
    input_resolution = _resolution_text(images)

    thumb = make_thumbnail_grid(images)
    save_image(vis_dir / "input_thumbnails.jpg", thumb)

    notes = ""
    panorama = None
    output_image = None
    crop_result: Optional[CropResult] = None

    if len(images) < 2:
        notes = "Need at least two readable images."
    else:
        processed_images = preprocess_images(images, preprocess_config)
        save_image(vis_dir / "preprocessed_thumbnails.jpg", make_thumbnail_grid(processed_images))

        no_exposure_panorama = _stitch(processed_images)
        if no_exposure_panorama is not None:
            save_image(panorama_dir / f"{dataset_name}_before_exposure.jpg", no_exposure_panorama)

        stitch_inputs = processed_images
        if exposure_enabled:
            stitch_inputs = match_mean_std_luminance(processed_images)
            save_image(vis_dir / "exposure_compensated_thumbnails.jpg", make_thumbnail_grid(stitch_inputs))

        panorama = _stitch(stitch_inputs)
        if panorama is None and no_exposure_panorama is not None:
            panorama = no_exposure_panorama
            notes = "Exposure version failed; saved panorama without exposure compensation."

        if panorama is not None:
            save_image(panorama_dir / f"{dataset_name}_raw.jpg", panorama)
            if exposure_enabled and no_exposure_panorama is not None:
                save_image(vis_dir / "before_after_exposure_stitch.jpg", make_side_by_side(no_exposure_panorama, panorama))

            before_ratio = black_area_ratio(panorama)
            after_ratio = before_ratio
            output_image = panorama
            if crop_enabled:
                crop_result = auto_crop_black_border(panorama)
                output_image = crop_result.cropped
                before_ratio = crop_result.black_ratio_before
                after_ratio = crop_result.black_ratio_after
                save_image(panorama_dir / f"{dataset_name}_cropped.jpg", output_image)
                save_image(vis_dir / "crop_mask.jpg", crop_result.mask)
                save_image(vis_dir / "before_after_crop.jpg", make_side_by_side(crop_result.original, crop_result.cropped))
            else:
                save_image(panorama_dir / f"{dataset_name}_final.jpg", output_image)

            save_image(panorama_dir / f"{dataset_name}_final.jpg", output_image)
            save_image(vis_dir / "input_vs_panorama.jpg", make_side_by_side(thumb, output_image))

            runtime = time.perf_counter() - start_time
            return StitchExperimentResult(
                dataset_name=dataset_name,
                image_count=len(valid_paths),
                input_resolution=input_resolution,
                preprocessing_method=preprocess_config.method_name(),
                exposure_compensation_enabled=exposure_enabled,
                crop_enabled=crop_enabled,
                stitch_success=True,
                runtime_seconds=runtime,
                output_width=int(output_image.shape[1]),
                output_height=int(output_image.shape[0]),
                black_area_ratio_before_crop=before_ratio,
                black_area_ratio_after_crop=after_ratio,
                notes=notes,
            )

        notes = notes or "OpenStitching failed to generate panorama."

    runtime = time.perf_counter() - start_time
    return StitchExperimentResult(
        dataset_name=dataset_name,
        image_count=len(valid_paths),
        input_resolution=input_resolution,
        preprocessing_method=preprocess_config.method_name(),
        exposure_compensation_enabled=exposure_enabled,
        crop_enabled=crop_enabled,
        stitch_success=False,
        runtime_seconds=runtime,
        output_width=0,
        output_height=0,
        black_area_ratio_before_crop=0.0,
        black_area_ratio_after_crop=0.0,
        notes=notes,
    )
