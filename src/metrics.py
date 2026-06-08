import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .stitching_pipeline import StitchExperimentResult


FIELDNAMES = [
    "dataset_name",
    "image_count",
    "input_resolution",
    "preprocessing_method",
    "exposure_compensation_enabled",
    "crop_enabled",
    "stitch_success",
    "runtime_seconds",
    "output_width",
    "output_height",
    "black_area_ratio_before_crop",
    "black_area_ratio_after_crop",
    "notes",
]


def write_metrics(output_path: Path, results: Iterable[StitchExperimentResult]) -> None:
    """Write experiment metrics to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["runtime_seconds"] = f"{result.runtime_seconds:.3f}"
            row["black_area_ratio_before_crop"] = f"{result.black_area_ratio_before_crop:.6f}"
            row["black_area_ratio_after_crop"] = f"{result.black_area_ratio_after_crop:.6f}"
            writer.writerow(row)
