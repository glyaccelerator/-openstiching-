import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
from stitching import Stitcher


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImprovedConfig:
    resize_width: int = 1200
    gamma: float = 1.0
    clahe: bool = False

    def method_name(self) -> str:
        methods = []
        if self.resize_width > 0:
            methods.append(f"resize_{self.resize_width}")
        methods.append("brightness_normalization")
        if abs(self.gamma - 1.0) > 1e-6:
            methods.append(f"gamma_{self.gamma:g}")
        if self.clahe:
            methods.append("clahe")
        methods.append("mean_std_exposure")
        methods.append("auto_crop")
        return "+".join(methods)


@dataclass
class ModeResult:
    success: bool
    runtime: float
    panorama: Optional[np.ndarray]
    final_image: Optional[np.ndarray]
    black_area_ratio: float
    output_size: str
    notes: str = ""


@dataclass
class ComparisonResult:
    dataset_name: str
    baseline_success: bool
    improved_success: bool
    baseline_runtime: float
    improved_runtime: float
    baseline_black_area_ratio: float
    improved_black_area_ratio: float
    baseline_output_size: str
    improved_output_size: str


def list_image_files(folder: Path) -> List[Path]:
    """Return sorted image paths and ignore reference files beginning with '_'."""
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("_")
        ],
        key=lambda path: path.name.lower(),
    )


def discover_datasets(batch_folder: Path) -> List[Path]:
    """Find child folders that contain at least one input image."""
    if not batch_folder.exists() or not batch_folder.is_dir():
        return []
    return [
        child
        for child in sorted(batch_folder.iterdir(), key=lambda path: path.name.lower())
        if child.is_dir() and list_image_files(child)
    ]


def read_images(image_paths: Iterable[Path]) -> List[np.ndarray]:
    """Read images as OpenCV BGR arrays and skip unreadable files."""
    images = []
    for path in image_paths:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
        else:
            print(f"[WARN] Cannot read image: {path}")
    return images


def save_image(path: Path, image: Optional[np.ndarray]) -> None:
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if success:
        encoded.tofile(str(path))
    else:
        print(f"[WARN] Cannot save image: {path}")


def resize_by_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize image while preserving aspect ratio."""
    if target_width <= 0 or image.shape[1] <= target_width:
        return image.copy()
    scale = target_width / image.shape[1]
    target_height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """Normalize the LAB luminance channel for global brightness consistency."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.normalize(l_channel, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction. gamma=1.0 keeps the image unchanged."""
    if gamma <= 0 or abs(gamma - 1.0) <= 1e-6:
        return image.copy()
    inv_gamma = 1.0 / gamma
    table = np.array(
        [(idx / 255.0) ** inv_gamma * 255 for idx in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, table)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Enhance local contrast on the LAB luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge([l_channel, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def preprocess_for_improved(images: List[np.ndarray], config: ImprovedConfig) -> List[np.ndarray]:
    """Run the improved preprocessing pipeline before exposure compensation."""
    processed = []
    for image in images:
        result = resize_by_width(image, config.resize_width)
        result = normalize_brightness(result)
        result = apply_gamma(result, config.gamma)
        if config.clahe:
            result = apply_clahe(result)
        processed.append(result)
    return processed


def match_mean_std_luminance(images: List[np.ndarray]) -> List[np.ndarray]:
    """Match each image LAB luminance mean/std to the first image.

    This is a simple exposure compensation algorithm that is easy to describe
    in a digital image processing report.
    """
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
        adjusted_l = (l_float - mean) * (ref_std / std) + ref_mean
        adjusted_l = np.clip(adjusted_l, 0, 255).astype(np.uint8)
        compensated.append(
            cv2.cvtColor(cv2.merge([adjusted_l, a_channel, b_channel]), cv2.COLOR_LAB2BGR)
        )
    return compensated


def stitch_images(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Run the OpenStitching default pipeline."""
    if len(images) < 2:
        return None
    try:
        panorama = Stitcher().stitch(images)
        if panorama is None or panorama.size == 0:
            return None
        return panorama
    except Exception as exc:
        print(f"[WARN] OpenStitching failed: {exc}")
        return None


def compute_non_black_mask(image: np.ndarray, threshold: int = 8) -> np.ndarray:
    """Create a valid-area mask from non-black pixels."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def black_area_ratio(image: Optional[np.ndarray], threshold: int = 8) -> float:
    """Return the ratio of near-black pixels."""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray <= threshold))


def auto_crop_black_border(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop panorama by the bounding rectangle of the largest non-black contour."""
    mask = compute_non_black_mask(image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image.copy(), mask
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return image[y : y + h, x : x + w], mask


def image_size_text(image: Optional[np.ndarray]) -> str:
    if image is None or image.size == 0:
        return "0x0"
    return f"{image.shape[1]}x{image.shape[0]}"


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_thumbnail_grid(images: List[np.ndarray], max_width: int = 240, pad: int = 8) -> np.ndarray:
    """Create a horizontal input thumbnail collage."""
    if not images:
        return make_placeholder("No input images")

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


def make_placeholder(text: str, width: int = 900, height: int = 260) -> np.ndarray:
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)
    cv2.putText(canvas, text, (30, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
    return canvas


def add_title(image: Optional[np.ndarray], title: str, target_height: int = 260) -> np.ndarray:
    """Resize an image and add an English title for report-friendly comparison."""
    if image is None:
        body = make_placeholder("Stitch failed", width=900, height=target_height)
    else:
        body = resize_to_height(image, target_height)

    title_h = 42
    canvas = np.full((body.shape[0] + title_h, body.shape[1], 3), 250, dtype=np.uint8)
    canvas[title_h:, : body.shape[1]] = body
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 30, 30), 2)
    return canvas


def stack_rows(rows: List[np.ndarray], pad: int = 14) -> np.ndarray:
    """Stack variable-width rows into one visualization canvas."""
    width = max(row.shape[1] for row in rows)
    height = sum(row.shape[0] for row in rows) + pad * (len(rows) - 1)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y_offset = 0
    for row in rows:
        canvas[y_offset : y_offset + row.shape[0], : row.shape[1]] = row
        y_offset += row.shape[0] + pad
    return canvas


def run_baseline(images: List[np.ndarray], dataset_name: str, output_root: Path) -> ModeResult:
    """Baseline: raw OpenStitching without preprocessing, exposure compensation or crop."""
    start_time = time.perf_counter()
    panorama = stitch_images(images)
    runtime = time.perf_counter() - start_time
    if panorama is None:
        return ModeResult(False, runtime, None, None, 0.0, "0x0", "baseline stitching failed")

    save_image(output_root / "panoramas" / f"{dataset_name}_baseline.jpg", panorama)
    return ModeResult(
        success=True,
        runtime=runtime,
        panorama=panorama,
        final_image=panorama,
        black_area_ratio=black_area_ratio(panorama),
        output_size=image_size_text(panorama),
    )


def run_improved(
    images: List[np.ndarray],
    dataset_name: str,
    output_root: Path,
    config: ImprovedConfig,
) -> ModeResult:
    """Improved: preprocessing, exposure compensation, stitching and black-border crop."""
    start_time = time.perf_counter()
    processed = preprocess_for_improved(images, config)
    compensated = match_mean_std_luminance(processed)
    panorama = stitch_images(compensated)
    runtime = time.perf_counter() - start_time
    vis_dir = output_root / "visualizations" / dataset_name

    save_image(vis_dir / "improved_preprocessed_thumbnails.jpg", make_thumbnail_grid(processed))
    save_image(vis_dir / "improved_exposure_thumbnails.jpg", make_thumbnail_grid(compensated))

    if panorama is None:
        return ModeResult(False, runtime, None, None, 0.0, "0x0", "improved stitching failed")

    cropped, mask = auto_crop_black_border(panorama)
    save_image(output_root / "panoramas" / f"{dataset_name}_improved_raw.jpg", panorama)
    save_image(output_root / "panoramas" / f"{dataset_name}_improved_cropped.jpg", cropped)
    save_image(output_root / "panoramas" / f"{dataset_name}_improved_final.jpg", cropped)
    save_image(vis_dir / "improved_crop_mask.jpg", mask)

    return ModeResult(
        success=True,
        runtime=runtime,
        panorama=panorama,
        final_image=cropped,
        black_area_ratio=black_area_ratio(cropped),
        output_size=image_size_text(cropped),
    )


def save_comparison_figure(
    dataset_name: str,
    images: List[np.ndarray],
    baseline: ModeResult,
    improved: ModeResult,
    output_root: Path,
) -> None:
    """Save one report-ready figure containing inputs and two algorithm modes."""
    input_row = add_title(make_thumbnail_grid(images), "Input thumbnails", target_height=220)
    baseline_row = add_title(baseline.final_image, "Baseline: OpenStitching original", target_height=280)
    improved_raw_row = add_title(improved.panorama, "Improved: stitched before crop", target_height=280)
    improved_crop_row = add_title(improved.final_image, "Improved: cropped final result", target_height=280)
    figure = stack_rows([input_row, baseline_row, improved_raw_row, improved_crop_row])
    save_image(output_root / "visualizations" / dataset_name / "baseline_vs_improved.jpg", figure)


def run_comparison_for_dataset(
    dataset_dir: Path,
    output_root: Path,
    config: ImprovedConfig,
    mode: str,
) -> ComparisonResult:
    """Run baseline, improved, or both for one dataset."""
    dataset_name = dataset_dir.name
    image_paths = list_image_files(dataset_dir)
    images = read_images(image_paths)
    vis_dir = output_root / "visualizations" / dataset_name
    vis_dir.mkdir(parents=True, exist_ok=True)
    save_image(vis_dir / "input_thumbnails.jpg", make_thumbnail_grid(images))

    if len(images) < 2:
        print(f"[WARN] {dataset_name}: need at least two readable images.")
        failed = ModeResult(False, 0.0, None, None, 0.0, "0x0", "not enough images")
        save_comparison_figure(dataset_name, images, failed, failed, output_root)
        return ComparisonResult(dataset_name, False, False, 0.0, 0.0, 0.0, 0.0, "0x0", "0x0")

    baseline = ModeResult(False, 0.0, None, None, 0.0, "0x0", "baseline not run")
    improved = ModeResult(False, 0.0, None, None, 0.0, "0x0", "improved not run")

    if mode in {"baseline", "compare"}:
        baseline = run_baseline(images, dataset_name, output_root)
    if mode in {"improved", "compare"}:
        improved = run_improved(images, dataset_name, output_root, config)
    if mode == "compare":
        save_comparison_figure(dataset_name, images, baseline, improved, output_root)

    return ComparisonResult(
        dataset_name=dataset_name,
        baseline_success=baseline.success,
        improved_success=improved.success,
        baseline_runtime=baseline.runtime,
        improved_runtime=improved.runtime,
        baseline_black_area_ratio=baseline.black_area_ratio,
        improved_black_area_ratio=improved.black_area_ratio,
        baseline_output_size=baseline.output_size,
        improved_output_size=improved.output_size,
    )


def write_comparison_csv(output_path: Path, results: List[ComparisonResult]) -> None:
    fieldnames = [
        "dataset_name",
        "baseline_success",
        "improved_success",
        "baseline_runtime",
        "improved_runtime",
        "baseline_black_area_ratio",
        "improved_black_area_ratio",
        "baseline_output_size",
        "improved_output_size",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "dataset_name": result.dataset_name,
                    "baseline_success": result.baseline_success,
                    "improved_success": result.improved_success,
                    "baseline_runtime": f"{result.baseline_runtime:.3f}",
                    "improved_runtime": f"{result.improved_runtime:.3f}",
                    "baseline_black_area_ratio": f"{result.baseline_black_area_ratio:.6f}",
                    "improved_black_area_ratio": f"{result.improved_black_area_ratio:.6f}",
                    "baseline_output_size": result.baseline_output_size,
                    "improved_output_size": result.improved_output_size,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baseline vs improved panorama stitching experiment for DIP report."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single image folder, e.g. data/set1")
    group.add_argument("--batch", type=Path, help="Batch folder containing image sets, e.g. data")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root folder")
    parser.add_argument(
        "--mode",
        choices=["baseline", "improved", "compare"],
        default="compare",
        help="Run original baseline, improved pipeline, or both.",
    )
    parser.add_argument("--resize-width", type=int, default=1200, help="Improved mode resize width")
    parser.add_argument("--gamma", type=float, default=1.0, help="Improved mode gamma correction")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE in improved mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (args.output / "panoramas").mkdir(parents=True, exist_ok=True)
    (args.output / "visualizations").mkdir(parents=True, exist_ok=True)

    datasets = [args.input] if args.input else discover_datasets(args.batch)
    if not datasets:
        print("[ERROR] No image datasets found. Please check --input or --batch.")
        return

    config = ImprovedConfig(
        resize_width=args.resize_width,
        gamma=args.gamma,
        clahe=args.clahe,
    )

    print(f"[INFO] mode={args.mode}, improved_method={config.method_name()}")
    results = []
    for dataset in datasets:
        print(f"[INFO] Processing {dataset}")
        result = run_comparison_for_dataset(dataset, args.output, config, args.mode)
        results.append(result)
        print(
            f"[INFO] {result.dataset_name}: baseline={result.baseline_success} "
            f"improved={result.improved_success} "
            f"black_ratio={result.baseline_black_area_ratio:.4f}->{result.improved_black_area_ratio:.4f}"
        )

    csv_path = args.output / "comparison.csv"
    write_comparison_csv(csv_path, results)
    print(f"[INFO] Comparison metrics saved to {csv_path}")


if __name__ == "__main__":
    main()
