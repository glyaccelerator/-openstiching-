import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from stitching import Stitcher

from pairwise_feature_analysis import analyze_pairwise_features
from .cropping import auto_crop_black_border, black_area_ratio
from .exposure import match_mean_std_luminance
from .io_utils import list_image_files, read_images
from .preprocessing import PreprocessConfig, preprocess_images, resize_by_width
from .visualization import make_side_by_side, make_thumbnail_grid, save_image


FEATURE_ALGOS = ["sift", "orb", "akaze"]


@dataclass
class ExperimentConfig:
    output: Path
    mode: str = "all"
    resize_width: int = 1200
    gamma: float = 1.0
    clahe: bool = False
    exposure: bool = False
    crop: bool = False
    feature: str = "all"
    debug_matches: bool = False
    save_report_assets: bool = False
    max_images: int = 6
    notes: str = ""


@dataclass
class DatasetInfo:
    path: Path
    dataset_name: str
    image_category: str


@dataclass
class BaselineMetric:
    dataset_name: str
    image_count: int
    input_resolution: str
    stitch_success: bool
    runtime_seconds: float
    output_width: int
    output_height: int
    notes: str


@dataclass
class ImprovedMetric:
    dataset_name: str
    image_count: int
    preprocessing_method: str
    gamma: float
    clahe_enabled: bool
    exposure_enabled: bool
    crop_enabled: bool
    stitch_success: bool
    runtime_seconds: float
    output_width_before_crop: int
    output_height_before_crop: int
    output_width_after_crop: int
    output_height_after_crop: int
    black_area_ratio_before_crop: float
    black_area_ratio_after_crop: float
    retained_area_ratio_after_crop: float
    notes: str


@dataclass
class FeatureMetric:
    dataset_name: str
    image_category: str
    feature_algo: str
    pair_index: str
    keypoints_count_img1: int
    keypoints_count_img2: int
    raw_matches_count: int
    good_matches_count: int
    ransac_inliers_count: int
    inlier_ratio: float
    homography_success: bool
    stitch_success: bool
    runtime_seconds: float
    failure_reason: str
    notes: str


@dataclass
class FailureCase:
    dataset_name: str
    image_category: str
    mode: str
    feature_algo: str
    failure_reason: str
    possible_cause: str
    suggested_solution: str


def discover_experiment_datasets(root: Path) -> List[DatasetInfo]:
    """Discover image groups from a single group, category folder, or data root."""
    root = root.resolve()
    if list_image_files(root):
        return [DatasetInfo(root, root.name, root.parent.name if root.parent.name[:2].isdigit() else root.name)]

    datasets: List[DatasetInfo] = []
    seen = set()
    for folder in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if not folder.is_dir():
            continue
        files = list_image_files(folder)
        rel = folder.relative_to(root)
        parts = rel.parts
        if any(part.startswith("_") for part in parts):
            continue
        if len(parts) >= 2 and parts[0][:2].isdigit():
            if len(parts) > 2:
                continue
            category = parts[0]
            name = f"{parts[0]}/{parts[-1]}"
        elif folder.parent.name[:2].isdigit():
            category = folder.parent.name
            name = f"{category}/{folder.name}"
        elif len(files) < 2:
            continue
        else:
            category = folder.name
            name = folder.name
        if folder not in seen:
            seen.add(folder)
            datasets.append(DatasetInfo(folder, name.replace("\\", "/"), category))
    return datasets


def load_limited_images(dataset: DatasetInfo, max_images: int):
    paths = list_image_files(dataset.path)
    if max_images and max_images > 0:
        paths = paths[:max_images]
    return read_images(paths)


def resolution_text(images: List[np.ndarray]) -> str:
    if not images:
        return "none"
    return f"{images[0].shape[1]}x{images[0].shape[0]}"


def safe_stitch(images: List[np.ndarray], detector: Optional[str] = None):
    try:
        if detector:
            panorama = Stitcher(detector=detector, nfeatures=4000).stitch(images)
        else:
            panorama = Stitcher().stitch(images)
        if panorama is None or panorama.size == 0:
            return None, "OpenStitching returned empty result."
        return panorama, ""
    except Exception as exc:
        return None, str(exc)


def output_dataset_dir(root: Path, dataset_name: str) -> Path:
    return root / Path(dataset_name)


def run_baseline(dataset: DatasetInfo, config: ExperimentConfig) -> BaselineMetric:
    start = time.perf_counter()
    images, valid_paths = load_limited_images(dataset, config.max_images)
    out_dir = output_dataset_dir(config.output / "baseline", dataset.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_image(out_dir / "input_contact_sheet.jpg", make_thumbnail_grid(images))

    if len(images) < 2:
        return BaselineMetric(dataset.dataset_name, len(images), resolution_text(images), False, 0.0, 0, 0, "Need at least two images.")

    stitch_images = images
    if config.resize_width and config.resize_width > 0:
        stitch_images = [resize_by_width(image, config.resize_width) for image in images]

    panorama, failure = safe_stitch(stitch_images)
    runtime = time.perf_counter() - start
    if panorama is None:
        return BaselineMetric(dataset.dataset_name, len(valid_paths), resolution_text(images), False, runtime, 0, 0, failure or config.notes)

    save_image(out_dir / "panorama_baseline.jpg", panorama)
    return BaselineMetric(
        dataset.dataset_name,
        len(valid_paths),
        resolution_text(images),
        True,
        runtime,
        int(panorama.shape[1]),
        int(panorama.shape[0]),
        config.notes,
    )


def run_improved(dataset: DatasetInfo, config: ExperimentConfig) -> ImprovedMetric:
    start = time.perf_counter()
    images, valid_paths = load_limited_images(dataset, config.max_images)
    out_dir = output_dataset_dir(config.output / "improved", dataset.dataset_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    preprocess_config = PreprocessConfig(config.resize_width, config.gamma, config.clahe)
    processed = preprocess_images(images, preprocess_config)
    save_image(out_dir / "preprocessing_preview.jpg", make_thumbnail_grid(processed))

    if len(processed) < 2:
        return ImprovedMetric(dataset.dataset_name, len(processed), preprocess_config.method_name(), config.gamma, config.clahe, config.exposure, config.crop, False, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, "Need at least two images.")

    stitch_inputs = match_mean_std_luminance(processed) if config.exposure else processed
    panorama, failure = safe_stitch(stitch_inputs)
    runtime = time.perf_counter() - start
    if panorama is None:
        return ImprovedMetric(dataset.dataset_name, len(valid_paths), preprocess_config.method_name(), config.gamma, config.clahe, config.exposure, config.crop, False, runtime, 0, 0, 0, 0, 0.0, 0.0, 0.0, failure or config.notes)

    before = panorama
    after = panorama
    before_ratio = black_area_ratio(before)
    after_ratio = before_ratio
    if config.crop:
        crop_result = auto_crop_black_border(before)
        after = crop_result.cropped
        before_ratio = crop_result.black_ratio_before
        after_ratio = crop_result.black_ratio_after
        save_image(out_dir / "crop_comparison.jpg", make_side_by_side(before, after))
    save_image(out_dir / "panorama_improved_before_crop.jpg", before)
    save_image(out_dir / "panorama_improved_after_crop.jpg", after)

    retained = float(after.shape[0] * after.shape[1]) / float(before.shape[0] * before.shape[1]) if before.size else 0.0
    return ImprovedMetric(
        dataset.dataset_name,
        len(valid_paths),
        preprocess_config.method_name(),
        config.gamma,
        config.clahe,
        config.exposure,
        config.crop,
        True,
        runtime,
        int(before.shape[1]),
        int(before.shape[0]),
        int(after.shape[1]),
        int(after.shape[0]),
        before_ratio,
        after_ratio,
        retained,
        config.notes,
    )


def selected_features(feature: str) -> List[str]:
    return FEATURE_ALGOS if feature == "all" else [feature]


def run_feature_experiment(dataset: DatasetInfo, config: ExperimentConfig) -> List[FeatureMetric]:
    images, valid_paths = load_limited_images(dataset, config.max_images)
    preprocess_config = PreprocessConfig(config.resize_width, config.gamma, config.clahe)
    processed = preprocess_images(images, preprocess_config)
    if config.exposure:
        processed = match_mean_std_luminance(processed)
    rows: List[FeatureMetric] = []

    if len(processed) < 2:
        rows.append(
            FeatureMetric(dataset.dataset_name, dataset.image_category, "all", "all", 0, 0, 0, 0, 0, 0.0, False, False, 0.0, "Need at least two images.", config.notes)
        )
        return rows

    for algo in selected_features(config.feature):
        algo_dir = output_dataset_dir(config.output / "feature_experiment", dataset.dataset_name) / algo
        pair_results = analyze_pairwise_features(processed, algo, algo_dir, debug_matches=config.debug_matches)

        stitch_start = time.perf_counter()
        panorama, stitch_failure = safe_stitch(processed, detector=algo)
        stitch_runtime = time.perf_counter() - stitch_start
        stitch_success = panorama is not None
        if stitch_success:
            save_image(algo_dir / f"panorama_{algo}.jpg", panorama)

        for pair in pair_results:
            rows.append(
                FeatureMetric(
                    dataset.dataset_name,
                    dataset.image_category,
                    algo,
                    pair.pair_index,
                    pair.keypoints_count_img1,
                    pair.keypoints_count_img2,
                    pair.raw_matches_count,
                    pair.good_matches_count,
                    pair.ransac_inliers_count,
                    pair.inlier_ratio,
                    pair.homography_success,
                    stitch_success,
                    pair.runtime_seconds + stitch_runtime / max(1, len(pair_results)),
                    pair.failure_reason or ("" if stitch_success else stitch_failure),
                    config.notes,
                )
            )
    return rows


def write_csv(path: Path, rows: Iterable, fieldnames: Optional[List[str]] = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames and rows:
        fieldnames = list(asdict(rows[0]).keys())
    if not fieldnames:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            data = asdict(row)
            for key, value in list(data.items()):
                if isinstance(value, float):
                    data[key] = f"{value:.6f}"
            writer.writerow(data)


def possible_cause(mode: str, failure_reason: str, category: str) -> str:
    if "two images" in failure_reason:
        return "Image group has fewer than two readable images."
    if "SIFT" in failure_reason:
        return "OpenCV SIFT API is unavailable."
    if "Homography" in failure_reason:
        return "Too few reliable matches or large parallax."
    if "09_failure" in category:
        return "Low overlap, smooth regions, repeated texture or moving objects."
    return "Feature matching, camera estimation or blending failed."


def suggested_solution(mode: str, category: str) -> str:
    if mode == "feature_experiment":
        return "Try SIFT/AKAZE, lower resize width less aggressively, or add more overlap."
    if "05_low_light" in category:
        return "Enable gamma, CLAHE and exposure compensation."
    return "Use images with more overlap, richer texture, and stable camera rotation."


def build_failure_cases(
    datasets: List[DatasetInfo],
    baseline_rows: List[BaselineMetric],
    improved_rows: List[ImprovedMetric],
    feature_rows: List[FeatureMetric],
) -> List[FailureCase]:
    category_by_name = {dataset.dataset_name: dataset.image_category for dataset in datasets}
    failures: List[FailureCase] = []
    for row in baseline_rows:
        if not row.stitch_success:
            category = category_by_name.get(row.dataset_name, "")
            failures.append(FailureCase(row.dataset_name, category, "baseline", "", row.notes, possible_cause("baseline", row.notes, category), suggested_solution("baseline", category)))
    for row in improved_rows:
        if not row.stitch_success:
            category = category_by_name.get(row.dataset_name, "")
            failures.append(FailureCase(row.dataset_name, category, "improved", "", row.notes, possible_cause("improved", row.notes, category), suggested_solution("improved", category)))
    seen = set()
    for row in feature_rows:
        key = (row.dataset_name, row.feature_algo, row.failure_reason, row.stitch_success)
        if (not row.homography_success or not row.stitch_success) and key not in seen:
            seen.add(key)
            failures.append(FailureCase(row.dataset_name, row.image_category, "feature_experiment", row.feature_algo, row.failure_reason, possible_cause("feature_experiment", row.failure_reason, row.image_category), suggested_solution("feature_experiment", row.image_category)))
    return failures


def summarize(
    datasets: List[DatasetInfo],
    baseline_rows: List[BaselineMetric],
    improved_rows: List[ImprovedMetric],
    feature_rows: List[FeatureMetric],
) -> List[Dict[str, object]]:
    baseline_by = {row.dataset_name: row for row in baseline_rows}
    improved_by = {row.dataset_name: row for row in improved_rows}
    rows = []
    for dataset in datasets:
        features = [row for row in feature_rows if row.dataset_name == dataset.dataset_name]
        by_algo: Dict[str, List[FeatureMetric]] = {}
        for row in features:
            by_algo.setdefault(row.feature_algo, []).append(row)
        avg_inliers = {
            algo: sum(item.inlier_ratio for item in values) / max(1, len(values))
            for algo, values in by_algo.items()
        }
        runtimes = {
            algo: sum(item.runtime_seconds for item in values)
            for algo, values in by_algo.items()
        }
        best_algo = max(avg_inliers, key=avg_inliers.get) if avg_inliers else ""
        fastest = min(runtimes, key=runtimes.get) if runtimes else ""
        base = baseline_by.get(dataset.dataset_name)
        imp = improved_by.get(dataset.dataset_name)
        before = imp.black_area_ratio_before_crop if imp else 0.0
        after = imp.black_area_ratio_after_crop if imp else 0.0
        recommendation = recommend_use(dataset.image_category, base, imp, features)
        rows.append(
            {
                "dataset_name": dataset.dataset_name,
                "image_category": dataset.image_category,
                "baseline_success": bool(base and base.stitch_success),
                "improved_success": bool(imp and imp.stitch_success),
                "best_feature_algo_by_inlier_ratio": best_algo,
                "fastest_feature_algo": fastest,
                "baseline_runtime": f"{base.runtime_seconds:.6f}" if base else "0.000000",
                "improved_runtime": f"{imp.runtime_seconds:.6f}" if imp else "0.000000",
                "black_area_ratio_before_crop": f"{before:.6f}",
                "black_area_ratio_after_crop": f"{after:.6f}",
                "recommended_report_use": recommendation,
                "conclusion_short": conclusion_short(dataset.image_category, best_algo, fastest, before, after),
            }
        )
    return rows


def recommend_use(category: str, base, imp, features: List[FeatureMetric]) -> str:
    if "09_failure" in category or (base and not base.stitch_success):
        return "failure_case"
    if imp and imp.stitch_success and imp.black_area_ratio_before_crop - imp.black_area_ratio_after_crop > 0.02:
        return "crop_or_improvement_case"
    if features:
        return "feature_algorithm_comparison"
    return "baseline_success_case"


def conclusion_short(category: str, best_algo: str, fastest: str, before: float, after: float) -> str:
    if "05_low_light" in category:
        return "Low-light data is suitable for showing preprocessing and exposure compensation."
    if "07_large_parallax" in category:
        return "Large parallax can weaken the single-homography assumption and cause ghosting."
    if "06_repetitive" in category:
        return "Repetitive texture may create false matches; RANSAC inlier ratio is important."
    if before > after:
        return "Automatic cropping reduces black borders and improves display quality."
    return f"{best_algo or 'Feature methods'} can be compared with {fastest or 'runtime'} for stability and speed."


def draw_text(canvas, text: str, x: int, y: int, scale: float = 0.7):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (35, 35, 35), 2)


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image is None:
        image = np.full((height, 420, 3), 235, dtype=np.uint8)
        draw_text(image, "missing", 30, height // 2)
        return image
    scale = height / image.shape[0]
    width = max(1, int(round(image.shape[1] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_output_image(path: Path):
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def feature_chart(rows: List[FeatureMetric], width: int = 900, height: int = 260) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    draw_text(canvas, "SIFT / ORB / AKAZE inlier ratio", 20, 34, 0.75)
    algos = [algo for algo in FEATURE_ALGOS if any(row.feature_algo == algo for row in rows)]
    if not algos:
        return canvas
    values = []
    for algo in algos:
        algo_rows = [row for row in rows if row.feature_algo == algo]
        values.append(sum(row.inlier_ratio for row in algo_rows) / max(1, len(algo_rows)))
    max_value = max(max(values), 1e-6)
    colors = [(65, 120, 230), (40, 165, 90), (230, 135, 55)]
    bar_w = 130
    for idx, (algo, value) in enumerate(zip(algos, values)):
        x = 80 + idx * 240
        bar_h = int((value / max_value) * 150)
        cv2.rectangle(canvas, (x, 220 - bar_h), (x + bar_w, 220), colors[idx % 3], -1)
        draw_text(canvas, algo.upper(), x, 245, 0.58)
        draw_text(canvas, f"{value:.2f}", x, 210 - bar_h, 0.55)
    return canvas


def make_comparison_images(datasets: List[DatasetInfo], config: ExperimentConfig, feature_rows: List[FeatureMetric]) -> None:
    comparison_root = config.output / "comparisons"
    comparison_root.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        images, _ = load_limited_images(dataset, config.max_images)
        input_sheet = make_thumbnail_grid(images)
        baseline = load_output_image(output_dataset_dir(config.output / "baseline", dataset.dataset_name) / "panorama_baseline.jpg")
        before = load_output_image(output_dataset_dir(config.output / "improved", dataset.dataset_name) / "panorama_improved_before_crop.jpg")
        after = load_output_image(output_dataset_dir(config.output / "improved", dataset.dataset_name) / "panorama_improved_after_crop.jpg")
        rows = [row for row in feature_rows if row.dataset_name == dataset.dataset_name]
        parts = [
            ("input thumbnails", input_sheet),
            ("baseline", baseline),
            ("improved before crop", before),
            ("improved after crop", after),
            ("feature metrics", feature_chart(rows)),
        ]
        resized = []
        for title, image in parts:
            body = resize_to_height(image, 220)
            title_bar = np.full((42, body.shape[1], 3), 245, dtype=np.uint8)
            draw_text(title_bar, title, 12, 28, 0.65)
            resized.append(np.vstack([title_bar, body]))
        width = max(item.shape[1] for item in resized)
        height = sum(item.shape[0] for item in resized) + 12 * (len(resized) - 1)
        canvas = np.full((height, width, 3), 245, dtype=np.uint8)
        y = 0
        for item in resized:
            canvas[y : y + item.shape[0], : item.shape[1]] = item
            y += item.shape[0] + 12
        safe_name = dataset.dataset_name.replace("/", "_")
        save_image(comparison_root / f"{safe_name}_comparison.jpg", canvas)


def generate_report_assets(
    datasets: List[DatasetInfo],
    summary_rows: List[Dict[str, object]],
    failure_rows: List[FailureCase],
    feature_rows: List[FeatureMetric],
    output_root: Path,
) -> None:
    report_root = output_root / "report_assets"
    for child in ["best_cases", "failure_cases", "feature_algorithm_comparison_charts"]:
        (report_root / child).mkdir(parents=True, exist_ok=True)
    tables_dir = output_root / "tables"
    for chart_name in ["summary_metrics.csv", "feature_metrics.csv"]:
        src = tables_dir / chart_name
        if src.exists():
            shutil_copy(src, report_root / chart_name)

    for dataset in datasets:
        rows = [row for row in feature_rows if row.dataset_name == dataset.dataset_name]
        if rows:
            safe_name = dataset.dataset_name.replace("/", "_")
            save_image(
                report_root / "feature_algorithm_comparison_charts" / f"{safe_name}_feature_chart.jpg",
                feature_chart(rows),
            )

    total = len(datasets)
    success_cases = [row for row in summary_rows if str(row.get("baseline_success")) == "True"]
    improved_cases = sorted(
        summary_rows,
        key=lambda row: float(row.get("black_area_ratio_before_crop", 0)) - float(row.get("black_area_ratio_after_crop", 0)),
        reverse=True,
    )
    feature_cases = [row for row in summary_rows if row.get("best_feature_algo_by_inlier_ratio")]
    failure_cases = [row for row in summary_rows if row.get("recommended_report_use") == "failure_case"]
    lines = [
        "# Report Assets Guide",
        "",
        f"- Total image groups tested: {total}",
        f"- Baseline success candidates: {len(success_cases)}",
        f"- Failure records: {len(failure_rows)}",
        "",
        "## Recommended Figures",
        f"- Baseline success case: {pick_name(success_cases)}",
        f"- Improved obvious case: {pick_name(improved_cases)}",
        f"- SIFT/ORB/AKAZE comparison case: {pick_name(feature_cases)}",
        f"- Auto-crop black border case: {pick_name(improved_cases)}",
        f"- Failure case: {pick_name(failure_cases)}",
        "",
        "## Category Analysis Templates",
        "- Landscape: rich textures usually provide many keypoints; ORB is fast while SIFT is often stable.",
        "- City/architecture: lines and corners are abundant, but repeated windows may create false matches.",
        "- Map/satellite: planar structure is suitable for homography, but repeated symbols or text can mislead matching.",
        "- Indoor low texture: fewer reliable keypoints may make matching unstable.",
        "- Low light/exposure: Gamma, CLAHE and exposure compensation can improve visible features.",
        "- Repetitive texture: RANSAC is important for removing false matches.",
        "- Large parallax: a single homography may not describe 3D viewpoint changes well.",
        "- Many images: useful for showing multi-image stitching, accumulated drift and cropping.",
        "- Failure cases: use them to explain low overlap, smooth regions, moving objects or strong parallax.",
        "",
        "## How To Use",
        "- Use outputs/comparisons/*_comparison.jpg for one-page visual comparison.",
        "- Use outputs/feature_experiment/<dataset>/<algo>/keypoints and matches for algorithm details.",
        "- Use outputs/tables/*.csv for quantitative tables in the report.",
    ]
    (report_root / "README_for_report.md").write_text("\n".join(lines), encoding="utf-8")

    copy_recommended_comparison(output_root, report_root / "best_cases", pick_name(success_cases), "baseline_success_case.jpg")
    copy_recommended_comparison(output_root, report_root / "best_cases", pick_name(improved_cases), "improved_or_crop_case.jpg")
    copy_recommended_comparison(output_root, report_root / "failure_cases", pick_name(failure_cases), "failure_case.jpg")


def copy_recommended_comparison(output_root: Path, target_dir: Path, dataset_name: str, target_name: str) -> None:
    if not dataset_name or dataset_name == "No suitable case found":
        return
    src = output_root / "comparisons" / f"{dataset_name.replace('/', '_')}_comparison.jpg"
    if src.exists():
        shutil_copy(src, target_dir / target_name)


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def pick_name(rows: List[Dict[str, object]]) -> str:
    return str(rows[0]["dataset_name"]) if rows else "No suitable case found"


def run_experiments(datasets: List[DatasetInfo], config: ExperimentConfig) -> None:
    tables_dir = config.output / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    config.output.mkdir(parents=True, exist_ok=True)

    baseline_rows: List[BaselineMetric] = []
    improved_rows: List[ImprovedMetric] = []
    feature_rows: List[FeatureMetric] = []

    if config.mode in {"baseline", "all"}:
        for dataset in datasets:
            print(f"[INFO] baseline: {dataset.dataset_name}")
            baseline_rows.append(run_baseline(dataset, config))
        write_csv(tables_dir / "baseline_metrics.csv", baseline_rows)

    if config.mode in {"improved", "all"}:
        improved_config = ExperimentConfig(**{**config.__dict__, "exposure": True if config.mode == "all" else config.exposure, "crop": True if config.mode == "all" else config.crop})
        for dataset in datasets:
            print(f"[INFO] improved: {dataset.dataset_name}")
            improved_rows.append(run_improved(dataset, improved_config))
        write_csv(tables_dir / "improved_metrics.csv", improved_rows)

    if config.mode in {"feature_experiment", "all"}:
        feature_config = ExperimentConfig(**{**config.__dict__, "feature": "all" if config.mode == "all" else config.feature})
        for dataset in datasets:
            print(f"[INFO] feature_experiment: {dataset.dataset_name}")
            feature_rows.extend(run_feature_experiment(dataset, feature_config))
        write_csv(tables_dir / "feature_metrics.csv", feature_rows)

    summary_rows = summarize(datasets, baseline_rows, improved_rows, feature_rows)
    if config.mode == "all":
        write_dict_csv(tables_dir / "summary_metrics.csv", summary_rows)
        failures = build_failure_cases(datasets, baseline_rows, improved_rows, feature_rows)
        write_csv(tables_dir / "failure_cases.csv", failures)
        make_comparison_images(datasets, config, feature_rows)
        generate_report_assets(datasets, summary_rows, failures, feature_rows, config.output)


def write_dict_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
