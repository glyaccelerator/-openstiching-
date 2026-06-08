from dataclasses import dataclass
from pathlib import Path
from typing import List

from .exposure import match_mean_std_luminance
from .feature_analysis import analyze_features
from .io_utils import ensure_output_dirs, list_image_files, read_images, save_image
from .preprocessing import PreprocessConfig, preprocess_images
from .reporting import save_comparison_charts, write_feature_metrics, write_markdown_table
from .stitching_runner import stitch_with_feature_algorithm
from .visualization import make_thumbnail_grid


FEATURE_ALGOS = ["sift", "orb", "akaze"]


@dataclass
class FeatureExperimentConfig:
    output_root: Path
    feature: str = "all"
    resize_width: int = 1200
    gamma: float = 1.0
    clahe: bool = False
    exposure: bool = False
    crop: bool = False
    debug_matches: bool = False


@dataclass
class FeatureMetricRow:
    dataset_name: str
    feature_algo: str
    keypoints_count_per_image: List[int]
    raw_matches_count: int
    good_matches_count: int
    ransac_inliers_count: int
    inlier_ratio: float
    homography_success: bool
    stitch_success: bool
    runtime_seconds: float


def selected_algorithms(feature: str) -> List[str]:
    return FEATURE_ALGOS if feature == "all" else [feature]


def prepare_images(images, config: FeatureExperimentConfig):
    preprocess_config = PreprocessConfig(
        resize_width=config.resize_width,
        gamma=config.gamma,
        clahe=config.clahe,
    )
    processed = preprocess_images(images, preprocess_config)
    if config.exposure:
        processed = match_mean_std_luminance(processed)
    return processed


def run_single_dataset(dataset_dir: Path, config: FeatureExperimentConfig) -> List[FeatureMetricRow]:
    dataset_name = dataset_dir.name
    dataset_vis_dir = config.output_root / "visualizations" / dataset_name
    dataset_panorama_dir = config.output_root / "panoramas" / dataset_name
    dataset_vis_dir.mkdir(parents=True, exist_ok=True)
    dataset_panorama_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_image_files(dataset_dir)
    images, valid_paths = read_images(image_paths)
    save_image(dataset_vis_dir / "input_thumbnails.jpg", make_thumbnail_grid(images))

    if len(images) < 2:
        print(f"[WARN] {dataset_name}: need at least two readable images.")
        return []

    prepared_images = prepare_images(images, config)
    save_image(dataset_vis_dir / "processed_thumbnails.jpg", make_thumbnail_grid(prepared_images))

    rows = []
    for feature_algo in selected_algorithms(config.feature):
        print(f"[INFO] {dataset_name}: feature={feature_algo}")
        algo_vis_dir = dataset_vis_dir / feature_algo
        algo_vis_dir.mkdir(parents=True, exist_ok=True)

        analysis = analyze_features(
            prepared_images,
            feature_algo,
            algo_vis_dir,
            debug_matches=config.debug_matches,
        )
        panorama_path = dataset_panorama_dir / f"{dataset_name}_{feature_algo}.jpg"
        stitch_result = stitch_with_feature_algorithm(
            prepared_images,
            feature_algo,
            panorama_path,
            crop_enabled=config.crop,
        )

        rows.append(
            FeatureMetricRow(
                dataset_name=dataset_name,
                feature_algo=feature_algo,
                keypoints_count_per_image=analysis.keypoints_count_per_image,
                raw_matches_count=analysis.raw_matches_count,
                good_matches_count=analysis.good_matches_count,
                ransac_inliers_count=analysis.ransac_inliers_count,
                inlier_ratio=analysis.inlier_ratio,
                homography_success=analysis.homography_success,
                stitch_success=stitch_result.stitch_success,
                runtime_seconds=stitch_result.runtime_seconds,
            )
        )

    write_feature_metrics(dataset_vis_dir / "feature_metrics.csv", rows)
    return rows


def run_feature_experiments(datasets: List[Path], config: FeatureExperimentConfig) -> None:
    ensure_output_dirs(config.output_root)
    all_rows = []
    for dataset_dir in datasets:
        print(f"[INFO] Processing dataset: {dataset_dir}")
        rows = run_single_dataset(dataset_dir, config)
        all_rows.extend(rows)

    metrics_path = config.output_root / "feature_metrics.csv"
    write_feature_metrics(metrics_path, all_rows)
    write_markdown_table(config.output_root / "tables" / "feature_comparison_table.md", all_rows)
    save_comparison_charts(config.output_root, all_rows)
    print(f"[INFO] Feature metrics saved to {metrics_path}")
