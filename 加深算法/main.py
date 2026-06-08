import argparse
from pathlib import Path

from src.experiment import FeatureExperimentConfig, run_feature_experiments
from src.io_utils import discover_datasets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature algorithm comparison experiments for panorama stitching."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single image folder, e.g. ../data/set1")
    group.add_argument("--batch", type=Path, help="Batch folder containing image sets")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root folder")
    parser.add_argument(
        "--feature",
        choices=["sift", "orb", "akaze", "all"],
        default="all",
        help="Feature algorithm used by both analysis and OpenStitching.",
    )
    parser.add_argument("--resize-width", type=int, default=1200, help="Resize images to this width")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE enhancement")
    parser.add_argument("--exposure", action="store_true", help="Enable simple exposure compensation")
    parser.add_argument("--crop", action="store_true", help="Enable automatic black border cropping")
    parser.add_argument(
        "--debug-matches",
        action="store_true",
        help="Save detailed keypoint, match and RANSAC inlier visualizations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [args.input] if args.input else discover_datasets(args.batch)
    if not datasets:
        print("[ERROR] No image datasets found. Please check --input or --batch.")
        return

    config = FeatureExperimentConfig(
        output_root=args.output,
        feature=args.feature,
        resize_width=args.resize_width,
        gamma=args.gamma,
        clahe=args.clahe,
        exposure=args.exposure,
        crop=args.crop,
        debug_matches=args.debug_matches,
    )
    run_feature_experiments(datasets, config)


if __name__ == "__main__":
    main()
