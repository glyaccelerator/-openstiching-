import argparse
from pathlib import Path

from src.experiment_system import ExperimentConfig, discover_experiment_datasets, run_experiments


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch panorama stitching experiment system based on OpenStitching and OpenCV."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single image group or category folder.")
    group.add_argument("--batch", type=Path, help="Batch root folder, e.g. data")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root folder")
    parser.add_argument(
        "--mode",
        choices=["baseline", "improved", "feature_experiment", "all"],
        default="all",
        help="Experiment mode.",
    )
    parser.add_argument("--resize-width", type=int, default=1200, help="Resize width before stitching")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE")
    parser.add_argument("--exposure", action="store_true", help="Enable exposure compensation")
    parser.add_argument("--crop", action="store_true", help="Enable black border cropping")
    parser.add_argument(
        "--feature",
        choices=["sift", "orb", "akaze", "all"],
        default="all",
        help="Feature algorithm for feature_experiment mode.",
    )
    parser.add_argument("--debug-matches", action="store_true", help="Save keypoint/match/RANSAC visuals")
    parser.add_argument("--save-report-assets", action="store_true", help="Generate report asset guide")
    parser.add_argument("--max-images", type=int, default=6, help="Maximum images per group")
    parser.add_argument("--notes", default="", help="Notes written to CSV files")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.input if args.input else args.batch
    datasets = discover_experiment_datasets(root)
    if not datasets:
        print("[ERROR] No valid image groups found. Each group needs at least two readable images.")
        return

    config = ExperimentConfig(
        output=args.output,
        mode=args.mode,
        resize_width=args.resize_width,
        gamma=args.gamma,
        clahe=args.clahe,
        exposure=args.exposure,
        crop=args.crop,
        feature=args.feature,
        debug_matches=args.debug_matches,
        save_report_assets=args.save_report_assets,
        max_images=args.max_images,
        notes=args.notes,
    )
    print(f"[INFO] Found {len(datasets)} image groups.")
    run_experiments(datasets, config)
    print(f"[INFO] Done. Results saved to {args.output}")


if __name__ == "__main__":
    main()
