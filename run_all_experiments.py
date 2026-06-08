import argparse
from pathlib import Path

from src.experiment_system import ExperimentConfig, discover_experiment_datasets, run_experiments


def parse_args():
    parser = argparse.ArgumentParser(description="Run the complete panorama stitching experiment pipeline.")
    parser.add_argument("--data", type=Path, default=Path("data"), help="Dataset root folder")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root folder")
    parser.add_argument("--resize-width", type=int, default=1200, help="Resize width")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE in improved/feature inputs")
    parser.add_argument("--max-images", type=int, default=6, help="Maximum images per group")
    parser.add_argument("--debug-matches", action="store_true", help="Save match visualizations")
    parser.add_argument("--notes", default="full batch experiment", help="Notes written to CSV files")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = discover_experiment_datasets(args.data)
    if not datasets:
        print("[ERROR] No image groups found.")
        return

    config = ExperimentConfig(
        output=args.output,
        mode="all",
        resize_width=args.resize_width,
        gamma=args.gamma,
        clahe=args.clahe,
        exposure=True,
        crop=True,
        feature="all",
        debug_matches=args.debug_matches,
        save_report_assets=True,
        max_images=args.max_images,
        notes=args.notes,
    )
    print(f"[INFO] Running full experiment on {len(datasets)} groups.")
    run_experiments(datasets, config)
    print(f"[INFO] Full experiment finished. Report guide: {args.output / 'report_assets' / 'README_for_report.md'}")


if __name__ == "__main__":
    main()
