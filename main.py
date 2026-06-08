import argparse
from pathlib import Path

from src.io_utils import discover_datasets, ensure_output_dirs
from src.metrics import write_metrics
from src.preprocessing import PreprocessConfig
from src.stitching_pipeline import run_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Panorama stitching experiment system based on OpenStitching."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Single image folder, e.g. data/set1")
    group.add_argument("--batch", type=Path, help="Batch folder containing image sets, e.g. data")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output root folder")
    parser.add_argument("--resize-width", type=int, default=1200, help="Resize images to this width")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma correction value")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement")
    parser.add_argument("--exposure", action="store_true", help="Enable simple exposure compensation")
    parser.add_argument("--crop", action="store_true", help="Enable automatic black border cropping")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_output_dirs(args.output)

    if args.input:
        datasets = [args.input]
    else:
        datasets = discover_datasets(args.batch)

    if not datasets:
        print("No image datasets found. Please check --input or --batch path.")
        return

    config = PreprocessConfig(
        resize_width=args.resize_width,
        gamma=args.gamma,
        clahe=args.clahe,
    )

    results = []
    for dataset in datasets:
        print(f"[INFO] Processing {dataset}")
        result = run_dataset(
            dataset_dir=dataset,
            output_root=args.output,
            preprocess_config=config,
            exposure_enabled=args.exposure,
            crop_enabled=args.crop,
        )
        results.append(result)
        print(
            f"[INFO] {result.dataset_name}: success={result.stitch_success}, "
            f"runtime={result.runtime_seconds:.2f}s, output={result.output_width}x{result.output_height}"
        )

    write_metrics(args.output / "metrics.csv", results)
    print(f"[INFO] Metrics saved to {args.output / 'metrics.csv'}")


if __name__ == "__main__":
    main()
