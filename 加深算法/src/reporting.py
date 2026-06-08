import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from .io_utils import save_image


FEATURE_FIELDNAMES = [
    "dataset_name",
    "feature_algo",
    "keypoints_count_per_image",
    "raw_matches_count",
    "good_matches_count",
    "ransac_inliers_count",
    "inlier_ratio",
    "homography_success",
    "stitch_success",
    "runtime_seconds",
]


def write_feature_metrics(path: Path, rows: Iterable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=FEATURE_FIELDNAMES)
        writer.writeheader()
        for result in rows:
            row = asdict(result)
            row["keypoints_count_per_image"] = ";".join(
                str(item) for item in result.keypoints_count_per_image
            )
            row["inlier_ratio"] = f"{result.inlier_ratio:.6f}"
            row["runtime_seconds"] = f"{result.runtime_seconds:.3f}"
            writer.writerow(row)


def write_markdown_table(path: Path, rows: List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| dataset | feature | good matches | inliers | inlier ratio | stitch | runtime(s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.dataset_name} | {row.feature_algo} | {row.good_matches_count} | "
            f"{row.ransac_inliers_count} | {row.inlier_ratio:.3f} | "
            f"{row.stitch_success} | {row.runtime_seconds:.3f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _bar_chart(
    title: str,
    labels: List[str],
    values: List[float],
    output_path: Path,
    width: int = 1200,
    height: int = 520,
) -> None:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    cv2.putText(canvas, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (30, 30, 30), 2)
    if not values:
        save_image(output_path, canvas)
        return

    chart_left, chart_top = 70, 85
    chart_width, chart_height = width - 120, height - 180
    max_value = max(max(values), 1e-6)
    bar_gap = 8
    bar_width = max(10, int((chart_width - bar_gap * (len(values) - 1)) / max(len(values), 1)))
    colors = [(52, 120, 246), (34, 160, 90), (230, 130, 40)]

    for idx, (label, value) in enumerate(zip(labels, values)):
        x1 = chart_left + idx * (bar_width + bar_gap)
        bar_h = int((value / max_value) * chart_height)
        y1 = chart_top + chart_height - bar_h
        color = colors[idx % len(colors)]
        cv2.rectangle(canvas, (x1, y1), (x1 + bar_width, chart_top + chart_height), color, -1)
        cv2.putText(
            canvas,
            f"{value:.2f}" if value < 10 else f"{value:.0f}",
            (x1, max(75, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (25, 25, 25),
            1,
        )
        short_label = label[-18:]
        cv2.putText(
            canvas,
            short_label,
            (x1, chart_top + chart_height + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
        )
    save_image(output_path, canvas)


def save_comparison_charts(output_root: Path, rows: List) -> None:
    labels = [f"{row.dataset_name}-{row.feature_algo}" for row in rows]
    chart_dir = output_root / "tables"
    _bar_chart(
        "Good matches count comparison",
        labels,
        [float(row.good_matches_count) for row in rows],
        chart_dir / "good_matches_comparison.jpg",
    )
    _bar_chart(
        "RANSAC inlier ratio comparison",
        labels,
        [float(row.inlier_ratio) for row in rows],
        chart_dir / "inlier_ratio_comparison.jpg",
    )
    _bar_chart(
        "Runtime comparison seconds",
        labels,
        [float(row.runtime_seconds) for row in rows],
        chart_dir / "runtime_comparison.jpg",
    )
    _bar_chart(
        "Stitch success comparison",
        labels,
        [1.0 if row.stitch_success else 0.0 for row in rows],
        chart_dir / "stitch_success_comparison.jpg",
    )
