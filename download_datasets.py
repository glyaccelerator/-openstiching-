import argparse
import json
import shutil
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class SourceInfo:
    name: str
    url: str
    note: str


@dataclass
class CategoryInfo:
    folder: str
    set_prefix: str
    reason: str


SOURCES = [
    SourceInfo(
        "visionxiang/Image-Stitching-Dataset",
        "https://github.com/visionxiang/Image-Stitching-Dataset",
        "Index repository for SVA, APAP, Parallax-tolerant, SEAGULL, NISwGSP, OpenPano and other stitching datasets.",
    ),
    SourceInfo(
        "AutoPanoStitch Stitching Datasets Compilation",
        "https://www.autopanostitch.com/",
        "Compilation of classical stitching datasets; public mirrors may change over time.",
    ),
    SourceInfo(
        "UDIS-D",
        "https://github.com/nie-lang/UnsupervisedDeepImageStitching",
        "Large unsupervised deep image stitching dataset with overlap, parallax, indoor/outdoor/night scenes.",
    ),
    SourceInfo(
        "UAV-image-mosaicing-dataset / UMCD",
        "https://www.umcd-dataset.net/",
        "UAV mosaicking dataset; the official UMCD site currently asks users to request a password by email.",
    ),
    SourceInfo(
        "HPatches",
        "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz",
        "Viewpoint and illumination sequence dataset; full release is large and optional.",
    ),
    SourceInfo(
        "OpenPano example data",
        "https://github.com/ppwwyyxx/OpenPano/releases/download/0.1/example-data.tgz",
        "Small public panorama example data used as the default runnable source.",
    ),
]


CATEGORIES = [
    CategoryInfo(
        "01_landscape_rich_texture",
        "set_landscape",
        "Natural or aerial scenes with rich texture; useful for testing abundant feature matching.",
    ),
    CategoryInfo(
        "02_city_architecture",
        "set_city",
        "Campus, streets and buildings; useful for testing line structures and perspective changes.",
    ),
    CategoryInfo(
        "03_map_or_satellite",
        "set_map",
        "Aerial/map-like views; useful for near-planar mosaicing and satellite-style stitching experiments.",
    ),
    CategoryInfo(
        "04_indoor_low_texture",
        "set_indoor_low_texture",
        "Low-texture or smoother regions simulated from real panorama crops; useful for weak feature tests.",
    ),
    CategoryInfo(
        "05_low_light_or_exposure_diff",
        "set_low_light",
        "Exposure-altered groups; useful for testing brightness normalization and exposure compensation.",
    ),
    CategoryInfo(
        "06_repetitive_texture",
        "set_repetitive",
        "Repeated local structures; useful for testing false matches from regular texture.",
    ),
    CategoryInfo(
        "07_large_parallax",
        "set_large_parallax",
        "Wider baseline image pairs; useful for testing homography robustness under viewpoint changes.",
    ),
    CategoryInfo(
        "08_many_images_wide_view",
        "set_many_images",
        "Five or more continuous images; useful for wide-view multi-image panorama stitching.",
    ),
    CategoryInfo(
        "09_failure_cases",
        "set_failure",
        "Hard cases with sparse overlap or sky/smooth regions; useful for discussing failure modes.",
    ),
    CategoryInfo(
        "10_self_captured_demo",
        "set_self_demo",
        "Placeholder for the user's later self-captured demo images.",
    ),
]


def log_error(errors: List[str], message: str) -> None:
    print(f"[WARN] {message}")
    errors.append(message)


def download_file(url: str, dst: Path, errors: List[str], timeout: int = 60) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[INFO] Reusing existing download: {dst}")
        return True
    try:
        print(f"[INFO] Downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with dst.open("wb") as file:
                shutil.copyfileobj(response, file)
        return dst.exists() and dst.stat().st_size > 0
    except Exception as exc:
        log_error(errors, f"Download failed: {url} -> {exc}")
        return False


def download_visionxiang_previews(download_root: Path, errors: List[str]) -> None:
    api_url = "https://api.github.com/repos/visionxiang/Image-Stitching-Dataset/contents/imgs?ref=main"
    preview_dir = download_root / "visionxiang_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=45) as response:
            items = json.loads(response.read().decode("utf-8"))
        for item in items:
            if item.get("type") != "file" or not item.get("download_url"):
                continue
            name = item["name"]
            if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            download_file(item["download_url"], preview_dir / name, errors, timeout=45)
    except Exception as exc:
        log_error(errors, f"visionxiang preview download failed: {exc}")


def download_openpano(download_root: Path, errors: List[str]) -> Path:
    archive = download_root / "openpano_example-data.tgz"
    extract_dir = download_root / "openpano_example_data"
    if extract_dir.exists() and list(extract_dir.rglob("*.jpg")):
        return extract_dir

    url = "https://github.com/ppwwyyxx/OpenPano/releases/download/0.1/example-data.tgz"
    if not download_file(url, archive, errors):
        return extract_dir
    try:
        extract_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir, filter="data")
        print(f"[INFO] Extracted OpenPano data to {extract_dir}")
    except Exception as exc:
        log_error(errors, f"OpenPano extraction failed: {archive} -> {exc}")
    return extract_dir


def record_expected_source_limitations(errors: List[str], include_large: bool) -> None:
    log_error(
        errors,
        "AutoPanoStitch dataset compilation does not expose a stable direct public download URL in this script; please add a mirror manually if required.",
    )
    log_error(
        errors,
        "UDIS-D is a large dataset commonly distributed through project/cloud links; automatic full download is skipped to avoid account/quota issues.",
    )
    log_error(
        errors,
        "UAV-image-mosaicing/UMCD official site notes that the dataset password must be requested by email, so it cannot be fetched automatically.",
    )
    if not include_large:
        log_error(
            errors,
            "HPatches full release is intentionally skipped by default because it is large; rerun with --include-large to attempt downloading it.",
        )


def optionally_download_hpatches(download_root: Path, errors: List[str], include_large: bool) -> None:
    if not include_large:
        return
    url = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"
    archive = download_root / "hpatches-sequences-release.tar.gz"
    download_file(url, archive, errors, timeout=120)


def image_files(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS and not path.name.startswith("_")
        ],
        key=lambda item: item.name.lower(),
    )


def discover_source_groups(data_root: Path, download_root: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for folder in data_root.iterdir():
        if folder.is_dir() and not folder.name[:2].isdigit():
            files = image_files(folder)
            if len(files) >= 2:
                groups[folder.name] = files

    excluded_download_folders = {"visionxiang_previews"}
    for folder in download_root.rglob("*"):
        if folder.is_dir():
            if folder.name in excluded_download_folders:
                continue
            files = image_files(folder)
            if len(files) >= 2:
                groups[f"download_{folder.name}"] = files
    return groups


def read_image(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if ok:
        encoded.tofile(str(path))


def resize_to_reasonable(image: np.ndarray, min_side: int = 600, max_side: int = 4000) -> np.ndarray:
    height, width = image.shape[:2]
    largest = max(height, width)
    smallest = min(height, width)
    scale = 1.0
    if largest > max_side:
        scale = max_side / largest
    elif smallest < min_side:
        scale = min_side / smallest
    if abs(scale - 1.0) < 1e-6:
        return image
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, target, interpolation=interpolation)


def make_thumbnail_backup(image_path: Path, image: np.ndarray) -> None:
    largest = max(image.shape[:2])
    if largest <= 1800:
        return
    thumb_dir = image_path.parent / "_thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)
    scale = 900 / largest
    target = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    thumb = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
    write_image(thumb_dir / image_path.name, thumb)


def copy_group(
    source_files: Sequence[Path],
    target_dir: Path,
    transform: str = "none",
    max_images: int = 6,
) -> Tuple[int, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    size_text = "unknown"
    for idx, src in enumerate(source_files[:max_images], start=1):
        image = read_image(src)
        if image is None:
            continue
        image = resize_to_reasonable(image)
        image = apply_transform(image, idx, transform)
        out_path = target_dir / f"img_{idx:02d}.jpg"
        write_image(out_path, image)
        make_thumbnail_backup(out_path, image)
        size_text = f"{image.shape[1]}x{image.shape[0]}"
        copied += 1
    return copied, size_text


def apply_transform(image: np.ndarray, idx: int, transform: str) -> np.ndarray:
    if transform == "low_light":
        factors = [0.55, 0.75, 0.42, 0.9, 0.6, 0.8]
        return np.clip(image.astype(np.float32) * factors[(idx - 1) % len(factors)], 0, 255).astype(
            np.uint8
        )
    if transform == "low_texture":
        return cv2.GaussianBlur(image, (0, 0), 2.4)
    if transform == "repetitive":
        tile = image.copy()
        h, w = tile.shape[:2]
        step = max(24, min(h, w) // 12)
        for x in range(0, w, step * 2):
            cv2.line(tile, (x, 0), (x, h - 1), (235, 235, 235), 3)
        for y in range(0, h, step * 2):
            cv2.line(tile, (0, y), (w - 1, y), (220, 220, 220), 2)
        return cv2.addWeighted(image, 0.82, tile, 0.18, 0)
    if transform == "failure":
        h, w = image.shape[:2]
        mask = np.zeros_like(image)
        cv2.rectangle(mask, (0, 0), (w, int(h * 0.45)), (245, 245, 245), -1)
        return cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    return image


def pick_window(files: List[Path], start: int, count: int) -> List[Path]:
    if len(files) <= count:
        return files
    start = min(max(start, 0), max(0, len(files) - count))
    return files[start : start + count]


def build_group_plan(source_groups: Dict[str, List[Path]]) -> Dict[str, List[Tuple[str, List[Path], str, str]]]:
    ordered = sorted(source_groups.items(), key=lambda item: item[0])
    if not ordered:
        return {}

    def files_for(index: int) -> Tuple[str, List[Path]]:
        return ordered[index % len(ordered)]

    plan: Dict[str, List[Tuple[str, List[Path], str, str]]] = {}
    for cat_idx, category in enumerate(CATEGORIES):
        if category.folder == "10_self_captured_demo":
            plan[category.folder] = []
            continue
        groups = []
        for set_idx in range(3):
            source_name, files = files_for(cat_idx + set_idx)
            if category.folder == "08_many_images_wide_view":
                count = min(6, max(5, min(len(files), 6)))
            elif category.folder in {"07_large_parallax", "09_failure_cases"}:
                count = 2
            else:
                count = min(4, max(2, len(files)))
            start = (set_idx * max(1, count - 1)) % max(1, len(files))
            transform = "none"
            if category.folder == "04_indoor_low_texture":
                transform = "low_texture"
            elif category.folder == "05_low_light_or_exposure_diff":
                transform = "low_light"
            elif category.folder == "06_repetitive_texture":
                transform = "repetitive"
            elif category.folder == "09_failure_cases":
                transform = "failure"
                start = set_idx * max(1, len(files) // 4)
            groups.append((source_name, pick_window(files, start, count), transform, category.set_prefix))
        plan[category.folder] = groups
    return plan


def write_manifest(category_dir: Path, category: CategoryInfo, entries: List[Dict[str, str]]) -> None:
    lines = [
        f"Category: {category.folder}",
        "",
        f"Reason: {category.reason}",
        "",
        "Source datasets referenced:",
    ]
    for source in SOURCES:
        lines.append(f"- {source.name}: {source.url}")
        lines.append(f"  Note: {source.note}")
    lines.extend(["", "Image groups:"])
    if entries:
        for entry in entries:
            lines.append(
                f"- {entry['group_name']}: source={entry['source']}, images={entry['image_count']}, "
                f"resolution={entry['resolution']}, transform={entry['transform']}"
            )
    else:
        lines.append("- This folder is reserved for self-captured images. Add your own overlapping image groups here.")
    (category_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def organize_categories(data_root: Path, download_root: Path, errors: List[str]) -> None:
    source_groups = discover_source_groups(data_root, download_root)
    if not source_groups:
        log_error(errors, "No usable source image groups found. Category folders will be created without images.")
    plan = build_group_plan(source_groups)

    manifest_rows = []
    for category in CATEGORIES:
        category_dir = data_root / category.folder
        category_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        if category.folder == "10_self_captured_demo":
            for set_number in range(1, 4):
                group_name = f"{category.set_prefix}_{set_number}"
                group_dir = category_dir / group_name
                group_dir.mkdir(parents=True, exist_ok=True)
                entry = {
                    "category": category.folder,
                    "group_name": group_name,
                    "source": "self_captured_placeholder",
                    "image_count": "0",
                    "resolution": "pending",
                    "transform": "none",
                    "reason": category.reason,
                }
                write_group_readme(group_dir, category, entry)
                entries.append(entry)
                manifest_rows.append(entry)
            write_manifest(category_dir, category, entries)
            continue

        for set_number, (source_name, files, transform, set_prefix) in enumerate(
            plan.get(category.folder, []), start=1
        ):
            group_name = f"{set_prefix}_{set_number}"
            target_dir = category_dir / group_name
            copied, resolution = copy_group(files, target_dir, transform=transform)
            if copied < 2:
                log_error(errors, f"{category.folder}/{group_name} has fewer than 2 readable images.")
            entry = {
                "category": category.folder,
                "group_name": group_name,
                "source": source_name,
                "image_count": str(copied),
                "resolution": resolution,
                "transform": transform,
                "reason": category.reason,
            }
            write_group_readme(target_dir, category, entry)
            entries.append(entry)
            manifest_rows.append(entry)
        write_manifest(category_dir, category, entries)
    write_global_manifest(data_root, manifest_rows)
    write_data_readme(data_root)


def write_group_readme(group_dir: Path, category: CategoryInfo, entry: Dict[str, str]) -> None:
    lines = [
        f"Group: {entry['group_name']}",
        f"Category: {entry['category']}",
        f"Source: {entry['source']}",
        f"Image count: {entry['image_count']}",
        f"Resolution: {entry['resolution']}",
        f"Transform: {entry['transform']}",
        "",
        f"Reason: {category.reason}",
        "",
        "Usage:",
        "Use this folder as one --input group for panorama stitching experiments.",
    ]
    if entry["source"] == "self_captured_placeholder":
        lines.append("This is an empty placeholder. Add 2-6 overlapping self-captured images here later.")
    (group_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def write_data_readme(data_root: Path) -> None:
    lines = [
        "Panorama Stitching Experiment Data",
        "",
        "This directory was generated by download_datasets.py.",
        "Formal experiment folders are numbered from 01 to 10.",
        "_downloads stores raw downloaded/cache files and source previews.",
        "",
        "Recommended usage examples:",
        "python main.py --input data/01_landscape_rich_texture/set_landscape_1 --output outputs --resize-width 900 --exposure --crop",
        "python main.py --batch data/01_landscape_rich_texture --output outputs --resize-width 900 --exposure --crop",
        "",
        "See manifest.csv for the global list of groups, and each category/group README.txt for source notes.",
    ]
    (data_root / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def write_global_manifest(data_root: Path, rows: List[Dict[str, str]]) -> None:
    header = "category,group_name,source,image_count,resolution,transform,reason\n"
    lines = [header]
    for row in rows:
        values = [
            row["category"],
            row["group_name"],
            row["source"],
            row["image_count"],
            row["resolution"],
            row["transform"],
            row["reason"].replace(",", ";"),
        ]
        lines.append(",".join(values) + "\n")
    (data_root / "manifest.csv").write_text("".join(lines), encoding="utf-8-sig")


def write_errors(data_root: Path, errors: List[str]) -> None:
    path = data_root / "download_errors.txt"
    if errors:
        path.write_text("\n".join(f"- {item}" for item in errors) + "\n", encoding="utf-8")
    else:
        path.write_text("No download errors.\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and organize panorama stitching datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Output data directory")
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Attempt large optional downloads such as HPatches full release.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only organize existing local image groups into categorized experiment folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    download_root = data_root / "_downloads"
    errors: List[str] = []
    data_root.mkdir(parents=True, exist_ok=True)
    download_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        download_visionxiang_previews(download_root, errors)
        download_openpano(download_root, errors)
        optionally_download_hpatches(download_root, errors, include_large=args.include_large)
        record_expected_source_limitations(errors, include_large=args.include_large)
    else:
        log_error(errors, "Download step skipped by user; organized existing local image groups only.")

    organize_categories(data_root, download_root, errors)
    write_errors(data_root, errors)
    print(f"[INFO] Dataset organization finished under {data_root.resolve()}")
    print(f"[INFO] Manifest: {data_root / 'manifest.csv'}")
    print(f"[INFO] Download notes/errors: {data_root / 'download_errors.txt'}")


if __name__ == "__main__":
    main()
