from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(folder: Path) -> List[Path]:
    """Return image files sorted by name; reference files starting with '_' are ignored."""
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
    """Find child folders that contain input images."""
    if not batch_folder.exists() or not batch_folder.is_dir():
        return []
    return [
        child
        for child in sorted(batch_folder.iterdir(), key=lambda path: path.name.lower())
        if child.is_dir() and list_image_files(child)
    ]


def read_image(path: Path):
    """Read an image from paths that may contain Chinese characters."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_images(paths: Iterable[Path]) -> Tuple[List[np.ndarray], List[Path]]:
    images = []
    valid_paths = []
    for path in paths:
        image = read_image(path)
        if image is None:
            print(f"[WARN] Cannot read image: {path}")
            continue
        images.append(image)
        valid_paths.append(path)
    return images, valid_paths


def save_image(path: Path, image) -> None:
    """Save an image to paths that may contain Chinese characters."""
    if image is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if ok:
        encoded.tofile(str(path))
    else:
        print(f"[WARN] Cannot save image: {path}")


def ensure_output_dirs(output_root: Path) -> None:
    for child in ["panoramas", "visualizations", "tables"]:
        (output_root / child).mkdir(parents=True, exist_ok=True)
