from pathlib import Path
from typing import List

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_files(folder: Path) -> List[Path]:
    """Return image files in a folder, sorted by file name."""
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
    """Find subfolders that contain at least one image."""
    if not batch_folder.exists() or not batch_folder.is_dir():
        return []
    datasets = []
    for child in sorted(batch_folder.iterdir(), key=lambda path: path.name.lower()):
        if child.is_dir() and list_image_files(child):
            datasets.append(child)
    return datasets


def read_images(image_paths: List[Path]):
    """Read images with OpenCV BGR format and skip unreadable files."""
    images = []
    valid_paths = []
    for path in image_paths:
        data = None
        try:
            import numpy as np

            data = np.fromfile(str(path), dtype=np.uint8)
        except Exception:
            data = None
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data is not None and data.size else cv2.imread(str(path))
        if image is not None:
            images.append(image)
            valid_paths.append(path)
    return images, valid_paths


def ensure_output_dirs(output_root: Path) -> None:
    """Create all output folders used by the experiment."""
    (output_root / "panoramas").mkdir(parents=True, exist_ok=True)
    (output_root / "visualizations").mkdir(parents=True, exist_ok=True)
