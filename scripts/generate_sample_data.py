from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


def save_image(path: Path, image: np.ndarray, quality: int = 92) -> None:
    """Save image to Unicode paths reliably on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def add_texture(canvas: np.ndarray, seed: int) -> np.ndarray:
    """Add deterministic texture so feature matching has enough keypoints."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 9, canvas.shape).astype(np.int16)
    textured = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for _ in range(180):
        x = int(rng.integers(0, canvas.shape[1]))
        y = int(rng.integers(0, canvas.shape[0]))
        radius = int(rng.integers(2, 7))
        color = tuple(int(v) for v in rng.integers(40, 230, size=3))
        cv2.circle(textured, (x, y), radius, color, -1)
    return textured


def create_base_scene(width: int, height: int, seed: int, title: str) -> np.ndarray:
    """Create a synthetic wide scene for panorama testing."""
    rng = np.random.default_rng(seed)
    x_grad = np.linspace(40, 210, width, dtype=np.uint8)
    y_grad = np.linspace(25, 95, height, dtype=np.uint8)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :, 0] = x_grad[None, :]
    canvas[:, :, 1] = y_grad[:, None] + 45
    canvas[:, :, 2] = 180 - x_grad[None, :] // 3

    for idx in range(20):
        x1 = int(rng.integers(20, width - 180))
        y1 = int(rng.integers(30, height - 100))
        x2 = x1 + int(rng.integers(60, 180))
        y2 = y1 + int(rng.integers(30, 110))
        color = tuple(int(v) for v in rng.integers(50, 240, size=3))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    for idx in range(14):
        center = (int(rng.integers(40, width - 40)), int(rng.integers(45, height - 45)))
        axes = (int(rng.integers(18, 75)), int(rng.integers(12, 45)))
        angle = int(rng.integers(0, 180))
        color = tuple(int(v) for v in rng.integers(50, 240, size=3))
        cv2.ellipse(canvas, center, axes, angle, 0, 360, color, 2)

    for x in range(0, width, 90):
        cv2.line(canvas, (x, 0), (x + 130, height), (235, 235, 235), 1)
    for y in range(0, height, 70):
        cv2.line(canvas, (0, y), (width, y + 50), (30, 30, 30), 1)

    cv2.putText(canvas, title, (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 4)
    cv2.putText(canvas, title, (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (245, 245, 245), 2)
    for idx, x in enumerate(range(120, width - 120, 260), start=1):
        cv2.putText(
            canvas,
            f"P{idx}",
            (x, height - 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            3,
        )

    return add_texture(canvas, seed)


def make_overlapping_crops(scene: np.ndarray, image_count: int, crop_width: int, crop_height: int):
    """Crop a wide scene into overlapping input images."""
    height, width = scene.shape[:2]
    y0 = (height - crop_height) // 2
    max_x = width - crop_width
    starts = np.linspace(0, max_x, image_count).astype(int)
    crops = []
    for idx, x0 in enumerate(starts):
        crop = scene[y0 : y0 + crop_height, x0 : x0 + crop_width].copy()
        angle = (idx - image_count / 2) * 0.35
        matrix = cv2.getRotationMatrix2D((crop_width / 2, crop_height / 2), angle, 1.0)
        crop = cv2.warpAffine(crop, matrix, (crop_width, crop_height), borderMode=cv2.BORDER_REFLECT)

        alpha = 0.86 + idx * 0.08
        beta = -10 + idx * 5
        crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        crops.append(crop)
    return crops


def generate_dataset(name: str, seed: int, image_count: int = 4) -> None:
    output_dir = DATA_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)

    scene = create_base_scene(1500, 520, seed, f"Synthetic Panorama {name}")
    crops = make_overlapping_crops(scene, image_count=image_count, crop_width=620, crop_height=420)

    save_image(output_dir / "_full_reference_not_input.jpg", scene)
    for idx, crop in enumerate(crops, start=1):
        save_image(output_dir / f"img_{idx:02d}.jpg", crop)


def main() -> None:
    generate_dataset("set1", seed=7, image_count=4)
    generate_dataset("set2", seed=21, image_count=4)
    generate_dataset("set3", seed=42, image_count=5)
    print(f"Sample datasets generated under: {DATA_DIR}")


if __name__ == "__main__":
    main()
