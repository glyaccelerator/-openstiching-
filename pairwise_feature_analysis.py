import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.visualization import save_image


@dataclass
class PairFeatureResult:
    pair_index: str
    keypoints_count_img1: int
    keypoints_count_img2: int
    raw_matches_count: int
    good_matches_count: int
    ransac_inliers_count: int
    inlier_ratio: float
    homography_success: bool
    runtime_seconds: float
    failure_reason: str = ""


def create_detector(feature_algo: str):
    if feature_algo == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is unavailable. Install opencv-contrib-python or a newer opencv-python.")
        return cv2.SIFT_create(nfeatures=4000)
    if feature_algo == "orb":
        return cv2.ORB_create(nfeatures=4000)
    if feature_algo == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported feature algorithm: {feature_algo}")


def create_matcher(feature_algo: str):
    if feature_algo == "sift":
        return cv2.BFMatcher(cv2.NORM_L2)
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def detect_features(image: np.ndarray, detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints or [], descriptors


def ratio_test(knn_matches, ratio: float = 0.75):
    good = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        first, second = candidates[0], candidates[1]
        if first.distance < ratio * second.distance:
            good.append(first)
    return good


def estimate_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    return cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)


def draw_keypoints(image: np.ndarray, keypoints, output_path: Path) -> None:
    visual = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    save_image(output_path, visual)


def draw_matches(image1, kp1, image2, kp2, matches, output_path: Path, mask: Optional[List[int]] = None) -> None:
    shown = sorted(matches, key=lambda item: item.distance)[:120]
    shown_mask = None
    if mask is not None:
        index_by_match = {id(match): idx for idx, match in enumerate(matches)}
        shown_mask = [mask[index_by_match[id(match)]] for match in shown]
    visual = cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        shown,
        None,
        matchesMask=shown_mask,
        matchColor=(0, 220, 0) if shown_mask else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_image(output_path, visual)


def analyze_pairwise_features(
    images: List[np.ndarray],
    feature_algo: str,
    output_dir: Path,
    debug_matches: bool = False,
) -> List[PairFeatureResult]:
    """Analyze adjacent image pairs with feature detection, matching, RANSAC and homography."""
    start_all = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    keypoint_dir = output_dir / "keypoints"
    match_dir = output_dir / "matches"
    inlier_dir = output_dir / "ransac_inliers"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    match_dir.mkdir(parents=True, exist_ok=True)
    inlier_dir.mkdir(parents=True, exist_ok=True)

    try:
        detector = create_detector(feature_algo)
        matcher = create_matcher(feature_algo)
    except Exception as exc:
        return [
            PairFeatureResult(
                pair_index="all",
                keypoints_count_img1=0,
                keypoints_count_img2=0,
                raw_matches_count=0,
                good_matches_count=0,
                ransac_inliers_count=0,
                inlier_ratio=0.0,
                homography_success=False,
                runtime_seconds=time.perf_counter() - start_all,
                failure_reason=str(exc),
            )
        ]

    keypoints = []
    descriptors = []
    for idx, image in enumerate(images, start=1):
        kp, desc = detect_features(image, detector)
        keypoints.append(kp)
        descriptors.append(desc)
        draw_keypoints(image, kp, keypoint_dir / f"keypoints_img{idx}.jpg")

    results = []
    for idx in range(len(images) - 1):
        pair_start = time.perf_counter()
        pair_name = f"{idx + 1:02d}"
        desc1, desc2 = descriptors[idx], descriptors[idx + 1]
        failure_reason = ""
        raw_count = good_count = inlier_count = 0
        homography_success = False
        good_matches = []
        mask_list = None

        try:
            if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
                raise RuntimeError("Not enough descriptors for matching.")
            knn = matcher.knnMatch(desc1, desc2, k=2)
            raw_count = len(knn)
            good_matches = ratio_test(knn)
            good_count = len(good_matches)
            homography, mask = estimate_homography(keypoints[idx], keypoints[idx + 1], good_matches)
            if mask is not None:
                mask_list = mask.ravel().astype(int).tolist()
                inlier_count = int(mask.sum())
            homography_success = homography is not None and inlier_count >= 4
            if not homography_success:
                failure_reason = "Homography failed or too few RANSAC inliers."
        except Exception as exc:
            failure_reason = str(exc)

        if debug_matches or good_matches:
            draw_matches(
                images[idx],
                keypoints[idx],
                images[idx + 1],
                keypoints[idx + 1],
                good_matches,
                match_dir / f"matches_pair_{pair_name}.jpg",
            )
            draw_matches(
                images[idx],
                keypoints[idx],
                images[idx + 1],
                keypoints[idx + 1],
                good_matches,
                inlier_dir / f"ransac_inliers_pair_{pair_name}.jpg",
                mask=mask_list,
            )

        results.append(
            PairFeatureResult(
                pair_index=pair_name,
                keypoints_count_img1=len(keypoints[idx]),
                keypoints_count_img2=len(keypoints[idx + 1]),
                raw_matches_count=raw_count,
                good_matches_count=good_count,
                ransac_inliers_count=inlier_count,
                inlier_ratio=(inlier_count / good_count) if good_count else 0.0,
                homography_success=homography_success,
                runtime_seconds=time.perf_counter() - pair_start,
                failure_reason=failure_reason,
            )
        )
    return results
