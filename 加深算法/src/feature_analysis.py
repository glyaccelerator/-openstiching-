from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .io_utils import save_image


@dataclass
class PairMatchResult:
    pair_name: str
    raw_matches_count: int
    good_matches_count: int
    ransac_inliers_count: int
    homography_success: bool


@dataclass
class FeatureAnalysisResult:
    keypoints_count_per_image: List[int]
    raw_matches_count: int
    good_matches_count: int
    ransac_inliers_count: int
    inlier_ratio: float
    homography_success: bool


def create_detector(feature_algo: str):
    """Create OpenCV feature detector/descriptor by algorithm name."""
    if feature_algo == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("Current OpenCV build does not provide SIFT_create.")
        return cv2.SIFT_create(nfeatures=4000)
    if feature_algo == "orb":
        return cv2.ORB_create(nfeatures=4000)
    if feature_algo == "akaze":
        return cv2.AKAZE_create()
    raise ValueError(f"Unsupported feature algorithm: {feature_algo}")


def create_matcher(feature_algo: str):
    """Create descriptor matcher. SIFT uses L2 distance; ORB/AKAZE use Hamming."""
    if feature_algo == "sift":
        return cv2.BFMatcher(cv2.NORM_L2)
    return cv2.BFMatcher(cv2.NORM_HAMMING)


def detect_keypoints_and_descriptors(images: List[np.ndarray], feature_algo: str):
    detector = create_detector(feature_algo)
    keypoints_list = []
    descriptors_list = []
    gray_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        keypoints_list.append(keypoints or [])
        descriptors_list.append(descriptors)
        gray_images.append(gray)
    return keypoints_list, descriptors_list, gray_images


def ratio_test_matches(knn_matches, ratio: float = 0.75):
    good_matches = []
    for candidates in knn_matches:
        if len(candidates) < 2:
            continue
        first, second = candidates[0], candidates[1]
        if first.distance < ratio * second.distance:
            good_matches.append(first)
    return good_matches


def estimate_homography(
    keypoints_a,
    keypoints_b,
    good_matches,
    reproj_threshold: float = 4.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(good_matches) < 4:
        return None, None
    points_a = np.float32([keypoints_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_b = np.float32([keypoints_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(points_a, points_b, cv2.RANSAC, reproj_threshold)
    return homography, mask


def draw_keypoint_visuals(
    images: List[np.ndarray],
    keypoints_list,
    output_dir: Path,
    feature_algo: str,
) -> None:
    keypoint_dir = output_dir / "keypoints"
    keypoint_dir.mkdir(parents=True, exist_ok=True)
    for idx, (image, keypoints) in enumerate(zip(images, keypoints_list), start=1):
        visual = cv2.drawKeypoints(
            image,
            keypoints,
            None,
            color=(0, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        save_image(keypoint_dir / f"{feature_algo}_image_{idx:02d}_keypoints.jpg", visual)


def draw_match_visuals(
    image_a: np.ndarray,
    image_b: np.ndarray,
    keypoints_a,
    keypoints_b,
    matches,
    inlier_mask,
    output_dir: Path,
    feature_algo: str,
    pair_index: int,
    debug_matches: bool,
) -> None:
    match_dir = output_dir / "matches"
    inlier_dir = output_dir / "ransac_inliers"
    match_dir.mkdir(parents=True, exist_ok=True)
    inlier_dir.mkdir(parents=True, exist_ok=True)

    draw_count = 120 if debug_matches else 50
    shown_matches = sorted(matches, key=lambda item: item.distance)[:draw_count]
    match_visual = cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        shown_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_image(match_dir / f"{feature_algo}_pair_{pair_index:02d}_good_matches.jpg", match_visual)

    if inlier_mask is None or not shown_matches:
        inlier_visual = match_visual
    else:
        full_mask = inlier_mask.ravel().tolist()
        shown_mask = [full_mask[matches.index(match)] for match in shown_matches]
        inlier_visual = cv2.drawMatches(
            image_a,
            keypoints_a,
            image_b,
            keypoints_b,
            shown_matches,
            None,
            matchesMask=shown_mask,
            matchColor=(0, 220, 0),
            singlePointColor=(80, 80, 80),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
    save_image(inlier_dir / f"{feature_algo}_pair_{pair_index:02d}_ransac_inliers.jpg", inlier_visual)


def analyze_features(
    images: List[np.ndarray],
    feature_algo: str,
    output_dir: Path,
    debug_matches: bool = False,
) -> FeatureAnalysisResult:
    """Analyze adjacent image pairs with a selected feature algorithm."""
    keypoints_list, descriptors_list, _ = detect_keypoints_and_descriptors(images, feature_algo)
    draw_keypoint_visuals(images, keypoints_list, output_dir, feature_algo)

    matcher = create_matcher(feature_algo)
    pair_results = []
    for idx in range(len(images) - 1):
        desc_a = descriptors_list[idx]
        desc_b = descriptors_list[idx + 1]
        if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
            pair_results.append(PairMatchResult(f"{idx + 1}-{idx + 2}", 0, 0, 0, False))
            continue

        knn_matches = matcher.knnMatch(desc_a, desc_b, k=2)
        good_matches = ratio_test_matches(knn_matches)
        homography, inlier_mask = estimate_homography(
            keypoints_list[idx], keypoints_list[idx + 1], good_matches
        )
        inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
        homography_success = homography is not None and inliers >= 4

        draw_match_visuals(
            images[idx],
            images[idx + 1],
            keypoints_list[idx],
            keypoints_list[idx + 1],
            good_matches,
            inlier_mask,
            output_dir,
            feature_algo,
            idx + 1,
            debug_matches,
        )
        pair_results.append(
            PairMatchResult(
                pair_name=f"{idx + 1}-{idx + 2}",
                raw_matches_count=len(knn_matches),
                good_matches_count=len(good_matches),
                ransac_inliers_count=inliers,
                homography_success=homography_success,
            )
        )

    raw_total = sum(item.raw_matches_count for item in pair_results)
    good_total = sum(item.good_matches_count for item in pair_results)
    inlier_total = sum(item.ransac_inliers_count for item in pair_results)
    return FeatureAnalysisResult(
        keypoints_count_per_image=[len(keypoints) for keypoints in keypoints_list],
        raw_matches_count=raw_total,
        good_matches_count=good_total,
        ransac_inliers_count=inlier_total,
        inlier_ratio=(inlier_total / good_total) if good_total > 0 else 0.0,
        homography_success=bool(pair_results) and all(item.homography_success for item in pair_results),
    )
