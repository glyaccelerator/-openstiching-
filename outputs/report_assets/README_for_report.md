# Report Assets Guide

- Total image groups tested: 33
- Baseline success candidates: 27
- Failure records: 18

## Recommended Figures
- Baseline success case: 01_landscape_rich_texture/set_landscape_1
- Improved obvious case: 06_repetitive_texture/set_repetitive_2
- SIFT/ORB/AKAZE comparison case: 01_landscape_rich_texture/set_landscape_1
- Auto-crop black border case: 06_repetitive_texture/set_repetitive_2
- Failure case: 07_large_parallax/set_large_parallax_1

## Category Analysis Templates
- Landscape: rich textures usually provide many keypoints; ORB is fast while SIFT is often stable.
- City/architecture: lines and corners are abundant, but repeated windows may create false matches.
- Map/satellite: planar structure is suitable for homography, but repeated symbols or text can mislead matching.
- Indoor low texture: fewer reliable keypoints may make matching unstable.
- Low light/exposure: Gamma, CLAHE and exposure compensation can improve visible features.
- Repetitive texture: RANSAC is important for removing false matches.
- Large parallax: a single homography may not describe 3D viewpoint changes well.
- Many images: useful for showing multi-image stitching, accumulated drift and cropping.
- Failure cases: use them to explain low overlap, smooth regions, moving objects or strong parallax.

## How To Use
- Use outputs/comparisons/*_comparison.jpg for one-page visual comparison.
- Use outputs/feature_experiment/<dataset>/<algo>/keypoints and matches for algorithm details.
- Use outputs/tables/*.csv for quantitative tables in the report.