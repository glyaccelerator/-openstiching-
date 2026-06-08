Category: 09_failure_cases

Reason: Hard cases with sparse overlap or sky/smooth regions; useful for discussing failure modes.

Source datasets referenced:
- visionxiang/Image-Stitching-Dataset: https://github.com/visionxiang/Image-Stitching-Dataset
  Note: Index repository for SVA, APAP, Parallax-tolerant, SEAGULL, NISwGSP, OpenPano and other stitching datasets.
- AutoPanoStitch Stitching Datasets Compilation: https://www.autopanostitch.com/
  Note: Compilation of classical stitching datasets; public mirrors may change over time.
- UDIS-D: https://github.com/nie-lang/UnsupervisedDeepImageStitching
  Note: Large unsupervised deep image stitching dataset with overlap, parallax, indoor/outdoor/night scenes.
- UAV-image-mosaicing-dataset / UMCD: https://www.umcd-dataset.net/
  Note: UAV mosaicking dataset; the official UMCD site currently asks users to request a password by email.
- HPatches: http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz
  Note: Viewpoint and illumination sequence dataset; full release is large and optional.
- OpenPano example data: https://github.com/ppwwyyxx/OpenPano/releases/download/0.1/example-data.tgz
  Note: Small public panorama example data used as the default runnable source.

Image groups:
- set_failure_1: source=set1_uav_aerial, images=2, resolution=2000x1500, transform=failure
- set_failure_2: source=set2_cmu_campus, images=2, resolution=1500x1112, transform=failure
- set_failure_3: source=set3_zijing_campus, images=2, resolution=1200x795, transform=failure