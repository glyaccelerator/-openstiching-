Category: 03_map_or_satellite

Reason: Aerial/map-like views; useful for near-planar mosaicing and satellite-style stitching experiments.

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
- set_map_1: source=download_CMU2, images=4, resolution=1300x867, transform=none
- set_map_2: source=download_NSH, images=4, resolution=1300x867, transform=none
- set_map_3: source=download_flower, images=4, resolution=1200x795, transform=none