Category: 08_many_images_wide_view

Reason: Five or more continuous images; useful for wide-view multi-image panorama stitching.

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
- set_many_images_1: source=download_zijing, images=6, resolution=1200x795, transform=none
- set_many_images_2: source=set1_uav_aerial, images=6, resolution=2000x1500, transform=none
- set_many_images_3: source=set2_cmu_campus, images=6, resolution=1500x1112, transform=none