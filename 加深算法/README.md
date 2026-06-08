# 加深算法：特征算法对比实验模块

本目录是独立于外层基础版的实验系统，用于数字图像处理课程大作业中的“算法加深 / 特征算法对比”部分。系统支持 SIFT、ORB、AKAZE 三种特征点算法，并输出关键点、匹配线、RANSAC 内点、拼接结果和指标表格，便于报告截图和定量分析。

## 环境安装

在项目根目录或本目录下均可安装依赖：

```bash
pip install -r requirements.txt
```

如果外层项目已经安装过 `opencv-python`、`numpy`、`stitching`，可以直接运行。

## 特征算法选择方法

通过 `--feature` 选择算法：

```bash
--feature sift
--feature orb
--feature akaze
--feature all
```

其中：

- `sift`：尺度不变特征，通常匹配质量较稳定，但耗时较高。
- `orb`：二进制特征，速度较快，适合实时或轻量实验。
- `akaze`：非线性尺度空间特征，速度和稳定性介于 SIFT 与 ORB 之间。
- `all`：对同一数据集依次运行 SIFT、ORB、AKAZE，适合报告对比实验。

## 运行批量实验

在本目录运行时，输入路径建议使用 `../data`：

```bash
python main.py --batch ../data --output outputs --feature all --resize-width 900 --gamma 1.0 --clahe --exposure --crop --debug-matches
```

只运行单个数据集：

```bash
python main.py --input ../data/set1_uav_aerial --output outputs --feature all --resize-width 900 --gamma 1.0 --clahe --exposure --crop --debug-matches
```

只对比 ORB：

```bash
python main.py --batch ../data --output outputs --feature orb --resize-width 900 --exposure --crop --debug-matches
```

如果你在外层项目根目录运行，则命令为：

```bash
python 加深算法/main.py --batch data --output 加深算法/outputs --feature all --resize-width 900 --gamma 1.0 --clahe --exposure --crop --debug-matches
```

## 参数说明

- `--input`：单个图片组目录，例如 `../data/set1_uav_aerial`
- `--batch`：批量图片组根目录，例如 `../data`
- `--output`：输出目录
- `--feature`：选择 `sift`、`orb`、`akaze` 或 `all`
- `--resize-width`：将输入图像按宽度缩放，降低计算量
- `--gamma`：Gamma 校正参数
- `--clahe`：启用 CLAHE 局部对比度增强
- `--exposure`：启用 LAB 亮度均值方差曝光补偿
- `--crop`：对拼接结果自动裁剪黑边
- `--debug-matches`：保存更完整的匹配可视化，便于报告截图

## 指标含义

输出总表为：

```text
outputs/feature_metrics.csv
```

字段含义：

- `dataset_name`：数据集名称
- `feature_algo`：特征算法，取值为 `sift`、`orb`、`akaze`
- `keypoints_count_per_image`：每张输入图检测到的关键点数量，使用分号分隔
- `raw_matches_count`：相邻图像之间 KNN 匹配的原始匹配数量总和
- `good_matches_count`：Lowe ratio test 筛选后的优质匹配数量总和
- `ransac_inliers_count`：通过 RANSAC 单应矩阵估计后保留的内点数量总和
- `inlier_ratio`：`ransac_inliers_count / good_matches_count`
- `homography_success`：所有相邻图像对是否都成功估计单应矩阵
- `stitch_success`：OpenStitching 是否成功生成全景图
- `runtime_seconds`：当前算法完成 OpenStitching 拼接的耗时

## 保存结果目录结构

```text
outputs/
├── feature_metrics.csv
├── panoramas/
│   └── set1_uav_aerial/
│       ├── set1_uav_aerial_sift.jpg
│       ├── set1_uav_aerial_orb.jpg
│       └── set1_uav_aerial_akaze.jpg
├── tables/
│   ├── feature_comparison_table.md
│   ├── good_matches_comparison.jpg
│   ├── inlier_ratio_comparison.jpg
│   ├── runtime_comparison.jpg
│   └── stitch_success_comparison.jpg
└── visualizations/
    └── set1_uav_aerial/
        ├── feature_metrics.csv
        ├── input_thumbnails.jpg
        ├── processed_thumbnails.jpg
        ├── sift/
        │   ├── keypoints/
        │   ├── matches/
        │   └── ransac_inliers/
        ├── orb/
        │   ├── keypoints/
        │   ├── matches/
        │   └── ransac_inliers/
        └── akaze/
            ├── keypoints/
            ├── matches/
            └── ransac_inliers/
```

## 如何使用可视化图片生成报告截图

建议报告中使用以下图片：

- 输入图展示：`visualizations/<dataset>/input_thumbnails.jpg`
- 预处理结果：`visualizations/<dataset>/processed_thumbnails.jpg`
- 关键点检测：`visualizations/<dataset>/<algo>/keypoints/*_keypoints.jpg`
- 匹配线对比：`visualizations/<dataset>/<algo>/matches/*_good_matches.jpg`
- RANSAC 内点：`visualizations/<dataset>/<algo>/ransac_inliers/*_ransac_inliers.jpg`
- 拼接结果：`panoramas/<dataset>/<dataset>_<algo>.jpg`
- 算法总体对比：`tables/good_matches_comparison.jpg`、`tables/inlier_ratio_comparison.jpg`、`tables/runtime_comparison.jpg`

写报告时可以按“景观、地图、人文”等不同图片类型分组，比较三种算法的匹配数量、内点率、拼接成功率和耗时。通常可从以下角度分析：

- SIFT 是否获得更高的内点率和更稳定的单应矩阵。
- ORB 是否在耗时方面更有优势。
- AKAZE 是否在纹理较复杂或尺度变化较明显的场景中表现更均衡。
- 预处理、曝光补偿和裁剪是否改善了最终拼接结果的可展示性。
