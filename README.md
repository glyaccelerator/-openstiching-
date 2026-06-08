# 基于 OpenStitching 的全景图像拼接与融合优化系统设计

本项目是数字图像处理课程大作业的批量实验系统，基于 `OpenStitching/stitching` 和 OpenCV 实现。系统不是只调用一次 `Stitcher().stitch()`，而是围绕全景拼接实验报告需要，封装了基础拼接、改进拼接、特征算法对比、CSV 指标统计和报告素材自动生成。

## 三种实验模式

`baseline` 基础版：主要验证 OpenStitching 默认流程能否完成图像拼接。该模式不启用 Gamma、CLAHE、曝光补偿和黑边裁剪，只在必要时按 `--resize-width` 缩放以控制运行时间。

`improved` 改进版：主要通过预处理和后处理提升拼接质量，例如增强暗光图像、减少曝光差异、裁剪黑边。该模式支持 resize、Gamma、CLAHE、曝光补偿和自动裁剪，并记录黑边比例变化。

`feature_experiment` 算法加深版：分析全景拼接核心过程中的特征点检测、匹配和 RANSAC 内点筛选。系统会比较 SIFT、ORB、AKAZE 在不同图像类型下的关键点数量、匹配数量、内点率和耗时。

三者关系可以理解为：

- `baseline` 解决“能不能拼起来”
- `improved` 解决“拼得是否更好看”
- `feature_experiment` 解决“为什么能拼起来、哪种算法更适合”

## 环境安装

建议使用 Python 3.9 或更高版本：

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果 SIFT 不可用，可以尝试：

```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

## 数据类别

推荐使用 `download_datasets.py` 自动整理数据：

```bash
python download_datasets.py
```

数据目录按实验目的分为：

```text
data/
├── 01_landscape_rich_texture/
├── 02_city_architecture/
├── 03_map_or_satellite/
├── 04_indoor_low_texture/
├── 05_low_light_or_exposure_diff/
├── 06_repetitive_texture/
├── 07_large_parallax/
├── 08_many_images_wide_view/
├── 09_failure_cases/
└── 10_self_captured_demo/
```

各类测试目的：

- 景观类：测试普通纹理丰富场景下的拼接稳定性。
- 建筑类：测试直线、角点丰富但重复结构较多的场景。
- 地图/卫星图：测试平面图像、线条结构、尺度变化下的匹配效果。
- 室内弱纹理：测试特征点不足时不同算法表现。
- 低曝光/曝光差异：测试 Gamma、CLAHE、曝光补偿是否有效。
- 重复纹理：测试误匹配和 RANSAC 剔除能力。
- 大视差：测试 Homography 模型的局限性和重影问题。
- 多图大视野：测试多图拼接能力、累积误差和自动裁剪效果。
- 失败案例：用于报告分析为什么拼接失败，不要求全部成功。
- 自采集演示：用于后续答辩展示，可以手动加入自己拍摄的 2-6 张重叠图片。

## 命令行入口

统一入口为 `main.py`：

```bash
python main.py --batch data --output outputs --mode all --resize-width 1200 --gamma 1.0 --clahe --exposure --crop --feature all --debug-matches --save-report-assets
```

参数说明：

- `--input`：指定单个图片组或某个类别目录
- `--batch`：指定批量目录，例如 `data`
- `--output`：输出目录，例如 `outputs`
- `--mode`：`baseline` / `improved` / `feature_experiment` / `all`
- `--resize-width`：统一缩放宽度
- `--gamma`：Gamma 校正参数
- `--clahe`：启用 CLAHE
- `--exposure`：启用曝光补偿
- `--crop`：启用自动裁剪黑边
- `--feature`：`sift` / `orb` / `akaze` / `all`
- `--debug-matches`：保存关键点、匹配线、RANSAC 内点图
- `--save-report-assets`：生成报告素材目录
- `--max-images`：每组最多处理多少张图片，防止过慢
- `--notes`：备注字段，写入 CSV

## 一键完整实验

推荐直接运行：

```bash
python run_all_experiments.py --data data --output outputs --resize-width 1200 --gamma 1.0 --clahe --debug-matches
```

该脚本会自动执行：

1. baseline 基础拼接
2. improved 改进拼接
3. SIFT / ORB / AKAZE 特征算法对比
4. 汇总 CSV 表格
5. 生成对比图
6. 生成 `outputs/report_assets/README_for_report.md`

## 单独运行示例

只跑基础版：

```bash
python main.py --input data/01_landscape_rich_texture/set_landscape_1 --output outputs --mode baseline --resize-width 1200
```

只跑改进版：

```bash
python main.py --input data/05_low_light_or_exposure_diff/set_low_light_1 --output outputs --mode improved --resize-width 1200 --gamma 1.2 --clahe --exposure --crop
```

只跑特征算法对比：

```bash
python main.py --input data/02_city_architecture/set_city_1 --output outputs --mode feature_experiment --feature all --resize-width 1200 --debug-matches
```

## 输出目录

```text
outputs/
├── baseline/
│   └── <dataset>/
│       ├── panorama_baseline.jpg
│       └── input_contact_sheet.jpg
├── improved/
│   └── <dataset>/
│       ├── panorama_improved_before_crop.jpg
│       ├── panorama_improved_after_crop.jpg
│       ├── crop_comparison.jpg
│       └── preprocessing_preview.jpg
├── feature_experiment/
│   └── <dataset>/
│       ├── sift/
│       ├── orb/
│       └── akaze/
├── comparisons/
├── tables/
│   ├── baseline_metrics.csv
│   ├── improved_metrics.csv
│   ├── feature_metrics.csv
│   ├── summary_metrics.csv
│   └── failure_cases.csv
└── report_assets/
    ├── best_cases/
    ├── failure_cases/
    ├── feature_algorithm_comparison_charts/
    └── README_for_report.md
```

## CSV 指标

`baseline_metrics.csv` 记录基础拼接是否成功、运行时间、输出尺寸等。

`improved_metrics.csv` 记录预处理方式、Gamma、CLAHE、曝光补偿、裁剪开关、裁剪前后黑边比例和保留面积比例。

`feature_metrics.csv` 记录每个相邻图片对的特征分析结果，包括关键点数量、raw matches、good matches、RANSAC 内点数、内点率、Homography 是否成功、拼接是否成功和失败原因。

`summary_metrics.csv` 汇总每组图片的 baseline/improved 成功情况、最佳内点率算法、最快算法、黑边比例变化和报告推荐用途。

`failure_cases.csv` 汇总失败原因、可能原因和建议解决方案。

## 核心算法分析模块

特征算法对比由 `pairwise_feature_analysis.py` 实现。它独立完成：

- SIFT / ORB / AKAZE 特征点检测
- 相邻图像 KNN 匹配
- Lowe ratio test 筛选 good matches
- RANSAC 估计 Homography
- 内点率统计
- 关键点图、匹配线图、RANSAC 内点图保存

当前版本也会尝试通过 `Stitcher(detector="sift/orb/akaze")` 调用 OpenStitching 的 detector 参数保存对应拼接图。如果某些环境下 OpenStitching 不支持底层替换，`pairwise_feature_analysis.py` 仍然可以作为报告中的核心算法分析模块。

## 报告素材建议

完整实验结束后，优先查看：

```text
outputs/report_assets/README_for_report.md
```

它会自动说明：

- 本次测试了多少组图片
- 哪些图适合展示基础版成功案例
- 哪些图适合展示改进版效果
- 哪些图适合展示 SIFT / ORB / AKAZE 对比
- 哪些图适合展示失败案例
- 每类图像的简短分析模板

## 后续手动工作

1. 检查 `data/10_self_captured_demo/`，放入自己拍摄的答辩演示图。
2. 运行 `python run_all_experiments.py --data data --output outputs --resize-width 1200 --clahe --debug-matches`。
3. 打开 `outputs/report_assets/README_for_report.md`，按推荐图片写实验报告。
4. 如果某些数据集太慢，可以把 `--resize-width` 调到 900，或用 `--max-images 4` 限制每组图片数量。
