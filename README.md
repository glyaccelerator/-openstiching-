# 基于 OpenStitching 的全景图像拼接与融合优化系统设计

本项目是一个面向数字图像处理课程大作业的 Python 实验工程。系统基于 [OpenStitching/stitching](https://github.com/OpenStitching/stitching) 与 OpenCV，实现从多张输入图像到全景图像输出的完整实验流程，并额外提供预处理、曝光补偿、自动裁剪、指标统计和结果可视化，方便后续写实验报告。

## 项目特点

- 支持单个图片组拼接，例如 `data/set1`
- 支持批量处理多个图片组，例如 `data/set1`、`data/set2`、`data/set3`
- 支持图像缩放、亮度归一化、Gamma 校正、CLAHE 增强
- 支持简单曝光补偿：LAB 亮度通道均值方差匹配
- 支持黑边自动裁剪，并保存裁剪前后对比图
- 输出 `metrics.csv`，记录运行时间、输出尺寸、黑边比例等实验指标
- 保存输入缩略图、拼接前后对比图、裁剪前后对比图

## 环境安装

建议使用 Python 3.9 或更高版本。

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果 `stitching` 安装失败，可以先升级 pip：

```bash
python -m pip install --upgrade pip
pip install stitching opencv-python numpy
```

## 文件夹结构

```text
.
├── main.py
├── requirements.txt
├── README.md
├── data/
│   ├── set1/
│   ├── set2/
│   └── set3/
├── outputs/
│   ├── panoramas/
│   ├── visualizations/
│   └── metrics.csv
└── src/
    ├── cropping.py
    ├── exposure.py
    ├── io_utils.py
    ├── metrics.py
    ├── preprocessing.py
    ├── stitching_pipeline.py
    └── visualization.py
```

## 运行方法

仓库已经包含三组真实多视角测试数据，来自 OpenPano 官方 example data：

```text
data/set1_uav_aerial       # UAV 航拍室外景观
data/set2_cmu_campus       # CMU 校园室外场景
data/set3_zijing_campus    # 清华紫荆公寓室外场景
```

数据来源：[OpenPano Example Data](https://github.com/ppwwyyxx/OpenPano/releases/tag/0.1)。OpenPano README 中说明这些是可下载的 original/example data，适合全景拼接测试。

说明：OpenStitching 在真实数据上可能提示 `Not all images are included in the final panorama`。这通常表示某些图片与主全景图匹配置信度不足，程序仍会输出可用全景图。报告中可以把它作为特征匹配和置信度阈值可扩展项来讨论。

处理单个图片组：

```bash
python main.py --input data/set1_uav_aerial --output outputs --resize-width 900 --gamma 1.0 --clahe --exposure --crop
```

批量处理 `data` 下的多个图片组：

```bash
python main.py --batch data --output outputs --resize-width 900 --gamma 1.0 --clahe --exposure --crop
```

如果想看更高分辨率结果，可以把宽度调大，例如：

```bash
python main.py --batch data --output outputs --resize-width 1200 --gamma 1.0 --clahe --exposure --crop
```

只运行基础拼接，不启用 CLAHE、曝光补偿和裁剪：

```bash
python main.py --input data/set1 --output outputs --resize-width 1200
```

## 输出结果

全景图输出在：

```text
outputs/panoramas/
```

常见文件包括：

- `set1_before_exposure.jpg`：曝光补偿前的拼接结果
- `set1_raw.jpg`：当前配置下的原始拼接结果
- `set1_cropped.jpg`：自动裁剪后的结果
- `set1_final.jpg`：最终结果

可视化结果输出在：

```text
outputs/visualizations/set1/
```

常见文件包括：

- `input_thumbnails.jpg`：输入图片缩略图拼图
- `preprocessed_thumbnails.jpg`：预处理后图片缩略图拼图
- `exposure_compensated_thumbnails.jpg`：曝光补偿后图片缩略图拼图
- `before_after_exposure_stitch.jpg`：曝光补偿前后拼接效果对比
- `before_after_crop.jpg`：裁剪前后对比
- `input_vs_panorama.jpg`：输入缩略图与最终全景图对比
- `crop_mask.jpg`：非黑色有效区域 mask

实验指标输出在：

```text
outputs/metrics.csv
```

字段包括：

- `dataset_name`
- `image_count`
- `input_resolution`
- `preprocessing_method`
- `exposure_compensation_enabled`
- `crop_enabled`
- `stitch_success`
- `runtime_seconds`
- `output_width`
- `output_height`
- `black_area_ratio_before_crop`
- `black_area_ratio_after_crop`
- `notes`

## 实验流程

1. 从输入文件夹读取多张图像。
2. 对输入图像进行缩放，降低计算量并统一尺度。
3. 对图像进行亮度归一化、Gamma 校正，可选 CLAHE 局部增强。
4. 可选执行曝光补偿，将各图像 LAB 空间中的亮度均值和方差匹配到第一张参考图像。
5. 调用 OpenStitching 完成特征匹配、相机估计、图像变换、缝合与融合。
6. 计算拼接结果中的黑边比例。
7. 可选根据非黑色区域 mask 自动裁剪全景图。
8. 保存全景图、对比图和 `metrics.csv`。

## 常见问题

### 1. 拼接失败怎么办？

请检查输入图片之间是否有足够重叠区域。全景拼接通常需要相邻图像有明显重叠纹理，建议重叠比例不低于 30%。如果图像分辨率过大，可以使用 `--resize-width 1200` 或更低数值减少计算压力。

### 2. 输出全景图有黑边怎么办？

使用 `--crop` 开启自动裁剪：

```bash
python main.py --input data/set1 --crop
```

系统会根据非黑色区域生成 mask，并保存裁剪前后对比图。

### 3. 图像亮度差异明显怎么办？

可以启用曝光补偿：

```bash
python main.py --input data/set1 --exposure
```

当前实现采用 LAB 亮度通道均值方差匹配，方法简单、速度快，也便于在报告中说明。

### 4. OpenStitching 参数还能扩展吗？

可以。当前工程优先保证流程可运行，因此主流程通过 `stitching.Stitcher()` 调用 OpenStitching 默认管线。后续可以继续扩展：

- 特征检测与描述：SIFT、ORB、AKAZE
- 特征匹配策略与匹配置信度
- 相机估计与 bundle adjustment 参数
- seam finder 缝合线查找方法
- blender 融合方法，例如 feather 或 multiband
- OpenCV 或 OpenStitching 内部 exposure compensator

这些参数可以在 `src/stitching_pipeline.py` 的 `_stitch()` 函数中进一步配置。

### 5. pip 提示 NumPy 或 OpenCV 版本冲突怎么办？

建议使用独立虚拟环境安装本项目依赖，避免和 Anaconda 全局环境中的其它包互相影响。当前 `requirements.txt` 将 OpenCV 与 NumPy 约束在较稳定的组合：

```text
opencv-python>=4.8.0,<4.12
numpy>=1.24.0,<2.0
stitching>=0.5.3
```

如果你的环境里已经安装了 `opencv-python-headless` 或 `opencv-contrib-python`，并且 pip 提示冲突，优先在新虚拟环境中重新安装本项目依赖。

## 文件说明

- `main.py`：命令行入口，解析参数并调度单组或批量实验。
- `src/io_utils.py`：图片文件发现、批量数据集发现、图像读取、输出目录创建。
- `src/preprocessing.py`：缩放、亮度归一化、Gamma 校正、CLAHE 增强。
- `src/exposure.py`：简单曝光补偿，使用 LAB 亮度均值方差匹配。
- `src/cropping.py`：非黑区域 mask、黑边比例统计、自动裁剪。
- `src/visualization.py`：输入缩略图拼图、前后对比图保存。
- `src/stitching_pipeline.py`：单个数据集的完整实验流程。
- `src/metrics.py`：将实验结果写入 `metrics.csv`。

## 后续改进方向

- 增加 OpenStitching 参数配置文件，例如 YAML 或 JSON。
- 对比不同特征算法：SIFT、ORB、AKAZE。
- 增加多种曝光补偿方法，例如直方图匹配、OpenCV 曝光补偿器。
- 增加更多评价指标，例如信息熵、清晰度、拼接缝附近亮度差。
- 增加 GUI 或 Web 页面，方便交互式选择数据集和参数。
- 增加自动生成实验报告的脚本，将 `metrics.csv` 和可视化结果整合到 Word 或 Markdown。
## 加深算法版：Baseline 与 Improved 对比实验

新增脚本：

```text
加深算法版.py
```

该脚本用于课程报告中的对比实验分析。它在同一组输入图片上分别运行 `baseline` 和 `improved` 两种模式，并输出全景图、对比图和 `comparison.csv`。

### 运行命令

单个数据集对比：

```bash
python 加深算法版.py --input data/set1_uav_aerial --output outputs --mode compare --resize-width 1200 --gamma 1.0 --clahe
```

批量数据集对比：

```bash
python 加深算法版.py --batch data --output outputs --mode compare --resize-width 1200 --gamma 1.0 --clahe
```

只运行原始 OpenStitching baseline：

```bash
python 加深算法版.py --input data/set1_uav_aerial --output outputs --mode baseline
```

只运行 improved 流程：

```bash
python 加深算法版.py --input data/set1_uav_aerial --output outputs --mode improved --resize-width 1200 --gamma 1.0 --clahe
```

### 输出文件

```text
outputs/
├── comparison.csv
├── panoramas/
│   ├── set1_baseline.jpg
│   ├── set1_improved_raw.jpg
│   ├── set1_improved_cropped.jpg
│   └── set1_improved_final.jpg
└── visualizations/
    └── set1_uav_aerial/
        ├── input_thumbnails.jpg
        ├── improved_preprocessed_thumbnails.jpg
        ├── improved_exposure_thumbnails.jpg
        ├── improved_crop_mask.jpg
        └── baseline_vs_improved.jpg
```

`baseline_vs_improved.jpg` 包含输入图片缩略图、baseline 拼接结果、improved 裁剪前结果和 improved 裁剪后结果，适合直接放入实验报告。

## 实验设计

### 为什么设置 baseline 和 improved

`baseline` 表示直接使用 OpenStitching 的原始拼接流程，不加入预处理、曝光补偿和黑边裁剪。它用于提供一个参考结果，反映默认拼接算法在当前数据集上的基础表现。

`improved` 在 baseline 基础上加入图像缩放、亮度归一化、Gamma 校正、可选 CLAHE、均值方差曝光补偿和自动黑边裁剪。通过对同一组图片分别运行两种模式，可以观察改进模块对视觉质量、黑边比例和运行时间的影响。

### 曝光补偿的作用

多张输入图片可能因为拍摄角度、自动曝光或光照变化产生亮度差异。曝光补偿模块将各图像转换到 LAB 色彩空间，并将 L 亮度通道的均值和标准差匹配到第一张参考图像，从而减小拼接后不同区域之间的亮度突变。

### 自动裁剪的作用

全景图拼接后通常会出现黑色无效区域。自动裁剪模块先根据非黑色像素生成 mask，再寻找最大有效区域的外接矩形，最后裁剪掉明显黑边。这样可以得到更规整、更适合展示的全景图。

### 评价指标含义

`comparison.csv` 中的字段含义如下：

- `dataset_name`：数据集名称，例如 `set1_uav_aerial`
- `baseline_success`：baseline 是否成功生成全景图
- `improved_success`：improved 是否成功生成全景图
- `baseline_runtime`：baseline 运行时间，单位为秒
- `improved_runtime`：improved 运行时间，单位为秒
- `baseline_black_area_ratio`：baseline 输出图像中的黑色区域比例
- `improved_black_area_ratio`：improved 裁剪后输出图像中的黑色区域比例
- `baseline_output_size`：baseline 输出图像尺寸，格式为 `宽x高`
- `improved_output_size`：improved 最终输出图像尺寸，格式为 `宽x高`

一般来说，`improved_black_area_ratio` 越低，说明自动裁剪对无效黑边的去除越明显；运行时间则用于分析改进模块带来的额外计算开销。
