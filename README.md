# DISTANCE  · v1.3.0

基于 LiDAR-相机单应矩阵（H）的可通行区域边界物理误差评估工具。

---

## 服务器数据集目录（4090，`/data1/`）

### 自采数据集 scene204（`/root/distance/WildScenes/DISTANCE/datasets/scene204/`）

```
scene204_wildscenes/
├── WildScenes2d/
│   ├── FRONT_CAMERA/            ← 前置摄像头（主力，17帧，标注完整）
│   │   ├── image/               ← 原图
│   │   ├── indexLabel/          ← GT标注（38类）
│   │   ├── finetune_pred_mask/  ← finetune模型预测结果
│   │   └── scratch_pred_mask/   ← scratch模型预测结果
│   ├── BACK_CAMERA/
│   ├── FRONT_LEFT_CAMERA/
│   ├── FRONT_RIGHT_CAMERA/
│   ├── BACK_LEFT_CAMERA/
│   ├── BACK_RIGHT_CAMERA/
│   └── merged/                  ← 训练用合并数据（train/val分割）
└── WildScenes3d/
    ├── Clouds/                  ← 点云文件
    └── Labels/                  ← 3D语义标注（2026-07-22起到位）
```

### GOOSE 公开数据集（`/root/distance/WildScenes/DISTANCE/datasets/Goose/`，7845张，64类）

```
data/Goose/
├── goose_2d_train/
│   ├── images/train/<场景名>/   ← 原图（23个场景，含城市/田野/森林/雪地）
│   └── labels/train/<场景名>/   ← GT标注（color / labelids / instanceids）
├── goose_2d_val/
└── goose_3d_val/
    ├── lidar/val/               ← 点云
    └── labels/val/              ← 3D标注
```

### WildScenes 公开数据集（`/data1/WildScenes/`，184G，5序列）

```
/data1/WildScenes/
│
├── Fullclouds/                       # 完整点云（未切分成帧），87G
│   ├── K-01/  *.ply   2271 个文件
│   ├── K-03/  *.ply   5290 个文件
│   ├── V-01/  *.ply   1079 个文件
│   ├── V-02/  *.ply   1100 个文件
│   └── V-03/  *.ply   2407 个文件
│       文件名格式：<unix时间戳>.ply（如 1624325374.450389996.ply）
│
├── WildScenes2d/                     # 2D 图像 + 语义分割标注，47G
│   ├── K-01/
│   │   ├── image/        1969 张   *.png
│   │   ├── label/        1969 张   *.png（分割标注，可视化色彩图）
│   │   ├── indexLabel/   1972 张   *.png（分割标注，类别索引图）
│   │   ├── poses2d.csv               # 相机位姿
│   │   └── camera_calibration.yaml   # 相机内参标定
│   ├── K-03/   同上结构，3913 张
│   ├── V-01/   同上结构，742 张，多一个 deeplabv3_pred_mask/（742 张预测掩码）
│   ├── V-02/   同上结构，833 张
│   └── V-03/   同上结构，1842 张
│       文件名格式：<秒>-<纳秒>.png，image/label/indexLabel 三者一一对应
│
└── WildScenes3d/                     # 3D 点云帧 + 标注，50G
    ├── K-01/
    │   ├── Clouds/   2271 个   *.bin（单帧点云，二进制）
    │   ├── Hists/    2271 个   *.csv（逐帧统计直方图）
    │   ├── Labels/   2271 个   *.label（逐点语义标注）
    │   └── poses3d.csv
    ├── K-03/   同上结构，5290 帧
    ├── V-01/   同上结构，1080 帧
    ├── V-02/   同上结构，1100 帧
    └── V-03/   同上结构，2407 帧
        Clouds/Hists/Labels 三者按时间戳一一对应（与 Fullclouds 时间戳同源）
```

### WildScenes Mini（`/root/distance/WildScenes/DISTANCE/datasets/wildscenes_mini/`，43 帧，19 类）

```
wildscenes_mini/
├── WildScenes2d/
│   ├── image/
│   ├── indexLabel/
│   └── deeplabv3_pred_mask/
└── WildScenes3d/
    ├── Clouds/
    └── Labels/
```

---

## 平台部署使用指南

> Docker 镜像 `distance_pipeline:v1` 支持两个独立任务，使用同一镜像、同一数据集卷挂载，只切换命令行。

### 公共配置

**输入卷挂载：**

| 名称 | 容器内路径 | 说明 |
|------|-----------|------|
| 数据集 | `/root/distance/WildScenes/DISTANCE/datasets/scene204/` | scene204 数据集（含 WildScenes2d + WildScenes3d） |
| 预训练模型 | `/root/distance/WildScenes/pretrained_models/` | 模型权重 `.pth` 文件 |

**环境变量：**

| 变量名 | 值 |
|--------|-----|
| `PYTHONUNBUFFERED` | `1` |
| `CAMERA` | `FRONT_CAMERA` |
| `MODEL_FILENAME` | `best_mIoU_iter_4000.pth` |

**资源配置：** CPU 8核，内存 16G，共享内存 8G，GPU 1张

---

### 任务一：Pipeline 评测

**命令行：**
```bash
cd /workspace/WildScenes/DISTANCE && export CUDA_VISIBLE_DEVICES=0 && bash run_pipeline.sh
```

**输出卷挂载：**

| 名称 | 容器内路径 |
|------|-----------|
| pipeline结果 | `/workspace/WildScenes/work_dirs/distance_output/` |

结果文件 `sequence_summary.json` 包含4类指标：边界误差、H重投影误差、Precision/Recall、地面倾斜角。

当前基准结果（scene204，18帧）：
- 边界误差 mean：**0.030 m**
- Precision：**99.87%**　Recall：**91.84%**
- H 重投影误差 mean：**2.174 px**

---

### 任务二：模型训练

**命令行：**
```bash
cd /workspace/WildScenes && export CUDA_VISIBLE_DEVICES=0 && \
python /workspace/WildScenes/DISTANCE/training/scene204/scripts/prepare_merged_split.py \
    --scenes /root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d \
    --out /root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d/merged && \
python scripts/benchmark/train2d.py \
    wildscenes/configs/deeplabv3/deeplabv3_r50-d8_5k_scene204-scratch-512x512.py \
    --launcher none \
    --work-dir /workspace/WildScenes/work_dirs/scene204_scratch \
    --cfg-options \
        train_dataloader.dataset.data_root=/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d/merged \
        val_dataloader.dataset.data_root=/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d/merged \
        train_dataloader.batch_size=4 \
        train_dataloader.num_workers=4
```

**输出卷挂载：**

| 名称 | 容器内路径 |
|------|-----------|
| 模型输出 | `/workspace/WildScenes/work_dirs/scene204_scratch/` |

---

### 收集完全量数据集后的操作

只需两步：

1. **替换数据集**：将新数据集上传到平台，挂载路径不变（`/root/distance/WildScenes/DISTANCE/datasets/scene204/`）

2. **调整训练命令**（增大 batch size 和迭代次数）：
```bash
# 将命令行中的配置改为 50k iters 版本，并调整参数
--cfg-options \
    train_dataloader.dataset.data_root=/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d/merged \
    val_dataloader.dataset.data_root=/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d/merged \
    train_dataloader.batch_size=8 \
    train_dataloader.num_workers=8 \
    train_cfg.max_iters=50000
```

镜像、挂载路径、输出路径全部不变。

---

针对每一帧，全链路依次执行4个阶段：

1. **模型推理**（`distance/inference.py`）——用语义分割模型对RGB图推理，产出预测mask
2. **H矩阵估计**（`distance/homography.py`）——将带标注的 LiDAR 点云投影到图像，估计地平面单应矩阵 **H**
3. **边界提取**（`distance/boundary.py`）——清理预测mask、提取 GT 与预测的 2D 边界
4. **扫描线物理误差**（`distance/scanline.py`）——均匀采样水平扫描线，配对左/右边界点，通过 H 矩阵将像素距离转换为**物理距离（米）**

`distance/run_sequence.py` 负责把以上4个阶段串联起来，跨帧汇总统计指标。

---

## 一键计算全部指标

运行一条命令即可同时得到以下 **4类指标**，结果写入 `sequence_summary.json`：

| # | 指标 | 说明 | 输出字段 |
|---|------|------|----------|
| ① | **模型分割质量**（Precision / Recall） | 逐帧对比预测 mask 与 GT mask，`Precision = TP/(TP+FP)`，`Recall = TP/(TP+FN)`，取算术均值 | `traversable_pr` |
| ② | **H矩阵重投影误差**（px） | 3D-2D点对的重投影残差，衡量标定+位姿精度 | `h_matrix.reproj_error_px` |
| ③ | **扫描线边界物理误差**（m） | 扫描线采样，逐点对比预测边界与GT边界的地面距离，报告 mean/median/P75/P90/P95 | `scanline` |
| ④ | **地面平面倾斜角**（deg） | H估计过程中拟合出的地面法向量与竖直方向的夹角，用于质量监控 | `h_matrix.plane_tilt_deg` |

> **前提**：Precision/Recall（指标①）需要 GT 二值 mask 文件（`masks/gt_<class>_binary.png`），
> 由流水线自动从 `indexLabel` 图生成。如果 GT 不存在，该帧自动跳过，其余三类指标不受影响。

```bash
# 跑完整序列，自动计算所有指标
python -m distance.run_sequence --config configs/evaluation.yaml --task scene204_road_finetune

# 断点续跑（跳过已处理帧）
python -m distance.run_sequence --config configs/evaluation.yaml --task scene204_road_finetune --skip-existing
```

终端会在结束时打印汇总：

```
Scanline boundary error (m):
                  mean    median     P75     P90  frames
  overall        0.030     0.010   0.035   0.071      18
  left           0.042     0.022   0.063   0.093      18
  right          0.019     0.007   0.013   0.021      18

Traversable Precision/Recall (18 frames):
  mean precision : 99.87%   (≥90% in 100.0% of frames)
  mean recall    : 91.84%   (≥90% in 61.1% of frames)
```

完整数值（含逐帧明细）在 `<output_dir>/sequence_summary.json` 的 `traversable_pr` / `scanline` / `h_matrix` 字段。

---

## 代码结构

```
distance/
├── inference.py      # ①模型推理：跑分割模型产出mask
├── homography.py     # ②H矩阵估计：点云投影+单应矩阵拟合+重投影误差诊断
├── boundary.py        # ③边界提取：mask清理+连通域分析+外边界抽取
├── scanline.py         # ④扫描线物理误差：像素距离→米
├── io_utils.py         # 共享：配置读写、位姿/标定解析、统计函数
├── pipeline.py         # 转发层：重新导出①②③④的全部函数，main()入口
└── run_sequence.py     # 编排：多进程跑一整个序列，汇总 sequence_summary.json

configs/          # 统一配置：常规任务、手动帧、Ray-Surface
scripts/          # 独立工具脚本（点云可视化、导出数据集等）
ray_surface/      # 独立小模块（射线-地面求交的另一种边界误差实现）
training/         # GOOSE/WildScenes 模型训练、TensorRT转换、推理评估脚本
onboard/          # 实车/部署链路：TRT推理→pred mask→边界提取（不含GT，区别于training/的离线模型评估）
tools/            # 数据整理工具（自采场景 → WildScenes风格目录结构等）
```

**只需要用某一个阶段的功能**（比如只要H矩阵），直接从对应模块导入即可，不需要引入整个pipeline：

```python
from distance.homography import estimate_h, extract_correspondences
from distance.boundary import create_boundaries, extract_external_boundary
from distance.scanline import scanline_error
```

已有代码里 `from distance.pipeline import xxx` 的写法完全不受影响——`pipeline.py` 是一个转发层，重新导出了以上四个模块的全部函数，保持向后兼容。

### `distance/inference.py` 与 `training/wildscenes/` 的依赖关系

如果模型是在 GOOSE/WildScenes 数据集上训练的（比如 `deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py`），这个 mmseg 配置文件依赖自定义数据集类（`GooseCategoryDataset`），注册代码在 `training/wildscenes/mmseg_wildscenes/dataset/`。调用 `distance.inference.run_inference()` 之前，需要先把 `training/wildscenes` 加进 `sys.path` 并 `import` 对应的数据集注册模块，`init_model()` 才能正确解析模型的类别/调色板信息：

```python
import sys
sys.path.insert(0, "/path/to/training")
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401

from distance.inference import run_inference
```

---

## 环境要求

- Docker（基于 `wildscenes_full:latest`，含 GPU 支持与语义分割环境）
- NVIDIA GPU（语义分割推理阶段需要）

---

## 部署脚本

### 脚本 1：构建 Docker 镜像

```bash
cd /path/to/distance-pipeline
docker build -t wildscenes_full:latest .
```

- **`/path/to/distance-pipeline`** 替换为服务器上本仓库的实际路径
- 构建完成后代码位于容器内 `/workspace/WildScenes/DISTANCE/`

### 脚本 2：进入 Docker 容器

```bash
docker run --rm -it --gpus all \
  --shm-size=16g \
  -v /path/to/data:/root/distance/WildScenes/DISTANCE/datasets \
  wildscenes_full:latest bash
```

- **`/path/to/data`** 替换为服务器上数据集的实际路径，挂载至容器内 `/root/distance/WildScenes/DISTANCE/datasets`
- 进入容器后建议先执行 `cd /workspace/WildScenes/DISTANCE && ls`

---

## 数据结构

### 模式 A — WildScenes 数据集

```
<数据集根目录>/
├── WildScenes2d/
│   └── <序列>/                      ← 对应配置中 "dir_2d"
│       ├── camera_calibration.yaml
│       ├── poses2d.csv
│       ├── image/                   ← RGB 图像，命名格式：{sec}-{nanosec}.png
│       ├── indexLabel/              ← GT 索引标注，命名格式：{sec}-{nanosec}.png
│       └── deeplabv3_pred_mask/     ← 预测 mask，命名格式：{sec}-{nanosec}.png
└── WildScenes3d/
    └── <序列>/                      ← 对应配置中 "dir_3d"
        ├── poses3d.csv
        ├── Clouds/                  ← 点云文件，命名格式：{sec}.{nanosec}.bin
        └── Labels/                  ← 3D 标注，命名格式：{sec}.{nanosec}.label
```

2D 与 3D 时间戳数字相同，2D 用 `-` 分隔，3D 用 `.` 分隔。
不满足 `min_class_pixels` 阈值或无法匹配点云的帧会自动跳过。

### 模式 B — 自定义帧列表

```
<项目根目录>/
├── calibration/
│   ├── camera_calibration.yaml
│   ├── poses2d.csv
│   └── poses3d.csv
├── raw/
│   └── <帧目录>/
│       ├── <时间戳>_rgb.png
│       └── <时间戳>.bin
└── labels/
    └── <帧目录>/
        ├── <时间戳>_3d.label
        ├── <时间戳>_gt_indexlabels.png
        └── <时间戳>_pred_color.png
```

---

## 配置文件与测评任务

配置已经收敛为三个文件：

| 文件 | 用途 |
|------|------|
| `configs/evaluation.yaml` | 常规 DISTANCE 测评任务目录，集中定义公共路径、数据集、模型、mask 规则和任务差异 |
| `configs/custom_frames.yaml` | 非标准数据结构的手动帧模式示例 |
| `configs/ray_surface.yaml` | Ray-Surface 独立算法配置 |

### 常规任务配置结构

`evaluation.yaml` 按以下层次组织：

- `project_root`、`datasets_root`、`outputs_root`：全局根目录；
- `defaults`：所有任务共享的默认参数；
- `datasets`：数据集目录和数据集级参数；
- `models`：在线推理模型，`cached_masks` 表示直接读取已有预测 mask；
- `mask_profiles`：不同标签体系对应的 mask 提取规则；
- `tasks`：选择数据集、模型和 mask profile，并只覆盖该任务特有参数。

程序会按上述顺序进行递归合并。任务中的值优先级最高，未填写项继续使用公共配置和代码内置默认值。配置支持 `${project_root}`、`${datasets_root}`、`${outputs_root}` 三个路径变量。

当前任务：

| 任务名 | 数据集 | 预测来源 | 说明 |
|--------|--------|----------|------|
| `scene204_road` | Scene204 | 在线 GOOSE Category DeepLabV3 | 生成 `deeplabv3_pred_mask` 后测评 |
| `scene204_road_scratch` | Scene204 | 已有 scratch mask | 从头训练模型结果 |
| `scene204_road_finetune` | Scene204 | 已有 finetune mask | 微调模型结果 |
| `wildscenes_mini` | WildScenes Mini | 在线 WildScenes DeepLabV3 | 小数据集在线推理 |
| `wildscenes_mini_cached` | WildScenes Mini | 已有 mask | 不加载分割模型，适合快速验证 |
| `wildscenes_basic` | WildScenes V01 | 在线 WildScenes DeepLabV3 | 日常参数 |
| `wildscenes_v01` | WildScenes V01 | 在线 WildScenes DeepLabV3 | 详细参数，较窄扫描区域 |

### 查看和运行任务

```bash
# 查看 evaluation.yaml 中的全部任务
python -m distance.run_sequence \
    --config configs/evaluation.yaml \
    --list-tasks

# 完整运行一个任务
python -m distance.run_sequence \
    --config configs/evaluation.yaml \
    --task scene204_road_finetune \
    --workers 8

# 断点续跑
python -m distance.run_sequence \
    --config configs/evaluation.yaml \
    --task scene204_road_finetune \
    --workers 8 \
    --skip-existing

# 只处理指定帧
python -m distance.run_sequence \
    --config configs/evaluation.yaml \
    --task wildscenes_mini_cached \
    --frames 1623379838.017682788
```

如果对包含 `tasks` 的配置省略 `--task`，程序会报错并列出可用任务。原有普通单任务 YAML/JSON 仍可直接使用，因此外部自定义配置保持兼容。

### 手动帧模式

手动指定 RGB、点云、3D 标签、GT 和预测 mask：

```bash
python -m distance.run_sequence --config configs/custom_frames.yaml
```

### Ray-Surface

Ray-Surface 使用独立入口和专有参数：

```bash
python ray_surface/run_ray.py --config configs/ray_surface.yaml
```

### 新增任务

通常只需在 `evaluation.yaml` 的 `tasks` 下增加一项，并引用已有的 `dataset`、`model` 和 `mask_profile`。只有目录结构、模型或标签体系发生变化时，才需要新增相应的公共定义。

---

## 输出结构

```
<output_dir>/
├── sequence_summary.json              ← 全序列汇总指标 + 逐帧结果
└── frames/
    └── <frame_id>/
        ├── pipeline_summary.json
        ├── h_estimation/
        │   ├── dirt_h_estimation.json              ← H 矩阵及重投影误差统计
        │   ├── dirt_pixel_correspondences.csv
        │   └── diagnostics/
        │       ├── dirt_reproj_error_heatmap.png   ← 重投影误差热图
        │       └── dirt_reproj_error_histogram.png
        ├── boundaries/
        │   └── dirt_gt_pred_boundary_overlay.png   ← GT 与预测边界叠加图
        ├── masks/
        │   ├── gt_dirt_binary.png
        │   └── pred_dirt_binary.png
        └── scanline/
            ├── dirt_scanline_rgb_overlay.png       ← 扫描线采样可视化
            └── dirt_scanline_samples.csv           ← 逐扫描线误差数据
```

### 终端输出示例

```
[INFO] 94 anchor frames ready  (skipped: 12)
[INFO] 1842 total images available as H candidates  (search window ±15s)

Phase 1 (H): 100%|████████| 94/94
Phase 2 (scanline): 100%|████████| 94/94

════════════════════════════════════════════════════════
  SEQUENCE SUMMARY  (94/94 frames with valid scanlines)
════════════════════════════════════════════════════════

Scanline boundary error (m):
                  median     P75     P90     P95   frames
  overall          0.100   0.581   2.764   7.831       43
  left             0.115   1.366   5.774  17.131       43
  right            0.012   0.489   1.069   1.738       43

H reprojection error (px):
  mean-of-means : 2.748
  mean-of-P95   : 6.498
```

---

## WildScenes 类别 ID 参考

| 类别                 | indexLabel id | 3D cidx | 预测 mask 颜色（R,G,B） |
| -------------------- | :-----------: | :-----: | :---------------------: |
| dirt（泥地）         |       2       |    1    |      (60, 180, 75)      |
| grass（草地）        |      18      |    3    |     (128, 128, 128)     |
| tree-foliage（树叶） |       8       |   12   |     (210, 245, 60)     |
| tree-trunk（树干）   |       7       |   13   |     (240, 50, 230)     |
| sky（天空）          |      17      |   10   |       (0, 0, 128)       |

---

## `training/` — 模型训练与推理评估

GOOSE / WildScenes 语义分割模型的训练、TensorRT转换、验证集评估脚本。等自采数据集（scene204等）标注完成后，会在这套基础上训练自己的模型。

| 脚本 | 用途 |
|---|---|
| `train.py` | DDRNet/DeepLabV3 训练 |
| `trt_convert.py` | PyTorch → TensorRT FP16 引擎转换 |
| `infer_val.py` / `infer_val_trt.py` | 验证集推理 + mIoU评估（PyTorch / TensorRT两个版本） |
| `eval_traversable.py` / `eval_traversable_12cls.py` | 可通行区域识别评估 |
| `vis_infer.py` | 推理可视化 |
| `audit_labels.py` | 标签数据审计 |
| `speed_compare.py` | 推理速度对比 |
| `wildscenes/` | mmseg 自定义数据集注册（`GooseDataset`/`GooseCategoryDataset`）+ DDRNet/DeepLabV3 训练配置 |

> **重要**：`wildscenes/` 这里只是一份**补丁/覆盖层**（8个文件：数据集注册代码 + GOOSE专属训练配置），
> 不是完整的 WildScenes 代码库。这些脚本里的 `_base_ = [...]` 配置继承链（`configs/_base_/models/`、
> `configs/_base_/schedules/`、`configs/_base_/default_runtime.py`）依赖一份**完整的上游 WildScenes 安装**
> （官方仓库：https://github.com/csiro-robotics/WildScenes ），这份完整安装已经打包进服务器的
> `wildscenes_goose:latest` Docker 镜像（`/workspace/WildScenes/wildscenes/`）。
>
> 实际调用时，`--config` 参数要指向**那份完整安装**里的 config 文件（比如
> `/workspace/WildScenes/wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py`），
> 而不是本仓库 `training/wildscenes/configs/` 下面的路径——本仓库这份只保证
> `import wildscenes.mmseg_wildscenes.dataset.goose_category` 能正确注册数据集类，
> 但它自己的 `configs/` 目录没有带全 `_base_` 依赖，直接指向它会报 `FileNotFoundError`
> （这是原始设计如此，不是迁移过程中漏文件——`inject_into_distance*.sh` 两个脚本记录了
> 这份补丁本来就是要拷贝合并进一个已存在的完整 WildScenes 安装里）。

---

## `ray_surface/` — 边界误差的另一种实现

用射线-地面求交（而非H矩阵单应投影）计算边界物理误差的独立模块，见 `ray_surface/README.md`。

---

## `onboard/` — 实车部署链路

跟 `training/`（离线模型训练评估，需要GT）不同，这里是**真车上实际要跑的链路**：TRT推理 → pred mask清理 → pred边界提取 → 画在原图上传给后端，全程不需要GT。

| 脚本 | 用途 |
|---|---|
| `run_e2e_pipeline.py` | 端到端：TRT推理 + 调用 `distance.run_sequence` 串联H矩阵/边界误差评估，一条命令跑完 |
| `profile_onboard_pipeline.py` | 真实车载链路耗时测量（TRT推理 + mask清理 + 边界提取 + 画图，逐阶段计时） |
| `profile_single_frame.py` | 绕开 `run_sequence.py` 的多进程调度，单进程跑几帧供 cProfile 精确剖析 |
| `project_lidar_to_image.py` | 点云投影到RGB图像，标定链路验证工具（含验证过的 cam2lidar 方向说明） |
| `infer_masks_trt.py` | 用TRT引擎批量推理生成二值mask，供 `distance.run_sequence` 读取 |
| `infer_scene204_vis.py` / `infer_scene204_vis_64class.py` | GOOSE 12类/64类模型在自采数据上的分割可视化示例 |

这些脚本依赖 `distance/` 包和/或 `training/wildscenes/` 的数据集注册（跟 `distance.inference.run_inference()` 的依赖关系一致，见上文）。

---

## `tools/` — 数据整理工具

| 脚本 | 用途 |
|---|---|
| `build_scene_wildscenes_format.py` | 把自采场景的原始数据（`car006.yaml`标定 + `point_data/<timestamp>/`逐帧文件夹）+ 标注公司交付的2D标注结果（`gtFine/`），整理成模仿WildScenes的目录结构（`WildScenes2d/<相机名>/` 每路相机一个伪序列 + `WildScenes3d/`），可以直接用现有`distance`包跑 |

用法：

```bash
python3 tools/build_scene_wildscenes_format.py \
    --raw-root "/path/to/sceneXXX（parameters和point_data所在目录）" \
    --scene-name scene204 \
    --annot-dir /path/to/annotation/gtFine \
    --out-root /path/to/output
```

处理逻辑：`world_from_lidar = ego2global(欧拉角) @ lidar2ego`，`world_from_camera = world_from_lidar @ cam2lidar`（`cam2lidar`直接用，不取逆，见上文验证过的方向约定）。3D标注还没有时，`WildScenes3d/Labels/`会生成为空文件夹，预留等标注到位后填入。已用scene204真实数据（138张图，2D标注全部到位）跑通验证，生成的`poses2d.csv`/`poses3d.csv`/`camera_calibration.yaml`均已用`distance.io_utils`的解析函数实际验证过格式兼容。
