# DISTANCE

DISTANCE 用于评估可通行区域的物理边界误差。程序以 LiDAR 点云帧为时间锚点，匹配相机图像和位姿，通过地面单应矩阵（H）将图像边界误差转换为实际距离（米）。

当前版本直接在本机 Conda 环境运行。

## 环境与目录

- 项目：`.`
- Conda 环境：`distance`
- Python：3.10
- PyTorch：2.2.0+cu118
- CUDA Runtime：11.8
- 数据集：`../data`
- 输出：`./outputs`

```bash
cd .
conda activate distance
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 数据集布局

数据集统一放在 `../data/` 下，可以使用真实目录或软连接：

```text
../data/
├── scene204/
│   ├── WildScenes2d/FRONT_CAMERA/
│   │   ├── camera_calibration.yaml
│   │   ├── poses2d.csv
│   │   ├── color/ 或 image/
│   │   ├── indexLabel/
│   │   ├── deeplabv3_pred_mask/
│   │   ├── finetune_pred_mask/
│   │   └── scratch_pred_mask/
│   └── WildScenes3d/
│       ├── poses3d.csv
│       ├── Clouds/
│       └── Labels/
├── wildscenes_mini/
│   ├── WildScenes2d/
│   └── WildScenes3d/
├── 61541v003/data/WildScenes/
│   ├── WildScenes2d/V-01/
│   └── WildScenes3d/V-01/
└── Goose/
```

检查失效软连接：

```bash
find ../data -xtype l -print
```

没有输出表示软连接均有效。

## 配置

| 文件 | 用途 |
|---|---|
| `configs/evaluation.yaml` | 常规 DISTANCE 测评任务 |
| `configs/custom_frames.yaml` | 手动指定少量帧 |
| `configs/ray_surface.yaml` | Ray-Surface 方法 |

`evaluation.yaml` 集中维护公共路径、默认参数、数据集、模型、mask 规则和任务差异。路径支持 `${project_root}`、`${datasets_root}`、`${outputs_root}`。迁移数据后通常只需修改文件顶部的根目录或对应的 `datasets` 项。

## 运行测评

查看任务：

```bash
python -m distance.run_sequence \
  --config configs/evaluation.yaml \
  --list-tasks
```

| 任务 | 数据集 | 预测来源 |
|---|---|---|
| `scene204_road` | scene204 | 在线 GOOSE-64 DeepLabV3，合并 Terrain 类 |
| `scene204_road_finetune` | scene204 | 已有 finetune mask |
| `scene204_road_scratch` | scene204 | 已有 scratch mask |
| `wildscenes_mini` | WildScenes Mini | 在线 DeepLabV3 |
| `wildscenes_mini_cached` | WildScenes Mini | 已有预测 mask |
| `wildscenes_basic` | WildScenes V-01 | 在线 DeepLabV3 |
| `wildscenes_v01` | WildScenes V-01 | 在线推理及详细参数 |

### 小范围验证

建议先用缓存预测结果验证数据路径、标定、位姿、点云、3D 标签和输出链路：

```bash
python -m distance.run_sequence \
  --config configs/evaluation.yaml \
  --task wildscenes_mini_cached \
  --frames 1623378025.856173264 \
  --workers 1
```

`--frames` 使用 `WildScenes3d/Clouds/` 中点云文件去掉扩展名后的 ID。3D 文件通常使用 `秒.纳秒`，2D 图像使用 `秒-纳秒`，程序会自动匹配。

### 完整运行与断点续跑

```bash
python -m distance.run_sequence \
  --config configs/evaluation.yaml \
  --task scene204_road_finetune \
  --workers 8

python -m distance.run_sequence \
  --config configs/evaluation.yaml \
  --task scene204_road_finetune \
  --workers 8 \
  --skip-existing
```

在线推理任务会加载 `evaluation.yaml` 中配置的模型，请先确认模型 config 和 checkpoint 存在。缓存任务不加载分割模型，更适合验证评估流水线。

非标准数据结构可编辑手动帧配置后运行：

```bash
python -m distance.run_sequence --config configs/custom_frames.yaml
```

## 流程与指标

每帧依次执行：

1. 读取已有预测 mask，或运行语义分割模型；
2. 匹配 2D/3D 位姿并估计 H；
3. 从 GT 和预测 mask 提取边界；
4. 沿水平扫描线匹配边界，将像素误差转换为米；
5. 汇总有效帧。

输出指标包括分割 Precision/Recall、H 重投影误差（px）、扫描线边界误差（m）和地面倾斜角（deg）。

## 输出

```text
outputs/<task_output>/
├── sequence_summary.json
└── frames/<frame_id>/
    ├── pipeline_summary.json
    ├── masks/
    ├── boundaries/
    ├── h_estimation/
    └── scanline/
```

`sequence_summary.json` 保存全序列汇总和逐帧结果；帧目录保存掩码、边界、H 诊断、可视化和扫描线 CSV。`datasets/`、`outputs/`、模型权重及运行产物已由 `.gitignore` 排除。

## 代码结构

```text
DISTANCE/
├── configs/               # 统一配置
├── distance/
│   ├── inference.py       # 分割推理
│   ├── homography.py      # 点云投影与 H 估计
│   ├── boundary.py        # 边界提取
│   ├── scanline.py        # 物理误差
│   ├── io_utils.py        # 配置、标定和位姿工具
│   ├── pipeline.py        # 单帧流水线
│   └── run_sequence.py    # 序列入口
├── ray_surface/           # 射线-地表求交
├── training/              # 模型训练
├── onboard/               # 推理与部署脚本
└── tools/                 # 数据整理工具
```

## 其他入口

Ray-Surface：

```bash
python ray_surface/run_ray.py --config configs/ray_surface.yaml
```

详见 [ray_surface/README.md](ray_surface/README.md)。scene204 数据准备和训练见 [training/scene204/README.md](training/scene204/README.md)。

## 常见问题

- 指定帧没有匹配：`--frames` 应使用点云 ID，例如 `1623378025.856173264`，不是 2D 图像名 `1623378025-856173264.png`。
- 扫描时跳过帧：检查点云、3D 标签、GT mask、预测 mask、标定文件和位姿记录是否完整。
- 在线推理找不到模型：检查 `evaluation.yaml` 中模型的 `config` 和 `checkpoint`。
- OpenMP 报 `OMP_NUM_THREADS` 无效：执行 `export OMP_NUM_THREADS=1` 后重试。
