# scene204 自采数据集 — 模型训练

## 当前两处 GOOSE 依赖（待解除）

| 依赖 | 现在 | 解除条件 | 怎么改 |
|------|------|----------|--------|
| **训练 backbone** | `load_from` 指向 GOOSE Category checkpoint，backbone 权重来自 GOOSE 数据 | 自采数据集扩充到足够规模 | 删掉 `load_from`，换用 `train_scratch.py`，拉长 `max_iters` |
| **H 矩阵 3D 标签** | `id_3d_for_h: 11`（GOOSE 3D 体系里 road 的 ID） | scene204 3D 标注接入 | 将 `configs/evaluation.yaml` 的 `scene204_road_finetune` 任务中 `id_3d_for_h: 11` 改为 scene204 自己的 ID（预计是 1，与 2D 一致），重跑 pipeline |

> 两处可以独立解除，互不阻塞。
> 推理 / 边界误差 / Precision/Recall 这三块已经全部用 scene204 自己的数据，无 GOOSE 依赖。

---

> **注意：训练模型和跑 DISTANCE Pipeline 是两件独立的事。**
>
> | 事情 | 做什么 | 在哪里 |
> |------|--------|--------|
> | **训练模型** | 用标注数据教模型识别38类场景 | 本文件夹 `training/scene204/` |
> | **DISTANCE Pipeline** | 用训练好的模型预测 mask，再算边界误差/Precision/Recall | 仓库根目录 `distance/` + `configs/` |
>
> 训练完得到 `.pth` checkpoint，才能拿去跑 DISTANCE Pipeline。两者不需要同时运行。

---

## 文件结构

```
training/scene204/
├── dataset/
│   ├── scene204.py                  # Scene204Dataset 类（38类，palette，label map）
│   └── mmseg_wildscenes_init.py     # 需要追加到服务器 __init__.py 的那一行
├── mmseg_configs/
│   ├── _base_dataset_scene204.py    # mmseg 数据集 base config（dataloader / pipeline）
│   ├── train_finetune.py            # 当前方案：从 GOOSE 权重 fine-tune（数据少时用）
│   └── train_scratch.py             # 未来方案：从头训练（数据够时换这个）
└── scripts/
    └── prepare_merged_split.py      # 数据准备：合并多相机/多场景 + train/val 分割
```

---

## 服务器部署（首次）

所有操作在 `goose_trt` 容器内进行：

```bash
docker exec -it goose_trt bash
```

### 1. 部署数据集类

```bash
# 复制 scene204 数据集类
cp training/scene204/dataset/scene204.py \
   ../wildscenes/mmseg_wildscenes/dataset/scene204.py

# 在 __init__.py 末尾追加注册行（只需做一次）
echo "from wildscenes.mmseg_wildscenes.dataset.scene204 import Scene204Dataset, SCENE204_LABEL_MAP" \
   >> ../wildscenes/mmseg_wildscenes/__init__.py
```

### 2. 部署 mmseg 配置文件

```bash
cp training/scene204/mmseg_configs/_base_dataset_scene204.py \
   ../wildscenes/configs/_base_/datasets/scene204.py

cp training/scene204/mmseg_configs/train_finetune.py \
   ../wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb4-5k_scene204-512x512.py
```

### 3. 准备数据（合并多相机 + train/val 分割）

```bash
python training/scene204/scripts/prepare_merged_split.py \
    --scenes ../data/scene204/WildScenes2d \
    --out    ../data/scene204/WildScenes2d/merged
```

输出：
```
  FRONT_CAMERA      : 23 frames  →  train 20  val 3
  BACK_CAMERA       : 19 frames  →  train 17  val 2
  ...
Done.  train=97  val=17  → .../merged
```

---

## 训练

### 方案 A：Fine-tune（现在，数据少）

从 GOOSE Category 12类模型迁移 backbone，head 重新初始化为38类。

```bash
cd ..
python wildscenes/scripts/benchmark/train2d.py \
    wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb4-5k_scene204-512x512.py \
    --work-dir work_dirs/scene204_deeplabv3_finetune
```

当前结果（scene204，114张图）：
- best mIoU: **61.64%**（iter 4000）
- road IoU: **97.93%**（iter 5000）
- checkpoint: `work_dirs/scene204_deeplabv3_finetune/best_mIoU_iter_4000.pth`

### 方案 B：从头训练（目标方案，数据集完整时使用）

把 `train_scratch.py` 部署到服务器，替换 fine-tune config 或另存一份：

```bash
cp training/scene204/mmseg_configs/train_scratch.py \
   ../wildscenes/configs/deeplabv3/deeplabv3_r50-d8_50k_scene204-scratch-512x512.py

python wildscenes/scripts/benchmark/train2d.py \
    wildscenes/configs/deeplabv3/deeplabv3_r50-d8_50k_scene204-scratch-512x512.py \
    --work-dir work_dirs/scene204_deeplabv3_scratch
```

与 fine-tune 的唯一区别：`load_from = None`，`max_iters = 50000`。

**端到端验证（2026-07-24）**：已用 114 张图 × 5k iter（`train_scratch.py` 的缩短版）跑通完整流水线，结果如下：

| 指标 | Fine-tune（GOOSE backbone） | Scratch（5k iter，仅验证） |
|------|---------------------------|--------------------------|
| best mIoU | 61.64% | 50.27% |
| road IoU | 97.93% | 48.47% |
| 边界误差 mean | 0.030 m | 0.077 m |
| Precision | 99.87% | 98.73% |
| Recall | 91.84% | 84.30% |

scratch 5k 数字差是正常的——114 张图 + 5k iter 严重不足。等 3D 标注到位、数据集扩充后，用 `train_scratch.py`（50k iter）从头训练，指标预期会超过 fine-tune。

---

## 加入新场景数据

```bash
# 把新 scene 也传给 prepare_merged_split.py，已有 scene204 的 symlink 不会被破坏
python training/scene204/scripts/prepare_merged_split.py \
    --scenes ../data/scene204/WildScenes2d \
             ../data/scene205_wildscenes/WildScenes2d \
    --out    ../data/all_scenes_merged

# 然后修改 _base_dataset_scene204.py 里的 data_root 指向新的 merged 目录，重新训练
```

---

## 训练好了之后 → 接 DISTANCE Pipeline

```bash
# 跑推理（生成 pred mask）
python onboard/infer_scene204_vis.py   # 或自定义推理脚本

# 跑 DISTANCE（算边界误差 + Precision/Recall）
python -m distance.run_sequence --config configs/evaluation.yaml --task scene204_road_finetune
```

详见仓库根目录 [README.md](../../README.md)。
