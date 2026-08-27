# GOOSE 训练部署说明（双 4090 平台）

在 WildScenes 的 DeepLabV3 框架上训练 GOOSE 数据集（64 类）。模型结构不变，仅替换数据集 + 类别数（18/15 → 64）。

---

## 0. 背景与镜像

- 基础环境镜像：`distance:latest`（含 WildScenes + mmseg + CUDA 全套环境）
- 已注入 GOOSE 支持的新镜像：`wildscenes_goose:latest`
  - 注入内容：4 个补丁文件 + 1 行数据集注册（已烤进镜像，运行时无需再挂载代码）

GOOSE 支持包含的 4 个文件（已在镜像内 `..` 树中）：

```
wildscenes/mmseg_wildscenes/dataset/goose.py                        # GooseDataset 类（64 类）
wildscenes/configs/_base_/datasets/goose.py                         # 数据集配置（data_root / num_classes=64）
wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose-512x512.py  # DeepLabV3 模型配置
scripts/data/setup_goose.py                                         # 数据预处理（建软链接）
```

注册行（已加在 `wildscenes/mmseg_wildscenes/__init__.py`）：

```python
from wildscenes.mmseg_wildscenes.dataset.goose import GooseDataset
```

---

## 1. 镜像传输（本地 → 平台）

```bash
# 本地：导出为纯 tar（镜像约 21GB，gzip 压不动，直接 tar）
docker save -o wildscenes_goose.tar wildscenes_goose:latest

# 传到平台（大文件，rsync 可断点续传）
rsync -a --partial --progress wildscenes_goose.tar 平台:/路径/

# 平台：导入
docker load -i wildscenes_goose.tar
docker images | grep wildscenes_goose   # 确认导入成功
```

---

## 2. 平台容器配置（填表项）

| 配置项 | 填写值 | 说明 |
|---|---|---|
| **镜像** | `wildscenes_goose:latest` | 已注入 GOOSE 支持 |
| **数据输入卷挂载** | 容器内路径 `../data/` | 宿主机侧指向平台上 GOOSE 数据目录 |
| **输出卷挂载** | 容器内路径 `../work_dirs` | checkpoint / 日志 / 最优权重持久保存 |
| **环境变量** | `PYTHONUNBUFFERED=1` | 非必需；让日志实时输出。无强制项可留空 |
| **配置文件** | 无 | 配置已烤进镜像，命令行直接引用 |

> 注意：代码已烤进镜像，**不要再 bind-mount 整个 `..`**，否则会盖掉镜像内的 GOOSE 代码。只挂 data 和 work_dirs。

---

## 3. 命令行

### 脚本 1：数据预处理（只跑一次）

```bash
cd ..

# 自动定位 goose_2d_train 的上一层作为 goose_root（无论挂载后是哪一层都能找到）
GROOT=$(dirname "$(find ../data -maxdepth 3 -type d -name goose_2d_train | head -1)")
echo "goose_root = $GROOT"

python scripts/data/setup_goose.py \
  --goose_root "$GROOT" \
  --save_path  ../data/processed/goose2d
```

预期输出：
```
[ok] train: linked 7845 image/label pairs -> .../goose2d/train
[ok] val:   linked 962 image/label pairs -> .../goose2d/val
Done. 8807 pairs total. ...
```

> `--save_path` 必须是 `../data/processed/goose2d`，因为数据集配置里的 `data_root` 就指向这里。

### 脚本 2：训练（双 4090 分布式）

```bash
cd ..

torchrun --nproc_per_node=2 \
  scripts/benchmark/train2d.py \
  wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose-512x512.py \
  --launcher pytorch \
  --work-dir ../work_dirs/deeplabv3_goose_2x4090 \
  --cfg-options train_dataloader.batch_size=16 train_dataloader.num_workers=4
```

- `--nproc_per_node=2` + `--launcher pytorch`：用满两张卡（SyncBN 跨卡同步）
- `batch_size=16` 是**每张卡**（2 卡 = 有效 32）。显存有余可提到 20（2×20=40，正好匹配 lr=0.01 原始设定）
- `num_workers=4` 每卡；平台 shm 小导致 DataLoader 崩则降为 0

### 脚本 3：评估（训练出权重后）

```bash
cd ..

python scripts/benchmark/eval2d.py \
  wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose-512x512.py \
  ../work_dirs/deeplabv3_goose_2x4090/best_mIoU_iter_*.pth \
  --launcher none \
  --work-dir ../vis_eval/deeplabv3_goose \
  --show-dir painted
```

> GOOSE 无官方预训练权重，用自己训练出的 `.pth`（在 work_dir 内）。

---

## 4. 成功标志与排错

**成功**：训练刷出 `Iter [50/80000]  ... loss: x.xx  decode.acc_seg: ...`，两张卡都有显存占用。

| 报错 | 原因 | 处理 |
|---|---|---|
| `GooseDataset is not registered` | 注册行缺失 | 镜像已注入，正常不会出现 |
| `persistent_workers needs num_workers > 0` | worker=0 时仍开 persistent | 命令已用 worker=4，不会触发；若设 0 则加 `*.persistent_workers=False` |
| `CUDA out of memory` | 显存不足 | `batch_size` 降到 8 |
| `shared memory` / DataLoader 崩 | 平台 shm 太小 | `num_workers=0` |
| `setup_goose` 写入失败 | 数据卷只读 | `--save_path` 改到输出卷下，并用 `--cfg-options ...data_root=` 同步指过去 |
| 找不到 `goose_2d_train` | 数据挂载层级异常 | 手动 `ls ../data/` 确认结构，把含 `goose_2d_train/val` 的目录传给 `--goose_root` |

---

## 5. 关键事实速查

- 数据集：GOOSE 2D，64 类（0=undefined … 63=military_vehicle），train 7845 / val 962
- 标签：`*_labelids.png`，像素值即类别索引（0–63），无需重映射
- 图像：用 `_windshield_vis.png`（RGB），非 `_nir`
- 模型：DeepLabV3 + ResNet-50（d8），ImageNet 预训练 backbone，SyncBN
- 训练计划：80k 迭代，SGD lr=0.01，PolyLR
