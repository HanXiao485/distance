"""
infer_val.py
在 GOOSE val 集上随机抽 N 张图推理，报告 per-class IoU、mIoU 和推理速度。

用法:
    # DeepLabV3
    python infer_val.py \
        --config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_deeplabv3/best_mIoU_iter_78000.pth

    # DDRNet
    python infer_val.py \
        --config wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_ddrnet/best_mIoU_iter_80000.pth
"""
import os, sys, argparse, time, random, glob
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ 目录，让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401
from mmseg.apis import init_model, inference_model

# 64-class → 12-category LUT (same as goose_category.py)
FINE_TO_CATEGORY = {
    0: 0, 8: 0, 56: 0,
    53: 1,
    5: 2, 16: 2, 17: 2, 18: 2, 27: 2, 28: 2,
    30: 2, 50: 2, 51: 2, 52: 2, 59: 2, 62: 2,
    2: 3, 3: 3, 23: 3, 24: 3, 31: 3,
    29: 4, 38: 4, 39: 4, 41: 4, 42: 4, 43: 4, 44: 4, 55: 4, 58: 4,
    12: 5, 13: 5, 15: 5, 20: 5, 34: 5, 35: 5,
    36: 5, 37: 5, 49: 5, 57: 5, 63: 5,
    7: 6, 9: 6, 11: 6, 21: 6, 22: 6, 26: 6,
    4: 7, 6: 7, 40: 7, 45: 7, 60: 7, 61: 7,
    1: 8, 10: 8, 19: 8, 25: 8, 46: 8, 47: 8, 48: 8,
    14: 9, 32: 9,
    54: 10,
    33: 11,
}
LABEL_LUT = np.full(256, 255, dtype=np.uint8)
for src, dst in FINE_TO_CATEGORY.items():
    LABEL_LUT[src] = dst

CLASS_NAMES = [
    "void", "sky", "vegetation", "terrain", "construction",
    "vehicle", "road", "object", "sign", "human", "water", "animal",
]
NUM_CLASSES  = 12
IGNORE_INDEX = 255
DATA_ROOT    = "../data/Goose/goose_2d_val"


def load_gt(ann_path):
    raw = np.array(Image.open(ann_path), dtype=np.uint8)
    return LABEL_LUT[raw]


def accumulate_intersect_union(pred, gt, total_inter, total_union):
    """按 mmseg IoUMetric 的方式在整个数据集上累积每类的交集/并集像素数。"""
    valid = gt != IGNORE_INDEX
    for c in range(NUM_CLASSES):
        p = (pred == c) & valid
        g = (gt == c) & valid
        total_inter[c] += int((p & g).sum())
        total_union[c] += int((p | g).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num",    type=int, default=50)
    ap.add_argument("--seed",   type=int, default=42)
    args = ap.parse_args()

    img_dir = os.path.join(DATA_ROOT, "images", "val")
    ann_dir = os.path.join(DATA_ROOT, "labels", "val")

    img_paths = sorted(glob.glob(os.path.join(img_dir, "**", "*_windshield_vis.png"), recursive=True))
    if not img_paths:
        raise FileNotFoundError(f"No images found under {img_dir}")

    random.seed(args.seed)
    sample = random.sample(img_paths, min(args.num, len(img_paths)))

    print(f"\n{'='*60}")
    print(f"模型  : {os.path.basename(args.ckpt)}")
    print(f"Val图 : {len(img_paths)} 张总数，抽样 {len(sample)} 张")
    print(f"{'='*60}")

    model = init_model(args.config, args.ckpt, device=args.device)

    total_inter = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_union = np.zeros(NUM_CLASSES, dtype=np.int64)
    times   = []
    skipped = 0

    for rank, img_path in enumerate(sample):
        rel     = os.path.relpath(img_path, img_dir)
        ann_path = os.path.join(ann_dir, rel.replace("_windshield_vis.png", "_labelids.png"))
        if not os.path.exists(ann_path):
            skipped += 1
            continue

        gt = load_gt(ann_path)

        t0 = time.perf_counter()
        result = inference_model(model, img_path)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        if pred.shape != gt.shape:
            pred = np.array(Image.fromarray(pred).resize(
                (gt.shape[1], gt.shape[0]), Image.NEAREST))

        accumulate_intersect_union(pred, gt, total_inter, total_union)

        if (rank + 1) % 10 == 0:
            print(f"  [{rank+1}/{len(sample)}] 完成")

    if times == []:
        print("没有有效样本，请检查数据路径")
        return

    with np.errstate(invalid="ignore", divide="ignore"):
        per_cls = np.where(total_union > 0, total_inter / total_union, np.nan) * 100
    miou     = float(np.nanmean(per_cls))
    avg_ms   = float(np.mean(times[3:])) if len(times) > 3 else float(np.mean(times))

    print(f"\n{'─'*42}")
    print(f"{'类别':<20} {'IoU (%)':>10}")
    print(f"{'─'*42}")
    for name, iou in zip(CLASS_NAMES, per_cls):
        mark = "  (nan)" if np.isnan(iou) else ""
        val  = f"{iou:>9.1f}%" if not np.isnan(iou) else "       n/a"
        print(f"{name:<20} {val}{mark}")
    print(f"{'─'*42}")
    print(f"{'mIoU':<20} {miou:>9.2f}%")
    print(f"{'─'*42}")
    print(f"\n推理速度 : {avg_ms:.1f} ms  ({1000/avg_ms:.1f} FPS)")
    if skipped:
        print(f"跳过     : {skipped} 张（无对应标注）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
