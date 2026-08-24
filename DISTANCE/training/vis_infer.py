#!/usr/bin/env python3
"""
GOOSE 12-category 推理可视化 + 推理时间统计
输出: 原图 | GT (Ground Truth) | Pred (Prediction) 三拼图，带标题和推理时间

用法 (在容器 .. 下):
    # DeepLabV3
    python vis_infer.py --model deeplabv3 \
        --config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_deeplabv3/best_mIoU_iter_78000.pth \
        --device cuda:0

    # DDRNet
    python vis_infer.py --model ddrnet \
        --config wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_ddrnet/best_mIoU_iter_80000.pth \
        --device cuda:1
"""
import os
import sys
import glob
import time
import argparse
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 注册 WildScenes 自定义 transforms (RemapLabel 等)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ 目录，让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401
from mmseg.apis import init_model, inference_model

# ── 数据路径（容器内路径） ────────────────────────────────────────────────────
GOOSE_VAL_ROOT = "../data/Goose/goose_2d_val"
IMG_GLOB       = os.path.join(GOOSE_VAL_ROOT, "images/val/**/*_windshield_vis.png")
LBL_ROOT       = os.path.join(GOOSE_VAL_ROOT, "labels/val")
OUT_BASE       = "../vis_output"  # 输出目录 ../vis_output

# ── 12 类名称 & 颜色 ──────────────────────────────────────────────────────────
CLASS_NAMES = [
    "void", "sky", "vegetation", "terrain", "construction",
    "vehicle", "road", "object", "sign", "human", "water", "animal"
]
PALETTE = np.array([
    [0,   0,   0],    # void
    [128, 200, 255],  # sky
    [50,  200,  50],  # vegetation
    [180, 130,  70],  # terrain
    [120, 120, 120],  # construction
    [255, 100,   0],  # vehicle
    [180, 180, 180],  # road
    [255, 200,   0],  # object
    [255,   0,   0],  # sign
    [0,     0, 255],  # human
    [0,   100, 255],  # water
    [255, 128, 200],  # animal
], dtype=np.uint8)

# ── GOOSE 64 类 → 12 类 LUT ───────────────────────────────────────────────────
FINE_TO_CATEGORY = {
    0: 0, 8: 0, 56: 0,
    53: 1,
    5: 2, 16: 2, 17: 2, 18: 2, 27: 2, 28: 2, 30: 2, 50: 2, 51: 2, 52: 2, 59: 2, 62: 2,
    2: 3, 3: 3, 23: 3, 24: 3, 31: 3,
    29: 4, 38: 4, 39: 4, 41: 4, 42: 4, 43: 4, 44: 4, 55: 4, 58: 4,
    12: 5, 13: 5, 15: 5, 20: 5, 34: 5, 35: 5, 36: 5, 37: 5, 49: 5, 57: 5, 63: 5,
    7: 6, 9: 6, 11: 6, 21: 6, 22: 6, 26: 6,
    4: 7, 6: 7, 40: 7, 45: 7, 60: 7, 61: 7,
    1: 8, 10: 8, 19: 8, 25: 8, 46: 8, 47: 8, 48: 8,
    14: 9, 32: 9,
    54: 10,
    33: 11,
}
LUT = np.full(256, 255, dtype=np.uint8)
for src, dst in FINE_TO_CATEGORY.items():
    LUT[src] = dst


def label_to_color(label_map):
    rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for cls_id in range(12):
        rgb[label_map == cls_id] = PALETTE[cls_id]
    return rgb


def make_legend():
    return [
        mpatches.Patch(facecolor=PALETTE[i] / 255.0, label=CLASS_NAMES[i])
        for i in range(12)
    ]


def find_label(img_path):
    """根据图像路径找对应的 labelids.png（跨 scene 子目录）。"""
    stem = os.path.basename(img_path).replace("_windshield_vis.png", "")
    lbl_name = stem + "_labelids.png"
    matches = glob.glob(os.path.join(LBL_ROOT, "**", lbl_name), recursive=True)
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="模型名称，用于输出子目录")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n",      type=int, default=20,
                        help="可视化图片数量，-1 = 全部 val 集")
    args = parser.parse_args()

    out_dir = os.path.join(OUT_BASE, args.model)
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(IMG_GLOB, recursive=True))
    assert imgs, f"找不到图像，检查路径: {IMG_GLOB}"
    if args.n > 0:
        imgs = imgs[:args.n]

    print(f"加载模型: {args.ckpt}")
    model = init_model(args.config, args.ckpt, device=args.device)

    print("GPU 预热...")
    _ = inference_model(model, imgs[0])
    torch.cuda.synchronize()

    ROAD_CLS = 3  # 12类中 terrain = 可通行类（asphalt/gravel/soil/cobble/snow）

    times = []
    num_classes = 12
    area_intersect = np.zeros(num_classes, dtype=np.float64)
    area_union     = np.zeros(num_classes, dtype=np.float64)
    area_gt        = np.zeros(num_classes, dtype=np.float64)
    area_pred      = np.zeros(num_classes, dtype=np.float64)
    # 可通行区域 per-image 精确率 / 召回率
    per_prec = []
    per_rec  = []

    print(f"\n开始推理，共 {len(imgs)} 张...\n")

    for i, img_path in enumerate(imgs):
        lbl_path = find_label(img_path)

        t0 = time.perf_counter()
        result = inference_model(model, img_path)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        pred_color = label_to_color(pred)

        img_rgb = np.array(Image.open(img_path).convert("RGB"))

        if lbl_path and os.path.exists(lbl_path):
            gt_fine = np.array(Image.open(lbl_path))
            gt_cat  = LUT[gt_fine.astype(np.uint8)]
            gt_color = label_to_color(gt_cat)
            has_gt = True

            valid   = gt_cat != 255
            p_valid = pred[valid].astype(np.int32)
            g_valid = gt_cat[valid].astype(np.int32)

            # ── IoU 统计 ────────────────────────────────────────────────
            for cls in range(num_classes):
                pm = p_valid == cls
                gm = g_valid == cls
                area_intersect[cls] += int((pm & gm).sum())
                area_union[cls]     += int((pm | gm).sum())
                area_gt[cls]        += int(gm.sum())
                area_pred[cls]      += int(pm.sum())

            # ── 可通行区域精确率 / 召回率（terrain = class 3，置信度阈值 0.45）
            conf = torch.softmax(result.seg_logits.data.float(), 0).max(0).values.cpu().numpy()
            low_conf  = conf < 0.45
            pred_road = (pred == ROAD_CLS) & valid & (~low_conf)
            gt_road   = (gt_cat == ROAD_CLS) & valid
            tp_px = int((pred_road & gt_road).sum())
            ps    = int(pred_road.sum())
            gs    = int(gt_road.sum())
            per_prec.append(tp_px / (ps + 1e-9))
            if gs > 0:
                per_rec.append(tp_px / gs)
        else:
            has_gt = False

        # ── 三拼图 ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(
            f"[{args.model}]  {os.path.basename(img_path)}  |  推理时间: {elapsed_ms:.1f} ms",
            fontsize=13, fontweight="bold", y=1.01
        )

        axes[0].imshow(img_rgb)
        axes[0].set_title("RGB Image", fontsize=13, fontweight="bold", pad=8)
        axes[0].axis("off")

        axes[1].imshow(gt_color if has_gt else np.zeros_like(img_rgb))
        axes[1].set_title("GT  (Ground Truth)", fontsize=13, fontweight="bold", pad=8)
        axes[1].axis("off")

        axes[2].imshow(pred_color)
        axes[2].set_title("Pred  (Prediction)", fontsize=13, fontweight="bold", pad=8)
        axes[2].axis("off")

        fig.legend(
            handles=make_legend(),
            loc="lower center",
            ncol=6,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.04),
            framealpha=0.9
        )

        plt.tight_layout()
        out_name = os.path.basename(img_path).replace("_windshield_vis.png", ".png")
        plt.savefig(os.path.join(out_dir, out_name), dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  [{i+1:3d}/{len(imgs)}] {os.path.basename(img_path):50s} {elapsed_ms:7.1f} ms")

    # ── 推理时间统计 ─────────────────────────────────────────────────────────
    times = np.array(times)
    t = times[1:] if len(times) > 1 else times
    print(f"\n{'='*60}")
    print(f"推理时间统计 ({len(t)} 张，已排除第一张预热):")
    print(f"  平均:  {t.mean():.1f} ms  ({1000/t.mean():.1f} FPS)")
    print(f"  中位:  {np.median(t):.1f} ms")
    print(f"  最快:  {t.min():.1f} ms")
    print(f"  最慢:  {t.max():.1f} ms")

    # ── mIoU / mAcc / aAcc ───────────────────────────────────────────────────
    iou  = area_intersect / (area_union + 1e-10)
    acc  = area_intersect / (area_gt    + 1e-10)
    valid_cls = area_gt > 0          # 只统计 val 集中实际出现的类
    mIoU = iou[valid_cls].mean() * 100
    mAcc = acc[valid_cls].mean() * 100
    aAcc = area_intersect.sum() / (area_gt.sum() + 1e-10) * 100

    print(f"\n{'='*60}")
    print(f"分割指标 (val {len(imgs)} 张):")
    print(f"  aAcc:  {aAcc:.2f}%")
    print(f"  mIoU:  {mIoU:.2f}%")
    print(f"  mAcc:  {mAcc:.2f}%")
    print(f"\n{'%-16s'% '类别'} {'IoU':>8}  {'Acc':>8}  {'GT像素':>12}")
    print("-" * 50)
    for cls in range(num_classes):
        tag = "" if valid_cls[cls] else "  (未出现)"
        print(f"  {CLASS_NAMES[cls]:<14} {iou[cls]*100:7.2f}%  {acc[cls]*100:7.2f}%  {int(area_gt[cls]):>12,}{tag}")
    print(f"{'='*60}")

    # ── 可通行区域评测（road, class 6）───────────────────────────────────────
    def print_dist(name, arr):
        bins   = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]
        labels = ["  <50%", "50-70%", "70-80%", "80-90%", " >=90%"]
        print(f"\n  [{name}] per-image 分布 (共 {len(arr)} 张):")
        for j in range(len(labels)):
            m = (arr >= bins[j]) & (arr < bins[j + 1])
            print(f"    {labels[j]} : {m.mean()*100:5.1f}%   ({int(m.sum())} 张)")
        print(f"    --> mean={arr.mean()*100:.2f}%  median={np.median(arr)*100:.2f}%  >=90%占比={( arr>=0.9).mean()*100:.1f}%")

    pp = np.array(per_prec)
    pr = np.array(per_rec)
    print(f"\n{'='*60}")
    print(f"可通行区域评测  (terrain = class {ROAD_CLS}, val {len(imgs)} 张)")
    print(f"  Precision = TP / (TP+FP)  只惩罚误报")
    print(f"  Recall    = TP / (TP+FN)  只惩罚漏报")
    if len(pp):
        print_dist("Precision 精确率", pp)
    if len(pr):
        print_dist("Recall    召回率", pr)
    print(f"{'='*60}")
    print(f"\n可视化结果: {out_dir}")


if __name__ == "__main__":
    main()
