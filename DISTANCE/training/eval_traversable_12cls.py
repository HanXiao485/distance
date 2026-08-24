"""
GOOSE 12类模型 可通行区域（Terrain, class 3）评测
置信度阈值 CONF_THRESH = 0.45

用法 (在容器 .. 下):
    python eval_traversable_12cls.py \
        --config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_deeplabv3/best_mIoU_iter_78000.pth \
        --device cuda:0

    python eval_traversable_12cls.py \
        --config wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_ddrnet/best_mIoU_iter_80000.pth \
        --device cuda:1
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # training/ 目录，让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401
from mmseg.apis import init_model, inference_model

CONF_THRESH   = 0.45
TERRAIN_CLS   = 3
GOOSE_VAL_ROOT = "../data/Goose/goose_2d_val"
IMG_GLOB      = os.path.join(GOOSE_VAL_ROOT, "images/val/**/*_windshield_vis.png")
LBL_ROOT      = os.path.join(GOOSE_VAL_ROOT, "labels/val")

FINE_TO_CATEGORY = {
    0: 0, 8: 0, 56: 0, 53: 1,
    5: 2, 16: 2, 17: 2, 18: 2, 27: 2, 28: 2, 30: 2, 50: 2, 51: 2, 52: 2, 59: 2, 62: 2,
    2: 3, 3: 3, 23: 3, 24: 3, 31: 3,
    29: 4, 38: 4, 39: 4, 41: 4, 42: 4, 43: 4, 44: 4, 55: 4, 58: 4,
    12: 5, 13: 5, 15: 5, 20: 5, 34: 5, 35: 5, 36: 5, 37: 5, 49: 5, 57: 5, 63: 5,
    7: 6, 9: 6, 11: 6, 21: 6, 22: 6, 26: 6,
    4: 7, 6: 7, 40: 7, 45: 7, 60: 7, 61: 7,
    1: 8, 10: 8, 19: 8, 25: 8, 46: 8, 47: 8, 48: 8,
    14: 9, 32: 9, 54: 10, 33: 11,
}
LUT = np.full(256, 255, dtype=np.uint8)
for src, dst in FINE_TO_CATEGORY.items():
    LUT[src] = dst


def find_label(img_path):
    stem = os.path.basename(img_path).replace("_windshield_vis.png", "")
    matches = glob.glob(os.path.join(LBL_ROOT, "**", stem + "_labelids.png"), recursive=True)
    return matches[0] if matches else None


def print_dist(name, arr):
    bins   = [0.0, 0.5, 0.7, 0.8, 0.9, 1.01]
    labels = ["  <50%", "50-70%", "70-80%", "80-90%", " >=90%"]
    print(f"\n  [{name}] per-image 分布 (共 {len(arr)} 张):")
    for j in range(len(labels)):
        m = (arr >= bins[j]) & (arr < bins[j + 1])
        print(f"    {labels[j]} : {m.mean()*100:5.1f}%   ({int(m.sum())} 张)")
    print(f"    --> mean={arr.mean()*100:.2f}%  median={np.median(arr)*100:.2f}%  >=90%占比={(arr>=0.9).mean()*100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt",   required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    imgs = sorted(glob.glob(IMG_GLOB, recursive=True))
    assert imgs, f"找不到图像: {IMG_GLOB}"

    print(f"加载模型: {args.ckpt}")
    model = init_model(args.config, args.ckpt, device=args.device)

    per_prec, per_rec = [], []

    for i, img_path in enumerate(imgs):
        lbl_path = find_label(img_path)
        if not lbl_path:
            continue

        result   = inference_model(model, img_path)
        pred     = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        conf     = torch.softmax(result.seg_logits.data.float(), 0).max(0).values.cpu().numpy()
        low_conf = conf < CONF_THRESH

        gt_fine  = np.array(Image.open(lbl_path))
        gt_cat   = LUT[gt_fine.astype(np.uint8)]
        valid    = gt_cat != 255

        pred_t = (pred == TERRAIN_CLS) & valid & (~low_conf)
        gt_t   = (gt_cat == TERRAIN_CLS) & valid

        tp = int((pred_t & gt_t).sum())
        ps = int(pred_t.sum())
        gs = int(gt_t.sum())

        per_prec.append(tp / (ps + 1e-9))
        if gs > 0:
            per_rec.append(tp / gs)

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(imgs)}")

    pp = np.array(per_prec)
    pr = np.array(per_rec)

    print(f"\n{'='*60}")
    print(f"可通行区域评测  (Terrain class {TERRAIN_CLS}, CONF≥{CONF_THRESH}, val {len(imgs)} 张)")
    print_dist("Precision 精确率", pp)
    print_dist("Recall    召回率", pr)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
