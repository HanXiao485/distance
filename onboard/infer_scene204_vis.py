"""
infer_scene204_vis.py
用 GOOSE 12-category DeepLabV3 模型对新场景（scene204）图片做推理，
输出彩色分割图 + 原图/分割并排对比图。没有GT，所以只做定性可视化，不算mIoU。

用法（容器内跑）：
    python infer_scene204_vis.py \
        --config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_goose_category-512x512.py \
        --ckpt work_dirs/goose_category_deeplabv3/best_mIoU_iter_78000.pth \
        --image-dir ../data/scene204_front \
        --output-dir ./outputs/scene204_vis
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "training"))  # 让 wildscenes 包可被 import
import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401
from mmseg.apis import init_model, inference_model

CLASS_NAMES = [
    "void", "sky", "vegetation", "terrain", "construction",
    "vehicle", "road", "object", "sign", "human", "water", "animal",
]
PALETTE = [
    (0,   0,   0),
    (128, 200, 255),
    (50,  200,  50),
    (180, 130,  70),
    (120, 120, 120),
    (255, 100,   0),
    (180, 180, 180),
    (255, 200,   0),
    (255,   0,   0),
    (0,     0, 255),
    (0,   100, 255),
    (255, 128, 200),
]


def colorize(pred: np.ndarray) -> np.ndarray:
    lut = np.array(PALETTE, dtype=np.uint8)
    return lut[pred]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "side_by_side"), exist_ok=True)

    model = init_model(args.config, args.ckpt, device=args.device)

    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.jpg")))
    if not img_paths:
        raise FileNotFoundError(f"未找到图片: {args.image_dir}")
    print(f"共 {len(img_paths)} 张图片，开始推理（GOOSE 12类模型，无GT，仅定性可视化）...")

    times = []
    class_pixel_totals = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    for i, img_path in enumerate(img_paths):
        stem = os.path.splitext(os.path.basename(img_path))[0]

        t0 = time.perf_counter()
        result = inference_model(model, img_path)
        times.append((time.perf_counter() - t0) * 1000)

        pred = result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        for c in range(len(CLASS_NAMES)):
            class_pixel_totals[c] += int((pred == c).sum())

        color = colorize(pred)
        Image.fromarray(color, mode="RGB").save(
            os.path.join(args.output_dir, "color", f"{stem}_color.png"))

        orig = np.array(Image.open(img_path).convert("RGB").resize(
            (pred.shape[1], pred.shape[0]), Image.BILINEAR))
        overlay = (0.5 * orig + 0.5 * color).astype(np.uint8)
        side = np.concatenate([orig, overlay, color], axis=1)
        Image.fromarray(side, mode="RGB").save(
            os.path.join(args.output_dir, "side_by_side", f"{stem}_sbs.png"))

        print(f"  [{i + 1}/{len(img_paths)}] {stem} 完成")

    avg_ms = float(np.mean(times[3:])) if len(times) > 3 else float(np.mean(times))
    total_px = int(class_pixel_totals.sum())

    print(f"\n{'=' * 60}")
    print(f"推理完成: {len(img_paths)} 张（无GT，以下为像素占比统计，非精度指标）")
    print(f"{'=' * 60}")
    for name, cnt in zip(CLASS_NAMES, class_pixel_totals):
        pct = 100.0 * cnt / total_px if total_px > 0 else 0.0
        print(f"  {name:<14}: {pct:5.1f}%")
    print(f"{'-' * 60}")
    print(f"平均推理耗时: {avg_ms:.2f} ms  ({1000 / avg_ms:.1f} FPS)")
    print(f"彩色分割图: {os.path.join(args.output_dir, 'color')}")
    print(f"原图/叠加/分割 并排对比图: {os.path.join(args.output_dir, 'side_by_side')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
