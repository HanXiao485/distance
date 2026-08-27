"""
infer_scene204_vis_64class.py
用 GOOSE 64-class（细粒度）DDRNet 模型对 scene204 图片做推理，
输出彩色分割图 + 原图/分割并排对比图。没有GT，仅定性可视化。

用法（容器内跑）：
    python infer_scene204_vis_64class.py \
        --config wildscenes/configs/ddrnet/ddrnet_23-slim_goose-512x512.py \
        --ckpt work_dirs/ddrnet_goose64_best_mIoU_iter_72000.pth \
        --image-dir ../data/scene204_all_cams \
        --output-dir ./outputs/scene204_vis_64class \
        --device cuda:0
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
import wildscenes.mmseg_wildscenes.dataset.goose  # noqa: F401
from mmseg.apis import init_model, inference_model

CLASS_NAMES = (
    "undefined", "traffic_cone", "snow", "cobble", "obstacle", "leaves",
    "street_light", "bikeway", "ego_vehicle", "pedestrian_crossing",
    "road_block", "road_marking", "car", "bicycle", "person", "bus",
    "forest", "bush", "moss", "traffic_light", "motorcycle", "sidewalk",
    "curb", "asphalt", "gravel", "boom_barrier", "rail_track", "tree_crown",
    "tree_trunk", "debris", "crops", "soil", "rider", "animal", "truck",
    "on_rails", "caravan", "trailer", "building", "wall", "rock", "fence",
    "guard_rail", "bridge", "tunnel", "pole", "traffic_sign", "misc_sign",
    "barrier_tape", "kick_scooter", "low_grass", "high_grass",
    "scenery_vegetation", "sky", "water", "wire", "outlier",
    "heavy_machinery", "container", "hedge", "barrel", "pipe", "tree_root",
    "military_vehicle",
)
PALETTE = [
    (0, 0, 0), (255, 255, 0), (209, 87, 160), (255, 52, 255), (255, 74, 70),
    (0, 137, 65), (0, 111, 166), (163, 0, 89), (255, 219, 229), (122, 73, 0),
    (0, 0, 166), (99, 255, 172), (183, 151, 98), (0, 77, 67), (143, 176, 255),
    (153, 125, 135), (90, 0, 7), (128, 150, 147), (180, 168, 189), (27, 68, 0),
    (79, 198, 1), (59, 93, 255), (74, 59, 83), (255, 47, 128), (97, 97, 90),
    (52, 54, 45), (107, 121, 0), (0, 194, 160), (255, 170, 146), (136, 111, 76),
    (0, 134, 237), (209, 97, 0), (221, 239, 255), (0, 0, 53), (123, 79, 75),
    (161, 194, 153), (48, 0, 24), (10, 166, 216), (1, 51, 73), (0, 132, 111),
    (55, 33, 1), (255, 181, 0), (194, 255, 237), (160, 121, 191), (204, 7, 68),
    (192, 185, 178), (194, 255, 153), (0, 30, 9), (190, 196, 89), (111, 0, 98),
    (12, 189, 102), (238, 195, 255), (69, 109, 117), (183, 123, 104),
    (122, 135, 161), (255, 140, 0), (120, 141, 102), (250, 208, 159),
    (255, 138, 154), (232, 211, 23), (208, 208, 0), (221, 0, 0),
    (196, 164, 132), (64, 64, 64),
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
    print(f"共 {len(img_paths)} 张图片，开始推理（GOOSE 64类模型，无GT，仅定性可视化）...")

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

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(img_paths)}] 完成")

    avg_ms = float(np.mean(times[3:])) if len(times) > 3 else float(np.mean(times))
    total_px = int(class_pixel_totals.sum())

    print(f"\n{'=' * 60}")
    print(f"推理完成: {len(img_paths)} 张（无GT，以下为像素占比>=0.5%的类别，非精度指标）")
    print(f"{'=' * 60}")
    order = np.argsort(-class_pixel_totals)
    for idx in order:
        pct = 100.0 * class_pixel_totals[idx] / total_px if total_px > 0 else 0.0
        if pct >= 0.5:
            print(f"  {CLASS_NAMES[idx]:<20}: {pct:5.1f}%")
    print(f"{'-' * 60}")
    print(f"平均推理耗时: {avg_ms:.2f} ms  ({1000 / avg_ms:.1f} FPS)")
    print(f"彩色分割图: {os.path.join(args.output_dir, 'color')}")
    print(f"原图/叠加/分割 并排对比图: {os.path.join(args.output_dir, 'side_by_side')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
