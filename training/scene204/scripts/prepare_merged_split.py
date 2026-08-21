#!/usr/bin/env python3
"""Merge all cameras from one or more scenes into a single train/val split.

Usage
-----
# 只有 scene204
python prepare_merged_split.py \
    --scenes /root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d \
    --out    /root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d/merged

# 加入更多 scene 后（把多个 WildScenes2d 目录都传进来）
python prepare_merged_split.py \
    --scenes /root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d \
             /root/distance/WildScenes/DISTANCE/datasets/scene205_wildscenes/WildScenes2d \
    --out    /root/distance/WildScenes/DISTANCE/datasets/all_scenes_merged \
    --val-ratio 0.15

Output layout
-------------
<out>/
├── train/
│   ├── image/        ← symlinks to jpg files
│   └── indexLabel/   ← symlinks to png files
└── val/
    ├── image/
    └── indexLabel/
"""
import argparse
import os
from pathlib import Path
import random


CAMERAS = [
    "FRONT_CAMERA",
    "BACK_CAMERA",
    "FRONT_LEFT_CAMERA",
    "FRONT_RIGHT_CAMERA",
    "REAR_LEFT_CAMERA",
    "REAR_RIGHT_CAMERA",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="+", required=True,
                   help="One or more WildScenes2d root directories")
    p.add_argument("--out", required=True,
                   help="Output merged directory")
    p.add_argument("--val-ratio", type=float, default=0.15,
                   help="Fraction of frames per camera to hold out for val (default 0.15)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def collect_pairs(scene_roots: list[Path]) -> dict[str, list[tuple[Path, Path]]]:
    """Return {camera: [(img_path, label_path), ...]} sorted by stem."""
    per_camera: dict[str, list[tuple[Path, Path]]] = {c: [] for c in CAMERAS}
    for scene_root in scene_roots:
        for cam in CAMERAS:
            img_dir   = scene_root / cam / "image"
            label_dir = scene_root / cam / "indexLabel"
            if not img_dir.exists() or not label_dir.exists():
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                label_path = label_dir / (img_path.stem + ".png")
                if label_path.exists():
                    per_camera[cam].append((img_path, label_path))
    return per_camera


def make_symlinks(pairs: list[tuple[Path, Path]], img_dir: Path, lbl_dir: Path):
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for img_src, lbl_src in pairs:
        img_dst = img_dir / img_src.name
        lbl_dst = lbl_dir / lbl_src.name
        if img_dst.exists() or img_dst.is_symlink():
            img_dst.unlink()
        if lbl_dst.exists() or lbl_dst.is_symlink():
            lbl_dst.unlink()
        img_dst.symlink_to(img_src.resolve())
        lbl_dst.symlink_to(lbl_src.resolve())


def main():
    args = parse_args()
    random.seed(args.seed)
    out = Path(args.out)
    scene_roots = [Path(s) for s in args.scenes]

    per_camera = collect_pairs(scene_roots)

    train_pairs: list[tuple[Path, Path]] = []
    val_pairs:   list[tuple[Path, Path]] = []

    for cam, pairs in per_camera.items():
        if not pairs:
            continue
        n_val = max(1, round(len(pairs) * args.val_ratio))
        # last N frames → val (temporal consistency; avoids leaking nearby frames)
        val_pairs.extend(pairs[-n_val:])
        train_pairs.extend(pairs[:-n_val])
        print(f"  {cam:25s}: {len(pairs):3d} frames  →  train {len(pairs)-n_val}  val {n_val}")

    make_symlinks(train_pairs, out / "train" / "image", out / "train" / "indexLabel")
    make_symlinks(val_pairs,   out / "val"   / "image", out / "val"   / "indexLabel")

    print(f"\nDone.  train={len(train_pairs)}  val={len(val_pairs)}  → {out}")


if __name__ == "__main__":
    main()
