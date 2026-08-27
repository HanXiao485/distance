"""
Export specific frames (e.g. the worst-error ones) into a self-contained mini dataset
for local single-frame diagnosis.

Copies RGB image, point cloud, 3D label, GT index mask, pred mask, plus shared
pose/calibration files — directly from the server dataset directories, keyed by frame_id.

Usage:
    python scripts/export_worst_frames.py \
        --dir_2d  path/to/WildScenes2d/V-01 \
        --dir_3d  path/to/WildScenes3d/V-01 \
        --pred_mask_dir path/to/deeplabv3_pred_mask \
        --output  path/to/worst_frames \
        --frames  1623378080.863696224 1623379765.054211146 1623379838.017682788
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# ── default: the three worst frames from sequence_summary.json ────────────────
DEFAULT_FRAMES = [
    "1623378080.863696224",
    "1623379765.054211146",
    "1623379838.017682788",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dir_2d", required=True, type=Path,
                   help="WildScenes2d sequence directory (contains image/, indexLabel/)")
    p.add_argument("--dir_3d", required=True, type=Path,
                   help="WildScenes3d sequence directory (contains Clouds/, Labels/)")
    p.add_argument("--pred_mask_dir", type=Path, default=None,
                   help="Directory of pred masks (default: dir_2d/deeplabv3_pred_mask)")
    p.add_argument("--output", required=True, type=Path,
                   help="Output folder")
    p.add_argument("--frames", nargs="*", default=DEFAULT_FRAMES,
                   help="frame_ids to export (dot format). Default: the 3 worst frames.")
    return p.parse_args()


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        print(f"  [WARN] not found: {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def dot_to_dash(frame_id: str) -> str:
    """1623378080.863696224 -> 1623378080-863696224"""
    return frame_id.replace(".", "-", 1)


def main() -> None:
    args = parse_args()

    pred_mask_dir = args.pred_mask_dir or (args.dir_2d / "deeplabv3_pred_mask")

    out_2d = args.output / "WildScenes2d"
    out_3d = args.output / "WildScenes3d"

    print(f"Exporting {len(args.frames)} frames -> {args.output}\n")

    for fid_dot in args.frames:
        fid_dash = dot_to_dash(fid_dot)
        print(f"frame: {fid_dot}")

        # 2D
        copy_file(args.dir_2d / "image"      / f"{fid_dash}.png",
                  out_2d      / "image"      / f"{fid_dash}.png")
        copy_file(args.dir_2d / "indexLabel" / f"{fid_dash}.png",
                  out_2d      / "indexLabel" / f"{fid_dash}.png")
        copy_file(pred_mask_dir              / f"{fid_dash}.png",
                  out_2d / "deeplabv3_pred_mask" / f"{fid_dash}.png")

        # 3D
        copy_file(args.dir_3d / "Clouds" / f"{fid_dot}.bin",
                  out_3d      / "Clouds" / f"{fid_dot}.bin")
        copy_file(args.dir_3d / "Labels" / f"{fid_dot}.label",
                  out_3d      / "Labels" / f"{fid_dot}.label")
        print()

    # shared files
    print("shared files:")
    for src, dst in [
        (args.dir_2d / "camera_calibration.yaml", out_2d / "camera_calibration.yaml"),
        (args.dir_2d / "poses2d.csv",             out_2d / "poses2d.csv"),
        (args.dir_3d / "poses3d.csv",             out_3d / "poses3d.csv"),
    ]:
        if copy_file(src, dst):
            print(f"  OK {src.name}")

    print(f"\nDone -> {args.output}")


if __name__ == "__main__":
    main()
