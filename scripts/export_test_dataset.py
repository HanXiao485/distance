"""
Build a WildScenes-mini dataset from frames used in a DISTANCE run.

Reads frame IDs from sequence_summary.json, then copies the corresponding
files directly from the server dataset directories.

Output structure mirrors WildScenes so it can be used as a drop-in replacement.

Usage:
    python scripts/export_test_dataset.py \
        --summary  /path/to/sequence_summary.json \
        --dir_2d   /path/to/WildScenes2d/V-01 \
        --dir_3d   /path/to/WildScenes3d/V-01 \
        --output   /path/to/wildscenes_mini
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, type=Path,
                   help="sequence_summary.json from a DISTANCE run")
    p.add_argument("--dir_2d", required=True, type=Path,
                   help="Server path to WildScenes2d sequence directory")
    p.add_argument("--dir_3d", required=True, type=Path,
                   help="Server path to WildScenes3d sequence directory")
    p.add_argument("--output", required=True, type=Path,
                   help="Output folder (wildscenes_mini)")
    return p.parse_args()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  [WARN] not found: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def dot_to_dash(frame_id: str) -> str:
    """1623378686.882864149  →  1623378686-882864149"""
    return frame_id.replace(".", "-", 1)


def main() -> None:
    args = parse_args()

    with args.summary.open() as f:
        summary = json.load(f)

    frames = [fr for fr in summary.get("per_frame", []) if "error" not in fr]
    print(f"Found {len(frames)} frames in summary\n")

    out_2d = args.output / "WildScenes2d"
    out_3d = args.output / "WildScenes3d"

    for fr in frames:
        fid_dot  = fr["frame_id"]          # e.g. 1623378686.882864149
        fid_dash = dot_to_dash(fid_dot)    # e.g. 1623378686-882864149

        # 2D files
        copy_file(args.dir_2d / "image"               / f"{fid_dash}.png",
                  out_2d      / "image"               / f"{fid_dash}.png")
        copy_file(args.dir_2d / "indexLabel"          / f"{fid_dash}.png",
                  out_2d      / "indexLabel"          / f"{fid_dash}.png")
        copy_file(args.dir_2d / "deeplabv3_pred_mask" / f"{fid_dash}.png",
                  out_2d      / "deeplabv3_pred_mask" / f"{fid_dash}.png")

        # 3D files
        copy_file(args.dir_3d / "Clouds" / f"{fid_dot}.bin",
                  out_3d      / "Clouds" / f"{fid_dot}.bin")
        copy_file(args.dir_3d / "Labels" / f"{fid_dot}.label",
                  out_3d      / "Labels" / f"{fid_dot}.label")

        print(f"  ✓ {fid_dot}")

    # shared files
    print("\nCopying shared files:")
    shared = [
        (args.dir_2d / "camera_calibration.yaml", out_2d / "camera_calibration.yaml"),
        (args.dir_2d / "poses2d.csv",             out_2d / "poses2d.csv"),
        (args.dir_3d / "poses3d.csv",             out_3d / "poses3d.csv"),
    ]
    for src, dst in shared:
        copy_file(src, dst)
        print(f"  ✓ {src.name}")

    print(f"\nDone. WildScenes-mini saved to: {args.output}")


if __name__ == "__main__":
    main()
