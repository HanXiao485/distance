"""Batch boundary-error pipeline over an ordered sequence of frames.

Config format:  configs/sequence_config_*.json
  {
    "project_root":  "...",
    "output_dir":    "outputs/sequence_001",
    "class":         { "name":"dirt", "id_2d":2, "id_3d_for_h":1 },
    "label3d_dtype": "uint32",
    "geometry":      { "pose2d_direction":"sensor_to_world",
                       "pose3d_direction":"sensor_to_world",
                       "max_timestamp_diff_sec":0.05,
                       "use_distortion":false },
    "shared_paths":  { "poses2d":"calibration/poses2d.csv",
                       "poses3d":"calibration/poses3d.csv",
                       "calibration":"calibration/camera_calibration.yaml" },
    "masks": {
      "gt":   { "type":"index" },
      "pred": { "type":"color_threshold", "threshold":{ ... } }
    },
    "mask_processing": { "min_component_area_px":5000, "close_iterations":2 },
    "h_estimation":    { "plane_inlier_threshold_m":0.5, "robust_method":"none" },
    "h_diagnostics":   { "enabled":true },
    "scanline":        { "step_px":10, "edge_margin_px":0,
                         "max_boundary_run_width_px":25 },
    "frames": [
      {
        "frame_id":   "1623378686.882864149",
        "rgb":        "raw/frames/.../1623378686.882864149_rgb.png",
        "pointcloud": "raw/frames/.../1623378686.882864149.bin",
        "label3d":    "labels/frames/.../1623378686.882864149_3d.label",
        "gt_index":   "labels/frames/.../..._gt_indexlabels.png",
        "pred_mask":  "labels/frames/.../..._pred_color.png"
      },
      ...
    ]
  }

Each frame is processed independently; per-frame outputs go to
  <output_dir>/frames/<frame_id>/

Aggregate statistics over all frames are saved to
  <output_dir>/sequence_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_boundary_error_pipeline import (
    ensure_dir,
    resolve_path,
    stats,
    extract_correspondences,
    estimate_h,
    create_boundaries,
    scanline_error,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full boundary-error pipeline over an ordered frame sequence."
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Sequence config JSON path.")
    parser.add_argument("--frames", nargs="*", default=None,
                        help="Optional subset of frame_ids to process (default: all).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames whose pipeline_summary.json already exists.")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["_config_path"] = str(path)
    return cfg


def _2d_stem_to_3d_stem(stem: str) -> str:
    """'1623377792-158486120' → '1623377792.158486120'"""
    return stem.replace("-", ".", 1)


def _count_class_pixels(gt_path: Path, class_id: int) -> int:
    import numpy as np
    from PIL import Image
    return int((np.array(Image.open(gt_path)) == class_id).sum())


def discover_frames(seq_cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Scan dataset dirs and auto-build the frames list.

    Anchors on image/ filenames; matches 3D files by exact timestamp
    (same digits, just dash→dot separator conversion). Skips any frame
    missing one or more required files or below the min_class_pixels threshold.
    """
    ds = seq_cfg["dataset"]
    dir_2d = Path(ds["dir_2d"])
    dir_3d = Path(ds["dir_3d"])
    class_id_2d = int(seq_cfg["class"]["id_2d"])
    min_pixels  = int(ds.get("min_class_pixels", 0))

    clouds  = {p.stem: p for p in (dir_3d / "Clouds").glob("*.bin")}
    labels3 = {p.stem: p for p in (dir_3d / "Labels").glob("*.label")}

    images = sorted((dir_2d / "image").glob("*.png"))
    frames: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for img_path in images:
        stem_3d   = _2d_stem_to_3d_stem(img_path.stem)
        gt_path   = dir_2d / "indexLabel"          / img_path.name
        pred_path = dir_2d / "deeplabv3_pred_mask" / img_path.name
        cloud_path  = clouds.get(stem_3d)
        label3_path = labels3.get(stem_3d)

        missing = []
        if not gt_path.exists():    missing.append("indexLabel")
        if not pred_path.exists():  missing.append("deeplabv3_pred_mask")
        if cloud_path  is None:     missing.append("pointcloud")
        if label3_path is None:     missing.append("label3d")

        if missing:
            skipped.append({"frame_id": img_path.stem, "missing": missing})
            continue

        if min_pixels > 0:
            px = _count_class_pixels(gt_path, class_id_2d)
            if px < min_pixels:
                skipped.append({"frame_id": img_path.stem,
                                 "missing": [f"class_pixels_below_threshold({px}<{min_pixels})"]})
                continue

        frames.append({
            "frame_id":   img_path.stem,
            "rgb":        str(img_path),
            "pointcloud": str(cloud_path),
            "label3d":    str(label3_path),
            "gt_index":   str(gt_path),
            "pred_mask":  str(pred_path),
        })

    meta = {
        "source":           "auto_discovery",
        "dir_2d":           str(dir_2d),
        "dir_3d":           str(dir_3d),
        "min_class_pixels": min_pixels,
        "total_images":     len(images),
        "matched_frames":   len(frames),
        "skipped_frames":   len(skipped),
        "skipped_detail":   skipped[:50],
    }
    return frames, meta


def _resolve_shared_paths(seq_cfg: dict[str, Any]) -> dict[str, str]:
    """Return shared paths, auto-deriving from dataset dirs if not explicit."""
    explicit = seq_cfg.get("shared_paths", {})
    if explicit:
        return explicit
    ds = seq_cfg.get("dataset", {})
    dir_2d = ds.get("dir_2d", "")
    dir_3d = ds.get("dir_3d", "")
    return {
        "poses2d":     str(Path(dir_2d) / "poses2d.csv"),
        "poses3d":     str(Path(dir_3d) / "poses3d.csv"),
        "calibration": str(Path(dir_2d) / "camera_calibration.yaml"),
    }


def build_frame_config(seq_cfg: dict[str, Any], frame: dict[str, Any]) -> dict[str, Any]:
    """Merge sequence-level shared settings with one frame's per-frame paths."""
    shared = _resolve_shared_paths(seq_cfg)
    masks_cfg = seq_cfg.get("masks", {})

    gt_cfg = dict(masks_cfg.get("gt", {}))
    if gt_cfg.get("type") == "index":
        gt_cfg["path"] = frame["gt_index"]

    pred_cfg = dict(masks_cfg.get("pred", {}))
    pred_cfg["path"] = frame["pred_mask"]

    return {
        "project_root":    seq_cfg["project_root"],
        "class":           seq_cfg["class"],
        "label3d_dtype":   seq_cfg.get("label3d_dtype", "uint32"),
        "geometry":        seq_cfg.get("geometry", {}),
        "h_estimation":    seq_cfg.get("h_estimation", {}),
        "h_diagnostics":   seq_cfg.get("h_diagnostics", {}),
        "mask_processing": seq_cfg.get("mask_processing", {}),
        "scanline":        seq_cfg.get("scanline", {}),
        "masks":           {"gt": gt_cfg, "pred": pred_cfg},
        "paths": {
            "rgb":         frame["rgb"],
            "pointcloud":  frame["pointcloud"],
            "label3d":     frame["label3d"],
            "poses2d":     shared["poses2d"],
            "poses3d":     shared["poses3d"],
            "calibration": shared["calibration"],
        },
    }


def aggregate_results(per_frame: list[dict[str, Any]]) -> dict[str, Any]:
    """Pool all per-frame scanline and H-matrix metrics into global statistics."""
    left_all, right_all, overall_all = [], [], []
    reproj_means, reproj_p95s = [], []
    tilt_degs: list[float] = []
    z_before_total, z_after_total = 0, 0
    valid_scan_frames = 0

    for f in per_frame:
        # --- scanline ---
        sl = f.get("scanline_error", {})
        if sl.get("sampled_scanline_count", 0) > 0:
            valid_scan_frames += 1
            left_all.append(sl.get("left_error_m", {}))
            right_all.append(sl.get("right_error_m", {}))
            overall_all.append(sl.get("overall_error_m", {}))

        # --- H reprojection ---
        h = f.get("h_estimation", {})
        reproj = h.get("image_reprojection_error_px", {})
        if reproj.get("count", 0) > 0:
            reproj_means.append(float(reproj["mean"]))
            reproj_p95s.append(float(reproj["p95"]))

        # --- plane tilt ---
        diag = h.get("diagnostics", {})
        tilt = diag.get("plane_normal_tilt_from_vertical_deg")
        if tilt is not None:
            tilt_degs.append(float(tilt))

        # --- z-filter counts ---
        zf = h.get("z_filter", {})
        if zf.get("enabled", False):
            z_before_total += int(zf.get("points_before", 0))
            z_after_total  += int(zf.get("points_after",  0))

    def pool_stat(stat_list: list[dict]) -> dict[str, Any]:
        counts = [s.get("count", 0) for s in stat_list]
        total  = sum(counts)
        if total == 0:
            return {"count": 0}
        w_mean   = sum(s.get("mean", 0) * s.get("count", 0) for s in stat_list) / total
        p95_vals = [s.get("p95", float("nan")) for s in stat_list if s.get("count", 0) > 0]
        p90_vals = [s.get("p90", float("nan")) for s in stat_list if s.get("count", 0) > 0]
        max_vals = [s.get("max", float("nan")) for s in stat_list if s.get("count", 0) > 0]
        return {
            "total_point_pairs":   total,
            "valid_frames":        len([s for s in stat_list if s.get("count", 0) > 0]),
            "weighted_mean_m":     float(w_mean),
            "mean_of_frame_p95_m": float(np.mean(p95_vals))  if p95_vals else float("nan"),
            "max_of_frame_p95_m":  float(np.max(p95_vals))   if p95_vals else float("nan"),
            "mean_of_frame_p90_m": float(np.mean(p90_vals))  if p90_vals else float("nan"),
            "max_frame_error_m":   float(np.max(max_vals))    if max_vals else float("nan"),
            "per_frame_p95_m":     p95_vals,
        }

    h_agg: dict[str, Any] = {}
    if reproj_p95s:
        h_agg["reproj_error_px"] = {
            "frame_count":        len(reproj_p95s),
            "mean_of_frame_means": float(np.mean(reproj_means)),
            "mean_of_frame_p95":   float(np.mean(reproj_p95s)),
            "max_of_frame_p95":    float(np.max(reproj_p95s)),
            "per_frame_p95":       reproj_p95s,
        }
    if tilt_degs:
        h_agg["plane_tilt_deg"] = {
            "frame_count": len(tilt_degs),
            "mean":        float(np.mean(tilt_degs)),
            "min":         float(np.min(tilt_degs)),
            "max":         float(np.max(tilt_degs)),
            "per_frame":   tilt_degs,
        }
    if z_before_total > 0:
        h_agg["z_filter"] = {
            "total_points_before": z_before_total,
            "total_points_after":  z_after_total,
            "keep_rate":           round(z_after_total / z_before_total, 4) if z_before_total else 0.0,
        }

    return {
        "valid_frame_count": valid_scan_frames,
        "total_frame_count": len(per_frame),
        "scanline": {
            "left_boundary":  pool_stat(left_all),
            "right_boundary": pool_stat(right_all),
            "overall":        pool_stat(overall_all),
        },
        "h_matrix": h_agg,
    }


def process_frame(
    seq_cfg: dict[str, Any],
    frame: dict[str, Any],
    seq_output_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    frame_id  = frame["frame_id"]
    frame_dir = seq_output_dir / "frames" / frame_id
    summary_path = frame_dir / "pipeline_summary.json"

    if skip_existing and summary_path.exists():
        print(f"  [SKIP] {frame_id} (already done)")
        with summary_path.open() as f:
            return json.load(f)

    print(f"  [RUN]  {frame_id}")
    ensure_dir(frame_dir)
    frame_cfg = build_frame_config(seq_cfg, frame)

    try:
        corr_csv, corr_meta   = extract_correspondences(frame_cfg, frame_dir)
        h_json,   h_meta      = estimate_h(frame_cfg, corr_csv, frame_dir)
        gt_bd,    pred_bd, bd_meta = create_boundaries(frame_cfg, frame_dir)
        sl_meta               = scanline_error(frame_cfg, h_json, gt_bd, pred_bd, frame_dir)
    except Exception as exc:
        print(f"  [ERR]  {frame_id}: {exc}")
        return {"frame_id": frame_id, "error": str(exc)}

    summary = {
        "frame_id":         frame_id,
        "output_dir":       str(frame_dir),
        "correspondences":  corr_meta,
        "h_estimation":     h_meta,
        "boundary":         bd_meta,
        "scanline_error":   sl_meta,
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _fmt(v: Any, unit: str = "") -> str:
    if isinstance(v, float) and not (v == v):  # nan check
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}{unit}"
    return str(v)


def _print_summary(agg: dict[str, Any], total_frames: int) -> None:
    valid = agg.get("valid_frame_count", 0)
    print(f"\n{'='*56}")
    print(f"  SEQUENCE SUMMARY  ({valid}/{total_frames} frames with valid scanlines)")
    print(f"{'='*56}")

    sc = agg.get("scanline", {})
    ov = sc.get("overall", {})
    lf = sc.get("left_boundary", {})
    rf = sc.get("right_boundary", {})
    print("\nScanline boundary error (m):")
    print(f"  {'':12s}  {'w-mean':>8s}  {'mean-P95':>8s}  {'max-P95':>8s}  {'pairs':>6s}")
    for label, d in (("overall", ov), ("left", lf), ("right", rf)):
        print(f"  {label:12s}  {_fmt(d.get('weighted_mean_m', float('nan'))):>8s}  "
              f"{_fmt(d.get('mean_of_frame_p95_m', float('nan'))):>8s}  "
              f"{_fmt(d.get('max_of_frame_p95_m', float('nan'))):>8s}  "
              f"{d.get('total_point_pairs', 0):>6d}")

    h = agg.get("h_matrix", {})
    reproj = h.get("reproj_error_px", {})
    if reproj:
        print("\nH reprojection error (px):")
        print(f"  mean-of-means : {_fmt(reproj.get('mean_of_frame_means', float('nan')))}")
        print(f"  mean-of-P95   : {_fmt(reproj.get('mean_of_frame_p95', float('nan')))}")
        print(f"  max-P95       : {_fmt(reproj.get('max_of_frame_p95', float('nan')))}")
        if reproj.get("per_frame_p95"):
            vals = "  ".join(f"{v:.2f}" for v in reproj["per_frame_p95"])
            print(f"  per-frame P95 : [{vals}]")

    tilt = h.get("plane_tilt_deg", {})
    if tilt:
        print("\nGround plane tilt from vertical (deg):")
        print(f"  mean {_fmt(tilt.get('mean', float('nan')))}  "
              f"min {_fmt(tilt.get('min', float('nan')))}  "
              f"max {_fmt(tilt.get('max', float('nan')))}")

    zf = h.get("z_filter", {})
    if zf:
        before = zf.get("total_points_before", 0)
        after  = zf.get("total_points_after", 0)
        rate   = zf.get("keep_rate", 0.0)
        print(f"\nZ-filter totals: {after}/{before} points kept ({rate*100:.1f}%)")

    print(f"{'='*56}\n")


def main() -> None:
    args   = parse_args()
    seq_cfg = read_config(args.config)

    project_root = Path(seq_cfg["project_root"])
    output_dir   = project_root / seq_cfg["output_dir"] if not Path(seq_cfg["output_dir"]).is_absolute() \
                   else Path(seq_cfg["output_dir"])
    ensure_dir(output_dir)

    discovery_meta: dict[str, Any] | None = None
    if "dataset" in seq_cfg:
        print("[INFO] Auto-discovering frames from dataset directories...")
        frames, discovery_meta = discover_frames(seq_cfg)
        print(f"[INFO] Found {discovery_meta['matched_frames']} complete frames "
              f"({discovery_meta['skipped_frames']} skipped due to missing files)")
    else:
        frames = seq_cfg["frames"]

    if args.frames:
        wanted = set(args.frames)
        frames = [f for f in frames if f["frame_id"] in wanted]
        if not frames:
            print(f"[WARN] No frames matched: {args.frames}")
            return

    print(f"[INFO] Processing {len(frames)} frame(s) → {output_dir}")
    per_frame_results: list[dict[str, Any]] = []
    for i, frame in enumerate(frames, 1):
        print(f"[{i}/{len(frames)}] frame_id={frame['frame_id']}")
        result = process_frame(seq_cfg, frame, output_dir, args.skip_existing)
        per_frame_results.append(result)

    aggregate = aggregate_results(per_frame_results)

    sequence_summary: dict[str, Any] = {
        "config_path":  str(args.config),
        "output_dir":   str(output_dir),
        "class":        seq_cfg["class"],
        "frame_count":  len(frames),
        "aggregate":    aggregate,
        "per_frame":    per_frame_results,
    }
    if discovery_meta is not None:
        sequence_summary["discovery"] = discovery_meta

    summary_path = output_dir / "sequence_summary.json"
    with summary_path.open("w") as f:
        json.dump(sequence_summary, f, indent=2)

    _print_summary(aggregate, len(frames))
    print(f"\n[INFO] Full details saved: {summary_path}")


if __name__ == "__main__":
    main()
