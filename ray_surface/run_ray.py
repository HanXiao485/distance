"""
Surface Reconstruction + Camera Ray Intersection boundary error pipeline.

For each frame:
  1. Load LiDAR point cloud → transform to world frame → build BEV elevation map
  2. Load GT + pred masks → extract external boundaries → sample scanlines
  3. For each scanline, cast camera rays through GT-left, GT-right, pred-left, pred-right
  4. Find 3D intersection of each ray with elevation map
  5. Compute GT-to-pred distance error in 3D

Usage (local mini):
    cd .
    python ray_surface/run_ray.py \
        --config ./configs/ray_surface.yaml
"""
from __future__ import annotations

import argparse
import bisect
import datetime
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── add DISTANCE package to path ────────────────────────────────────────────
_DISTANCE_ROOT = Path(__file__).parent.parent.parent / "DISTANCE"
if str(_DISTANCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_DISTANCE_ROOT))

from distance.pipeline import (
    ensure_dir,
    stats,
    parse_pose_csv,
    load_camera_calibration,
    load_point_cloud,
    load_3d_labels,
    transform_points,
    make_transform,
    invert_transform,
    pose_record_to_transform,
    infer_timestamp_from_name,
    find_pose_by_timestamp,
    uniform_scanline_rows,
    outer_boundary_points,
)
from PIL import Image

try:
    from .modules.surface import build_elevation_map, ElevationMap
    from .modules.fast_mask import fast_clean_prediction_mask, fast_extract_external_boundary
    from .modules.visualize import draw_scanline_overlay, draw_surface_reconstruction, draw_error_histogram
except ImportError:  # Direct script execution: python ray_surface/run_ray.py
    from modules.surface import build_elevation_map, ElevationMap
    from modules.fast_mask import fast_clean_prediction_mask, fast_extract_external_boundary
    from modules.visualize import draw_scanline_overlay, draw_surface_reconstruction, draw_error_histogram


# ── Config ────────────────────────────────────────────────────────────────────

_DEFAULTS: dict[str, Any] = {
    "geometry": {
        "pose2d_direction": "sensor_to_world",
        "pose3d_direction": "sensor_to_world",
        "max_timestamp_diff_sec": 0.1,
    },
    "surface": {
        "cell_size_m": 0.25,
        "lidar_z_min": -1.5,
        "lidar_z_max": 1.0,
    },
    "ray": {
        "t_min": 0.5,
        "t_max": 60.0,
        "n_coarse": 300,
        "n_bisect": 25,
        "max_distance_m": 30.0,
    },
    "mask_processing": {
        "min_component_area_px": 5000,
        "close_iterations": 2,
    },
    "scanline": {
        "step_px": 10,
        "u_min": 0,
        "u_max": 9999,
        "max_boundary_run_width_px": 25,
        "min_scanlines_per_frame": 5,
    },
    "class": {
        "name": "dirt",
        "id_2d": 2,
    },
    "label3d_dtype": "uint32",
    "min_scanlines_per_frame": 5,
}


def _apply_defaults(cfg: dict, defaults: dict) -> None:
    for key, val in defaults.items():
        if key not in cfg:
            cfg[key] = val
        elif isinstance(val, dict) and isinstance(cfg.get(key), dict):
            _apply_defaults(cfg[key], val)


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            import yaml
            cfg = yaml.safe_load(f)
        else:
            cfg = json.load(f)
    _apply_defaults(cfg, _DEFAULTS)
    return cfg


# ── Frame discovery (reuses same pattern as run_sequence.py) ─────────────────

def _stem_to_ts(stem: str) -> float:
    return float(stem.replace("-", ".", 1))


def _count_class_pixels(gt_path: Path, class_id: int) -> int:
    return int((np.array(Image.open(gt_path)) == class_id).sum())


def discover_frames(cfg: dict[str, Any]) -> tuple[list[dict], dict]:
    ds = cfg["dataset"]
    dir_2d = Path(ds["dir_2d"])
    dir_3d = Path(ds["dir_3d"])
    class_id = int(cfg["class"]["id_2d"])
    min_px = int(ds.get("min_class_pixels", 0))
    max_diff = float(cfg["geometry"]["max_timestamp_diff_sec"])

    img_by_ts: dict[float, Path] = {}
    gt_by_ts:  dict[float, Path] = {}
    pred_by_ts: dict[float, Path] = {}

    for p in (dir_2d / "image").glob("*.png"):
        try: img_by_ts[_stem_to_ts(p.stem)] = p
        except ValueError: pass
    for p in (dir_2d / "indexLabel").glob("*.png"):
        try: gt_by_ts[_stem_to_ts(p.stem)] = p
        except ValueError: pass

    pred_dir = Path(ds.get("pred_mask_dir", str(dir_2d / "deeplabv3_pred_mask")))
    if pred_dir.exists():
        for p in pred_dir.glob("*.png"):
            try: pred_by_ts[_stem_to_ts(p.stem)] = p
            except ValueError: pass

    img_ts = sorted(img_by_ts.keys())
    clouds  = sorted((dir_3d / "Clouds").glob("*.bin"))
    labels3 = {p.stem: p for p in (dir_3d / "Labels").glob("*.label")}

    frames, skipped = [], []
    for cloud in clouds:
        fid = cloud.stem
        ts  = float(cloud.stem)
        lbl = labels3.get(fid)
        if lbl is None:
            skipped.append({"frame_id": fid, "reason": "no label3d"}); continue
        if not img_ts:
            skipped.append({"frame_id": fid, "reason": "no images"}); continue

        pos = bisect.bisect_left(img_ts, ts)
        cands = []
        if pos > 0: cands.append(img_ts[pos - 1])
        if pos < len(img_ts): cands.append(img_ts[pos])
        closest = min(cands, key=lambda t: abs(t - ts))

        if abs(closest - ts) > max_diff:
            skipped.append({"frame_id": fid, "reason": "no matching image"}); continue

        gt   = gt_by_ts.get(closest)
        pred = pred_by_ts.get(closest)
        if gt is None:
            skipped.append({"frame_id": fid, "reason": "no gt mask"}); continue
        if pred is None:
            skipped.append({"frame_id": fid, "reason": "no pred mask"}); continue

        if min_px > 0:
            px = _count_class_pixels(gt, class_id)
            if px < min_px:
                skipped.append({"frame_id": fid, "reason": f"class_pixels={px}<{min_px}"}); continue

        frames.append({
            "frame_id":   fid,
            "rgb":        str(img_by_ts[closest]),
            "pointcloud": str(cloud),
            "label3d":    str(lbl),
            "gt_index":   str(gt),
            "pred_mask":  str(pred),
        })

    return frames, {
        "total_clouds": len(clouds),
        "matched": len(frames),
        "skipped": len(skipped),
        "skipped_detail": skipped[:30],
    }


# ── Mask loading ──────────────────────────────────────────────────────────────

def load_gt_mask(gt_path: Path, class_id: int) -> np.ndarray:
    return np.array(Image.open(gt_path)) == class_id


def load_pred_mask(pred_path: Path, pred_cfg: dict) -> np.ndarray:
    ptype = pred_cfg.get("type", "color_threshold")
    if ptype == "binary":
        return np.array(Image.open(pred_path).convert("L")) > 0
    if ptype == "index":
        return np.array(Image.open(pred_path)) == int(pred_cfg.get("class_id", 2))
    # color_threshold
    rgb = np.array(Image.open(pred_path).convert("RGB"))
    r, g, b = rgb[:,:,0].astype(np.int16), rgb[:,:,1].astype(np.int16), rgb[:,:,2].astype(np.int16)
    t = pred_cfg.get("threshold", {})
    return (
        (r >= t.get("r_min", 0))   & (r <= t.get("r_max", 255)) &
        (g >= t.get("g_min", 0))   & (g <= t.get("g_max", 255)) &
        (b >= t.get("b_min", 0))   & (b <= t.get("b_max", 255)) &
        ((g - r) >= t.get("g_minus_r_min", -255)) &
        ((g - b) >= t.get("g_minus_b_min", -255))
    )


# ── Camera ray ────────────────────────────────────────────────────────────────

def pixel_to_ray_world(
    u: float, v: float,
    fx: float, fy: float, cx: float, cy: float,
    world_from_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (origin, direction) of the camera ray for pixel (u, v) in world frame."""
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float64)
    R = world_from_camera[:3, :3]
    origin = world_from_camera[:3, 3]
    direction = R @ dir_cam
    return origin, direction


# ── Per-frame processing ──────────────────────────────────────────────────────

def process_frame(
    frame: dict[str, Any],
    cfg: dict[str, Any],
    poses2d: dict,
    poses3d: dict,
    K: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    """Process one frame. Returns per-frame result dict."""
    fid = frame["frame_id"]
    t0  = time.perf_counter()

    geom_cfg   = cfg["geometry"]
    surf_cfg   = cfg["surface"]
    ray_cfg    = cfg["ray"]
    scan_cfg   = cfg["scanline"]
    mask_cfg   = cfg["mask_processing"]
    class_cfg  = cfg["class"]
    pred_cfg   = cfg.get("masks", {}).get("pred", {
        "type": "color_threshold",
        "threshold": {"r_min":0,"r_max":144,"g_min":110,"g_max":255,
                      "b_min":0,"b_max":144,"g_minus_r_min":35,"g_minus_b_min":35},
    })

    max_diff = float(geom_cfg["max_timestamp_diff_sec"])
    fx, fy, cx, cy = K

    # ── Poses ────────────────────────────────────────────────────────────────
    rgb_path   = Path(frame["rgb"])
    cloud_path = Path(frame["pointcloud"])
    label_path = Path(frame["label3d"])

    cam_ts   = infer_timestamp_from_name(rgb_path)
    cloud_ts = float(cloud_path.stem)

    cam_pose   = find_pose_by_timestamp(poses2d, cam_ts,   max_diff)
    lidar_pose = find_pose_by_timestamp(poses3d, cloud_ts, max_diff)

    world_from_cam   = pose_record_to_transform(cam_pose,   geom_cfg["pose2d_direction"])
    world_from_lidar = pose_record_to_transform(lidar_pose, geom_cfg["pose3d_direction"])

    cam_origin_world = world_from_cam[:3, 3]

    # ── Point cloud → world frame ────────────────────────────────────────────
    pts_lidar = load_point_cloud(cloud_path)

    # Use ALL points in Z band (not just dirt class) for better surface coverage
    z_lidar_min = float(surf_cfg["lidar_z_min"])
    z_lidar_max = float(surf_cfg["lidar_z_max"])
    z_mask = (pts_lidar[:, 2] >= z_lidar_min) & (pts_lidar[:, 2] <= z_lidar_max)
    pts_ground_lidar = pts_lidar[z_mask]
    pts_world = transform_points(world_from_lidar, pts_ground_lidar)

    # ── Build elevation map ──────────────────────────────────────────────────
    cell_size = float(surf_cfg["cell_size_m"])
    elev = build_elevation_map(
        pts_world,
        cell_size_m=cell_size,
        z_min_world=-np.inf,   # already filtered in lidar frame
        z_max_world=np.inf,
    )
    surf_stats = elev.stats()

    # ── Masks and boundaries (fast scipy path) ───────────────────────────────
    class_id = int(class_cfg["id_2d"])
    gt_mask   = load_gt_mask(Path(frame["gt_index"]), class_id)
    pred_mask = load_pred_mask(Path(frame["pred_mask"]), pred_cfg)

    pred_cleaned = fast_clean_prediction_mask(
        pred_mask,
        int(mask_cfg["min_component_area_px"]),
        int(mask_cfg["close_iterations"]),
    )
    gt_boundary   = fast_extract_external_boundary(gt_mask)
    pred_boundary = fast_extract_external_boundary(pred_cleaned)

    u_min = int(scan_cfg["u_min"])
    u_max = int(scan_cfg["u_max"])
    step  = int(scan_cfg["step_px"])
    max_run = int(scan_cfg["max_boundary_run_width_px"])
    min_lines = int(cfg.get("min_scanlines_per_frame", 5))

    rows, row_meta = uniform_scanline_rows(
        pred_boundary, step, 0, max_run, u_min, u_max
    )

    if len(rows) < min_lines:
        return {
            "frame_id": fid,
            "skipped": True,
            "reason": f"only {len(rows)} valid scanlines (min={min_lines})",
            "elapsed_s": round(time.perf_counter() - t0, 2),
        }

    # ── Collect valid scanlines and build ray arrays (one numpy batch) ───────
    t_min     = float(ray_cfg["t_min"])
    t_max_cfg = float(ray_cfg["t_max"])
    n_coarse  = int(ray_cfg["n_coarse"])
    n_bisect  = int(ray_cfg["n_bisect"])
    max_dist  = float(ray_cfg["max_distance_m"])

    left_errors:  list[float] = []
    right_errors: list[float] = []
    all_errors:   list[float] = []
    samples:      list[dict]  = []
    skipped_rows: dict[str, int] = {
        "no_gt_boundary":   0,
        "no_pred_boundary": 0,
        "gt_ray_miss":      0,
        "pred_ray_miss":    0,
        "beyond_max_dist":  0,
    }
    gt_pts_world:   list[np.ndarray] = []
    pred_pts_world: list[np.ndarray] = []

    R_world_cam = world_from_cam[:3, :3]
    cam_origin  = world_from_cam[:3, 3]

    valid_rows: list[dict] = []
    all_origins:    list[np.ndarray] = []
    all_directions: list[np.ndarray] = []

    for y in rows:
        gt_pts   = outer_boundary_points(gt_boundary,   y, 0, u_min, u_max)
        pred_pts = outer_boundary_points(pred_boundary, y, 0, u_min, u_max)

        if gt_pts is None:
            skipped_rows["no_gt_boundary"] += 1; continue
        if pred_pts is None:
            skipped_rows["no_pred_boundary"] += 1; continue

        gt_left_x, gt_right_x     = gt_pts
        pred_left_x, pred_right_x = pred_pts

        ray_offset = len(all_origins)
        for u in (gt_left_x, gt_right_x, pred_left_x, pred_right_x):
            dir_cam = np.array([(u - cx) / fx, (float(y) - cy) / fy, 1.0])
            all_origins.append(cam_origin)
            all_directions.append(R_world_cam @ dir_cam)

        valid_rows.append({
            "y": y,
            "gt_left_x": gt_left_x, "gt_right_x": gt_right_x,
            "pred_left_x": pred_left_x, "pred_right_x": pred_right_x,
            "ray_offset": ray_offset,  # gt_l=+0, gt_r=+1, pred_l=+2, pred_r=+3
        })

    # ── Single vectorised batch intersection ─────────────────────────────────
    hits: list = []
    if all_origins:
        hits = elev.intersect_rays_batch(
            np.stack(all_origins), np.stack(all_directions),
            t_min, t_max_cfg, n_coarse, n_bisect,
        )

    # ── Decode results per scanline ───────────────────────────────────────────
    for row in valid_rows:
        off = row["ray_offset"]
        gt_l   = hits[off + 0]
        gt_r   = hits[off + 1]
        pred_l = hits[off + 2]
        pred_r = hits[off + 3]

        y          = row["y"]
        gt_left_x  = row["gt_left_x"];  gt_right_x  = row["gt_right_x"]
        pred_left_x= row["pred_left_x"]; pred_right_x= row["pred_right_x"]

        left_hit  = (gt_l is not None) and (pred_l is not None)
        right_hit = (gt_r is not None) and (pred_r is not None)

        if not left_hit and not right_hit:
            skipped_rows["gt_ray_miss"]   += int(gt_l is None or gt_r is None)
            skipped_rows["pred_ray_miss"] += int(pred_l is None or pred_r is None)
            continue

        left_err:  Optional[float] = None
        right_err: Optional[float] = None

        if left_hit:
            gt_pt, gt_t = gt_l
            pr_pt, pr_t = pred_l
            if gt_t > max_dist or pr_t > max_dist:
                skipped_rows["beyond_max_dist"] += 1
                left_hit = False
            else:
                left_err = float(np.linalg.norm(gt_pt - pr_pt))
                left_errors.append(left_err)
                all_errors.append(left_err)
                gt_pts_world.append(gt_pt)
                pred_pts_world.append(pr_pt)

        if right_hit:
            gt_pt, gt_t = gt_r
            pr_pt, pr_t = pred_r
            if gt_t > max_dist or pr_t > max_dist:
                skipped_rows["beyond_max_dist"] += 1
                right_hit = False
            else:
                right_err = float(np.linalg.norm(gt_pt - pr_pt))
                right_errors.append(right_err)
                all_errors.append(right_err)
                gt_pts_world.append(gt_pt)
                pred_pts_world.append(pr_pt)

        if left_err is None and right_err is None:
            continue

        samples.append({
            "y":           y,
            "gt_left_x":   gt_left_x,
            "gt_right_x":  gt_right_x,
            "pred_left_x": pred_left_x,
            "pred_right_x":pred_right_x,
            "left_hit":    left_hit,
            "right_hit":   right_hit,
            "left_err_m":  round(left_err, 4)  if left_err  is not None else None,
            "right_err_m": round(right_err, 4) if right_err is not None else None,
        })

    # ── Visualizations (non-fatal) ────────────────────────────────────────────
    frame_out = output_dir / fid
    ensure_dir(frame_out)

    cam_forward = world_from_cam[:3, :3] @ np.array([0.0, 0.0, 1.0])

    try:
        if samples:
            draw_scanline_overlay(
                rgb_path=rgb_path,
                samples=samples,
                output_path=frame_out / "scanline_overlay.png",
                vmax_m=2.0,
            )
        if gt_pts_world and pred_pts_world:
            draw_surface_reconstruction(
                elev=elev,
                coverage_mask=elev.coverage_mask,
                gt_intersections=gt_pts_world,
                pred_intersections=pred_pts_world,
                errors_m=all_errors,
                camera_origin=cam_origin_world,
                cam_forward=cam_forward,
                output_path=frame_out / "surface_reconstruction.png",
                view_radius_m=25.0,
            )
            draw_error_histogram(
                errors_m=all_errors,
                output_path=frame_out / "error_histogram.png",
                title=f"Frame {fid[:10]}",
            )
    except Exception as viz_err:
        pass  # visualization errors must not affect stats

    # ── Per-frame stats ──────────────────────────────────────────────────────
    frame_stats: dict[str, Any] = {
        "frame_id":     fid,
        "skipped":      False,
        "scanlines": {
            "candidates":  len(rows),
            "used":        len(samples),
            "skipped":     skipped_rows,
        },
        "ray_pairs": {
            "left_hits":  len(left_errors),
            "right_hits": len(right_errors),
            "total_hits": len(all_errors),
        },
        "surface": surf_stats,
        "error_m": {
            "left":  stats(left_errors)  if left_errors  else {"count": 0},
            "right": stats(right_errors) if right_errors else {"count": 0},
            "all":   stats(all_errors)   if all_errors   else {"count": 0},
        },
        "elapsed_s": round(time.perf_counter() - t0, 2),
    }

    # Save per-frame JSON
    with open(frame_out / "frame_result.json", "w") as f:
        json.dump(frame_stats, f, indent=2)

    return frame_stats


# ── Sequence runner ───────────────────────────────────────────────────────────

def pool_stat(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    arr = np.asarray(values)
    return {
        "count":  int(arr.size),
        "mean":   round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "p75":    round(float(np.percentile(arr, 75)), 4),
        "p90":    round(float(np.percentile(arr, 90)), 4),
        "p95":    round(float(np.percentile(arr, 95)), 4),
        "p99":    round(float(np.percentile(arr, 99)), 4),
        "max":    round(float(arr.max()), 4),
    }


def run_sequence(cfg_path: Path) -> None:
    cfg = read_config(cfg_path)
    output_dir = Path(cfg["output_dir"])
    ensure_dir(output_dir)

    ds = cfg["dataset"]
    dir_2d = Path(ds["dir_2d"])
    dir_3d = Path(ds["dir_3d"])

    # Shared data
    poses2d = parse_pose_csv(Path(dir_2d) / "poses2d.csv")
    poses3d = parse_pose_csv(Path(dir_3d) / "poses3d.csv")
    K_arr, _, _ = load_camera_calibration(Path(dir_2d) / "camera_calibration.yaml")
    # K_arr = [fx, fy, cx, cy]

    print("Discovering frames...")
    frames, disc_meta = discover_frames(cfg)
    print(f"  Found {len(frames)} frames  (skipped {disc_meta['skipped']})")

    all_errors:   list[float] = []
    left_errors:  list[float] = []
    right_errors: list[float] = []
    frame_means:  list[float] = []
    results:      list[dict]  = []
    processed = skipped = 0

    for i, frame in enumerate(frames):
        fid = frame["frame_id"]
        print(f"  [{i+1:3d}/{len(frames)}] {fid}", end="", flush=True)
        try:
            res = process_frame(frame, cfg, poses2d, poses3d, K_arr, output_dir)
        except Exception as e:
            res = {"frame_id": fid, "skipped": True, "reason": str(e), "elapsed_s": 0}
            print(f"  ERROR: {e}")
        else:
            if res.get("skipped"):
                print(f"  SKIPPED: {res.get('reason')}")
            else:
                err_all = res["error_m"]["all"]
                if err_all.get("count", 0) > 0:
                    fm = err_all["mean"]
                    frame_means.append(fm)
                    all_errors += [fm]  # use frame means for aggregate
                    print(f"  mean={err_all['mean']:.3f}m  median={err_all['median']:.3f}m"
                          f"  hits={err_all['count']}  t={res['elapsed_s']}s")
                else:
                    print("  no valid hits")

        results.append(res)
        if res.get("skipped"):
            skipped += 1
        else:
            processed += 1

    # ── Pool all per-sample errors from saved frame JSONs for aggregate stats
    all_sample_errors: list[float] = []
    all_left:  list[float] = []
    all_right: list[float] = []
    for res in results:
        if res.get("skipped"):
            continue
        em = res.get("error_m", {})
        cnt = em.get("all", {}).get("count", 0)
        if cnt == 0:
            continue
        # Re-read per-sample errors would require storing them; use frame stats instead
        # For median-of-medians and per-side, reconstruct from saved stats
        all_left.append(em.get("left", {}).get("mean", 0) or 0)
        all_right.append(em.get("right", {}).get("mean", 0) or 0)

    # Aggregate: mean-of-frame-means, median-of-frame-medians
    frame_medians = [
        res["error_m"]["all"]["median"]
        for res in results
        if not res.get("skipped") and res.get("error_m", {}).get("all", {}).get("count", 0) > 0
    ]
    frame_means_list = [
        res["error_m"]["all"]["mean"]
        for res in results
        if not res.get("skipped") and res.get("error_m", {}).get("all", {}).get("count", 0) > 0
    ]

    aggregate = {
        "mean_of_frame_means_m":     round(float(np.mean(frame_means_list)), 4) if frame_means_list else None,
        "median_of_frame_medians_m": round(float(np.median(frame_medians)), 4)   if frame_medians  else None,
        "frames_with_data":          len(frame_means_list),
        "frames_total":              len(frames),
        "frames_skipped":            skipped,
    }

    if frame_means_list:
        arr = np.asarray(frame_means_list)
        aggregate["p75"]  = round(float(np.percentile(arr, 75)), 4)
        aggregate["p90"]  = round(float(np.percentile(arr, 90)), 4)
        aggregate["p95"]  = round(float(np.percentile(arr, 95)), 4)
        aggregate["max"]  = round(float(arr.max()), 4)

    summary = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method":       "ray_surface",
        "config":       str(cfg_path),
        "discovery":    disc_meta,
        "overall":      aggregate,
        "frames":       results,
    }

    summary_path = output_dir / "sequence_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Aggregate histogram
    draw_error_histogram(
        errors_m=frame_means_list,
        output_path=output_dir / "aggregate_error_histogram.png",
        title="Frame-mean error distribution (all frames)",
    )

    print("\n── Summary ──────────────────────────────────────────────────────────")
    print(f"  mean_of_frame_means_m    = {aggregate.get('mean_of_frame_means_m')}")
    print(f"  median_of_frame_medians_m= {aggregate.get('median_of_frame_medians_m')}")
    print(f"  P90                      = {aggregate.get('p90')}")
    print(f"  P95                      = {aggregate.get('p95')}")
    print(f"  frames with data         = {aggregate['frames_with_data']} / {aggregate['frames_total']}")
    print(f"  output → {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ray-surface boundary error pipeline.")
    p.add_argument("--config", required=True, type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_sequence(args.config)
