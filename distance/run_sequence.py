"""Batch boundary-error pipeline over an ordered sequence of frames.

Two usage modes
---------------
1. Auto-discovery (WildScenes-style flat dataset):
   Set "dataset": {"dir_2d": "...", "dir_3d": "...", "min_class_pixels": 600000}
   in the config. Files are discovered automatically, anchored on 3D point clouds.
   For each anchor cloud P_i, nearby images (within ±h_search_window_sec) are
   evaluated to find the best H matrix (lowest reprojection error).

2. Explicit frame list:
   Set "frames": [...] and "shared_paths": {...} in the config.
   Each frame entry must have: frame_id, rgb, pointcloud, label3d, gt_index, pred_mask.

Run
---
    python -m distance.run_sequence --config configs/evaluation.yaml --list-tasks
    python -m distance.run_sequence --config configs/evaluation.yaml --task wildscenes_mini_cached
    python -m distance.run_sequence --config configs/evaluation.yaml --task wildscenes_mini_cached --frames 1623379838.017682788
"""
from __future__ import annotations

import argparse
import bisect
import copy
import datetime
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from distance.pipeline import (
    ensure_dir,
    resolve_path,
    stats,
    extract_correspondences,
    estimate_h,
    create_boundaries,
    scanline_error,
)


# ---------------------------------------------------------------------------
# Traversable precision / recall
# ---------------------------------------------------------------------------

def _compute_traversable_pr(frame_dir: Path, class_name: str) -> dict:
    """Per-frame traversable precision / recall (binary mask pixel comparison).

    Precision = |Pred_road ∩ GT_road| / |Pred_road|
                Only penalises false positives; does NOT penalise missing coverage.
    Recall    = |Pred_road ∩ GT_road| / |GT_road|
                Only penalises false negatives.

    Both are computed per image and returned as floats in [0, 1].
    Returns empty dict when masks are absent or GT has no positive pixels.
    """
    from PIL import Image as _PILImage

    masks_dir = frame_dir / "masks"
    gt_path   = masks_dir / f"gt_{class_name}_binary.png"
    pred_path = masks_dir / f"pred_{class_name}_cleaned_binary.png"
    if not pred_path.exists():
        pred_path = masks_dir / f"pred_{class_name}_binary.png"

    if not gt_path.exists() or not pred_path.exists():
        return {}

    gt   = np.asarray(_PILImage.open(gt_path).convert("L"))   > 127
    pred = np.asarray(_PILImage.open(pred_path).convert("L")) > 127

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    gt_total   = tp + fn
    pred_total = tp + fp

    if gt_total == 0:
        return {}   # no GT road pixels — skip frame

    return {
        "precision": round(tp / (pred_total + 1e-9), 6),
        "recall":    round(tp / (gt_total   + 1e-9), 6),
        "tp":        tp,
        "fp":        fp,
        "fn":        fn,
        "gt_pixels": gt_total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the DISTANCE boundary-error pipeline over a frame sequence."
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Config YAML/JSON path (see configs/ for examples).")
    parser.add_argument("--task", default=None,
                        help="Task name when the config contains a top-level tasks mapping.")
    parser.add_argument("--list-tasks", action="store_true",
                        help="List tasks defined by --config and exit.")
    parser.add_argument("--frames", nargs="*", default=None,
                        help="Optional subset of frame_ids to process (default: all).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames whose pipeline_summary.json already exists.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel worker processes (default: all CPU cores).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS: dict[str, Any] = {
    "label3d_dtype": "uint32",
    "geometry": {
        "pose2d_direction":       "sensor_to_world",
        "pose3d_direction":       "sensor_to_world",
        "max_timestamp_diff_sec": 0.1,
        "use_distortion":         False,
    },
    "masks": {
        "gt": {"type": "index"},
        "pred": {
            "type": "color_threshold",
            "threshold": {
                "r_min": 0,   "r_max": 144,
                "g_min": 110, "g_max": 255,
                "b_min": 0,   "b_max": 144,
                "g_minus_r_min": 35,
                "g_minus_b_min": 35,
            },
        },
    },
    "mask_processing": {
        "min_component_area_px": 5000,
        "close_iterations":      2,
    },
    "h_estimation": {
        "plane_inlier_threshold_m": 0.5,
        "robust_method":            "none",
        "z_filter": {
            "enabled":     True,
            "lidar_z_min": -0.75,
            "lidar_z_max":  0.5,
            "u_min":        300,
            "u_max":        1700,
        },
    },
    "h_diagnostics": {
        "enabled":             True,
        "image_edge_margin_px": 100,
        "near_distance_m":      5.0,
        "far_distance_m":       20.0,
    },
    "h_search_window_sec":            15,
    "h_candidate_reproj_tolerance_px": 1.0,
    "min_scanlines_per_frame":         5,
    "scanline": {
        "step_px":                   10,
        "edge_margin_px":             0,
        "u_min":                      0,
        "u_max":                   9999,
        "max_boundary_run_width_px": 25,
        "top_margin_ratio":          0.10,
        "bottom_margin_ratio":       0.10,
        "line_width":                 4,
        "point_radius":               7,
    },
}


def _apply_defaults(cfg: dict, defaults: dict) -> None:
    """Recursively fill missing keys from defaults (does not overwrite existing values)."""
    for key, val in defaults.items():
        if key not in cfg:
            cfg[key] = val
        elif isinstance(val, dict) and isinstance(cfg.get(key), dict):
            _apply_defaults(cfg[key], val)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two mappings without mutating either input."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand_config_variables(value: Any, variables: dict[str, str]) -> Any:
    """Expand ${project_root}, ${datasets_root}, and ${outputs_root}."""
    if isinstance(value, dict):
        return {key: _expand_config_variables(val, variables) for key, val in value.items()}
    if isinstance(value, list):
        return [_expand_config_variables(item, variables) for item in value]
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("${" + key + "}", replacement)
    return value


def _load_config_document(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        if path.suffix in (".yaml", ".yml"):
            import yaml
            document = yaml.safe_load(f)
        else:
            document = json.load(f)
    if not isinstance(document, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return document


def list_config_tasks(path: Path) -> list[str]:
    """Return task names from a task-based config document."""
    return sorted(_load_config_document(path).get("tasks", {}))


def read_config(path: Path, task: str | None = None) -> dict[str, Any]:
    """Load a legacy single-task config or resolve one task from a task catalog."""
    document = _load_config_document(path)
    cfg = copy.deepcopy(document)

    if "tasks" in document:
        tasks = document.get("tasks", {})
        if not isinstance(tasks, dict) or not tasks:
            raise ValueError(f"Config has no usable tasks: {path}")
        if task is None:
            available = ", ".join(sorted(tasks))
            raise ValueError(f"--task is required for {path}; available tasks: {available}")
        if task not in tasks:
            available = ", ".join(sorted(tasks))
            raise ValueError(f"Unknown task {task!r}; available tasks: {available}")
        task_cfg = tasks[task]
        if not isinstance(task_cfg, dict):
            raise ValueError(f"Task {task!r} must contain a mapping")

        cfg = copy.deepcopy(document.get("defaults", {}))
        for root_key in ("project_root", "datasets_root", "outputs_root"):
            if root_key in document:
                cfg[root_key] = document[root_key]

        dataset_name = task_cfg.get("dataset")
        if dataset_name:
            datasets = document.get("datasets", {})
            if dataset_name not in datasets:
                raise ValueError(f"Task {task!r} references unknown dataset {dataset_name!r}")
            cfg["dataset"] = _deep_merge(
                datasets[dataset_name], task_cfg.get("dataset_overrides", {}))

        model_name = task_cfg.get("model")
        if model_name:
            models = document.get("models", {})
            if model_name not in models:
                raise ValueError(f"Task {task!r} references unknown model {model_name!r}")
            cfg["segmentation"] = copy.deepcopy(models[model_name])

        mask_name = task_cfg.get("mask_profile")
        if mask_name:
            profiles = document.get("mask_profiles", {})
            if mask_name not in profiles:
                raise ValueError(f"Task {task!r} references unknown mask profile {mask_name!r}")
            cfg["masks"] = copy.deepcopy(profiles[mask_name])

        selectors = {"dataset", "dataset_overrides", "model", "mask_profile"}
        overrides = {key: value for key, value in task_cfg.items() if key not in selectors}
        cfg = _deep_merge(cfg, overrides)
        variables = {
            key: str(document[key])
            for key in ("project_root", "datasets_root", "outputs_root")
            if key in document
        }
        cfg = _expand_config_variables(cfg, variables)
        cfg["_task"] = task

    _apply_defaults(cfg, _CONFIG_DEFAULTS)
    cfg["_config_path"] = str(path)
    return cfg

def _resolve_shared_paths(seq_cfg: dict[str, Any]) -> dict[str, str]:
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


def build_frame_config_for_h_candidate(
    seq_cfg: dict[str, Any],
    anchor_frame: dict[str, Any],
    candidate_image: dict[str, Any],
) -> dict[str, Any]:
    """Build config for H estimation using anchor P_i's cloud + candidate I_j's image."""
    shared = _resolve_shared_paths(seq_cfg)
    return {
        "project_root":  seq_cfg["project_root"],
        "class":         seq_cfg["class"],
        "label3d_dtype": seq_cfg.get("label3d_dtype", "uint32"),
        "geometry":      seq_cfg.get("geometry", {}),
        "h_estimation":  seq_cfg.get("h_estimation", {}),
        "h_diagnostics": seq_cfg.get("h_diagnostics", {}),
        "paths": {
            "rgb":         candidate_image["rgb"],      # I_j provides camera pose
            "pointcloud":  anchor_frame["pointcloud"],  # P_i provides 3D data
            "label3d":     anchor_frame["label3d"],
            "poses2d":     shared["poses2d"],
            "poses3d":     shared["poses3d"],
            "calibration": shared["calibration"],
        },
    }


# ---------------------------------------------------------------------------
# Auto-discovery (point-cloud anchored)
# ---------------------------------------------------------------------------

def _stem_to_ts(stem: str) -> float:
    """Convert file stem to float timestamp (handles both dash and dot formats)."""
    return float(stem.replace("-", ".", 1))


def _count_class_pixels(gt_path: Path, class_id: int) -> int:
    import numpy as np
    from PIL import Image
    return int((np.array(Image.open(gt_path)) == class_id).sum())


def discover_frames(seq_cfg: dict[str, Any], require_pred_mask: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Anchor on 3D point clouds; match each to closest 2D image; filter by class pixels.

    Frame IDs use the point-cloud stem format: '1623378674.755743672'.
    If require_pred_mask=False, frames without an existing pred mask are still included
    (pred_mask will be filled in later by the inference phase).
    """
    ds = seq_cfg["dataset"]
    dir_2d = Path(ds["dir_2d"])
    dir_3d = Path(ds["dir_3d"])
    class_id_2d = int(seq_cfg["class"]["id_2d"])
    min_pixels  = int(ds.get("min_class_pixels", 0))
    max_ts_diff = float(seq_cfg.get("geometry", {}).get("max_timestamp_diff_sec", 0.1))

    # Build timestamp → path maps for all 2D file types
    img_by_ts:  dict[float, Path] = {}
    gt_by_ts:   dict[float, Path] = {}
    pred_by_ts: dict[float, Path] = {}

    for _img_ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in (dir_2d / "image").glob(_img_ext):
            try:
                img_by_ts[_stem_to_ts(p.stem)] = p
            except ValueError:
                pass
    for p in (dir_2d / "indexLabel").glob("*.png"):
        try:
            gt_by_ts[_stem_to_ts(p.stem)] = p
        except ValueError:
            pass
    pred_mask_dir = Path(ds["pred_mask_dir"]) if "pred_mask_dir" in ds else dir_2d / "deeplabv3_pred_mask"
    if pred_mask_dir.exists():
        for p in pred_mask_dir.glob("*.png"):
            try:
                pred_by_ts[_stem_to_ts(p.stem)] = p
            except ValueError:
                pass

    img_timestamps = sorted(img_by_ts.keys())

    clouds  = sorted(
        list((dir_3d / "Clouds").glob("*.bin")) +
        list((dir_3d / "Clouds").glob("*.pcd"))
    )
    labels3 = {p.stem: p for p in (dir_3d / "Labels").glob("*.label")}

    frames:  list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for cloud_path in tqdm(clouds, desc="Scanning dataset", unit="frame", leave=False):
        cloud_ts  = float(cloud_path.stem)
        label3_path = labels3.get(cloud_path.stem)
        frame_id  = cloud_path.stem

        if label3_path is None:
            skipped.append({"frame_id": frame_id, "missing": ["label3d"]})
            continue

        # Nearest image by timestamp (binary search)
        if not img_timestamps:
            skipped.append({"frame_id": frame_id, "missing": ["no_2d_images"]})
            continue

        pos = bisect.bisect_left(img_timestamps, cloud_ts)
        near_ts_candidates = []
        if pos > 0:
            near_ts_candidates.append(img_timestamps[pos - 1])
        if pos < len(img_timestamps):
            near_ts_candidates.append(img_timestamps[pos])
        closest_ts = min(near_ts_candidates, key=lambda t: abs(t - cloud_ts))

        if abs(closest_ts - cloud_ts) > max_ts_diff:
            skipped.append({"frame_id": frame_id,
                             "missing": [f"matched_image(diff={abs(closest_ts-cloud_ts):.4f}>{max_ts_diff})"]})
            continue

        img_path  = img_by_ts[closest_ts]
        gt_path   = gt_by_ts.get(closest_ts)
        pred_path = pred_by_ts.get(closest_ts)

        missing = []
        if gt_path is None: missing.append("indexLabel")
        if pred_path is None and require_pred_mask: missing.append("deeplabv3_pred_mask")
        if missing:
            skipped.append({"frame_id": frame_id, "missing": missing})
            continue

        if min_pixels > 0:
            px = _count_class_pixels(gt_path, class_id_2d)
            if px < min_pixels:
                skipped.append({"frame_id": frame_id,
                                 "missing": [f"class_pixels_below_threshold({px}<{min_pixels})"]})
                continue

        frames.append({
            "frame_id":   frame_id,
            "rgb":        str(img_path),
            "pointcloud": str(cloud_path),
            "label3d":    str(label3_path),
            "gt_index":   str(gt_path),
            "pred_mask":  str(pred_path) if pred_path else "",
        })

    meta = {
        "source":           "auto_discovery_pc_anchored",
        "dir_2d":           str(dir_2d),
        "dir_3d":           str(dir_3d),
        "min_class_pixels": min_pixels,
        "total_clouds":     len(clouds),
        "matched_frames":   len(frames),
        "skipped_frames":   len(skipped),
        "skipped_detail":   skipped[:50],
    }
    return frames, meta


def discover_all_images(seq_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan ALL images in dir_2d/image/ (no class-pixel filter) as H candidate pool."""
    ds = seq_cfg["dataset"]
    dir_2d = Path(ds["dir_2d"])
    result: list[dict[str, Any]] = []
    for _img_ext in ("*.png", "*.jpg", "*.jpeg"):
        for img_path in sorted((dir_2d / "image").glob(_img_ext)):
            try:
                ts = _stem_to_ts(img_path.stem)
            except ValueError:
                continue
            result.append({"image_id": img_path.stem, "rgb": str(img_path), "timestamp": ts})
    return result


def find_candidate_images(
    anchor_ts: float,
    all_images: list[dict[str, Any]],
    window_sec: float,
) -> list[dict[str, Any]]:
    """Return all images within ±window_sec of anchor_ts."""
    return [img for img in all_images if abs(img["timestamp"] - anchor_ts) <= window_sec]


# ---------------------------------------------------------------------------
# Per-frame processing (two-phase)
# ---------------------------------------------------------------------------

def process_frame_phase1(
    seq_cfg: dict[str, Any],
    frame: dict[str, Any],
    candidate_images: list[dict[str, Any]],
    seq_output_dir: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Phase 1: try all candidate images with this anchor's point cloud; keep best H.

    Selection rule: among candidates whose reprojection error is within
    h_candidate_reproj_tolerance_px of the global minimum, prefer the one
    temporally closest to the anchor.  This avoids using a geometrically
    distant viewpoint when a nearby image is nearly as accurate.
    """
    frame_id   = frame["frame_id"]
    frame_dir  = seq_output_dir / "frames" / frame_id
    class_name = seq_cfg["class"]["name"]
    best_h_path    = frame_dir / "h_estimation" / f"{class_name}_h_estimation.json"
    best_cand_path = frame_dir / "h_estimation" / "best_candidate.txt"

    if skip_existing and best_h_path.exists():
        with best_h_path.open() as f:
            h_meta = json.load(f)
        cand_id = best_cand_path.read_text().strip() if best_cand_path.exists() else "unknown"
        return {"frame_id": frame_id, "h_estimation": h_meta, "h_json": str(best_h_path),
                "best_candidate": cand_id}

    ensure_dir(frame_dir)

    anchor_ts   = float(frame["frame_id"])
    tolerance   = float(seq_cfg.get("h_candidate_reproj_tolerance_px", 1.0))

    # Collect all valid (reproj_error, ts_diff, cand_id, h_meta) tuples
    valid: list[tuple[float, float, str, dict[str, Any]]] = []

    for cand in candidate_images:
        cand_id   = cand["image_id"]
        cand_dir  = frame_dir / "h_candidates" / cand_id
        cand_h_path = cand_dir / "h_estimation" / f"{class_name}_h_estimation.json"

        if skip_existing and cand_h_path.exists():
            try:
                with cand_h_path.open() as f:
                    h_meta = json.load(f)
            except Exception:
                continue
        else:
            ensure_dir(cand_dir)
            cand_cfg = build_frame_config_for_h_candidate(seq_cfg, frame, cand)
            try:
                corr_csv, _, corr_arrays = extract_correspondences(cand_cfg, cand_dir)
                _, h_meta = estimate_h(cand_cfg, corr_csv, cand_dir, precomputed=corr_arrays)
            except Exception as exc:
                tqdm.write(f"  [WARN] {frame_id} cand={cand_id}: {exc}")
                continue

        reproj  = h_meta.get("image_reprojection_error_px", {}).get("mean", float("inf"))
        ts_diff = abs(cand["timestamp"] - anchor_ts)
        valid.append((reproj, ts_diff, cand_id, h_meta))

    if not valid:
        return {"frame_id": frame_id, "error": "no valid H found among candidates"}

    best_reproj = min(v[0] for v in valid)

    # Within tolerance window, pick temporally closest; break ties by reproj error
    near_best = [v for v in valid if v[0] <= best_reproj + tolerance]
    near_best.sort(key=lambda v: (v[1], v[0]))
    _, _, best_candidate_id, best_h_meta = near_best[0]

    ensure_dir(best_h_path.parent)
    with best_h_path.open("w") as f:
        json.dump(best_h_meta, f, indent=2)
    best_cand_path.write_text(best_candidate_id)

    return {"frame_id": frame_id, "h_estimation": best_h_meta, "h_json": str(best_h_path),
            "best_candidate": best_candidate_id}


def process_frame_phase2(
    seq_cfg: dict[str, Any],
    frame: dict[str, Any],
    seq_output_dir: Path,
    best_h_json: Path,
    skip_existing: bool,
) -> dict[str, Any]:
    """Phase 2: create boundaries + scanline error using the provided H."""
    frame_id     = frame["frame_id"]
    frame_dir    = seq_output_dir / "frames" / frame_id
    summary_path = frame_dir / "pipeline_summary.json"

    if skip_existing and summary_path.exists():
        with summary_path.open() as f:
            return json.load(f)

    ensure_dir(frame_dir)
    frame_cfg = build_frame_config(seq_cfg, frame)
    try:
        gt_bd, pred_bd, bd_meta, boundary_arrays = create_boundaries(frame_cfg, frame_dir)
        sl_meta = scanline_error(frame_cfg, best_h_json, gt_bd, pred_bd, frame_dir,
                                  precomputed_boundaries=boundary_arrays)
    except Exception as exc:
        tqdm.write(f"  [ERR-SL] {frame_id}: {exc}")
        return {"frame_id": frame_id, "error": str(exc)}

    h_json_path = frame_dir / "h_estimation" / f"{seq_cfg['class']['name']}_h_estimation.json"
    h_meta: dict[str, Any] = {}
    if h_json_path.exists():
        with h_json_path.open() as f:
            h_meta = json.load(f)

    pr_meta = _compute_traversable_pr(frame_dir, seq_cfg["class"]["name"])

    summary = {
        "frame_id":       frame_id,
        "output_dir":     str(frame_dir),
        "h_estimation":   h_meta,
        "boundary":       bd_meta,
        "scanline_error": sl_meta,
        "traversable_pr": pr_meta,
        "h_source":       str(best_h_json),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _phase1_worker(packed: tuple) -> dict[str, Any]:
    seq_cfg, frame, candidate_images, seq_output_dir, skip_existing = packed
    return process_frame_phase1(seq_cfg, frame, candidate_images, Path(seq_output_dir), skip_existing)


def _phase2_worker(packed: tuple) -> dict[str, Any]:
    seq_cfg, frame, seq_output_dir, best_h_json, skip_existing = packed
    return process_frame_phase2(seq_cfg, frame, Path(seq_output_dir), Path(best_h_json), skip_existing)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_results(per_frame: list[dict[str, Any]], seq_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    left_all, right_all, overall_all = [], [], []
    reproj_means, reproj_p95s = [], []
    tilt_degs: list[float] = []
    z_before_total, z_after_total = 0, 0
    valid_scan_frames = 0
    pr_precisions: list[float] = []
    pr_recalls:    list[float] = []

    min_scanlines = int(seq_cfg.get("min_scanlines_per_frame", 5)) if seq_cfg else 5

    for f in per_frame:
        sl = f.get("scanline_error", {})
        if sl.get("sampled_scanline_count", 0) >= min_scanlines:
            valid_scan_frames += 1
            left_all.append(sl.get("left_error_m", {}))
            right_all.append(sl.get("right_error_m", {}))
            overall_all.append(sl.get("overall_error_m", {}))

        h = f.get("h_estimation", {})
        reproj = h.get("image_reprojection_error_px", {})
        if reproj.get("count", 0) > 0:
            reproj_means.append(float(reproj["mean"]))
            reproj_p95s.append(float(reproj["p95"]))

        diag = h.get("diagnostics", {})
        tilt = diag.get("plane_normal_tilt_from_vertical_deg")
        if tilt is not None:
            tilt_degs.append(float(tilt))

        zf = h.get("z_filter", {})
        if zf.get("enabled", False):
            z_before_total += int(zf.get("points_before", 0))
            z_after_total  += int(zf.get("points_after",  0))

        pr = f.get("traversable_pr", {})
        if pr.get("gt_pixels", 0) > 0:
            pr_precisions.append(float(pr["precision"]))
            pr_recalls.append(float(pr["recall"]))

    def pool_stat(stat_list: list[dict]) -> dict[str, Any]:
        valid_stats = [s for s in stat_list if s.get("count", 0) > 0]
        if not valid_stats:
            return {"valid_frames": 0}
        frame_means   = [s.get("mean",    float("nan")) for s in valid_stats]
        frame_medians = [s.get("median",  float("nan")) for s in valid_stats]
        frame_p75s    = [s.get("p75",     float("nan")) for s in valid_stats]
        frame_p90s    = [s.get("p90",     float("nan")) for s in valid_stats]
        frame_p95s    = [s.get("p95",     float("nan")) for s in valid_stats]
        return {
            "valid_frames":                len(valid_stats),
            "mean_of_frame_means_m":       float(np.mean(frame_means)),
            "median_of_frame_medians_m":   float(np.median(frame_medians)),
            "mean_of_frame_p75_m":         float(np.mean(frame_p75s)),
            "mean_of_frame_p90_m":         float(np.mean(frame_p90s)),
            "mean_of_frame_p95_m":         float(np.mean(frame_p95s)),
        }

    h_agg: dict[str, Any] = {}
    if reproj_p95s:
        h_agg["reproj_error_px"] = {
            "frame_count":         len(reproj_p95s),
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
            "keep_rate":           round(z_after_total / z_before_total, 4),
        }

    pr_agg: dict = {}
    if pr_precisions:
        pp = np.array(pr_precisions)
        pr = np.array(pr_recalls)
        pr_agg = {
            "frame_count":         len(pp),
            "mean_precision":      round(float(pp.mean()), 4),
            "mean_recall":         round(float(pr.mean()), 4),
            "median_precision":    round(float(np.median(pp)), 4),
            "median_recall":       round(float(np.median(pr)), 4),
            "prec_ge_90pct_ratio": round(float((pp >= 0.9).mean()), 4),
            "rec_ge_90pct_ratio":  round(float((pr >= 0.9).mean()), 4),
            "per_frame_precision": [round(v, 4) for v in pp.tolist()],
            "per_frame_recall":    [round(v, 4) for v in pr.tolist()],
        }

    return {
        "valid_frame_count": valid_scan_frames,
        "total_frame_count": len(per_frame),
        "scanline": {
            "left_boundary":  pool_stat(left_all),
            "right_boundary": pool_stat(right_all),
            "overall":        pool_stat(overall_all),
        },
        "h_matrix":       h_agg,
        "traversable_pr": pr_agg,
    }


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    if isinstance(v, float) and v != v:
        return "n/a"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def _print_summary(agg: dict[str, Any], total_frames: int) -> None:
    valid = agg.get("valid_frame_count", 0)
    print(f"\n{'='*56}")
    print(f"  SEQUENCE SUMMARY  ({valid}/{total_frames} frames with valid scanlines)")
    print(f"{'='*56}")

    sc = agg.get("scanline", {})
    print("\nScanline boundary error (m):")
    print(f"  {'':12s}  {'mean':>8s}  {'median':>8s}  {'P75':>8s}  {'P90':>8s}  {'P95':>8s}  {'frames':>6s}")
    for label, d in (("overall",  sc.get("overall", {})),
                     ("left",     sc.get("left_boundary", {})),
                     ("right",    sc.get("right_boundary", {}))):
        print(f"  {label:12s}  {_fmt(d.get('mean_of_frame_means_m', float('nan'))):>8s}  "
              f"{_fmt(d.get('median_of_frame_medians_m', float('nan'))):>8s}  "
              f"{_fmt(d.get('mean_of_frame_p75_m', float('nan'))):>8s}  "
              f"{_fmt(d.get('mean_of_frame_p90_m', float('nan'))):>8s}  "
              f"{_fmt(d.get('mean_of_frame_p95_m', float('nan'))):>8s}  "
              f"{d.get('valid_frames', 0):>6d}")

    h = agg.get("h_matrix", {})
    reproj = h.get("reproj_error_px", {})
    if reproj:
        print("\nH reprojection error (px):")
        print(f"  mean-of-means : {_fmt(reproj.get('mean_of_frame_means', float('nan')))}")
        print(f"  mean-of-P95   : {_fmt(reproj.get('mean_of_frame_p95', float('nan')))}")
        print(f"  max-P95       : {_fmt(reproj.get('max_of_frame_p95', float('nan')))}")
        vals = "  ".join(f"{v:.2f}" for v in reproj.get("per_frame_p95", []))
        if vals:
            print(f"  per-frame P95 : [{vals}]")

    tilt = h.get("plane_tilt_deg", {})
    if tilt:
        print(f"\nGround plane tilt (deg):  "
              f"mean {_fmt(tilt.get('mean', float('nan')))}  "
              f"min {_fmt(tilt.get('min', float('nan')))}  "
              f"max {_fmt(tilt.get('max', float('nan')))}")

    zf = h.get("z_filter", {})
    if zf:
        before = zf.get("total_points_before", 0)
        after  = zf.get("total_points_after", 0)
        print(f"\nZ-filter totals: {after}/{before} points kept ({zf.get('keep_rate',0)*100:.1f}%)")

    pr = agg.get("traversable_pr", {})
    if pr:
        print(f"\nTraversable Precision/Recall ({pr['frame_count']} frames):")
        print(f"  mean precision : {pr['mean_precision']*100:.2f}%"
              f"   (≥90% in {pr['prec_ge_90pct_ratio']*100:.1f}% of frames)")
        print(f"  mean recall    : {pr['mean_recall']*100:.2f}%"
              f"   (≥90% in {pr['rec_ge_90pct_ratio']*100:.1f}% of frames)")

    print(f"{'='*56}\n")


def _print_visualizations(per_frame: list[dict[str, Any]]) -> None:
    rgb_overlays = []
    h_heatmaps   = []

    for f in per_frame:
        if "error" in f:
            continue
        sl_paths = f.get("scanline_error", {}).get("paths", {})
        rgb = sl_paths.get("rgb_overlay_png")
        if rgb:
            rgb_overlays.append((f["frame_id"], rgb))

        h_diag = f.get("h_estimation", {}).get("diagnostics", {}).get("diagnostic_files", {})
        hm = h_diag.get("reproj_error_heatmap_png")
        if hm:
            h_heatmaps.append((f["frame_id"], hm))

    if rgb_overlays:
        print("Scanline RGB overlays:")
        for fid, p in rgb_overlays:
            print(f"  {fid}")
            print(f"    {p}")

    if h_heatmaps:
        print("\nH-matrix reprojection heatmaps:")
        for fid, p in h_heatmaps:
            print(f"  {fid}")
            print(f"    {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if args.list_tasks:
        tasks = list_config_tasks(args.config)
        print("\n".join(tasks) if tasks else f"No tasks defined in {args.config}")
        return
    try:
        seq_cfg = read_config(args.config, args.task)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc

    output_dir = Path(seq_cfg["output_dir"]) if Path(seq_cfg["output_dir"]).is_absolute() \
                 else Path(seq_cfg["project_root"]) / seq_cfg["output_dir"]
    ensure_dir(output_dir)

    discovery_meta: dict[str, Any] | None = None
    all_images: list[dict[str, Any]] = []
    h_window: float = 0.0

    if "dataset" in seq_cfg:
        seg_enabled = seq_cfg.get("segmentation", {}).get("enabled", False)
        print("[INFO] Scanning dataset directories (anchoring on point clouds)...")
        frames, discovery_meta = discover_frames(seq_cfg, require_pred_mask=not seg_enabled)
        print(f"[INFO] {discovery_meta['matched_frames']} anchor frames ready  "
              f"({discovery_meta['skipped_frames']} skipped)")

        h_window   = float(seq_cfg.get("h_search_window_sec", 15.0))
        all_images = discover_all_images(seq_cfg)
        print(f"[INFO] {len(all_images)} total images available as H candidates  "
              f"(search window ±{h_window}s)\n")
    else:
        frames = seq_cfg["frames"]

    if args.frames:
        wanted = set(args.frames)
        frames = [f for f in frames if f["frame_id"] in wanted]
        if not frames:
            print(f"[WARN] No frames matched: {args.frames}")
            return

    n_workers  = min(args.workers, len(frames))
    class_name = seq_cfg["class"]["name"]
    frame_id_to_idx = {f["frame_id"]: i for i, f in enumerate(frames)}

    print(f"[INFO] Processing {len(frames)} frame(s) → {output_dir}  (workers={n_workers})\n")

    # --- Phase 0: segmentation inference (optional) ---
    seg_cfg = seq_cfg.get("segmentation", {})
    if seg_cfg.get("enabled", False):
        from distance.inference import run_inference
        seg_config     = seg_cfg["config"]
        seg_checkpoint = seg_cfg["checkpoint"]
        seg_output_dir = Path(seg_cfg["pred_mask_output_dir"])
        image_paths    = [f["rgb"] for f in frames]
        pred_mask_map  = run_inference(
            image_paths, seg_output_dir, seg_config, seg_checkpoint,
            device=seg_cfg.get("device", "cuda:0"),
        )
        # update each frame's pred_mask path to the newly generated mask
        for frame in frames:
            generated = pred_mask_map.get(frame["rgb"])
            if generated:
                frame["pred_mask"] = str(generated)
        print()

    # --- Phase 1: estimate best H for each anchor frame via candidate search ---
    h_results: dict[str, dict[str, Any]] = {}

    p1_packed: list[tuple] = []
    for frame in frames:
        if all_images:
            anchor_ts  = float(frame["frame_id"])
            candidates = find_candidate_images(anchor_ts, all_images, h_window)
        else:
            # Explicit frames mode: use the frame's own image as sole candidate
            candidates = [{"image_id": frame["frame_id"],
                           "rgb":      frame["rgb"],
                           "timestamp": _stem_to_ts(frame["frame_id"])}]
        p1_packed.append((seq_cfg, frame, candidates, str(output_dir), args.skip_existing))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_id = {executor.submit(_phase1_worker, p): p[1]["frame_id"] for p in p1_packed}
        with tqdm(total=len(frames), desc="Phase 1 (H)", unit="frame") as pbar:
            for future in as_completed(future_to_id):
                result = future.result()
                h_results[result["frame_id"]] = result
                cand = result.get("best_candidate", "?")
                n_cands = next(
                    (len(p[2]) for p in p1_packed if p[1]["frame_id"] == result["frame_id"]), "?")
                pbar.set_postfix({"best": cand, "pool": n_cands})
                pbar.update(1)

    # Best H for each anchor is already at the standard path
    frame_to_best_h = {
        frame["frame_id"]: output_dir / "frames" / frame["frame_id"] / "h_estimation" / f"{class_name}_h_estimation.json"
        for frame in frames
    }

    # --- Phase 2: scanline error using best H ---
    per_frame_results: list[dict[str, Any]] = [None] * len(frames)  # type: ignore[list-item]
    p2_packed = [
        (seq_cfg, frame, str(output_dir), str(frame_to_best_h[frame["frame_id"]]), args.skip_existing)
        for frame in frames
    ]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_id = {executor.submit(_phase2_worker, p): p[1]["frame_id"] for p in p2_packed}
        with tqdm(total=len(frames), desc="Phase 2 (scanline)", unit="frame") as pbar:
            for future in as_completed(future_to_id):
                result = future.result()
                per_frame_results[frame_id_to_idx[result["frame_id"]]] = result
                pbar.update(1)
                if "error" in result:
                    tqdm.write(f"  [ERR] {result['frame_id']}: {result['error']}")

    aggregate = aggregate_results(per_frame_results, seq_cfg)

    sequence_summary: dict[str, Any] = {
        "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config_path": str(args.config),
        "output_dir":  str(output_dir),
        "class":       seq_cfg["class"],
        "frame_count": len(frames),
        "aggregate":   aggregate,
        "per_frame":   per_frame_results,
    }
    if discovery_meta is not None:
        sequence_summary["discovery"] = discovery_meta

    summary_path = output_dir / "sequence_summary.json"
    with summary_path.open("w") as f:
        json.dump(sequence_summary, f, indent=2)

    _print_summary(aggregate, len(frames))

    print("Results saved in:")
    print(f"  Summary JSON  : {summary_path}")
    print(f"  Per-frame dirs: {output_dir}/frames/<frame_id>/")
    print(f"  H candidates  : {output_dir}/frames/<frame_id>/h_candidates/")
    print(f"  Visualizations: {output_dir}/frames/<frame_id>/scanline/")
    print(f"  H diagnostics : {output_dir}/frames/<frame_id>/h_estimation/diagnostics/")


if __name__ == "__main__":
    main()
