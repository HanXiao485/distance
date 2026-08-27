"""In-memory adapters used only by the real-time benchmark."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import distance.homography as homography_module
from distance.boundary import (
    clean_prediction_mask, extract_external_boundary, extract_gt_mask, extract_pred_mask,
)
from distance.io_utils import (
    POSE_DIRECTION_SENSOR_TO_WORLD, find_pose_by_timestamp, infer_timestamp_from_name,
    make_transform, parse_pose_csv, resolve_path,
)
from distance.run_sequence import build_frame_config
from distance.scanline import scanline_error

_POINT_CACHE: dict[str, np.ndarray] = {}
_ORIGINAL_LOAD_POINT_CLOUD = homography_module.load_point_cloud


def install_point_cloud_cache() -> None:
    """Make DISTANCE H estimation and the local BEV renderer share one NumPy cloud."""
    def cached(path: Path) -> np.ndarray:
        key = str(Path(path).resolve())
        if key not in _POINT_CACHE:
            _POINT_CACHE.clear()
            _POINT_CACHE[key] = _ORIGINAL_LOAD_POINT_CLOUD(Path(path))
        return _POINT_CACHE[key]
    homography_module.load_point_cloud = cached


def get_shared_point_cloud(path: str | Path) -> np.ndarray:
    return homography_module.load_point_cloud(Path(path))


def create_boundaries_in_memory(config: dict[str, Any], output_dir: Path):
    """Equivalent numerical boundary extraction without GT/Pred boundary PNG files."""
    class_name = config["class"]["name"]
    mask_cfg = config.get("mask_processing", {})
    gt_mask, gt_mask_path = extract_gt_mask(config, output_dir)
    pred_mask, pred_mask_path = extract_pred_mask(config, output_dir)
    pred_cleaned = clean_prediction_mask(
        pred_mask, int(mask_cfg.get("min_component_area_px", 5000)),
        int(mask_cfg.get("close_iterations", 2)),
    )
    gt_boundary = extract_external_boundary(gt_mask)
    pred_boundary = extract_external_boundary(pred_cleaned)
    # Paths remain descriptive metadata only. scanline_error consumes the arrays below.
    gt_path = output_dir / "boundaries" / f"gt_{class_name}_boundary.png"
    pred_path = output_dir / "boundaries" / f"pred_{class_name}_boundary.png"
    # Remove stale visual artifacts when an existing output directory is reused.
    for stale in (gt_path, pred_path,
                  output_dir / "boundaries" / f"{class_name}_gt_pred_boundary_overlay.png",
                  output_dir / "scanline" / f"{class_name}_scanline_rgb_overlay.png",
                  output_dir / "scanline" / f"{class_name}_scanline_boundary_overlay.png"):
        stale.unlink(missing_ok=True)
    metadata = {
        "gt_mask_pixels": int(gt_mask.sum()), "pred_mask_pixels": int(pred_mask.sum()),
        "pred_cleaned_pixels": int(pred_cleaned.sum()),
        "gt_boundary_pixels": int(gt_boundary.sum()),
        "pred_boundary_pixels": int(pred_boundary.sum()),
        "gt_mask": str(gt_mask_path), "pred_mask": str(pred_mask_path),
        "boundary_png_output": False,
    }
    return gt_path, pred_path, metadata, (gt_boundary, pred_boundary)


def process_frame_phase2_in_memory(seq_cfg: dict[str, Any], frame: dict[str, Any],
                                   seq_output_dir: Path, best_h_json: Path, _skip: bool) -> dict[str, Any]:
    frame_id = frame["frame_id"]
    frame_dir = seq_output_dir / "frames" / frame_id
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_cfg = build_frame_config(seq_cfg, frame)
    try:
        gt_path, pred_path, boundary_meta, arrays = create_boundaries_in_memory(frame_cfg, frame_dir)
        scan_meta = scanline_error(frame_cfg, best_h_json, gt_path, pred_path, frame_dir,
                                   precomputed_boundaries=arrays)
    except Exception as exc:
        return {"frame_id": frame_id, "error": str(exc)}
    h_meta = json.loads(best_h_json.read_text())
    summary = {
        "frame_id": frame_id, "output_dir": str(frame_dir), "h_estimation": h_meta,
        "boundary": boundary_meta, "scanline_error": scan_meta,
        "traversable_pr": {}, "h_source": str(best_h_json),
    }
    (frame_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_bev_context(cfg: dict[str, Any], size: tuple[int, int] = (1000, 900)) -> dict[str, Any]:
    width, height = size
    bg = np.full((height, width, 3), (18, 18, 32), np.uint8)
    for metres in range(-30, 31, 10):
        x = int((metres + 30) / 60 * (width - 1)); cv2.line(bg, (x, 0), (x, height-1), (45,45,65), 1)
    for metres in range(-5, 46, 5):
        y = int((45 - metres) / 53 * (height - 1)); cv2.line(bg, (0, y), (width-1, y), (45,45,65), 1)
    shared = cfg.get("shared_paths", {})
    poses_path = resolve_path(cfg, shared.get("poses3d", Path(cfg["dataset"]["dir_3d"]) / "poses3d.csv"))
    return {"background": bg, "poses": parse_pose_csv(poses_path), "poses_path": poses_path,
            "pose_direction": cfg.get("geometry", {}).get("pose3d_direction", POSE_DIRECTION_SENSOR_TO_WORLD),
            "max_diff": float(cfg.get("geometry", {}).get("max_timestamp_diff_sec", .3))}


def _world_to_lidar(world: np.ndarray, frame: dict[str, Any], ctx: dict[str, Any]) -> np.ndarray:
    pose = find_pose_by_timestamp(ctx["poses"], infer_timestamp_from_name(Path(frame["pointcloud"])), ctx["max_diff"])
    world_from_lidar = make_transform(pose.translation, pose.quaternion_wxyz)
    if ctx["pose_direction"] != POSE_DIRECTION_SENSOR_TO_WORLD:
        world_from_lidar = np.linalg.inv(world_from_lidar)
    return (world - world_from_lidar[:3, 3]) @ world_from_lidar[:3, :3]


def _boundary_xy(rows: list[dict[str, str]], col: str, h: dict[str, Any], frame: dict[str, Any], ctx: dict[str, Any]):
    if not rows: return np.empty((0, 2))
    uv = np.array([[float(r[col]), float(r["y"])] for r in rows if r.get(col) not in (None, "", "None")])
    if not len(uv): return np.empty((0, 2))
    H=np.asarray(h["H_image_to_plane"]); q=np.c_[uv,np.ones(len(uv))]@H.T; plane=q[:,:2]/q[:,2:3]
    c=np.asarray(h["plane_centroid_world"]); ax=np.asarray(h["plane_axis_x_world"]); ay=np.asarray(h["plane_axis_y_world"])
    return _world_to_lidar(c + plane[:,0:1]*ax + plane[:,1:2]*ay, frame, ctx)[:,:2]


def render_bev_opencv(ctx: dict[str, Any], frame: dict[str, Any], phase2: dict[str, Any] | None) -> tuple[np.ndarray, float]:
    started=time.perf_counter(); canvas=ctx["background"].copy(); height,width=canvas.shape[:2]
    pts=get_shared_point_cloud(frame["pointcloud"]); labels=np.fromfile(frame["label3d"],dtype=np.uint8)
    valid=(np.abs(pts[:,0])<45)&(np.abs(pts[:,1])<30)&(pts[:,2]>-3)&(pts[:,2]<12)
    pts=pts[valid]; labels=labels[valid]; stride=max(1,len(pts)//60000); pts=pts[::stride]; labels=labels[::stride]
    px=np.clip(((-pts[:,1]+30)/60*(width-1)).astype(int),0,width-1)
    py=np.clip(((45-pts[:,0])/53*(height-1)).astype(int),0,height-1)
    colors=np.stack(((labels*47+40)%220+25,(labels*83+30)%220+25,(labels*131+20)%220+25),axis=1).astype(np.uint8)
    canvas[py,px]=colors; canvas=cv2.dilate(canvas,np.ones((2,2),np.uint8),iterations=1)
    title = "LiDAR-only BEV"
    if phase2 is not None:
        scan_path=Path(phase2["scanline_error"]["paths"]["samples_csv"])
        with scan_path.open(newline="") as f: rows=list(csv.DictReader(f))
        h=phase2["h_estimation"]
        for col,color in (("gt_left_x",(50,50,255)),("gt_right_x",(50,50,255)),("pred_left_x",(255,150,40)),("pred_right_x",(255,150,40))):
            xy=_boundary_xy(rows,col,h,frame,ctx)
            if len(xy):
                q=np.column_stack(((-xy[:,1]+30)/60*(width-1),(45-xy[:,0])/53*(height-1))).astype(np.int32)
                good=(q[:,0]>=0)&(q[:,0]<width)&(q[:,1]>=0)&(q[:,1]<height)
                if good.any(): cv2.polylines(canvas,[q[good]],False,color,3,cv2.LINE_AA)
        mean=phase2["scanline_error"].get("overall_error_m",{}).get("mean",float("nan"))
        title=f"BEV  boundary error mean={mean:.3f} m"
    cv2.drawMarker(canvas,(width//2,int(45/53*(height-1))),(0,255,255),cv2.MARKER_TRIANGLE_UP,22,3)
    cv2.putText(canvas,title,(22,38),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
    return canvas,time.perf_counter()-started


def render_fpv_in_memory(image: np.ndarray, mask_path: Path) -> tuple[np.ndarray, float]:
    started=time.perf_counter(); mask=cv2.imread(str(mask_path),cv2.IMREAD_GRAYSCALE)
    if mask is None: raise FileNotFoundError(mask_path)
    if mask.shape!=image.shape[:2]: mask=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    binary=mask>127; tint=image.copy(); tint[binary]=(0.55*tint[binary]+0.45*np.array([40,210,80])).astype(np.uint8)
    contours,_=cv2.findContours(binary.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tint,contours,-1,(40,255,80),3,cv2.LINE_AA); cv2.putText(tint,"Prediction overlay",(25,45),cv2.FONT_HERSHEY_SIMPLEX,.9,(255,255,255),2,cv2.LINE_AA)
    return tint,time.perf_counter()-started
