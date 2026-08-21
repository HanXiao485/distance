from __future__ import annotations

import argparse
import colorsys
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


POSE_DIRECTION_SENSOR_TO_WORLD = "sensor_to_world"
POSE_DIRECTION_WORLD_TO_SENSOR = "world_to_sensor"


@dataclass
class PoseRecord:
    timestamp: float
    translation: np.ndarray
    quaternion_wxyz: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full physical boundary-error pipeline from one JSON config.")
    parser.add_argument("--config", required=True, type=Path, help="Pipeline config JSON path.")
    return parser.parse_args()


def read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["_config_path"] = str(path)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(config["project_root"])
    return root / path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_binary(mask: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def load_binary_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


def stats(values: list[float] | np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def stats_extended(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "mean": mean,
        "std": std,
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
        "outlier_ratio_2sigma": float(np.mean(arr > mean + 2.0 * std)),
        "outlier_ratio_3sigma": float(np.mean(arr > mean + 3.0 * std)),
    }


def _error_to_rgb_array(errors: np.ndarray, vmax: float | None = None) -> np.ndarray:
    if errors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    effective_vmax = float(vmax) if (vmax is not None and vmax > 0) else float(np.percentile(errors, 95))
    if effective_vmax <= 0:
        effective_vmax = 1.0
    normalized = np.clip(errors / effective_vmax, 0.0, 1.0)
    rgb = np.zeros((len(errors), 3), dtype=np.uint8)
    for i, t in enumerate(normalized):
        hue = (1.0 - t) * 240.0 / 360.0  # blue (240°) → red (0°) as error increases
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        rgb[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return rgb


def _draw_colorbar(draw: ImageDraw.ImageDraw, w: int, h: int, vmin: float, vmax: float) -> None:
    bar_w, bar_h = 200, 15
    bar_x, bar_y = w - bar_w - 10, h - bar_h - 28
    for i in range(bar_w):
        t = i / max(bar_w - 1, 1)
        hue = (1.0 - t) * 240.0 / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        draw.rectangle((bar_x + i, bar_y, bar_x + i, bar_y + bar_h), fill=(int(r * 255), int(g * 255), int(b * 255)))
    try:
        draw.text((bar_x, bar_y + bar_h + 2), f"{vmin:.0f}px", fill=(255, 255, 255))
        draw.text((bar_x + bar_w - 35, bar_y + bar_h + 2), f"{vmax:.1f}px", fill=(255, 255, 255))
    except Exception:
        pass


def save_reproj_heatmap(
    image_path: Path,
    pixels_uv: np.ndarray,
    errors: np.ndarray,
    output_path: Path,
    vmax_px: float | None = None,
    point_radius: int = 3,
) -> None:
    if pixels_uv.shape[0] == 0 or not image_path.exists():
        return
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    effective_vmax = float(vmax_px) if (vmax_px is not None and vmax_px > 0) else float(np.percentile(errors, 95))
    colors = _error_to_rgb_array(errors, vmax=effective_vmax)
    for (u, v), color in zip(pixels_uv.astype(int), colors):
        c = (int(color[0]), int(color[1]), int(color[2]))
        draw.ellipse((u - point_radius, v - point_radius, u + point_radius, v + point_radius), fill=c, outline=c)
    _draw_colorbar(draw, image.width, image.height, 0.0, effective_vmax)
    ensure_dir(output_path.parent)
    image.save(output_path)


def save_reproj_histogram(errors: np.ndarray, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors, bins=min(60, max(10, len(errors) // 20)), color="steelblue", edgecolor="white", linewidth=0.5)
    for pct, color, ls in ((90, "orange", "--"), (95, "tomato", "--"), (99, "darkred", ":")):
        val = float(np.percentile(errors, pct))
        ax.axvline(val, color=color, linestyle=ls, linewidth=1.5, label=f"P{pct}={val:.1f}px")
    ax.set_xlabel("Reprojection error (px)")
    ax.set_ylabel("Count")
    ax.set_title("H reprojection error distribution")
    ax.legend(fontsize=9)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def reproj_by_image_region(
    uv: np.ndarray,
    errors: np.ndarray,
    image_width: int,
    image_height: int,
    edge_margin_px: int,
) -> dict[str, Any]:
    is_edge = (
        (uv[:, 0] < edge_margin_px) |
        (uv[:, 0] >= image_width - edge_margin_px) |
        (uv[:, 1] < edge_margin_px) |
        (uv[:, 1] >= image_height - edge_margin_px)
    )
    is_center = ~is_edge
    return {
        "edge_margin_px": edge_margin_px,
        "center_count": int(is_center.sum()),
        "edge_count": int(is_edge.sum()),
        "center": stats_extended(errors[is_center]) if is_center.any() else {"count": 0},
        "edge": stats_extended(errors[is_edge]) if is_edge.any() else {"count": 0},
    }


def reproj_by_distance(
    cam_pts: np.ndarray,
    errors: np.ndarray,
    near_m: float,
    far_m: float,
) -> dict[str, Any]:
    dists = np.linalg.norm(cam_pts, axis=1)
    near_mask = dists < near_m
    mid_mask = (dists >= near_m) & (dists < far_m)
    far_mask = dists >= far_m
    return {
        "thresholds_m": {"near": near_m, "far": far_m},
        "near_count": int(near_mask.sum()),
        "mid_count": int(mid_mask.sum()),
        "far_count": int(far_mask.sum()),
        f"near_lt{near_m:.0f}m": stats_extended(errors[near_mask]) if near_mask.any() else {"count": 0},
        f"mid_{near_m:.0f}to{far_m:.0f}m": stats_extended(errors[mid_mask]) if mid_mask.any() else {"count": 0},
        f"far_ge{far_m:.0f}m": stats_extended(errors[far_mask]) if far_mask.any() else {"count": 0},
        "all_distances_m": stats_extended(dists),
    }


def sigma_filter_correspondences(
    plane_xy: np.ndarray,
    uv: np.ndarray,
    threshold_sigma: float,
    max_iters: int,
) -> tuple[np.ndarray, list[int]]:
    mask = np.ones(len(plane_xy), dtype=bool)
    removed_per_iter: list[int] = []
    for _ in range(max_iters):
        if mask.sum() < 4:
            break
        H_try = fit_homography(plane_xy[mask], uv[mask])
        uv_reproj = apply_homography(H_try, plane_xy[mask])
        errs = np.linalg.norm(uv_reproj - uv[mask], axis=1)
        cutoff = errs.mean() + threshold_sigma * errs.std()
        keep_inner = errs <= cutoff
        n_removed = int((~keep_inner).sum())
        removed_per_iter.append(n_removed)
        if n_removed == 0:
            break
        new_mask = mask.copy()
        new_mask[np.flatnonzero(mask)] = keep_inner
        mask = new_mask
    return mask, removed_per_iter


def ransac_fit_h(
    plane_xy: np.ndarray,
    uv: np.ndarray,
    threshold_px: float,
    max_iters: int,
    min_inliers: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(plane_xy)
    best_mask = np.ones(n, dtype=bool)
    best_count = 0
    rng = np.random.default_rng(seed)
    for _ in range(max_iters):
        idx = rng.choice(n, size=4, replace=False)
        try:
            H_try = fit_homography(plane_xy[idx], uv[idx])
        except (np.linalg.LinAlgError, ValueError):
            continue
        uv_reproj = apply_homography(H_try, plane_xy)
        errs = np.linalg.norm(uv_reproj - uv, axis=1)
        inliers = errs <= threshold_px
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_mask = inliers.copy()
    if best_count < min_inliers:
        return fit_homography(plane_xy, uv), np.ones(n, dtype=bool)
    return fit_homography(plane_xy[best_mask], uv[best_mask]), best_mask


def quaternion_wxyz_to_rotation_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat_wxyz.astype(np.float64)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Quaternion norm is zero.")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def make_transform(translation: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_rotation_matrix(quat_wxyz)
    transform[:3, 3] = translation.astype(np.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def choose_direction(transform: np.ndarray, direction: str) -> np.ndarray:
    if direction == POSE_DIRECTION_SENSOR_TO_WORLD:
        return transform
    if direction == POSE_DIRECTION_WORLD_TO_SENSOR:
        return invert_transform(transform)
    raise ValueError(f"Unsupported transform direction: {direction}")


def parse_pose_csv(csv_path: Path) -> dict[float, PoseRecord]:
    pose_map: dict[float, PoseRecord] = {}
    with csv_path.open("r", newline="") as f:
        lines = [line.strip() for line in f if line.strip()]
    for raw_line in lines[1:]:
        row = next(csv.reader([raw_line], delimiter=" ", skipinitialspace=True))
        timestamp = float(row[1])
        x, y, z = map(float, row[2:5])
        qw, qx, qy, qz = map(float, row[5:9])
        pose_map[timestamp] = PoseRecord(
            timestamp=timestamp,
            translation=np.array([x, y, z], dtype=np.float64),
            quaternion_wxyz=np.array([qw, qx, qy, qz], dtype=np.float64),
        )
    return pose_map


def find_pose_by_timestamp(
    poses: dict[float, PoseRecord],
    timestamp: float,
    max_timestamp_diff_sec: float,
) -> PoseRecord:
    nearest_timestamp = min(poses, key=lambda ts: abs(ts - timestamp))
    if abs(nearest_timestamp - timestamp) > max_timestamp_diff_sec:
        raise ValueError(
            f"No pose close enough for timestamp {timestamp}; nearest={nearest_timestamp}, "
            f"diff={abs(nearest_timestamp - timestamp)}"
        )
    return poses[nearest_timestamp]


def pose_record_to_transform(pose: PoseRecord, direction: str) -> np.ndarray:
    raw_transform = make_transform(pose.translation, pose.quaternion_wxyz)
    return choose_direction(raw_transform, direction)


def parse_float_list_from_line(line: str, field_name: str, expected_length: int | None = None) -> np.ndarray:
    match = re.search(r"\[(.*?)\]", line)
    if not match:
        raise ValueError(f"Failed to parse {field_name} from: {line}")
    values = [float(item.strip()) for item in match.group(1).split(",") if item.strip()]
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"{field_name} expected {expected_length} values, got {len(values)}")
    return np.array(values, dtype=np.float64)


def load_camera_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with calib_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]
    k_line = next(line for line in lines if line.startswith("K:"))
    d_line = next(line for line in lines if line.startswith("D:"))
    translation_line = next(line for line in lines if line.startswith("translation:"))
    rotation_line = next(line for line in lines if line.startswith("rotation:"))

    fx, fy, cx, cy = parse_float_list_from_line(k_line, "K", expected_length=4)
    dist_coeffs = parse_float_list_from_line(d_line, "D")
    translation = parse_float_list_from_line(translation_line, "translation", expected_length=3)
    qx, qy, qz, qw = parse_float_list_from_line(rotation_line, "rotation", expected_length=4)
    quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
    frame_to_child = make_transform(translation, quat_wxyz)
    return np.array([fx, fy, cx, cy], dtype=np.float64), dist_coeffs, frame_to_child


def infer_timestamp_from_name(path: Path) -> float:
    try:
        return float(path.stem.split("_")[0])
    except ValueError as exc:
        raise ValueError(f"Cannot infer timestamp from file name: {path.name}") from exc


def load_point_cloud(pointcloud_path: Path) -> np.ndarray:
    suffix = pointcloud_path.suffix.lower()
    if suffix == ".bin":
        points = np.fromfile(pointcloud_path, dtype=np.float32)
        if points.size % 3 != 0:
            raise ValueError(f"Unexpected point count in {pointcloud_path}")
        return points.reshape(-1, 3).astype(np.float64)
    if suffix in {".ply", ".pcd"}:
        try:
            import open3d as o3d
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("open3d is required for .ply/.pcd point clouds.") from exc
        point_cloud = o3d.io.read_point_cloud(str(pointcloud_path))
        if point_cloud.is_empty():
            raise ValueError(f"Loaded point cloud is empty: {pointcloud_path}")
        return np.asarray(point_cloud.points, dtype=np.float64)
    raise ValueError(f"Unsupported point cloud suffix: {suffix}")


def load_3d_labels(label_path: Path, dtype: str) -> np.ndarray:
    return np.fromfile(label_path, dtype=np.dtype(dtype)).astype(np.int64)


def transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points_xyz, ones])
    return (transform @ points_h.T).T[:, :3]


def distort_normalized_points(x: np.ndarray, y: np.ndarray, dist_coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if dist_coeffs.size == 0:
        return x, y
    k1 = dist_coeffs[0] if dist_coeffs.size > 0 else 0.0
    k2 = dist_coeffs[1] if dist_coeffs.size > 1 else 0.0
    p1 = dist_coeffs[2] if dist_coeffs.size > 2 else 0.0
    p2 = dist_coeffs[3] if dist_coeffs.size > 3 else 0.0
    k3 = dist_coeffs[4] if dist_coeffs.size > 4 else 0.0
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_distorted, y_distorted


def project_camera_points(
    camera_points: np.ndarray,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    z = camera_points[:, 2]
    valid_depth = z > 0.0
    if not np.any(valid_depth):
        return np.empty((0, 2), dtype=np.int32), valid_depth, np.zeros(camera_points.shape[0], dtype=bool)
    valid_points = camera_points[valid_depth]
    x = valid_points[:, 0] / valid_points[:, 2]
    y = valid_points[:, 1] / valid_points[:, 2]
    x, y = distort_normalized_points(x, y, dist_coeffs)
    u = fx * x + cx
    v = fy * y + cy
    inside = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    pixels = np.rint(np.stack([u[inside], v[inside]], axis=1)).astype(np.int32)
    inside_full = np.zeros(camera_points.shape[0], dtype=bool)
    valid_indices = np.flatnonzero(valid_depth)
    inside_full[valid_indices[inside]] = True
    return pixels, valid_depth, inside_full


def zbuffer_visible_mask(pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    best: dict[tuple[int, int], tuple[float, int]] = {}
    for idx, ((u, v), depth) in enumerate(zip(pixels, depths)):
        key = (int(u), int(v))
        prev = best.get(key)
        if prev is None or depth < prev[0]:
            best[key] = (float(depth), idx)
    visible = np.zeros(len(pixels), dtype=bool)
    for _, idx in best.values():
        visible[idx] = True
    return visible


def draw_points(image_path: Path, pixels: np.ndarray, output_path: Path, color: tuple[int, int, int]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for u, v in pixels:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=color, outline=color)
    ensure_dir(output_path.parent)
    image.save(output_path)


def extract_correspondences(config: dict[str, Any], output_dir: Path) -> tuple[Path, dict[str, Any]]:
    paths = config["paths"]
    class_cfg = config["class"]
    geom_cfg = config.get("geometry", {})
    output_csv = output_dir / "h_estimation" / f"{class_cfg['name']}_pixel_correspondences.csv"
    overlay_path = output_dir / "h_estimation" / f"{class_cfg['name']}_projected_overlay.png"
    ensure_dir(output_csv.parent)

    image_path = resolve_path(config, paths["rgb"])
    pointcloud_path = resolve_path(config, paths["pointcloud"])
    label_path = resolve_path(config, paths["label3d"])
    poses2d_path = resolve_path(config, paths["poses2d"])
    poses3d_path = resolve_path(config, paths["poses3d"])
    calib_path = resolve_path(config, paths["calibration"])

    intrinsics, dist_coeffs, _ = load_camera_calibration(calib_path)
    if not geom_cfg.get("use_distortion", False):
        dist_coeffs = np.array([], dtype=np.float64)

    points_lidar = load_point_cloud(pointcloud_path)
    labels = load_3d_labels(label_path, config.get("label3d_dtype", "uint32"))
    if labels.shape[0] != points_lidar.shape[0]:
        raise ValueError(f"Label count {labels.shape[0]} does not match point count {points_lidar.shape[0]}")

    label_id = int(class_cfg["id_3d_for_h"])
    target_mask = labels == label_id
    target_points_lidar = points_lidar[target_mask]
    target_indices = np.flatnonzero(target_mask)

    poses2d = parse_pose_csv(poses2d_path)
    poses3d = parse_pose_csv(poses3d_path)
    max_diff = float(geom_cfg.get("max_timestamp_diff_sec", 0.05))
    camera_pose = find_pose_by_timestamp(poses2d, infer_timestamp_from_name(image_path), max_diff)
    pointcloud_pose = find_pose_by_timestamp(poses3d, infer_timestamp_from_name(pointcloud_path), max_diff)
    world_from_lidar = pose_record_to_transform(pointcloud_pose, geom_cfg.get("pose3d_direction", POSE_DIRECTION_SENSOR_TO_WORLD))
    world_from_camera = pose_record_to_transform(camera_pose, geom_cfg.get("pose2d_direction", POSE_DIRECTION_SENSOR_TO_WORLD))
    camera_from_world = invert_transform(world_from_camera)

    points_world = transform_points(world_from_lidar, target_points_lidar)
    points_camera = transform_points(camera_from_world, points_world)
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    pixels, valid_depth_mask, inside_image_mask = project_camera_points(
        points_camera, intrinsics, dist_coeffs, image_width, image_height
    )
    visible_subset = inside_image_mask.copy()
    inside_indices = np.flatnonzero(inside_image_mask)
    if inside_indices.size > 0:
        visible_inside = zbuffer_visible_mask(pixels, points_camera[inside_image_mask, 2])
        visible_subset[:] = False
        visible_subset[inside_indices[visible_inside]] = True

    inside_points_camera = points_camera[inside_image_mask]
    inside_points_world = points_world[inside_image_mask]
    inside_points_lidar = target_points_lidar[inside_image_mask]
    inside_point_ids = target_indices[inside_image_mask]
    visible_pixels = pixels[visible_subset[inside_image_mask]]
    visible_points_world = inside_points_world[visible_subset[inside_image_mask]]
    visible_points_camera = inside_points_camera[visible_subset[inside_image_mask]]
    visible_points_lidar = inside_points_lidar[visible_subset[inside_image_mask]]
    visible_point_ids = inside_point_ids[visible_subset[inside_image_mask]]

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "label_id", "u", "v", "lidar_x", "lidar_y", "lidar_z", "world_x", "world_y", "world_z", "cam_x", "cam_y", "cam_z"])
        for pid, px, p_lidar, p_world, p_cam in zip(
            visible_point_ids, visible_pixels, visible_points_lidar, visible_points_world, visible_points_camera
        ):
            writer.writerow([
                int(pid), label_id, int(px[0]), int(px[1]),
                float(p_lidar[0]), float(p_lidar[1]), float(p_lidar[2]),
                float(p_world[0]), float(p_world[1]), float(p_world[2]),
                float(p_cam[0]), float(p_cam[1]), float(p_cam[2]),
            ])

    draw_points(image_path, visible_pixels, overlay_path, tuple(config.get("h_overlay_color", [0, 255, 0])))
    return output_csv, {
        "total_points": int(points_lidar.shape[0]),
        "target_label_points": int(target_points_lidar.shape[0]),
        "positive_depth_points": int(np.count_nonzero(valid_depth_mask)),
        "inside_image_points": int(inside_points_lidar.shape[0]),
        "visible_correspondences": int(visible_pixels.shape[0]),
        "overlay_png": str(overlay_path),
    }


def load_correspondences(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    uv_list = []
    world_list = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            uv_list.append([float(row["u"]), float(row["v"])])
            world_list.append([float(row["world_x"]), float(row["world_y"]), float(row["world_z"])])
    return np.asarray(uv_list, dtype=np.float64), np.asarray(world_list, dtype=np.float64)


def load_correspondences_full(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    uv_list, world_list, cam_list, lz_list = [], [], [], []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            uv_list.append([float(row["u"]), float(row["v"])])
            world_list.append([float(row["world_x"]), float(row["world_y"]), float(row["world_z"])])
            cam_list.append([float(row["cam_x"]), float(row["cam_y"]), float(row["cam_z"])])
            lz_list.append(float(row["lidar_z"]))
    return (
        np.asarray(uv_list,   dtype=np.float64),
        np.asarray(world_list, dtype=np.float64),
        np.asarray(cam_list,   dtype=np.float64),
        np.asarray(lz_list,    dtype=np.float64),
    )


def fit_plane(points_world: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = points_world.mean(axis=0)
    centered = points_world - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    d = -float(normal @ centroid)
    residual = np.abs(points_world @ normal + d)
    return normal, d, residual


def choose_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(ref, normal)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_x = ref - np.dot(ref, normal) * normal
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    return axis_x, axis_y


def world_to_plane(points_world: np.ndarray, origin: np.ndarray, axis_x: np.ndarray, axis_y: np.ndarray) -> np.ndarray:
    centered = points_world - origin
    return np.column_stack([centered @ axis_x, centered @ axis_y])


def normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    dist = np.sqrt((centered ** 2).sum(axis=1))
    scale = np.sqrt(2.0) / max(float(dist.mean()), 1e-12)
    T = np.array([[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]])
    normalized_h = (T @ np.column_stack([points, np.ones(len(points))]).T).T
    return normalized_h[:, :2], T


def fit_homography(src_xy: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    src_norm, T_src = normalize_points_2d(src_xy)
    dst_norm, T_dst = normalize_points_2d(dst_uv)
    A = []
    for (x, y), (u, v) in zip(src_norm, dst_norm):
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(A, dtype=np.float64), full_matrices=False)
    H_norm = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[2, 2]


def apply_homography(H: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points_uv, np.ones(len(points_uv), dtype=np.float64)])
    mapped_h = (H @ points_h.T).T
    return mapped_h[:, :2] / mapped_h[:, 2:3]


def estimate_h(config: dict[str, Any], correspondence_csv: Path, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    class_name = config["class"]["name"]
    h_cfg = config.get("h_estimation", {})
    diag_cfg = config.get("h_diagnostics", {})
    plane_inlier_threshold_m = float(h_cfg.get("plane_inlier_threshold_m", 0.5))

    uv, world, cam, lidar_z = load_correspondences_full(correspondence_csv)
    if len(uv) < 4:
        raise ValueError("At least 4 correspondences are required to fit H.")

    # Optional Z-filter: subset points before plane fitting and H estimation
    z_cfg = h_cfg.get("z_filter", {})
    z_filter_meta: dict[str, Any] = {"enabled": False}
    if z_cfg.get("enabled", False):
        z_min = float(z_cfg.get("lidar_z_min", -np.inf))
        z_max = float(z_cfg.get("lidar_z_max",  np.inf))
        u_min = float(z_cfg.get("u_min", 0))
        u_max = float(z_cfg.get("u_max", np.inf))
        z_mask = (
            (lidar_z >= z_min) & (lidar_z <= z_max) &
            (uv[:, 0] >= u_min) & (uv[:, 0] <= u_max)
        )
        z_filter_meta = {
            "enabled": True,
            "lidar_z_min": z_min, "lidar_z_max": z_max,
            "u_min": u_min, "u_max": u_max,
            "points_before": int(len(uv)),
            "points_after":  int(z_mask.sum()),
        }
        uv, world, cam = uv[z_mask], world[z_mask], cam[z_mask]
        if len(uv) < 4:
            raise ValueError(f"Z-filter left only {len(uv)} points; need at least 4.")

    # --- Plane fitting ---
    normal, d, residual = fit_plane(world)
    inlier_mask = residual <= plane_inlier_threshold_m
    uv_inliers = uv[inlier_mask]
    world_inliers = world[inlier_mask]
    cam_inliers = cam[inlier_mask]
    centroid = world_inliers.mean(axis=0)
    axis_x, axis_y = choose_plane_basis(normal)
    plane_xy = world_to_plane(world_inliers, centroid, axis_x, axis_y)

    # --- Baseline DLT homography (always computed, preserves original behaviour) ---
    H_baseline = fit_homography(plane_xy, uv_inliers)
    uv_reproj_baseline = apply_homography(H_baseline, plane_xy)
    reproj_err_baseline = np.linalg.norm(uv_reproj_baseline - uv_inliers, axis=1)

    # --- Optional robust H fitting (controlled by h_estimation.robust_method) ---
    robust_method = h_cfg.get("robust_method", "none")
    robust_meta: dict[str, Any] = {"method": robust_method}
    H_fit = H_baseline
    fit_mask = np.ones(len(plane_xy), dtype=bool)

    if robust_method == "sigma_filter":
        sigma_thr = float(h_cfg.get("sigma_filter_threshold", 2.5))
        sigma_iters = int(h_cfg.get("sigma_filter_max_iters", 5))
        fit_mask, removed = sigma_filter_correspondences(plane_xy, uv_inliers, sigma_thr, sigma_iters)
        if fit_mask.sum() >= 4:
            H_fit = fit_homography(plane_xy[fit_mask], uv_inliers[fit_mask])
        robust_meta.update({
            "sigma_threshold": sigma_thr,
            "removed_per_iter": removed,
            "inlier_count": int(fit_mask.sum()),
            "outlier_count": int((~fit_mask).sum()),
        })
    elif robust_method == "ransac":
        ransac_thr = float(h_cfg.get("ransac_threshold_px", 5.0))
        ransac_iters = int(h_cfg.get("ransac_max_iters", 1000))
        H_fit, fit_mask = ransac_fit_h(plane_xy, uv_inliers, ransac_thr, ransac_iters)
        robust_meta.update({
            "ransac_threshold_px": ransac_thr,
            "ransac_max_iters": ransac_iters,
            "inlier_count": int(fit_mask.sum()),
            "outlier_count": int((~fit_mask).sum()),
        })

    # --- Primary H is robust H when enabled, baseline otherwise ---
    H_primary = H_fit
    uv_reproj_primary = apply_homography(H_primary, plane_xy)
    reproj_err_primary = np.linalg.norm(uv_reproj_primary - uv_inliers, axis=1)
    H_image_to_plane = np.linalg.inv(H_primary)
    H_image_to_plane = H_image_to_plane / H_image_to_plane[2, 2]

    # --- Diagnostics (controlled by h_diagnostics.enabled, default: true) ---
    diag_enabled = diag_cfg.get("enabled", True)
    diag: dict[str, Any] = {}
    if diag_enabled:
        edge_margin = int(diag_cfg.get("image_edge_margin_px", 100))
        near_m = float(diag_cfg.get("near_distance_m", 10.0))
        far_m = float(diag_cfg.get("far_distance_m", 30.0))
        heatmap_vmax = diag_cfg.get("error_heatmap_vmax_px", None)

        diag["baseline_reproj_error_px"] = stats_extended(reproj_err_baseline)
        if robust_method != "none":
            diag["robust_reproj_error_px"] = stats_extended(reproj_err_primary)
            diag["robust_fitting"] = robust_meta

        image_path = resolve_path(config, config["paths"]["rgb"])
        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
            diag["reproj_error_by_image_region"] = reproj_by_image_region(
                uv_inliers, reproj_err_primary, img_w, img_h, edge_margin
            )
        except Exception:
            pass

        diag["reproj_error_by_distance"] = reproj_by_distance(cam_inliers, reproj_err_primary, near_m, far_m)
        plane_tilt_deg = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])), 0.0, 1.0))))
        diag["plane_normal_tilt_from_vertical_deg"] = plane_tilt_deg

        diag_dir = output_dir / "h_estimation" / "diagnostics"
        ensure_dir(diag_dir)
        heatmap_path = diag_dir / f"{class_name}_reproj_error_heatmap.png"
        hist_path = diag_dir / f"{class_name}_reproj_error_histogram.png"
        vmax = float(heatmap_vmax) if heatmap_vmax is not None else None
        save_reproj_heatmap(image_path, uv_inliers, reproj_err_primary, heatmap_path, vmax_px=vmax)
        save_reproj_histogram(reproj_err_primary, hist_path)
        diag["diagnostic_files"] = {
            "reproj_error_heatmap_png": str(heatmap_path),
            "reproj_error_histogram_png": str(hist_path),
        }

    result: dict[str, Any] = {
        "source_csv": str(correspondence_csv),
        "total_correspondences": int(len(uv)),
        "z_filter": z_filter_meta,
        "plane_inlier_threshold_m": plane_inlier_threshold_m,
        "plane_inlier_count": int(inlier_mask.sum()),
        "plane_normal_world": normal.tolist(),
        "plane_d_world": float(d),
        "plane_centroid_world": centroid.tolist(),
        "plane_axis_x_world": axis_x.tolist(),
        "plane_axis_y_world": axis_y.tolist(),
        "plane_residual_m": stats(residual),
        "image_reprojection_error_px": stats(reproj_err_primary),
        "H_plane_to_image": H_primary.tolist(),
        "H_image_to_plane": H_image_to_plane.tolist(),
    }
    if diag_enabled and diag:
        result["diagnostics"] = diag

    json_path = output_dir / "h_estimation" / f"{class_name}_h_estimation.json"
    ensure_dir(json_path.parent)
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    return json_path, result


def extract_gt_mask(config: dict[str, Any], output_dir: Path) -> tuple[np.ndarray, Path]:
    gt_cfg = config["masks"]["gt"]
    class_name = config["class"]["name"]
    path = resolve_path(config, gt_cfg["path"])
    if gt_cfg["type"] == "index":
        mask = np.array(Image.open(path)) == int(config["class"]["id_2d"])
    elif gt_cfg["type"] == "binary":
        mask = load_binary_mask(path)
    else:
        raise ValueError(f"Unsupported GT mask type: {gt_cfg['type']}")
    out_path = output_dir / "masks" / f"gt_{class_name}_binary.png"
    save_binary(mask, out_path)
    return mask, out_path


def extract_pred_mask(config: dict[str, Any], output_dir: Path) -> tuple[np.ndarray, Path]:
    pred_cfg = config["masks"]["pred"]
    class_name = config["class"]["name"]
    path = resolve_path(config, pred_cfg["path"])
    if pred_cfg["type"] == "binary":
        mask = load_binary_mask(path)
    elif pred_cfg["type"] == "index":
        mask = np.array(Image.open(path)) == int(config["class"]["id_2d"])
    elif pred_cfg["type"] == "color_threshold":
        rgb = np.array(Image.open(path).convert("RGB"))
        r, g, b = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
        t = pred_cfg["threshold"]
        mask = (
            (r >= int(t.get("r_min", 0))) & (r <= int(t.get("r_max", 255))) &
            (g >= int(t.get("g_min", 0))) & (g <= int(t.get("g_max", 255))) &
            (b >= int(t.get("b_min", 0))) & (b <= int(t.get("b_max", 255))) &
            ((g - r) >= int(t.get("g_minus_r_min", -255))) &
            ((g - b) >= int(t.get("g_minus_b_min", -255)))
        )
    else:
        raise ValueError(f"Unsupported pred mask type: {pred_cfg['type']}")
    out_path = output_dir / "masks" / f"pred_{class_name}_binary.png"
    save_binary(mask, out_path)
    return mask, out_path


def binary_erode_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = [padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]] for dy in range(3) for dx in range(3)]
    return np.all(np.stack(neighbors, axis=0), axis=0)


def binary_dilate_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = [padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]] for dy in range(3) for dx in range(3)]
    return np.any(np.stack(neighbors, axis=0), axis=0)


def binary_close_3x3(mask: np.ndarray, iterations: int) -> np.ndarray:
    closed = mask.copy()
    for _ in range(iterations):
        closed = binary_dilate_3x3(closed)
    for _ in range(iterations):
        closed = binary_erode_3x3(closed)
    return closed


def connected_components(mask: np.ndarray, target_value: bool = True) -> list[np.ndarray]:
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for start_y, start_x in np.argwhere((mask == target_value) & (~visited)):
        component = []
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if visited[ny, nx] or mask[ny, nx] != target_value:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        components.append(np.asarray(component, dtype=np.int32))
    return components


def keep_large_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    cleaned = np.zeros_like(mask, dtype=bool)
    for component in connected_components(mask, True):
        if len(component) >= min_area_px:
            cleaned[component[:, 0], component[:, 1]] = True
    return cleaned


def clean_prediction_mask(mask: np.ndarray, min_area_px: int, close_iterations: int) -> np.ndarray:
    large_components = keep_large_components(mask, min_area_px)
    closed = binary_close_3x3(large_components, close_iterations)
    return keep_large_components(closed, min_area_px)


def border_connected_background(mask: np.ndarray) -> np.ndarray:
    background = ~mask
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    stack = []
    for x in range(width):
        if background[0, x]:
            stack.append((0, x))
        if background[height - 1, x]:
            stack.append((height - 1, x))
    for y in range(height):
        if background[y, 0]:
            stack.append((y, 0))
        if background[y, width - 1]:
            stack.append((y, width - 1))
    while stack:
        y, x = stack.pop()
        if visited[y, x] or not background[y, x]:
            continue
        visited[y, x] = True
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                stack.append((ny, nx))
    return visited


def extract_external_boundary(mask: np.ndarray) -> np.ndarray:
    outside_background = border_connected_background(mask)
    padded = np.pad(outside_background, 1, mode="constant", constant_values=True)
    touches_outside = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            touches_outside |= padded[dy:dy + height, dx:dx + width]
    return mask & touches_outside


def make_boundary_overlay(gt_boundary: np.ndarray, pred_boundary: np.ndarray, path: Path) -> None:
    canvas = np.zeros((*gt_boundary.shape, 3), dtype=np.uint8)
    canvas[gt_boundary & pred_boundary] = [45, 190, 90]
    canvas[gt_boundary & ~pred_boundary] = [255, 70, 70]
    canvas[~gt_boundary & pred_boundary] = [55, 135, 255]
    ensure_dir(path.parent)
    Image.fromarray(canvas, mode="RGB").save(path)


def create_boundaries(config: dict[str, Any], output_dir: Path) -> tuple[Path, Path, dict[str, Any]]:
    class_name = config["class"]["name"]
    mask_cfg = config.get("mask_processing", {})
    gt_mask, gt_mask_path = extract_gt_mask(config, output_dir)
    pred_mask, pred_mask_path = extract_pred_mask(config, output_dir)
    pred_cleaned = clean_prediction_mask(
        pred_mask,
        int(mask_cfg.get("min_component_area_px", 5000)),
        int(mask_cfg.get("close_iterations", 2)),
    )
    gt_boundary = extract_external_boundary(gt_mask)
    pred_boundary = extract_external_boundary(pred_cleaned)
    pred_cleaned_path = output_dir / "boundaries" / f"pred_{class_name}_cleaned_binary.png"
    gt_boundary_path = output_dir / "boundaries" / f"gt_{class_name}_boundary.png"
    pred_boundary_path = output_dir / "boundaries" / f"pred_{class_name}_boundary.png"
    overlay_path = output_dir / "boundaries" / f"{class_name}_gt_pred_boundary_overlay.png"
    save_binary(pred_cleaned, pred_cleaned_path)
    save_binary(gt_boundary, gt_boundary_path)
    save_binary(pred_boundary, pred_boundary_path)
    make_boundary_overlay(gt_boundary, pred_boundary, overlay_path)
    metadata = {
        "gt_mask_pixels": int(gt_mask.sum()),
        "pred_mask_pixels": int(pred_mask.sum()),
        "pred_cleaned_pixels": int(pred_cleaned.sum()),
        "gt_boundary_pixels": int(gt_boundary.sum()),
        "pred_boundary_pixels": int(pred_boundary.sum()),
        "gt_mask": str(gt_mask_path),
        "pred_mask": str(pred_mask_path),
        "pred_cleaned_mask": str(pred_cleaned_path),
        "boundary_overlay": str(overlay_path),
    }
    return gt_boundary_path, pred_boundary_path, metadata


def runs_from_bool_row(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    gaps = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[xs[0], xs[gaps + 1]]
    ends = np.r_[xs[gaps], xs[-1]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def filtered_boundary_runs(boundary: np.ndarray, y: int, edge_margin_px: int, max_boundary_run_width_px: int) -> list[tuple[int, int]]:
    width = boundary.shape[1]
    valid_left = edge_margin_px
    valid_right = width - 1 - edge_margin_px
    filtered = []
    for left, right in runs_from_bool_row(boundary[y]):
        left = max(left, valid_left)
        right = min(right, valid_right)
        if left > right:
            continue
        if right - left + 1 > max_boundary_run_width_px:
            continue
        filtered.append((left, right))
    return filtered


def outer_boundary_points(boundary: np.ndarray, y: int, edge_margin_px: int) -> tuple[float, float] | None:
    width = boundary.shape[1]
    xs = np.flatnonzero(boundary[y])
    xs = xs[(xs >= edge_margin_px) & (xs <= width - 1 - edge_margin_px)]
    if xs.size < 2:
        return None
    left_x = float(xs.min())
    right_x = float(xs.max())
    if left_x >= right_x:
        return None
    return left_x, right_x


def uniform_scanline_rows(
    pred_boundary: np.ndarray,
    step_px: int,
    edge_margin_px: int,
    max_boundary_run_width_px: int,
) -> tuple[list[int], dict[str, Any]]:
    top_cap_y = -1
    row_candidates = []
    for y in range(pred_boundary.shape[0]):
        runs = filtered_boundary_runs(pred_boundary, y, edge_margin_px, max_boundary_run_width_px)
        if top_cap_y < 0 and len(runs) >= 1:
            top_cap_y = y
        if len(runs) >= 2:
            row_candidates.append(y)
    if not row_candidates:
        return [], {"top_cap_y": top_cap_y, "first_valid_y": -1, "last_valid_y": -1}
    first_valid = row_candidates[0]
    last_valid = row_candidates[-1]
    rows = list(range(first_valid, last_valid + 1, step_px))
    if rows[-1] != last_valid:
        rows.append(last_valid)
    return rows, {
        "row_source": "pred_boundary_image",
        "range_rule": "first_to_last_pred_row_with_at_least_two_non_edge_narrow_boundary_runs",
        "sampling_rule": "uniform_y_spacing_then_outermost_boundary_pixels",
        "top_cap_y": int(top_cap_y),
        "first_valid_y": int(first_valid),
        "last_valid_y": int(last_valid),
    }


def distance_m(H: np.ndarray, a_uv: tuple[float, float], b_uv: tuple[float, float]) -> float:
    plane = apply_homography(H, np.asarray([a_uv, b_uv], dtype=np.float64))
    return float(np.linalg.norm(plane[0] - plane[1]))


def make_boundary_canvas(gt_boundary: np.ndarray, pred_boundary: np.ndarray) -> Image.Image:
    canvas = np.zeros((*gt_boundary.shape, 3), dtype=np.uint8)
    canvas[gt_boundary & pred_boundary] = [45, 190, 90]
    canvas[gt_boundary & ~pred_boundary] = [255, 70, 70]
    canvas[~gt_boundary & pred_boundary] = [55, 135, 255]
    return Image.fromarray(canvas, mode="RGB").convert("RGBA")


def draw_samples_on_image(
    base_image: Image.Image,
    samples: list[dict[str, float]],
    output_path: Path,
    line_width: int,
    point_radius: int,
) -> None:
    image = base_image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, _ = image.size
    for row in samples:
        y = int(row["y"])
        draw.line((0, y, width - 1, y), fill=(255, 230, 80, 215), width=line_width)
        draw.line((row["gt_left_x"], y, row["gt_right_x"], y), fill=(255, 80, 80, 190), width=line_width + 1)
        draw.line((row["pred_left_x"], y, row["pred_right_x"], y), fill=(60, 150, 255, 190), width=line_width + 1)
        for key, color in (
            ("gt_left_x", (255, 70, 70, 245)),
            ("gt_right_x", (255, 70, 70, 245)),
            ("pred_left_x", (40, 140, 255, 245)),
            ("pred_right_x", (40, 140, 255, 245)),
        ):
            x = float(row[key])
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=color)
            draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), outline=(255, 255, 255, 230), width=2)
    ensure_dir(output_path.parent)
    Image.alpha_composite(image, overlay).save(output_path)


def scanline_error(config: dict[str, Any], h_json: Path, gt_boundary_path: Path, pred_boundary_path: Path, output_dir: Path) -> dict[str, Any]:
    class_name = config["class"]["name"]
    scan_cfg = config.get("scanline", {})
    gt_boundary = load_binary_mask(gt_boundary_path)
    pred_boundary = load_binary_mask(pred_boundary_path)
    with h_json.open() as f:
        H_image_to_plane = np.asarray(json.load(f)["H_image_to_plane"], dtype=np.float64)
    step_px = int(scan_cfg.get("step_px", 10))
    edge_margin_px = int(scan_cfg.get("edge_margin_px", 0))
    max_boundary_run_width_px = int(scan_cfg.get("max_boundary_run_width_px", 25))
    rows, row_limits = uniform_scanline_rows(pred_boundary, step_px, edge_margin_px, max_boundary_run_width_px)
    samples = []
    left_errors = []
    right_errors = []
    skipped = {"missing_gt_outer_points": 0, "missing_pred_outer_points": 0}
    for y in rows:
        gt_points = outer_boundary_points(gt_boundary, y, edge_margin_px)
        pred_points = outer_boundary_points(pred_boundary, y, edge_margin_px)
        if gt_points is None:
            skipped["missing_gt_outer_points"] += 1
            continue
        if pred_points is None:
            skipped["missing_pred_outer_points"] += 1
            continue
        gt_left, gt_right = gt_points
        pred_left, pred_right = pred_points
        left_error = distance_m(H_image_to_plane, (gt_left, y), (pred_left, y))
        right_error = distance_m(H_image_to_plane, (gt_right, y), (pred_right, y))
        left_errors.append(left_error)
        right_errors.append(right_error)
        samples.append({
            "y": int(y),
            "gt_left_x": float(gt_left),
            "gt_right_x": float(gt_right),
            "pred_left_x": float(pred_left),
            "pred_right_x": float(pred_right),
            "left_error_m": left_error,
            "right_error_m": right_error,
        })
    scan_dir = output_dir / "scanline"
    ensure_dir(scan_dir)
    csv_path = scan_dir / f"{class_name}_scanline_samples.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["y", "gt_left_x", "gt_right_x", "pred_left_x", "pred_right_x", "left_error_m", "right_error_m"])
        writer.writeheader()
        writer.writerows(samples)
    rgb_path = resolve_path(config, config["paths"]["rgb"])
    rgb_overlay_path = scan_dir / f"{class_name}_scanline_rgb_overlay.png"
    boundary_overlay_path = scan_dir / f"{class_name}_scanline_boundary_overlay.png"
    if rgb_path.exists():
        rgb_canvas = Image.open(rgb_path).convert("RGBA")
    else:
        rgb_canvas = Image.new("RGBA", (gt_boundary.shape[1], gt_boundary.shape[0]), (20, 20, 20, 255))
    draw_samples_on_image(
        rgb_canvas, samples, rgb_overlay_path,
        int(scan_cfg.get("line_width", 4)),
        int(scan_cfg.get("point_radius", 7)),
    )
    draw_samples_on_image(
        make_boundary_canvas(gt_boundary, pred_boundary), samples, boundary_overlay_path,
        int(scan_cfg.get("line_width", 4)),
        int(scan_cfg.get("point_radius", 7)),
    )
    all_errors = left_errors + right_errors
    return {
        "method": "uniform_scanline_outermost_boundary_error",
        "control_variable": "uniform_y_scanlines_shared_by_gt_and_pred",
        "scanline_step_px": step_px,
        "edge_margin_px": edge_margin_px,
        "max_boundary_run_width_px": max_boundary_run_width_px,
        "valid_row_rule": row_limits,
        "candidate_scanline_count": int(len(rows)),
        "sampled_scanline_count": int(len(samples)),
        "skipped_scanlines": skipped,
        "sampled_point_count": int(len(samples) * 4),
        "left_error_m": stats(left_errors),
        "right_error_m": stats(right_errors),
        "overall_error_m": stats(all_errors),
        "paths": {
            "samples_csv": str(csv_path),
            "rgb_overlay_png": str(rgb_overlay_path),
            "boundary_overlay_png": str(boundary_overlay_path),
        },
    }


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    output_dir = resolve_path(config, config["output_dir"])
    ensure_dir(output_dir)

    correspondence_csv, correspondence_summary = extract_correspondences(config, output_dir)
    h_json, h_summary = estimate_h(config, correspondence_csv, output_dir)
    gt_boundary_path, pred_boundary_path, boundary_summary = create_boundaries(config, output_dir)
    scanline_summary = scanline_error(config, h_json, gt_boundary_path, pred_boundary_path, output_dir)

    summary = {
        "config_path": str(args.config),
        "project_root": str(resolve_path(config, ".")),
        "class": config["class"],
        "outputs_root": str(output_dir),
        "correspondences": correspondence_summary,
        "h_estimation": h_summary,
        "boundary": boundary_summary,
        "scanline_error": scanline_summary,
    }
    summary_path = output_dir / "pipeline_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"[INFO] Saved pipeline summary: {summary_path}")


if __name__ == "__main__":
    main()
