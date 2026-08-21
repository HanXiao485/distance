"""
project_pointcloud_to_rgb.py

独立、可复用的"点云 -> RGB 图像"投影模块（含 z-buffer 遮挡剔除）。

从 scripts/error_metric/extract_grass_correspondences.py 中抽取出投影步骤，
去掉了写死的 grass 类别，语义标签过滤改为可选参数，便于其他脚本 import 复用。

用法（CLI）:
    python scripts/project_pointcloud_to_rgb.py \
        --image raw/current_frame/1623379838.017682788_rgb.png \
        --bin raw/current_frame/1623379838.017682788.bin \
        --poses2d calibration/poses2d.csv \
        --poses3d calibration/poses3d.csv \
        --calib calibration/camera_calibration.yaml \
        --label-file labels/3d_point_labels/1623379838.017682788_3d.label \
        --label-id 3 \
        --output-csv outputs/projection/grass_pixel_correspondences.csv \
        --output-overlay outputs/projection/grass_projected_overlay.png

不传 --label-file / --label-id 时，默认投影全部点云。

作为模块复用:
    from scripts.project_pointcloud_to_rgb import project_point_cloud_to_image
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

USE_DISTORTION = False
REQUIRE_EXACT_TIMESTAMP_MATCH = False
MAX_TIMESTAMP_DIFF_SEC = 0.05

POSE_DIRECTION_SENSOR_TO_WORLD = "sensor_to_world"
POSE_DIRECTION_WORLD_TO_SENSOR = "world_to_sensor"
POSE3D_DIRECTION = POSE_DIRECTION_SENSOR_TO_WORLD
POSE2D_DIRECTION = POSE_DIRECTION_SENSOR_TO_WORLD

EXTRINSIC_DIRECTION_FRAME_TO_CHILD = "frame_to_child"
EXTRINSIC_DIRECTION_CHILD_TO_FRAME = "child_to_frame"
CALIBRATION_EXTRINSIC_DIRECTION = EXTRINSIC_DIRECTION_FRAME_TO_CHILD

POINT_RADIUS = 2
POINT_COLOR = (0, 255, 0)


@dataclass
class PoseRecord:
    timestamp: float
    translation: np.ndarray
    quaternion_wxyz: np.ndarray


# ── 姿态 / 外参 ────────────────────────────────────────────────────────────

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


def choose_direction(transform: np.ndarray, is_source_to_target: bool) -> np.ndarray:
    return transform if is_source_to_target else invert_transform(transform)


def pose_direction_to_is_forward(direction: str) -> bool:
    if direction not in (POSE_DIRECTION_SENSOR_TO_WORLD, POSE_DIRECTION_WORLD_TO_SENSOR):
        raise ValueError(f"Invalid pose direction: {direction}")
    return direction == POSE_DIRECTION_SENSOR_TO_WORLD


def extrinsic_direction_to_is_forward(direction: str) -> bool:
    if direction not in (EXTRINSIC_DIRECTION_FRAME_TO_CHILD, EXTRINSIC_DIRECTION_CHILD_TO_FRAME):
        raise ValueError(f"Invalid extrinsic direction: {direction}")
    return direction == EXTRINSIC_DIRECTION_FRAME_TO_CHILD


def pose_record_to_transform(pose: PoseRecord, direction: str) -> np.ndarray:
    raw_transform = make_transform(pose.translation, pose.quaternion_wxyz)
    return choose_direction(raw_transform, pose_direction_to_is_forward(direction))


def parse_pose_csv(csv_path: Path) -> Dict[float, PoseRecord]:
    pose_map: Dict[float, PoseRecord] = {}
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
    pose_map: Dict[float, PoseRecord],
    target_timestamp: float,
    require_exact: bool,
    max_diff_sec: float,
) -> PoseRecord:
    if not pose_map:
        raise ValueError("Pose table is empty.")
    if target_timestamp in pose_map:
        return pose_map[target_timestamp]
    nearest_timestamp = min(pose_map.keys(), key=lambda ts: abs(ts - target_timestamp))
    diff = abs(nearest_timestamp - target_timestamp)
    if require_exact or diff > max_diff_sec:
        raise KeyError(
            f"No pose found for timestamp {target_timestamp:.9f}. "
            f"Nearest is {nearest_timestamp:.9f} (diff={diff:.6f}s)."
        )
    print(
        f"[INFO] Using nearest pose for {target_timestamp:.9f}: "
        f"{nearest_timestamp:.9f} (diff={diff:.6f}s)"
    )
    return pose_map[nearest_timestamp]


def parse_float_list_from_line(line: str, field_name: str, expected_length: Optional[int] = None) -> np.ndarray:
    import re

    match = re.search(r"\[(.*?)\]", line)
    if not match:
        raise ValueError(f"Failed to parse {field_name} from: {line}")
    values = [float(item.strip()) for item in match.group(1).split(",") if item.strip()]
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"{field_name} expected {expected_length} values, got {len(values)}")
    return np.array(values, dtype=np.float64)


def load_camera_calibration(calib_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    extrinsic = choose_direction(frame_to_child, extrinsic_direction_to_is_forward(CALIBRATION_EXTRINSIC_DIRECTION))
    return np.array([fx, fy, cx, cy], dtype=np.float64), dist_coeffs, extrinsic


def infer_timestamp_from_name(path: Path) -> float:
    # 与 run_boundary_error_pipeline.py 保持一致：文件名可能带 "_rgb" 等后缀
    # （如 1623379838.017682788_rgb.png），只取下划线前的数字前缀作为时间戳。
    try:
        return float(path.stem.split("_")[0])
    except ValueError as exc:
        raise ValueError(f"Cannot infer timestamp from file name: {path.name}") from exc


# ── 点云 / 标签 加载 ────────────────────────────────────────────────────────

def load_point_cloud_from_bin(bin_path: Path) -> np.ndarray:
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 3 != 0:
        raise ValueError(f"Unexpected point count in {bin_path}")
    return points.reshape(-1, 3).astype(np.float64)


def load_labels(label_path: Path) -> np.ndarray:
    labels = np.fromfile(label_path, dtype=np.uint32)
    return labels.astype(np.int64)


# ── 几何变换 / 投影 ─────────────────────────────────────────────────────────

def transform_points(transform: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    ones = np.ones((points_xyz.shape[0], 1), dtype=np.float64)
    points_h = np.hstack([points_xyz, ones])
    transformed_h = (transform @ points_h.T).T
    return transformed_h[:, :3]


def distort_normalized_points(x: np.ndarray, y: np.ndarray, dist_coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if dist_coeffs.size == 0:
        return x, y
    k1 = dist_coeffs[0] if dist_coeffs.size > 0 else 0.0
    k2 = dist_coeffs[1] if dist_coeffs.size > 1 else 0.0
    p1 = dist_coeffs[2] if dist_coeffs.size > 2 else 0.0
    p2 = dist_coeffs[3] if dist_coeffs.size > 3 else 0.0
    k3 = dist_coeffs[4] if dist_coeffs.size > 4 else 0.0
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_distorted, y_distorted


def project_camera_points(
    camera_points: np.ndarray,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把相机坐标系下的 3D 点投影到像素坐标。

    返回:
        pixels: 落在图像内的点的 (u, v) 整数像素坐标，形状 (K, 2)
        valid_depth: 深度 > 0 的掩码，形状 (N,)
        inside_full: 落在图像内（且深度有效）的掩码，形状 (N,)
    """
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

    pixels = np.stack([u[inside], v[inside]], axis=1)
    pixels = np.rint(pixels).astype(np.int32)

    inside_full = np.zeros(camera_points.shape[0], dtype=bool)
    valid_indices = np.flatnonzero(valid_depth)
    inside_full[valid_indices[inside]] = True
    return pixels, valid_depth, inside_full


def zbuffer_visible_mask(pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """每个像素 (int(u), int(v)) 只保留深度最小的点；深度并列时保留最先出现的索引。"""
    n = len(pixels)
    if n == 0:
        return np.zeros(0, dtype=bool)
    u = pixels[:, 0].astype(np.int64)
    v = pixels[:, 1].astype(np.int64)
    mult = int(u.max()) + 1
    key = v * mult + u
    depth_arr = np.asarray(depths, dtype=np.float64)
    order = np.lexsort((depth_arr, key))
    ks = key[order]
    first_in_group = np.ones(n, dtype=bool)
    first_in_group[1:] = ks[1:] != ks[:-1]
    visible = np.zeros(n, dtype=bool)
    visible[order[first_in_group]] = True
    return visible


def draw_points(image_path: Path, pixels: np.ndarray, output_path: Path,
                 color: Tuple[int, int, int] = POINT_COLOR, radius: int = POINT_RADIUS) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for u, v in pixels:
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=color, outline=color)
    image.save(output_path)


# ── 顶层封装：点云 -> RGB 投影（可复用） ────────────────────────────────────

def project_point_cloud_to_image(
    image_path: Path,
    bin_path: Path,
    poses2d_path: Path,
    poses3d_path: Path,
    calib_path: Path,
    label_path: Optional[Path] = None,
    label_id: Optional[int] = None,
    use_zbuffer: bool = True,
) -> dict:
    """点云投影到 RGB 图像的完整流程（含可选语义过滤 + z-buffer 遮挡剔除）。

    返回一个 dict，字段:
        target_indices: 参与投影的原始点云下标（若 label_id 为 None 则是全部点）
        points_lidar / points_world / points_camera: 对应坐标 (M, 3)
        pixels_all_inside: 落在图像内的点的像素坐标 (K, 2)
        inside_mask: 相对于 target_indices 的"落在图像内"掩码 (M,)
        visible_mask_inside: 相对于 inside 子集的"z-buffer 可见"掩码 (K,)
        image_width / image_height
    """
    intrinsics, dist_coeffs, _ = load_camera_calibration(calib_path)
    if not USE_DISTORTION:
        dist_coeffs = np.array([], dtype=np.float64)

    image_timestamp = infer_timestamp_from_name(image_path)
    pointcloud_timestamp = infer_timestamp_from_name(bin_path)
    poses2d = parse_pose_csv(poses2d_path)
    poses3d = parse_pose_csv(poses3d_path)
    camera_pose = find_pose_by_timestamp(poses2d, image_timestamp, REQUIRE_EXACT_TIMESTAMP_MATCH, MAX_TIMESTAMP_DIFF_SEC)
    pointcloud_pose = find_pose_by_timestamp(poses3d, pointcloud_timestamp, REQUIRE_EXACT_TIMESTAMP_MATCH, MAX_TIMESTAMP_DIFF_SEC)

    world_from_lidar = pose_record_to_transform(pointcloud_pose, POSE3D_DIRECTION)
    world_from_camera = pose_record_to_transform(camera_pose, POSE2D_DIRECTION)
    camera_from_world = invert_transform(world_from_camera)

    points_lidar_all = load_point_cloud_from_bin(bin_path)

    if label_path is not None and label_id is not None:
        labels = load_labels(label_path)
        if labels.shape[0] != points_lidar_all.shape[0]:
            raise ValueError(f"Label count {labels.shape[0]} does not match point count {points_lidar_all.shape[0]}")
        target_mask = labels == label_id
    else:
        target_mask = np.ones(points_lidar_all.shape[0], dtype=bool)

    target_points_lidar = points_lidar_all[target_mask]
    target_indices = np.flatnonzero(target_mask)

    points_world = transform_points(world_from_lidar, target_points_lidar)
    points_camera = transform_points(camera_from_world, points_world)

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    pixels, valid_depth_mask, inside_image_mask = project_camera_points(
        points_camera, intrinsics, dist_coeffs, image_width, image_height,
    )

    visible_subset = inside_image_mask.copy()
    inside_indices = np.flatnonzero(inside_image_mask)
    if use_zbuffer and inside_indices.size > 0:
        visible_inside = zbuffer_visible_mask(pixels, points_camera[inside_image_mask, 2])
        visible_subset[:] = False
        visible_subset[inside_indices[visible_inside]] = True
    elif not use_zbuffer:
        visible_subset = inside_image_mask.copy()

    return {
        "target_indices": target_indices,
        "points_lidar": target_points_lidar,
        "points_world": points_world,
        "points_camera": points_camera,
        "valid_depth_mask": valid_depth_mask,
        "inside_image_mask": inside_image_mask,
        "visible_mask": visible_subset,
        "pixels_inside": pixels,
        "image_width": image_width,
        "image_height": image_height,
    }


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project a point cloud onto an RGB image (with optional semantic label filter and z-buffer occlusion culling).")
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "raw" / "current_frame" / "1623379838.017682788_rgb.png")
    parser.add_argument("--bin", type=Path, default=PROJECT_ROOT / "raw" / "current_frame" / "1623379838.017682788.bin")
    parser.add_argument("--poses2d", type=Path, default=PROJECT_ROOT / "calibration" / "poses2d.csv")
    parser.add_argument("--poses3d", type=Path, default=PROJECT_ROOT / "calibration" / "poses3d.csv")
    parser.add_argument("--calib", type=Path, default=PROJECT_ROOT / "calibration" / "camera_calibration.yaml")
    parser.add_argument("--label-file", type=Path, default=None, help="可选：逐点语义标签文件（.label）")
    parser.add_argument("--label-id", type=int, default=None, help="可选：只保留该标签 id 的点")
    parser.add_argument("--no-zbuffer", action="store_true", help="禁用 z-buffer 遮挡剔除")
    parser.add_argument("--output-csv", type=Path, default=PROJECT_ROOT / "outputs" / "projection" / "pixel_correspondences.csv")
    parser.add_argument("--output-overlay", type=Path, default=PROJECT_ROOT / "outputs" / "projection" / "projected_overlay.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_overlay.parent.mkdir(parents=True, exist_ok=True)

    result = project_point_cloud_to_image(
        image_path=args.image,
        bin_path=args.bin,
        poses2d_path=args.poses2d,
        poses3d_path=args.poses3d,
        calib_path=args.calib,
        label_path=args.label_file,
        label_id=args.label_id,
        use_zbuffer=not args.no_zbuffer,
    )

    target_indices = result["target_indices"]
    inside_image_mask = result["inside_image_mask"]
    visible_mask = result["visible_mask"]
    pixels_inside = result["pixels_inside"]

    inside_points_camera = result["points_camera"][inside_image_mask]
    inside_points_world = result["points_world"][inside_image_mask]
    inside_points_lidar = result["points_lidar"][inside_image_mask]
    inside_point_ids = target_indices[inside_image_mask]

    visible_sub = visible_mask[inside_image_mask]
    visible_pixels = pixels_inside[visible_sub]
    visible_points_world = inside_points_world[visible_sub]
    visible_points_camera = inside_points_camera[visible_sub]
    visible_points_lidar = inside_points_lidar[visible_sub]
    visible_point_ids = inside_point_ids[visible_sub]

    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["point_id", "u", "v", "lidar_x", "lidar_y", "lidar_z",
                          "world_x", "world_y", "world_z", "cam_x", "cam_y", "cam_z"])
        for pid, px, p_lidar, p_world, p_cam in zip(
            visible_point_ids, visible_pixels, visible_points_lidar, visible_points_world, visible_points_camera,
        ):
            writer.writerow([
                int(pid), int(px[0]), int(px[1]),
                float(p_lidar[0]), float(p_lidar[1]), float(p_lidar[2]),
                float(p_world[0]), float(p_world[1]), float(p_world[2]),
                float(p_cam[0]), float(p_cam[1]), float(p_cam[2]),
            ])

    draw_points(args.image, visible_pixels, args.output_overlay)

    print(f"[INFO] Total input points (after label filter): {len(target_indices)}")
    print(f"[INFO] Points inside image: {inside_points_lidar.shape[0]}")
    print(f"[INFO] Visible correspondences after z-buffer: {visible_pixels.shape[0]}")
    print(f"[INFO] Saved CSV: {args.output_csv}")
    print(f"[INFO] Saved overlay: {args.output_overlay}")


if __name__ == "__main__":
    main()
