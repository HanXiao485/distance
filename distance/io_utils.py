"""Shared I/O, config, pose, and calibration utilities used across all pipeline stages."""
from __future__ import annotations

import argparse
import csv
import functools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


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
        "p75": float(np.percentile(arr, 75)),
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


@functools.lru_cache(maxsize=8)
def parse_pose_csv(csv_path: Path) -> dict[float, PoseRecord]:
    """结果按 csv_path 缓存：H候选搜索会对同一帧反复用同一份 poses2d/poses3d.csv，
    没必要每个候选都重新读盘解析一次。"""
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


@functools.lru_cache(maxsize=8)
def load_camera_calibration(calib_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标定文件在整个序列里通常只有一份，按路径缓存避免每个候选重新解析。"""
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
        stem = path.stem.split("_")[0]
        return float(stem.replace("-", ".", 1))
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
