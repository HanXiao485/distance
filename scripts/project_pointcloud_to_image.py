import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


# =========================
# User configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set explicit names for the current validation target.
RGB_IMAGE_PATH = PROJECT_ROOT / "raw" / "current_frame" / "1623379838.017682788_rgb.png"
POINTCLOUD_BIN_PATH = PROJECT_ROOT / "raw" / "current_frame" / "1623379838.017682788.bin"
OUTPUT_IMAGE_PATH = PROJECT_ROOT / "outputs" / "projection" / "1623379838.017682788_projected_result.png"
USE_DISTORTION = False


# =========================
# Geometric assumptions
# =========================
# Important dataset-specific finding from this folder:
# - poses3d.csv and poses2d.csv are already related by the calibration extrinsic.
# - In other words, poses2d.csv appears to already be the camera pose.
# - Therefore the default implementation below uses poses2d.csv directly as the
#   camera pose and does NOT multiply camera_calibration.yaml extrinsics again.
#
# Direction options:
# - "sensor_to_world": pose/extrinsic maps local sensor frame -> world/parent frame
# - "world_to_sensor": pose/extrinsic maps world/parent frame -> local sensor frame
#
# Default implementation used here:
# 1. poses3d.csv is treated as the point-cloud sensor pose.
# 2. poses2d.csv is treated as the camera pose.
# 3. camera_calibration.yaml is still read for intrinsics and kept available for
#    optional validation, but its extrinsic is NOT reapplied by default.
# 4. Distortion is disabled by default for this geometric validation because
#    the provided RGB image appears to behave more like an already-rectified
#    image; applying D again shifts many in-image points by tens to hundreds
#    of pixels.
#
# Therefore the default transform chain is:
#   point_in_world = T_world_lidar * point_in_lidar
#   T_world_camera = T_world_camera_from_poses2d
#   point_in_camera = inv(T_world_camera) * point_in_world
#
# Alternative validation mode:
# - If you want to test the old assumption that poses2d.csv is a base pose,
#   switch CAMERA_POSE_SOURCE to CAMERA_POSE_FROM_POSES3D_AND_EXTRINSIC.
# - In that mode, poses3d.csv is interpreted as the parent/base frame and
#   camera_calibration.yaml extrinsic is applied once to recover the camera pose.
POSE_DIRECTION_SENSOR_TO_WORLD = "sensor_to_world"
POSE_DIRECTION_WORLD_TO_SENSOR = "world_to_sensor"

EXTRINSIC_DIRECTION_FRAME_TO_CHILD = "frame_to_child"
EXTRINSIC_DIRECTION_CHILD_TO_FRAME = "child_to_frame"

CAMERA_POSE_FROM_POSES2D = "poses2d_camera_pose"
CAMERA_POSE_FROM_POSES3D_AND_EXTRINSIC = "poses3d_plus_extrinsic"

# Default choice:
# - poses3d.csv: "sensor_to_world"
#   Alternative to try: "world_to_sensor"
POSE3D_DIRECTION = POSE_DIRECTION_SENSOR_TO_WORLD

# Default choice:
# - poses2d.csv: "sensor_to_world"
#   Alternative to try: "world_to_sensor"
POSE2D_DIRECTION = POSE_DIRECTION_SENSOR_TO_WORLD

# Default choice:
# - poses2d.csv already stores camera pose
#   Alternative to try: CAMERA_POSE_FROM_POSES3D_AND_EXTRINSIC
CAMERA_POSE_SOURCE = CAMERA_POSE_FROM_POSES2D

# Default choice:
# - camera_calibration.yaml extrinsic: "frame_to_child"
#   Alternative to try: "child_to_frame"
CALIBRATION_EXTRINSIC_DIRECTION = EXTRINSIC_DIRECTION_FRAME_TO_CHILD


# Whether poses are required to match timestamps exactly.
# If False, the nearest pose within MAX_TIMESTAMP_DIFF_SEC is used.
REQUIRE_EXACT_TIMESTAMP_MATCH = False
MAX_TIMESTAMP_DIFF_SEC = 0.05


# Drawing style
POINT_RADIUS = 2
POINT_COLOR = (255, 0, 0)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


@dataclass
class PoseRecord:
    timestamp: float
    translation: np.ndarray
    quaternion_wxyz: np.ndarray


def find_first_file(base_dir: Path, suffixes: Iterable[str]) -> Optional[Path]:
    for path in sorted(base_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in suffixes:
            return path
    return None


def resolve_input_paths(base_dir: Path) -> Tuple[Path, Path, Path, Path, Path]:
    image_path = RGB_IMAGE_PATH if RGB_IMAGE_PATH else find_first_file(base_dir, IMAGE_EXTENSIONS)
    bin_path = POINTCLOUD_BIN_PATH if POINTCLOUD_BIN_PATH else find_first_file(base_dir, (".bin",))
    poses2d_path = base_dir / "calibration" / "poses2d.csv"
    poses3d_path = base_dir / "calibration" / "poses3d.csv"
    calib_path = base_dir / "calibration" / "camera_calibration.yaml"

    missing = []
    for label, path in {
        "RGB image": image_path,
        "point cloud": bin_path,
        "poses2d.csv": poses2d_path,
        "poses3d.csv": poses3d_path,
        "camera_calibration.yaml": calib_path,
    }.items():
        if path is None or not path.exists():
            missing.append(f"{label}: {path}")

    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    return image_path, bin_path, poses2d_path, poses3d_path, calib_path


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


def validate_direction(name: str, value: str, valid_values: Tuple[str, ...]) -> None:
    if value not in valid_values:
        raise ValueError(f"{name} must be one of {valid_values}, got {value!r}")


def pose_direction_to_is_forward(direction: str) -> bool:
    validate_direction(
        "pose direction",
        direction,
        (POSE_DIRECTION_SENSOR_TO_WORLD, POSE_DIRECTION_WORLD_TO_SENSOR),
    )
    return direction == POSE_DIRECTION_SENSOR_TO_WORLD


def extrinsic_direction_to_is_forward(direction: str) -> bool:
    validate_direction(
        "calibration extrinsic direction",
        direction,
        (EXTRINSIC_DIRECTION_FRAME_TO_CHILD, EXTRINSIC_DIRECTION_CHILD_TO_FRAME),
    )
    return direction == EXTRINSIC_DIRECTION_FRAME_TO_CHILD


def validate_camera_pose_source(source: str) -> None:
    validate_direction(
        "camera pose source",
        source,
        (CAMERA_POSE_FROM_POSES2D, CAMERA_POSE_FROM_POSES3D_AND_EXTRINSIC),
    )


def pose_record_to_transform(pose: PoseRecord, direction: str) -> np.ndarray:
    raw_transform = make_transform(pose.translation, pose.quaternion_wxyz)
    return choose_direction(raw_transform, pose_direction_to_is_forward(direction))


def parse_pose_csv(csv_path: Path) -> Dict[float, PoseRecord]:
    pose_map: Dict[float, PoseRecord] = {}

    with csv_path.open("r", newline="") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"{csv_path} is empty.")

    for raw_line in lines[1:]:
        row = next(csv.reader([raw_line], delimiter=" ", skipinitialspace=True))
        if len(row) < 9:
            raise ValueError(f"Unexpected row format in {csv_path}: {raw_line}")

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
    if require_exact and target_timestamp in pose_map:
        return pose_map[target_timestamp]

    if not pose_map:
        raise ValueError("Pose table is empty.")

    if target_timestamp in pose_map:
        return pose_map[target_timestamp]

    nearest_timestamp = min(pose_map.keys(), key=lambda ts: abs(ts - target_timestamp))
    diff = abs(nearest_timestamp - target_timestamp)

    if require_exact or diff > max_diff_sec:
        raise KeyError(
            f"No pose found for timestamp {target_timestamp:.9f}. "
            f"Nearest timestamp is {nearest_timestamp:.9f} (diff={diff:.6f}s)."
        )

    print(
        f"[INFO] Using nearest pose for {target_timestamp:.9f}: "
        f"{nearest_timestamp:.9f} (diff={diff:.6f}s)"
    )
    return pose_map[nearest_timestamp]


def parse_float_list_from_line(line: str, field_name: str, expected_length: Optional[int] = None) -> np.ndarray:
    match = re.search(r"\[(.*?)\]", line)
    if not match:
        raise ValueError(f"Failed to parse list for {field_name}: {line}")
    values = [float(item.strip()) for item in match.group(1).split(",") if item.strip()]
    if expected_length is not None and len(values) != expected_length:
        raise ValueError(f"{field_name} expected {expected_length} values, got {len(values)}")
    return np.array(values, dtype=np.float64)


def load_camera_calibration(calib_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with calib_path.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Calibration file is empty: {calib_path}")

    camera_name = lines[0].split(":")[0].strip()

    k_line = next((line for line in lines if line.startswith("K:")), None)
    d_line = next((line for line in lines if line.startswith("D:")), None)
    translation_line = next((line for line in lines if line.startswith("translation:")), None)
    rotation_line = next((line for line in lines if line.startswith("rotation:")), None)
    frame_line = next((line for line in lines if line.startswith("frame-id:")), None)
    child_frame_line = next((line for line in lines if line.startswith("child-frame-id:")), None)

    if not all([k_line, d_line, translation_line, rotation_line]):
        raise ValueError(f"Failed to parse required calibration fields from: {calib_path}")

    fx, fy, cx, cy = parse_float_list_from_line(k_line, "K", expected_length=4)
    dist_coeffs = parse_float_list_from_line(d_line, "D")

    translation = parse_float_list_from_line(translation_line, "translation", expected_length=3)
    qx, qy, qz, qw = parse_float_list_from_line(rotation_line, "rotation", expected_length=4)
    quat_wxyz = np.array([qw, qx, qy, qz], dtype=np.float64)
    frame_to_child = make_transform(translation, quat_wxyz)
    extrinsic = choose_direction(
        frame_to_child,
        extrinsic_direction_to_is_forward(CALIBRATION_EXTRINSIC_DIRECTION),
    )

    print(f"[INFO] Loaded camera calibration section: {camera_name}")
    if frame_line or child_frame_line:
        frame_id = frame_line.split(":", 1)[1].strip() if frame_line else "UNKNOWN"
        child_frame_id = child_frame_line.split(":", 1)[1].strip() if child_frame_line else "UNKNOWN"
        print(f"[INFO] Calibration frame chain: {frame_id} -> {child_frame_id}")
    return np.array([fx, fy, cx, cy], dtype=np.float64), dist_coeffs, extrinsic


def load_point_cloud_from_bin(bin_path: Path) -> np.ndarray:
    # Reuse the same loading logic already used in local bin.py:
    # np.fromfile(..., dtype=np.float32).reshape(-1, 3)
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Point cloud is empty: {bin_path}")
    if points.size % 3 != 0:
        raise ValueError(
            f"Point cloud element count ({points.size}) is not divisible by 3. "
            "This does not match the read logic in bin.py."
        )
    points = points.reshape(-1, 3).astype(np.float64)
    if points.shape[0] == 0:
        raise ValueError(f"Point cloud has zero points after reshape: {bin_path}")
    return points


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


def project_camera_points_to_image(
    camera_points: np.ndarray,
    intrinsics: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = intrinsics
    z = camera_points[:, 2]
    valid_depth = z > 0.0
    if not np.any(valid_depth):
        return np.empty((0, 2), dtype=np.int32), valid_depth

    points = camera_points[valid_depth]
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    x, y = distort_normalized_points(x, y, dist_coeffs)

    u = fx * x + cx
    v = fy * y + cy

    inside = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height)
    pixels = np.stack([u[inside], v[inside]], axis=1)
    pixels = np.rint(pixels).astype(np.int32)
    return pixels, valid_depth


def draw_projected_points(image_path: Path, pixels: np.ndarray, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    for u, v in pixels:
        draw.ellipse(
            (u - POINT_RADIUS, v - POINT_RADIUS, u + POINT_RADIUS, v + POINT_RADIUS),
            fill=POINT_COLOR,
            outline=POINT_COLOR,
        )

    image.save(output_path)


def infer_timestamp_from_name(path: Path) -> float:
    # 文件名可能带 "_rgb" 等后缀（如 1623379838.017682788_rgb.png），
    # 只取下划线前的数字前缀作为时间戳。
    stem = path.stem.split("_")[0]
    try:
        return float(stem)
    except ValueError as exc:
        raise ValueError(
            f"Cannot infer timestamp from file name: {path.name}. "
            "Expected the stem to be a numeric timestamp like 1623377796.111739618."
        ) from exc


def main() -> None:
    image_path, bin_path, poses2d_path, poses3d_path, calib_path = resolve_input_paths(PROJECT_ROOT)
    output_path = OUTPUT_IMAGE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] RGB image: {image_path.name}")
    print(f"[INFO] Point cloud: {bin_path.name}")
    print(f"[INFO] poses3d direction: {POSE3D_DIRECTION}")
    print(f"[INFO] poses2d direction: {POSE2D_DIRECTION}")
    print(f"[INFO] camera pose source: {CAMERA_POSE_SOURCE}")
    print(f"[INFO] calibration extrinsic direction: {CALIBRATION_EXTRINSIC_DIRECTION}")
    print(f"[INFO] use distortion: {USE_DISTORTION}")

    if image_path.stem != bin_path.stem:
        print(
            f"[WARN] Image stem ({image_path.stem}) and point cloud stem ({bin_path.stem}) are different. "
            "The script will use the individual file stems to look up poses."
        )

    image_timestamp = infer_timestamp_from_name(image_path)
    pointcloud_timestamp = infer_timestamp_from_name(bin_path)

    poses2d = parse_pose_csv(poses2d_path)
    poses3d = parse_pose_csv(poses3d_path)

    camera_pose = find_pose_by_timestamp(
        poses2d,
        image_timestamp,
        REQUIRE_EXACT_TIMESTAMP_MATCH,
        MAX_TIMESTAMP_DIFF_SEC,
    )
    pointcloud_pose = find_pose_by_timestamp(
        poses3d,
        pointcloud_timestamp,
        REQUIRE_EXACT_TIMESTAMP_MATCH,
        MAX_TIMESTAMP_DIFF_SEC,
    )

    intrinsics, dist_coeffs, base_to_camera = load_camera_calibration(calib_path)
    if not USE_DISTORTION:
        dist_coeffs = np.array([], dtype=np.float64)
    points_lidar = load_point_cloud_from_bin(bin_path)
    validate_camera_pose_source(CAMERA_POSE_SOURCE)

    world_from_lidar = pose_record_to_transform(pointcloud_pose, POSE3D_DIRECTION)
    if CAMERA_POSE_SOURCE == CAMERA_POSE_FROM_POSES2D:
        world_from_camera = pose_record_to_transform(camera_pose, POSE2D_DIRECTION)
    else:
        world_from_base = pose_record_to_transform(camera_pose, POSE2D_DIRECTION)
        world_from_camera = world_from_base @ base_to_camera
    camera_from_world = invert_transform(world_from_camera)

    points_world = transform_points(world_from_lidar, points_lidar)
    points_camera = transform_points(camera_from_world, points_world)

    with Image.open(image_path) as image:
        image_width, image_height = image.size

    pixels, valid_depth_mask = project_camera_points_to_image(
        points_camera,
        intrinsics,
        dist_coeffs,
        image_width,
        image_height,
    )

    if pixels.shape[0] == 0:
        raise RuntimeError(
            "No projected pixels landed inside the image. "
            "Please check the pose direction flags and extrinsic assumptions at the top of the script."
        )

    draw_projected_points(image_path, pixels, output_path)

    print(f"[INFO] Total input points: {points_lidar.shape[0]}")
    print(f"[INFO] Points with positive camera depth: {int(np.count_nonzero(valid_depth_mask))}")
    print(f"[INFO] Points drawn inside image: {pixels.shape[0]}")
    print(f"[INFO] Saved result to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
