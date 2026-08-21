from pathlib import Path

import numpy as np


# =========================
# User configuration
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Set to None to auto-pick the first supported file.
# You can also set it explicitly, for example:
# POINTCLOUD_FILE_NAME = "1623377796.111739618.bin"
# POINTCLOUD_FILE_NAME = "1623377796.111739618.ply"
POINTCLOUD_PATH = PROJECT_ROOT / "raw" / "current_frame" / "1623379838.017682788.bin"

# Viewer controls
MAX_POINTS_TO_SHOW = 0
RANDOM_SEED = 42
POINT_SIZE = 2.0
SHOW_COORDINATE_FRAME = True
COORDINATE_FRAME_SIZE = 3.0
BACKGROUND_COLOR = np.array([0.05, 0.05, 0.05], dtype=np.float64)

SUPPORTED_SUFFIXES = (".bin", ".ply")


def find_first_pointcloud_file(base_dir: Path) -> Path:
    for suffix in SUPPORTED_SUFFIXES:
        for path in sorted(base_dir.iterdir()):
            if path.is_file() and path.suffix.lower() == suffix:
                return path
    raise FileNotFoundError(f"No supported point cloud file found in {base_dir}")


def resolve_pointcloud_path(base_dir: Path) -> Path:
    pointcloud_path = POINTCLOUD_PATH if POINTCLOUD_PATH else find_first_pointcloud_file(base_dir)
    if not pointcloud_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {pointcloud_path}")
    if pointcloud_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported point cloud suffix: {pointcloud_path.suffix}")
    return pointcloud_path


def load_bin_point_cloud(bin_path: Path) -> np.ndarray:
    # Reuse the same local logic from bin.py.
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Point cloud is empty: {bin_path}")
    if points.size % 3 != 0:
        raise ValueError(
            f"Point cloud element count ({points.size}) is not divisible by 3. "
            "This does not match the local bin.py read logic."
        )
    return points.reshape(-1, 3).astype(np.float64)


def load_ply_point_cloud(ply_path: Path) -> np.ndarray:
    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    if point_cloud.is_empty():
        raise ValueError(f"Loaded PLY point cloud is empty: {ply_path}")
    return np.asarray(point_cloud.points, dtype=np.float64)


def load_point_cloud_auto(pointcloud_path: Path) -> np.ndarray:
    suffix = pointcloud_path.suffix.lower()
    if suffix == ".bin":
        return load_bin_point_cloud(pointcloud_path)
    if suffix == ".ply":
        return load_ply_point_cloud(pointcloud_path)
    raise ValueError(f"Unsupported point cloud suffix: {pointcloud_path.suffix}")


def maybe_downsample(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points

    rng = np.random.default_rng(seed)
    indices = rng.choice(points.shape[0], size=max_points, replace=False)
    sampled = points[indices]
    print(f"[INFO] Randomly sampled {sampled.shape[0]} / {points.shape[0]} points for viewing.")
    return sampled


def main() -> None:
    try:
        import open3d as o3d
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "open3d is not installed in the current environment. "
            "Please install it first, then rerun this script."
        ) from exc

    pointcloud_path = resolve_pointcloud_path(PROJECT_ROOT)
    print(f"[INFO] Loading point cloud: {pointcloud_path}")

    points = load_point_cloud_auto(pointcloud_path)
    print(f"[INFO] Total points: {points.shape[0]}")
    print(f"[INFO] Min:  {points.min(axis=0)}")
    print(f"[INFO] Max:  {points.max(axis=0)}")
    print(f"[INFO] Mean: {points.mean(axis=0)}")

    view_points = maybe_downsample(points, MAX_POINTS_TO_SHOW, RANDOM_SEED)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(view_points)

    geometries = [point_cloud]
    if SHOW_COORDINATE_FRAME:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=COORDINATE_FRAME_SIZE)
        geometries.append(frame)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Open3D Viewer - {pointcloud_path.name}", width=1600, height=1000)
    for geometry in geometries:
        vis.add_geometry(geometry)

    render_option = vis.get_render_option()
    render_option.point_size = POINT_SIZE
    render_option.background_color = BACKGROUND_COLOR

    print("[INFO] Controls: left-drag rotate, right-drag pan, mouse-wheel zoom, R reset view.")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
