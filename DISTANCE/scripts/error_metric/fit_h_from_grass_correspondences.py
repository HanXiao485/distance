from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = PROJECT_ROOT / "outputs" / "grass_correspondences" / "grass_pixel_correspondences.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "grass_correspondences"
PLANE_INLIER_THRESHOLD_M = 0.5


def load_correspondences(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    uv_list = []
    world_list = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            uv_list.append([float(row["u"]), float(row["v"])])
            world_list.append([float(row["world_x"]), float(row["world_y"]), float(row["world_z"])])
    return np.asarray(uv_list, dtype=np.float64), np.asarray(world_list, dtype=np.float64)


def fit_plane(points_world: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    centroid = points_world.mean(axis=0)
    centered = points_world - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
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
    xp = centered @ axis_x
    yp = centered @ axis_y
    return np.column_stack([xp, yp])


def normalize_points_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    centered = points - centroid
    dist = np.sqrt((centered ** 2).sum(axis=1))
    mean_dist = np.maximum(dist.mean(), 1e-12)
    scale = np.sqrt(2.0) / mean_dist
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    points_h = np.column_stack([points, np.ones(len(points))])
    normalized_h = (T @ points_h.T).T
    return normalized_h[:, :2], T


def fit_homography(src_xy: np.ndarray, dst_uv: np.ndarray) -> np.ndarray:
    src_norm, T_src = normalize_points_2d(src_xy)
    dst_norm, T_dst = normalize_points_2d(dst_uv)

    A = []
    for (x, y), (u, v) in zip(src_norm, dst_norm):
        A.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        A.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    A = np.asarray(A, dtype=np.float64)

    _, _, vh = np.linalg.svd(A, full_matrices=False)
    H_norm = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    H = H / H[2, 2]
    return H


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return proj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a plane and H matrix from projected semantic correspondences.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="CSV path produced by the correspondence extractor.")
    parser.add_argument("--tag", type=str, default="grass", help="Readable tag used in the JSON output name.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    tag = str(args.tag)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    uv, world = load_correspondences(csv_path)

    normal, d, residual = fit_plane(world)
    inlier_mask = residual <= PLANE_INLIER_THRESHOLD_M
    uv_inliers = uv[inlier_mask]
    world_inliers = world[inlier_mask]

    centroid = world_inliers.mean(axis=0)
    axis_x, axis_y = choose_plane_basis(normal)
    plane_xy = world_to_plane(world_inliers, centroid, axis_x, axis_y)

    H_plane_to_image = fit_homography(plane_xy, uv_inliers)
    H_image_to_plane = np.linalg.inv(H_plane_to_image)
    H_image_to_plane = H_image_to_plane / H_image_to_plane[2, 2]

    uv_reproj = apply_homography(H_plane_to_image, plane_xy)
    reproj_error = np.linalg.norm(uv_reproj - uv_inliers, axis=1)

    stats = {
        "source_csv": str(csv_path),
        "total_correspondences": int(len(uv)),
        "plane_inlier_threshold_m": float(PLANE_INLIER_THRESHOLD_M),
        "plane_inlier_count": int(inlier_mask.sum()),
        "plane_normal_world": normal.tolist(),
        "plane_d_world": float(d),
        "plane_centroid_world": centroid.tolist(),
        "plane_axis_x_world": axis_x.tolist(),
        "plane_axis_y_world": axis_y.tolist(),
        "plane_residual_m": {
            "mean": float(residual.mean()),
            "p50": float(np.percentile(residual, 50)),
            "p90": float(np.percentile(residual, 90)),
            "p95": float(np.percentile(residual, 95)),
            "p99": float(np.percentile(residual, 99)),
            "max": float(residual.max()),
        },
        "image_reprojection_error_px": {
            "mean": float(reproj_error.mean()),
            "p50": float(np.percentile(reproj_error, 50)),
            "p90": float(np.percentile(reproj_error, 90)),
            "p95": float(np.percentile(reproj_error, 95)),
            "p99": float(np.percentile(reproj_error, 99)),
            "max": float(reproj_error.max()),
        },
        "H_plane_to_image": H_plane_to_image.tolist(),
        "H_image_to_plane": H_image_to_plane.tolist(),
    }

    json_path = OUTPUT_DIR / f"{tag}_h_estimation.json"
    with json_path.open("w") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    print(f"[INFO] Saved H estimation JSON: {json_path}")


if __name__ == "__main__":
    main()
