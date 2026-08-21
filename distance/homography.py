"""Stage 2 — H-matrix (ground-plane homography) estimation.

Projects labelled LiDAR points onto the image, fits a ground-plane homography H,
and reports reprojection-error diagnostics.
"""
from __future__ import annotations

import colorsys
import csv
import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .io_utils import (
    POSE_DIRECTION_SENSOR_TO_WORLD,
    ensure_dir,
    find_pose_by_timestamp,
    infer_timestamp_from_name,
    invert_transform,
    load_3d_labels,
    load_camera_calibration,
    load_point_cloud,
    parse_pose_csv,
    pose_record_to_transform,
    resolve_path,
    stats,
    stats_extended,
    transform_points,
)


def _hsv_full_sv_to_rgb(hue: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S=1, V=1 时的 HSV->RGB 向量化实现，逐元素与 colorsys.hsv_to_rgb(h,1,1) 完全一致
    （已用 5000 组随机 hue 验证浮点/取整后结果零误差）。hue 取值范围 [0, 1)。"""
    hp = hue * 6.0
    i = (np.floor(hp).astype(np.int64)) % 6
    f = hp - np.floor(hp)
    q = 1.0 - f
    r = np.zeros_like(hue); g = np.zeros_like(hue); b = np.zeros_like(hue)
    m0 = i == 0; r[m0] = 1; g[m0] = f[m0]; b[m0] = 0
    m1 = i == 1; r[m1] = q[m1]; g[m1] = 1; b[m1] = 0
    m2 = i == 2; r[m2] = 0; g[m2] = 1; b[m2] = f[m2]
    m3 = i == 3; r[m3] = 0; g[m3] = q[m3]; b[m3] = 1
    m4 = i == 4; r[m4] = f[m4]; g[m4] = 0; b[m4] = 1
    m5 = i == 5; r[m5] = 1; g[m5] = 0; b[m5] = q[m5]
    return r, g, b


def _error_to_rgb_array(errors: np.ndarray, vmax: float | None = None) -> np.ndarray:
    if errors.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    effective_vmax = float(vmax) if (vmax is not None and vmax > 0) else float(np.percentile(errors, 95))
    if effective_vmax <= 0:
        effective_vmax = 1.0
    normalized = np.clip(errors / effective_vmax, 0.0, 1.0)
    hue = (1.0 - normalized) * 240.0 / 360.0  # blue (240°) → red (0°) as error increases
    r, g, b = _hsv_full_sv_to_rgb(hue)
    rgb = np.stack([r, g, b], axis=1) * 255.0
    return rgb.astype(np.uint8)


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


@functools.lru_cache(maxsize=16)
def _prepare_frame_world_points(
    pointcloud_path: Path,
    label_path: Path,
    label3d_dtype: str,
    label_id: int,
    poses3d_path: Path,
    pose3d_direction: str,
    max_timestamp_diff_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """H候选搜索里，同一个锚点帧的点云/3D标签/目标类别过滤/世界系投影，
    在所有候选图像之间是完全不变的（只有相机自己的位姿会变），
    按锚点帧的关键路径缓存，避免每个候选都重新读盘+重新算一遍。

    返回: (target_points_lidar, points_world, target_indices, total_point_count)
    """
    points_lidar = load_point_cloud(pointcloud_path)
    labels = load_3d_labels(label_path, label3d_dtype)
    if labels.shape[0] != points_lidar.shape[0]:
        raise ValueError(f"Label count {labels.shape[0]} does not match point count {points_lidar.shape[0]}")

    target_mask = labels == label_id
    target_points_lidar = points_lidar[target_mask]
    target_indices = np.flatnonzero(target_mask)

    poses3d = parse_pose_csv(poses3d_path)
    pointcloud_pose = find_pose_by_timestamp(
        poses3d, infer_timestamp_from_name(pointcloud_path), max_timestamp_diff_sec)
    world_from_lidar = pose_record_to_transform(pointcloud_pose, pose3d_direction)
    points_world = transform_points(world_from_lidar, target_points_lidar)

    return target_points_lidar, points_world, target_indices, int(points_lidar.shape[0])


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
    """每个像素 (int(u), int(v)) 只保留深度最小的点；深度并列时保留最先出现的索引。

    向量化实现，与原逐点字典版本逐点等价（已用 200 组含碰撞/并列的随机用例验证），
    在 ~10 万点规模下从 ~50ms 降到 ~13ms。pixels 要求非负整数坐标（来自
    project_camera_points 的输出恰好满足：inside 掩码已保证 u>=0, v>=0）。
    """
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


def draw_points(image_path: Path, pixels: np.ndarray, output_path: Path, color: tuple[int, int, int]) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for u, v in pixels:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=color, outline=color)
    ensure_dir(output_path.parent)
    image.save(output_path)


def extract_correspondences(
    config: dict[str, Any], output_dir: Path
) -> tuple[Path, dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """返回值第三项是 (uv, world, cam, lidar_z) 内存数组，跟 load_correspondences_full
    从 CSV 读出来的格式/dtype完全一致，供 estimate_h 直接使用、跳过磁盘往返。"""
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

    label_id = int(class_cfg["id_3d_for_h"])
    max_diff = float(geom_cfg.get("max_timestamp_diff_sec", 0.05))

    # 点云加载 + 3D标签加载 + 目标类别过滤 + 投影到世界系：这部分只取决于锚点帧自己的
    # 点云/位姿，跟当前候选图像无关，H候选搜索时同一帧的所有候选都会命中缓存。
    target_points_lidar, points_world, target_indices, total_point_count = _prepare_frame_world_points(
        pointcloud_path, label_path, config.get("label3d_dtype", "uint32"), label_id,
        poses3d_path, geom_cfg.get("pose3d_direction", POSE_DIRECTION_SENSOR_TO_WORLD), max_diff,
    )

    poses2d = parse_pose_csv(poses2d_path)
    camera_pose = find_pose_by_timestamp(poses2d, infer_timestamp_from_name(image_path), max_diff)
    world_from_camera = pose_record_to_transform(camera_pose, geom_cfg.get("pose2d_direction", POSE_DIRECTION_SENSOR_TO_WORLD))
    camera_from_world = invert_transform(world_from_camera)

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

    # CSV 落盘同样只是留痕用的诊断产物：estimate_h 在同一进程里紧接着就会把这份文件
    # 整个读回内存（load_correspondences_full），profile 显示这一写一读的磁盘往返
    # 占了 Phase1 总耗时的 66%。跟 overlay PNG 共用 h_diagnostics.enabled 开关，
    # 关掉时直接跳过写盘，把内存里的数组通过返回值传给调用方即可。
    save_csv = config.get("h_diagnostics", {}).get("enabled", True)
    if save_csv:
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

    # overlay PNG 属于纯诊断可视化产物，跟 h_diagnostics.enabled 共用同一个开关。
    # H候选搜索时同一帧要测试多个候选，每个候选都写一张全分辨率PNG是最大的性能瓶颈
    # （profile 显示 PNG 编码占了单帧总耗时的 70%+），关掉后不影响任何数值计算结果。
    if config.get("h_diagnostics", {}).get("enabled", True):
        draw_points(image_path, visible_pixels, overlay_path, tuple(config.get("h_overlay_color", [0, 255, 0])))

    in_memory = (
        visible_pixels.astype(np.float64),
        visible_points_world,
        visible_points_camera,
        visible_points_lidar[:, 2],
    )
    return output_csv, {
        "total_points": int(total_point_count),
        "target_label_points": int(target_points_lidar.shape[0]),
        "positive_depth_points": int(np.count_nonzero(valid_depth_mask)),
        "inside_image_points": int(inside_points_lidar.shape[0]),
        "visible_correspondences": int(visible_pixels.shape[0]),
        "overlay_png": str(overlay_path),
    }, in_memory


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
    # 向量化构建 DLT 矩阵 A，等价于原来逐点 Python 循环 append 两行。
    x, y = src_norm[:, 0], src_norm[:, 1]
    u, v = dst_norm[:, 0], dst_norm[:, 1]
    n = len(x)
    zeros = np.zeros(n)
    ones_neg = -np.ones(n)
    row1 = np.column_stack([-x, -y, ones_neg, zeros, zeros, zeros, u * x, u * y, u])
    row2 = np.column_stack([zeros, zeros, zeros, -x, -y, ones_neg, v * x, v * y, v])
    A = np.empty((2 * n, 9), dtype=np.float64)
    A[0::2] = row1
    A[1::2] = row2
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    H_norm = vh[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_norm @ T_src
    return H / H[2, 2]


def apply_homography(H: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points_uv, np.ones(len(points_uv), dtype=np.float64)])
    mapped_h = (H @ points_h.T).T
    return mapped_h[:, :2] / mapped_h[:, 2:3]


def estimate_h(
    config: dict[str, Any], correspondence_csv: Path, output_dir: Path,
    precomputed: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """precomputed 若提供，直接用 extract_correspondences 返回的内存数组
    (uv, world, cam, lidar_z)，跳过重新读取 correspondence_csv 文件——两者数值
    完全等价（Python 的 csv 往返对 float64 是无损的），但省掉了磁盘I/O
    （profile 显示 H候选搜索里这一读一写占了 Phase1 总耗时的66%）。"""
    class_name = config["class"]["name"]
    h_cfg = config.get("h_estimation", {})
    diag_cfg = config.get("h_diagnostics", {})
    plane_inlier_threshold_m = float(h_cfg.get("plane_inlier_threshold_m", 0.5))

    if precomputed is not None:
        uv, world, cam, lidar_z = precomputed
    else:
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

    # Recover camera world position via Umeyama rigid-body estimation.
    # cam_inliers (camera frame) ↔ world_inliers (world frame): world = R_cw @ cam + t_cw
    # Camera origin in world frame = t_cw (when cam = 0).
    _cam_c   = cam_inliers.mean(axis=0)
    _world_c = world_inliers.mean(axis=0)
    _H_cov   = (cam_inliers - _cam_c).T @ (world_inliers - _world_c)
    _U, _S, _Vt = np.linalg.svd(_H_cov)
    _det_sign = np.linalg.det(_Vt.T @ _U.T)
    _R_cw    = _Vt.T @ np.diag([1.0, 1.0, _det_sign]) @ _U.T
    camera_world = _world_c - _R_cw @ _cam_c       # camera origin in world frame
    # Coplanar LiDAR points leave the out-of-plane rotation ambiguous: camera must be
    # above the fitted ground plane (positive dot with outward normal).  If SVD chose
    # the wrong hemisphere, reflect the camera position across the plane.
    _plane_normal_unit = normal / np.linalg.norm(normal)
    _height = float(np.dot(camera_world - centroid, _plane_normal_unit))
    if _height < 0:
        camera_world = camera_world - 2.0 * _height * _plane_normal_unit
    camera_plane_xy = world_to_plane(                # camera foot on ground plane
        camera_world.reshape(1, 3), centroid, axis_x, axis_y
    )[0]

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
        "camera_world_m": camera_world.tolist(),
        "camera_plane_xy_m": camera_plane_xy.tolist(),
    }
    if diag_enabled and diag:
        result["diagnostics"] = diag

    json_path = output_dir / "h_estimation" / f"{class_name}_h_estimation.json"
    ensure_dir(json_path.parent)
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)
    return json_path, result
