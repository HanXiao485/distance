"""Stage 4 — uniform scanline sampling and pixel-to-metre physical boundary error."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .homography import apply_homography
from .io_utils import ensure_dir, load_binary_mask, resolve_path, stats


def runs_from_bool_row(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    gaps = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[xs[0], xs[gaps + 1]]
    ends = np.r_[xs[gaps], xs[-1]]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def filtered_boundary_runs(boundary: np.ndarray, y: int, edge_margin_px: int, max_boundary_run_width_px: int,
                           u_min: int = 0, u_max: int | None = None) -> list[tuple[int, int]]:
    width = boundary.shape[1]
    valid_left  = max(edge_margin_px, u_min)
    valid_right = min(width - 1 - edge_margin_px, u_max if u_max is not None else width - 1)
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


def outer_boundary_points(boundary: np.ndarray, y: int, edge_margin_px: int,
                          u_min: int = 0, u_max: int | None = None) -> tuple[float, float] | None:
    """Return the original outermost points inside the configured horizontal ROI.

    ``edge_margin_px`` is intentionally not used to clip points here. Clipping
    before taking min/max can silently replace a rejected image-edge point with
    an unrelated interior boundary point. Edge eligibility is decided later,
    per GT/pred point pair, so rejected points remain available for gray
    visualization without contributing to the error.
    """
    width = boundary.shape[1]
    valid_left = max(0, u_min)
    valid_right = min(width - 1, u_max if u_max is not None else width - 1)
    xs = np.flatnonzero(boundary[y])
    xs = xs[(xs >= valid_left) & (xs <= valid_right)]
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
    u_min: int = 0,
    u_max: int | None = None,
) -> tuple[list[int], dict[str, Any]]:
    top_cap_y = -1
    row_candidates = []
    for y in range(pred_boundary.shape[0]):
        runs = filtered_boundary_runs(pred_boundary, y, edge_margin_px, max_boundary_run_width_px, u_min, u_max)
        if top_cap_y < 0 and len(runs) >= 1:
            top_cap_y = y
        if len(runs) >= 2:
            row_candidates.append(y)
    if not row_candidates:
        return [], {"top_cap_y": top_cap_y, "first_valid_y": -1, "last_valid_y": -1}
    first_valid = row_candidates[0]
    last_valid = row_candidates[-1]
    rows = list(range(first_valid, last_valid + 1, step_px))
    if not rows or rows[-1] != last_valid:
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


def run_midpoints(
    boundary: np.ndarray,
    y: int,
    edge_margin_px: int,
    max_boundary_run_width_px: int,
    u_min: int = 0,
    u_max: int | None = None,
) -> list[float]:
    """Return midpoint x of every valid boundary run on scanline y."""
    runs = filtered_boundary_runs(boundary, y, edge_margin_px, max_boundary_run_width_px, u_min, u_max)
    return [float(l + r) / 2.0 for l, r in runs]


def match_run_midpoints(
    gt_mids: list[float],
    pred_mids: list[float],
    max_match_dist_px: float | None = None,
) -> tuple[list[tuple[float, float]], list[float], list[float]]:
    """Greedily match GT run midpoints to nearest Pred run midpoints (left-to-right).

    Returns:
        matched      : [(gt_x, pred_x), ...]
        unmatched_gt : GT midpoints with no close Pred match
        unmatched_pred: Pred midpoints not matched to any GT
    """
    if not gt_mids or not pred_mids:
        return [], list(gt_mids), list(pred_mids)

    used_pred: set[int] = set()
    matched: list[tuple[float, float]] = []
    unmatched_gt: list[float] = []

    for gt_x in sorted(gt_mids):
        best_dist = float("inf")
        best_j: int | None = None
        for j, pred_x in enumerate(pred_mids):
            if j in used_pred:
                continue
            d = abs(gt_x - pred_x)
            if d < best_dist:
                best_dist, best_j = d, j

        if best_j is not None and (max_match_dist_px is None or best_dist <= max_match_dist_px):
            matched.append((gt_x, pred_mids[best_j]))
            used_pred.add(best_j)
        else:
            unmatched_gt.append(gt_x)

    unmatched_pred = [pred_mids[j] for j in range(len(pred_mids)) if j not in used_pred]
    return matched, unmatched_gt, unmatched_pred


def ground_distance_m(
    H: np.ndarray,
    uv: tuple[float, float],
    cam_plane_xy: np.ndarray | None = None,
) -> float:
    """Project image pixel to ground plane; return distance from camera foot.

    cam_plane_xy: camera origin projected onto the ground plane (in plane_xy coords).
    If None (old behaviour), measures from the plane coordinate origin (dirt centroid).
    """
    plane = apply_homography(H, np.asarray([uv], dtype=np.float64))
    ref = cam_plane_xy if cam_plane_xy is not None else np.zeros(2)
    return float(np.linalg.norm(plane[0] - ref))


def wprime(H: np.ndarray, u: float, v: float) -> float:
    """透视分母 w' = h20·u + h21·v + h22。

    w' 趋近于 0 时,该像素投影到消失线附近,H 投影数值上爆炸。
    |w'| 越大,投影越稳定。用 min|w'| 阈值过滤不可靠扫描线,
    比固定地面距离更自适应:阈值含义与具体 H 矩阵和数据集无关。
    """
    return float(H[2, 0] * u + H[2, 1] * v + H[2, 2])


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
    excluded_samples: list[dict[str, float]] | None = None,
    excluded_pairs: list[dict[str, float]] | None = None,
) -> None:
    image = base_image.convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, _ = image.size
    for row in (excluded_samples or []):
        y = int(row["y"])
        gray = (145, 145, 145, 190)
        draw.line((0, y, width - 1, y), fill=gray, width=line_width)
        draw.line((row["gt_left_x"], y, row["gt_right_x"], y), fill=gray, width=line_width + 1)
        draw.line((row["pred_left_x"], y, row["pred_right_x"], y), fill=gray, width=line_width + 1)
        for key in ("gt_left_x", "gt_right_x", "pred_left_x", "pred_right_x"):
            x = float(row[key])
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=(145, 145, 145, 230), outline=(225, 225, 225, 230), width=2,
            )
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
    for pair in (excluded_pairs or []):
        y = int(pair["y"])
        gt_x = float(pair["gt_x"])
        pred_x = float(pair["pred_x"])
        draw.line((gt_x, y, pred_x, y), fill=(145, 145, 145, 230), width=line_width + 1)
        for x in (gt_x, pred_x):
            draw.ellipse(
                (x - point_radius, y - point_radius, x + point_radius, y + point_radius),
                fill=(145, 145, 145, 230), outline=(225, 225, 225, 230), width=2,
            )
    ensure_dir(output_path.parent)
    Image.alpha_composite(image, overlay).save(output_path)


def scanline_error(
    config: dict[str, Any], h_json: Path, gt_boundary_path: Path, pred_boundary_path: Path, output_dir: Path,
    precomputed_boundaries: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    """precomputed_boundaries 若提供，直接用 create_boundaries 返回的内存数组
    (gt_boundary, pred_boundary)，跳过重新从 gt_boundary_path/pred_boundary_path
    读盘（PNG是无损格式，读回来的数组跟内存里的原始数组逐像素相同）。"""
    class_name = config["class"]["name"]
    scan_cfg = config.get("scanline", {})
    if precomputed_boundaries is not None:
        gt_boundary, pred_boundary = precomputed_boundaries
    else:
        gt_boundary = load_binary_mask(gt_boundary_path)
        pred_boundary = load_binary_mask(pred_boundary_path)
    with h_json.open() as f:
        _h_data = json.load(f)
    H_image_to_plane = np.asarray(_h_data["H_image_to_plane"], dtype=np.float64)
    _cam_plane_raw = _h_data.get("camera_plane_xy_m")
    cam_plane_xy = np.array(_cam_plane_raw, dtype=np.float64) if _cam_plane_raw is not None else None
    step_px = int(scan_cfg.get("step_px", 10))
    edge_margin_px = int(scan_cfg.get("edge_margin_px", 0))
    max_boundary_run_width_px = int(scan_cfg.get("max_boundary_run_width_px", 25))
    u_min = int(scan_cfg.get("u_min", 0))
    u_max = scan_cfg.get("u_max", None)
    u_max = int(u_max) if u_max is not None else None
    # 有效采样区域上界（米）:4个边界点中任一投影地面距离超过此值则整条跳过。
    # 同时过滤远场不稳定行（接近消失线）和边界贴近图像边缘的横向投影虚大行。
    # 0 表示不启用。
    max_ground_distance_m = float(scan_cfg.get("max_ground_distance_m", 0.0))

    multi_run = bool(scan_cfg.get("multi_run_matching", False))
    max_match_dist_px = scan_cfg.get("max_match_dist_px", None)
    if max_match_dist_px is not None:
        max_match_dist_px = float(max_match_dist_px)
    # GT 边缘过滤：图像最外侧像素始终跳过；正值会进一步扩大 GT 边缘排除范围
    gt_edge_margin_px = int(scan_cfg.get("gt_edge_margin_px", 0))
    image_width = gt_boundary.shape[1]

    rows, row_limits = uniform_scanline_rows(
        pred_boundary, step_px, edge_margin_px, max_boundary_run_width_px,
        u_min, u_max,
    )
    samples = []
    excluded_visual_samples = []
    excluded_visual_pairs = []
    left_errors: list[float] = []
    right_errors: list[float] = []
    all_matched_errors: list[float] = []
    skipped = {"missing_gt_outer_points": 0, "missing_pred_outer_points": 0,
               "beyond_max_ground_distance": 0}
    unmatched_gt_total = 0
    unmatched_pred_total = 0

    for y in rows:
        if not multi_run:
            # ── 原始方法：只取最左和最右边界点 ─────────────────────────────
            gt_points = outer_boundary_points(gt_boundary, y, edge_margin_px, u_min, u_max)
            pred_points = outer_boundary_points(pred_boundary, y, edge_margin_px, u_min, u_max)
            if gt_points is None:
                skipped["missing_gt_outer_points"] += 1
                continue
            if pred_points is None:
                skipped["missing_pred_outer_points"] += 1
                continue
            gt_left, gt_right = gt_points
            pred_left, pred_right = pred_points

            if max_ground_distance_m > 0.0:
                dists = [
                    ground_distance_m(H_image_to_plane, (gt_left,   y), cam_plane_xy),
                    ground_distance_m(H_image_to_plane, (gt_right,  y), cam_plane_xy),
                    ground_distance_m(H_image_to_plane, (pred_left, y), cam_plane_xy),
                    ground_distance_m(H_image_to_plane, (pred_right,y), cam_plane_xy),
                ]
                if max(dists) > max_ground_distance_m:
                    skipped["beyond_max_ground_distance"] += 1
                    excluded_visual_samples.append({
                        "y": int(y),
                        "gt_left_x": float(gt_left), "gt_right_x": float(gt_right),
                        "pred_left_x": float(pred_left), "pred_right_x": float(pred_right),
                    })
                    continue

            # GT 边缘过滤：GT 点贴着图像边界 → 跳过该侧
            edge_right_x = image_width - 1 - edge_margin_px
            gt_right_x = image_width - gt_edge_margin_px if gt_edge_margin_px > 0 else image_width - 1
            do_left = (
                gt_left >= edge_margin_px
                and pred_left >= edge_margin_px
                and gt_left > gt_edge_margin_px
            )
            do_right = (
                gt_right <= edge_right_x
                and pred_right <= edge_right_x
                and gt_right < gt_right_x
            )
            if not do_left:
                excluded_visual_pairs.append({"y": int(y), "gt_x": gt_left, "pred_x": pred_left})
            if not do_right:
                excluded_visual_pairs.append({"y": int(y), "gt_x": gt_right, "pred_x": pred_right})
            if not do_left and not do_right:
                skipped["missing_gt_outer_points"] += 1
                continue

            left_error  = distance_m(H_image_to_plane, (gt_left,  y), (pred_left,  y)) if do_left  else None
            right_error = distance_m(H_image_to_plane, (gt_right, y), (pred_right, y)) if do_right else None
            if left_error  is not None: left_errors.append(left_error);   all_matched_errors.append(left_error)
            if right_error is not None: right_errors.append(right_error); all_matched_errors.append(right_error)
            samples.append({
                "y": int(y),
                "gt_left_x": float(gt_left), "gt_right_x": float(gt_right),
                "pred_left_x": float(pred_left), "pred_right_x": float(pred_right),
                "left_error_m": left_error, "right_error_m": right_error,
            })

        else:
            # ── 改进方法：所有 runs 按近邻匹配 ──────────────────────────────
            gt_mids   = run_midpoints(gt_boundary,   y, edge_margin_px, max_boundary_run_width_px, u_min, u_max)
            pred_mids = run_midpoints(pred_boundary, y, edge_margin_px, max_boundary_run_width_px, u_min, u_max)

            if not gt_mids:
                skipped["missing_gt_outer_points"] += 1
                continue
            if not pred_mids:
                skipped["missing_pred_outer_points"] += 1
                continue

            matched, unmatched_gt, unmatched_pred = match_run_midpoints(
                gt_mids, pred_mids, max_match_dist_px
            )
            unmatched_gt_total   += len(unmatched_gt)
            unmatched_pred_total += len(unmatched_pred)

            row_has_sample = False
            gt_left_for_viz  = min(gt_mids)
            gt_right_for_viz = max(gt_mids)
            pr_left_for_viz  = min(pred_mids)
            pr_right_for_viz = max(pred_mids)

            for gt_x, pred_x in matched:
                if max_ground_distance_m > 0.0:
                    dists = [
                        ground_distance_m(H_image_to_plane, (gt_x,   y), cam_plane_xy),
                        ground_distance_m(H_image_to_plane, (pred_x, y), cam_plane_xy),
                    ]
                    if max(dists) > max_ground_distance_m:
                        skipped["beyond_max_ground_distance"] += 1
                        continue

                err = distance_m(H_image_to_plane, (gt_x, y), (pred_x, y))
                all_matched_errors.append(err)
                # 保持与旧方法兼容：最左配对 → left_errors，最右配对 → right_errors
                if gt_x == min(gt_mids):
                    left_errors.append(err)
                elif gt_x == max(gt_mids):
                    right_errors.append(err)
                row_has_sample = True

            if row_has_sample:
                samples.append({
                    "y": int(y),
                    "gt_left_x":   gt_left_for_viz,
                    "gt_right_x":  gt_right_for_viz,
                    "pred_left_x": pr_left_for_viz,
                    "pred_right_x":pr_right_for_viz,
                    "left_error_m":  left_errors[-1] if left_errors else 0.0,
                    "right_error_m": right_errors[-1] if right_errors else 0.0,
                    "n_gt_runs":    len(gt_mids),
                    "n_pred_runs":  len(pred_mids),
                    "n_matched":    len(matched),
                    "n_unmatched_gt":   len(unmatched_gt),
                    "n_unmatched_pred": len(unmatched_pred),
                })
    scan_dir = output_dir / "scanline"
    ensure_dir(scan_dir)
    csv_path = scan_dir / f"{class_name}_scanline_samples.csv"
    base_fields = ["y", "gt_left_x", "gt_right_x", "pred_left_x", "pred_right_x", "left_error_m", "right_error_m"]
    extra_fields = [k for k in (samples[0].keys() if samples else []) if k not in base_fields]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + extra_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(samples)
    rgb_path = resolve_path(config, config["paths"]["rgb"])
    rgb_overlay_path = scan_dir / f"{class_name}_scanline_rgb_overlay.png"
    boundary_overlay_path = scan_dir / f"{class_name}_scanline_boundary_overlay.png"
    # 跟 h_diagnostics.enabled 一样是纯可视化产物（画扫描线采样点叠加图），
    # profile 显示这两张全分辨率PNG单帧就要占大约0.7秒，用 scanline.save_overlays
    # 开关控制，默认保留原行为(True)，追求速度时可以关掉，不影响任何数值结果。
    if scan_cfg.get("save_overlays", True):
        if rgb_path.exists():
            rgb_canvas = Image.open(rgb_path).convert("RGBA")
        else:
            rgb_canvas = Image.new("RGBA", (gt_boundary.shape[1], gt_boundary.shape[0]), (20, 20, 20, 255))
        draw_samples_on_image(
            rgb_canvas, samples, rgb_overlay_path,
            int(scan_cfg.get("line_width", 4)),
            int(scan_cfg.get("point_radius", 7)),
            excluded_visual_samples,
            excluded_visual_pairs,
        )
        draw_samples_on_image(
            make_boundary_canvas(gt_boundary, pred_boundary), samples, boundary_overlay_path,
            int(scan_cfg.get("line_width", 4)),
            int(scan_cfg.get("point_radius", 7)),
            excluded_visual_samples,
            excluded_visual_pairs,
        )
    return {
        "method": "uniform_scanline_multirun_matching" if multi_run else "uniform_scanline_outermost_boundary_error",
        "multi_run_matching": multi_run,
        "control_variable": "uniform_y_scanlines_shared_by_gt_and_pred",
        "scanline_step_px": step_px,
        "edge_margin_px": edge_margin_px,
        "max_boundary_run_width_px": max_boundary_run_width_px,
        "max_ground_distance_m": max_ground_distance_m,
        "gt_edge_margin_px": gt_edge_margin_px,
        "valid_row_rule": row_limits,
        "candidate_scanline_count": int(len(rows)),
        "sampled_scanline_count": int(len(samples)),
        "skipped_scanlines": skipped,
        "unmatched_gt_runs": unmatched_gt_total,
        "unmatched_pred_runs": unmatched_pred_total,
        "sampled_point_count": int(len(all_matched_errors)),
        "left_error_m": stats(left_errors),
        "right_error_m": stats(right_errors),
        "overall_error_m": stats(all_matched_errors),
        "paths": {
            "samples_csv": str(csv_path),
            "rgb_overlay_png": str(rgb_overlay_path),
            "boundary_overlay_png": str(boundary_overlay_path),
        },
    }
