from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIN_COMPONENT_AREA_PX = 5000
CLOSE_ITERATIONS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute physical boundary error between GT and Pred masks.")
    parser.add_argument("--mask-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "mask_dirt_preview",
                        help="Directory containing gt_dirt_binary.png and pred_dirt_binary.png.")
    parser.add_argument("--h-json", type=Path,
                        default=PROJECT_ROOT / "outputs" / "best_h" / "h_strategy_o_distorted.json",
                        help="JSON file containing H_image_to_plane matrix.")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT_ROOT / "outputs" / "boundary_physical_error",
                        help="Output directory.")
    return parser.parse_args()


def load_binary_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    return arr > 0


def binary_erode_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = []
    for dy in range(3):
        for dx in range(3):
            neighbors.append(padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]])
    stacked = np.stack(neighbors, axis=0)
    return np.all(stacked, axis=0)


def binary_dilate_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = []
    for dy in range(3):
        for dx in range(3):
            neighbors.append(padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]])
    stacked = np.stack(neighbors, axis=0)
    return np.any(stacked, axis=0)


def binary_close_3x3(mask: np.ndarray, iterations: int) -> np.ndarray:
    closed = mask.copy()
    for _ in range(iterations):
        closed = binary_dilate_3x3(closed)
    for _ in range(iterations):
        closed = binary_erode_3x3(closed)
    return closed


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


def clean_prediction_mask(mask: np.ndarray) -> np.ndarray:
    large_components = keep_large_components(mask, MIN_COMPONENT_AREA_PX)
    closed = binary_close_3x3(large_components, CLOSE_ITERATIONS)
    return keep_large_components(closed, MIN_COMPONENT_AREA_PX)


def boundary_points_from_mask(boundary_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(boundary_mask)
    return np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])


def apply_homography(H: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    points_h = np.column_stack([points_uv, np.ones(len(points_uv), dtype=np.float64)])
    mapped_h = (H @ points_h.T).T
    denom = mapped_h[:, 2:3]
    valid = np.abs(denom[:, 0]) > 1e-9
    mapped = np.empty((len(points_uv), 2), dtype=np.float64)
    mapped[:] = np.nan
    mapped[valid] = mapped_h[valid, :2] / denom[valid]
    return mapped


def min_distances_chunked(src: np.ndarray, dst: np.ndarray, chunk_size: int = 512) -> np.ndarray:
    if len(src) == 0 or len(dst) == 0:
        return np.array([], dtype=np.float64)
    mins = np.empty(len(src), dtype=np.float64)
    for start in range(0, len(src), chunk_size):
        end = min(start + chunk_size, len(src))
        chunk = src[start:end]
        diff = chunk[:, None, :] - dst[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        mins[start:end] = np.sqrt(np.min(dist2, axis=1))
    return mins


def stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def save_binary(mask: np.ndarray, path: Path) -> None:
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def save_overlap(gt_boundary: np.ndarray, pred_boundary: np.ndarray, path: Path) -> None:
    canvas = np.zeros((*gt_boundary.shape, 3), dtype=np.uint8)
    canvas[gt_boundary & pred_boundary] = [0, 220, 80]
    canvas[gt_boundary & ~pred_boundary] = [255, 80, 80]
    canvas[~gt_boundary & pred_boundary] = [60, 130, 255]
    Image.fromarray(canvas, mode="RGB").save(path)


def main() -> None:
    args = parse_args()
    MASK_DIR = args.mask_dir
    H_JSON_PATH = args.h_json
    OUTPUT_DIR = args.output_dir

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt_mask = load_binary_mask(MASK_DIR / "gt_dirt_binary.png")
    pred_mask = load_binary_mask(MASK_DIR / "pred_dirt_binary.png")
    pred_mask_cleaned = clean_prediction_mask(pred_mask)

    gt_boundary = extract_external_boundary(gt_mask)
    pred_boundary = extract_external_boundary(pred_mask_cleaned)
    save_binary(pred_mask_cleaned, OUTPUT_DIR / "pred_dirt_cleaned_binary.png")
    save_binary(gt_boundary, OUTPUT_DIR / "gt_dirt_boundary.png")
    save_binary(pred_boundary, OUTPUT_DIR / "pred_dirt_boundary.png")
    save_overlap(gt_boundary, pred_boundary, OUTPUT_DIR / "gt_pred_boundary_overlap.png")

    gt_uv = boundary_points_from_mask(gt_boundary)
    pred_uv = boundary_points_from_mask(pred_boundary)

    with H_JSON_PATH.open() as f:
        h_data = json.load(f)
    H_image_to_plane = np.asarray(h_data["H_image_to_plane"], dtype=np.float64)

    gt_plane = apply_homography(H_image_to_plane, gt_uv)
    pred_plane = apply_homography(H_image_to_plane, pred_uv)
    gt_plane = gt_plane[np.isfinite(gt_plane).all(axis=1)]
    pred_plane = pred_plane[np.isfinite(pred_plane).all(axis=1)]

    pred_to_gt = min_distances_chunked(pred_plane, gt_plane)
    gt_to_pred = min_distances_chunked(gt_plane, pred_plane)
    chamfer = float(np.mean(pred_to_gt) + np.mean(gt_to_pred)) if len(pred_to_gt) and len(gt_to_pred) else float("nan")

    result = {
        "gt_mask_pixels": int(gt_mask.sum()),
        "pred_mask_pixels": int(pred_mask.sum()),
        "pred_mask_cleaned_pixels": int(pred_mask_cleaned.sum()),
        "pred_mask_removed_noise_pixels": int(pred_mask.sum() - (pred_mask & pred_mask_cleaned).sum()),
        "pred_mask_filled_hole_pixels": int(pred_mask_cleaned.sum() - (pred_mask & pred_mask_cleaned).sum()),
        "pred_cleaning": {
            "method": "keep_large_components_then_3x3_binary_closing_external_boundary_only",
            "min_component_area_px": int(MIN_COMPONENT_AREA_PX),
            "close_iterations": int(CLOSE_ITERATIONS),
            "boundary_type": "external_boundary_connected_to_image_outside",
        },
        "gt_boundary_pixels": int(gt_boundary.sum()),
        "pred_boundary_pixels": int(pred_boundary.sum()),
        "gt_boundary_point_count": int(len(gt_plane)),
        "pred_boundary_point_count": int(len(pred_plane)),
        "pred_to_gt_m": stats(pred_to_gt),
        "gt_to_pred_m": stats(gt_to_pred),
        "symmetric": {
            "mean_of_means_m": float((np.mean(pred_to_gt) + np.mean(gt_to_pred)) / 2.0),
            "median_of_medians_m": float((np.median(pred_to_gt) + np.median(gt_to_pred)) / 2.0),
            "p95_avg_m": float((np.percentile(pred_to_gt, 95) + np.percentile(gt_to_pred, 95)) / 2.0),
            "chamfer_m": chamfer,
        },
        "paths": {
            "pred_cleaned_mask_png": str(OUTPUT_DIR / "pred_dirt_cleaned_binary.png"),
            "gt_boundary_png": str(OUTPUT_DIR / "gt_dirt_boundary.png"),
            "pred_boundary_png": str(OUTPUT_DIR / "pred_dirt_boundary.png"),
            "overlap_png": str(OUTPUT_DIR / "gt_pred_boundary_overlap.png"),
        },
    }

    out_json = OUTPUT_DIR / "dirt_boundary_physical_error.json"
    with out_json.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"[INFO] Saved metrics: {out_json}")


if __name__ == "__main__":
    main()
