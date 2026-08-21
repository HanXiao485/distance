from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOUNDARY_DIR = PROJECT_ROOT / "outputs" / "boundary_physical_error"
DEFAULT_H_JSON_PATH = PROJECT_ROOT / "outputs" / "best_h" / "h_strategy_o_distorted.json"
DEFAULT_RGB_PATH = PROJECT_ROOT / "raw" / "current_frame" / "1623379838.017682788_rgb.png"
DEFAULT_GT_BOUNDARY_PATH = DEFAULT_BOUNDARY_DIR / "gt_dirt_boundary.png"
DEFAULT_PRED_BOUNDARY_PATH = DEFAULT_BOUNDARY_DIR / "pred_dirt_boundary.png"

GT_COLOR = (255, 70, 70, 240)
PRED_COLOR = (40, 140, 255, 240)
CONTROL_LINE_COLOR = (255, 230, 80, 210)
GT_SEGMENT_COLOR = (255, 80, 80, 185)
PRED_SEGMENT_COLOR = (60, 150, 255, 185)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute physical left/right boundary error by shared uniform horizontal scanline sampling."
    )
    parser.add_argument("--gt-boundary", type=Path, default=DEFAULT_GT_BOUNDARY_PATH)
    parser.add_argument("--pred-boundary", type=Path, default=DEFAULT_PRED_BOUNDARY_PATH)
    parser.add_argument("--h-json", type=Path, default=DEFAULT_H_JSON_PATH)
    parser.add_argument("--rgb", type=Path, default=DEFAULT_RGB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BOUNDARY_DIR / "scanline")
    parser.add_argument("--class-name", default="dirt")
    parser.add_argument("--scanline-step-px", type=int, default=10,
                        help="Uniform vertical step between scanlines (px).")
    parser.add_argument("--cap-run-width-px", type=int, default=25,
                        help="Max run width (px) used ONLY to determine valid y range "
                             "(excludes top-cap and bottom-closure rows).")
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--point-radius", type=int, default=7)
    return parser.parse_args()


def load_binary_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 0


# ---------------------------------------------------------------------------
# Range determination: narrow-run filter used ONLY here to detect top cap
# and bottom closure rows.  NOT used for point extraction.
# ---------------------------------------------------------------------------

def _runs_from_row(row: np.ndarray) -> list[tuple[int, int]]:
    xs = np.flatnonzero(row)
    if xs.size == 0:
        return []
    gaps = np.flatnonzero(np.diff(xs) > 1)
    starts = np.r_[xs[0], xs[gaps + 1]]
    ends   = np.r_[xs[gaps], xs[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _narrow_runs(boundary: np.ndarray, y: int, max_width: int) -> list[tuple[int, int]]:
    return [r for r in _runs_from_row(boundary[y]) if r[1] - r[0] + 1 <= max_width]


def valid_y_range(pred_boundary: np.ndarray, cap_run_width_px: int) -> tuple[int | None, int | None]:
    """Return (first_valid_y, last_valid_y) using narrow-run filter to exclude
    top-convergence rows (single wide cap) and bottom-closure rows."""
    first_valid: int | None = None
    last_valid:  int | None = None
    for y in range(pred_boundary.shape[0]):
        if len(_narrow_runs(pred_boundary, y, cap_run_width_px)) >= 2:
            if first_valid is None:
                first_valid = y
            last_valid = y
    return first_valid, last_valid


def uniform_scanline_rows(pred_boundary: np.ndarray, step_px: int,
                          cap_run_width_px: int) -> tuple[list[int], dict]:
    """Generate a uniform grid y = first_valid, first_valid + step, ... last_valid.

    The grid is generated first; individual rows are not pre-filtered by
    whether they have boundary pixels.  That check happens in build_samples.
    """
    first_valid, last_valid = valid_y_range(pred_boundary, cap_run_width_px)
    if first_valid is None:
        return [], {"first_valid_y": -1, "last_valid_y": -1}

    rows = list(range(first_valid, last_valid + 1, step_px))
    if rows[-1] != last_valid:
        rows.append(last_valid)

    return rows, {
        "sampling": "uniform_grid_within_pred_boundary_range",
        "first_valid_y": int(first_valid),
        "last_valid_y":  int(last_valid),
        "step_px":       int(step_px),
        "grid_row_count": int(len(rows)),
    }


# ---------------------------------------------------------------------------
# Point extraction: outermost pixels in the full boundary row, NO run filter.
# ---------------------------------------------------------------------------

def outermost_points(boundary: np.ndarray, y: int) -> tuple[float, float] | None:
    """Return (left_x, right_x) as the leftmost and rightmost boundary pixel
    in row y.  Returns None if the row has fewer than 2 distinct x positions."""
    xs = np.flatnonzero(boundary[y])
    if xs.size < 2:
        return None
    left_x  = float(xs[0])
    right_x = float(xs[-1])
    if left_x >= right_x:
        return None
    return left_x, right_x


# ---------------------------------------------------------------------------
# Homography helpers
# ---------------------------------------------------------------------------

def apply_homography(H: np.ndarray, points_uv: np.ndarray) -> np.ndarray:
    ph = np.column_stack([points_uv, np.ones(len(points_uv))])
    r  = (H @ ph.T).T
    return r[:, :2] / r[:, 2:3]


def distance_m(H: np.ndarray, a_uv: tuple[float, float], b_uv: tuple[float, float]) -> float:
    plane = apply_homography(H, np.array([a_uv, b_uv], dtype=np.float64))
    return float(np.linalg.norm(plane[0] - plane[1]))


# ---------------------------------------------------------------------------
# Build samples: uniform rows × outermost points × left-left / right-right
# ---------------------------------------------------------------------------

def build_samples(
    gt_boundary: np.ndarray,
    pred_boundary: np.ndarray,
    rows: list[int],
    H_image_to_plane: np.ndarray,
) -> tuple[list[dict], list[float], list[float]]:
    samples: list[dict] = []
    left_errors:  list[float] = []
    right_errors: list[float] = []

    for y in rows:
        gt_pts   = outermost_points(gt_boundary,   y)
        pred_pts = outermost_points(pred_boundary, y)
        if gt_pts is None or pred_pts is None:
            continue  # no valid left/right pair on this row; skip silently

        gt_left,   gt_right   = gt_pts
        pred_left, pred_right = pred_pts

        # Semantic pairing: left↔left, right↔right
        left_err  = distance_m(H_image_to_plane, (gt_left,  y), (pred_left,  y))
        right_err = distance_m(H_image_to_plane, (gt_right, y), (pred_right, y))

        left_errors.append(left_err)
        right_errors.append(right_err)
        samples.append({
            "y":             int(y),
            "gt_left_x":     float(gt_left),
            "gt_right_x":    float(gt_right),
            "pred_left_x":   float(pred_left),
            "pred_right_x":  float(pred_right),
            "left_error_m":  left_err,
            "right_error_m": right_err,
        })

    return samples, left_errors, right_errors


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def make_boundary_canvas(gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    canvas = np.zeros((*gt.shape, 3), dtype=np.uint8)
    canvas[gt & pred]   = [45, 190, 90]
    canvas[gt & ~pred]  = [255, 70, 70]
    canvas[~gt & pred]  = [55, 135, 255]
    return Image.fromarray(canvas, mode="RGB").convert("RGBA")


def draw_samples_on_image(
    base: Image.Image,
    samples: list[dict],
    out_path: Path,
    line_width: int,
    point_radius: int,
) -> None:
    img     = base.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    W       = img.size[0]

    for s in samples:
        y = int(s["y"])
        draw.line((0, y, W - 1, y), fill=CONTROL_LINE_COLOR, width=line_width)
        draw.line((int(s["gt_left_x"]),   y, int(s["gt_right_x"]),   y),
                  fill=GT_SEGMENT_COLOR,   width=line_width + 1)
        draw.line((int(s["pred_left_x"]), y, int(s["pred_right_x"]), y),
                  fill=PRED_SEGMENT_COLOR, width=line_width + 1)

        for key, color in (("gt_left_x",   GT_COLOR),  ("gt_right_x",   GT_COLOR),
                           ("pred_left_x", PRED_COLOR), ("pred_right_x", PRED_COLOR)):
            x = float(s[key])
            draw.ellipse((x - point_radius, y - point_radius,
                          x + point_radius, y + point_radius), fill=color)
            draw.ellipse((x - point_radius, y - point_radius,
                          x + point_radius, y + point_radius),
                         outline=(255, 255, 255, 230), width=2)

    Image.alpha_composite(img, overlay).save(out_path)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count":  int(arr.size),
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90":    float(np.percentile(arr, 90)),
        "p95":    float(np.percentile(arr, 95)),
        "p99":    float(np.percentile(arr, 99)),
        "max":    float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt_boundary   = load_binary_mask(args.gt_boundary)
    pred_boundary = load_binary_mask(args.pred_boundary)

    with args.h_json.open() as f:
        H_image_to_plane = np.asarray(json.load(f)["H_image_to_plane"], dtype=np.float64)

    # Step 1: generate uniform grid using pred boundary for range only
    rows, row_meta = uniform_scanline_rows(
        pred_boundary, args.scanline_step_px, args.cap_run_width_px
    )

    # Step 2: for each grid row, extract outermost points + compute error
    samples, left_errors, right_errors = build_samples(
        gt_boundary, pred_boundary, rows, H_image_to_plane
    )

    # Save CSV
    csv_path = args.output_dir / "scanline_boundary_samples.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "y", "gt_left_x", "gt_right_x", "pred_left_x", "pred_right_x",
            "left_error_m", "right_error_m",
        ])
        writer.writeheader()
        writer.writerows(samples)

    # Visualisation
    rgb_overlay_path  = args.output_dir / "scanline_samples_rgb_overlay.png"
    mask_overlay_path = args.output_dir / "scanline_samples_mask_overlay.png"

    rgb_base = (Image.open(args.rgb).convert("RGBA") if args.rgb.exists()
                else Image.new("RGBA", (gt_boundary.shape[1], gt_boundary.shape[0]), (20, 20, 20, 255)))

    draw_samples_on_image(rgb_base,
                          samples, rgb_overlay_path,  args.line_width, args.point_radius)
    draw_samples_on_image(make_boundary_canvas(gt_boundary, pred_boundary),
                          samples, mask_overlay_path, args.line_width, args.point_radius)

    # Output JSON
    all_errors = left_errors + right_errors
    result = {
        "method": "uniform_horizontal_scanline_outermost_boundary_points",
        "class_name": args.class_name,
        "sampling_strategy": {
            "description": "uniform grid y=first_valid,first_valid+step,...,last_valid; "
                           "GT and Pred share identical y values; "
                           "each row: take leftmost and rightmost boundary pixel; "
                           "left paired with left, right paired with right; "
                           "physical distance computed via H_image_to_plane",
            **row_meta,
            "cap_run_width_px_for_range_only": int(args.cap_run_width_px),
        },
        "sampled_scanline_count": int(len(samples)),
        "sampled_point_count":    int(len(samples) * 4),
        "left_error_m":    stats(left_errors),
        "right_error_m":   stats(right_errors),
        "overall_error_m": stats(all_errors),
        "paths": {
            "gt_boundary":       str(args.gt_boundary),
            "pred_boundary":     str(args.pred_boundary),
            "samples_csv":       str(csv_path),
            "rgb_overlay_png":   str(rgb_overlay_path),
            "mask_overlay_png":  str(mask_overlay_path),
        },
    }

    json_path = args.output_dir / "scanline_boundary_error.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"[INFO] Saved: {json_path}")


if __name__ == "__main__":
    main()
