"""Full boundary-error pipeline entry point.

This module re-exports every function from the per-stage modules
(io_utils / homography / boundary / scanline) so existing code that does
`from distance.pipeline import X` keeps working unchanged. New code that only
needs one stage (e.g. just the H-matrix estimation) should import directly
from that stage's module instead:

    from distance.homography import estimate_h, extract_correspondences
    from distance.boundary import create_boundaries, extract_external_boundary
    from distance.scanline import scanline_error
"""
from __future__ import annotations

import json

from .io_utils import (  # noqa: F401
    POSE_DIRECTION_SENSOR_TO_WORLD,
    POSE_DIRECTION_WORLD_TO_SENSOR,
    PoseRecord,
    choose_direction,
    ensure_dir,
    find_pose_by_timestamp,
    infer_timestamp_from_name,
    invert_transform,
    load_3d_labels,
    load_binary_mask,
    load_camera_calibration,
    load_point_cloud,
    make_transform,
    parse_args,
    parse_float_list_from_line,
    parse_pose_csv,
    pose_record_to_transform,
    quaternion_wxyz_to_rotation_matrix,
    read_config,
    resolve_path,
    save_binary,
    stats,
    stats_extended,
    transform_points,
)
from .homography import (  # noqa: F401
    apply_homography,
    choose_plane_basis,
    distort_normalized_points,
    draw_points,
    estimate_h,
    extract_correspondences,
    fit_homography,
    fit_plane,
    load_correspondences,
    load_correspondences_full,
    normalize_points_2d,
    project_camera_points,
    ransac_fit_h,
    reproj_by_distance,
    reproj_by_image_region,
    save_reproj_heatmap,
    save_reproj_histogram,
    sigma_filter_correspondences,
    world_to_plane,
    zbuffer_visible_mask,
)
from .boundary import (  # noqa: F401
    binary_close_3x3,
    binary_dilate_3x3,
    binary_erode_3x3,
    border_connected_background,
    clean_prediction_mask,
    connected_components,
    create_boundaries,
    extract_external_boundary,
    extract_gt_mask,
    extract_pred_mask,
    keep_large_components,
    make_boundary_overlay,
)
from .scanline import (  # noqa: F401
    distance_m,
    filtered_boundary_runs,
    ground_distance_m,
    make_boundary_canvas,
    match_run_midpoints,
    outer_boundary_points,
    run_midpoints,
    runs_from_bool_row,
    scanline_error,
    uniform_scanline_rows,
    wprime,
)


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    output_dir = resolve_path(config, config["output_dir"])
    ensure_dir(output_dir)

    correspondence_csv, correspondence_summary, corr_arrays = extract_correspondences(config, output_dir)
    h_json, h_summary = estimate_h(config, correspondence_csv, output_dir, precomputed=corr_arrays)
    gt_boundary_path, pred_boundary_path, boundary_summary, boundary_arrays = create_boundaries(config, output_dir)
    scanline_summary = scanline_error(config, h_json, gt_boundary_path, pred_boundary_path, output_dir,
                                       precomputed_boundaries=boundary_arrays)

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
