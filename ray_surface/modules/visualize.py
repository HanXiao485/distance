"""Visualization utilities for the ray-surface method."""
from __future__ import annotations

import colorsys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _err_to_rgb(err: float, vmax: float) -> tuple[int, int, int]:
    t = min(err / max(vmax, 1e-9), 1.0)
    hue = (1.0 - t) * 240.0 / 360.0  # blue→red
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_scanline_overlay(
    rgb_path: Path,
    samples: list[dict[str, Any]],
    output_path: Path,
    point_radius: int = 6,
    line_width: int = 3,
    vmax_m: float = 2.0,
) -> None:
    """Draw boundary points and scanlines on the original RGB image.

    Each sample has keys: y, gt_left_x, gt_right_x, pred_left_x, pred_right_x,
    left_err_m, right_err_m, left_hit, right_hit.
    """
    img = Image.open(rgb_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    W = img.width

    for s in samples:
        y = int(s["y"])
        # Faint horizontal scanline
        draw.line((0, y, W - 1, y), fill=(255, 230, 80, 100), width=1)

        # GT segment (red)
        draw.line((s["gt_left_x"], y, s["gt_right_x"], y),
                  fill=(255, 70, 70, 170), width=line_width)
        # Pred segment (blue)
        draw.line((s["pred_left_x"], y, s["pred_right_x"], y),
                  fill=(55, 140, 255, 170), width=line_width)

        for side, gt_key, pred_key, err_key, hit_key in (
            ("left",  "gt_left_x",  "pred_left_x",  "left_err_m",  "left_hit"),
            ("right", "gt_right_x", "pred_right_x", "right_err_m", "right_hit"),
        ):
            gx = int(s[gt_key])
            px = int(s[pred_key])
            hit = s.get(hit_key, False)
            err = s.get(err_key, 0.0) or 0.0

            if hit:
                color = _err_to_rgb(err, vmax_m) + (230,)
            else:
                color = (150, 150, 150, 180)  # grey = no intersection

            # GT point (solid circle)
            draw.ellipse(
                (gx - point_radius, y - point_radius,
                 gx + point_radius, y + point_radius),
                fill=(255, 70, 70, 220), outline=(255, 255, 255, 200), width=2,
            )
            # Pred point (colored by error or grey if miss)
            draw.ellipse(
                (px - point_radius, y - point_radius,
                 px + point_radius, y + point_radius),
                fill=color, outline=(255, 255, 255, 200), width=2,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(img, overlay).convert("RGB").save(output_path)


def draw_surface_reconstruction(
    elev: "ElevationMap",  # type: ignore[name-defined]
    coverage_mask: np.ndarray,
    gt_intersections: list[np.ndarray],
    pred_intersections: list[np.ndarray],
    errors_m: list[float],
    camera_origin: np.ndarray,
    cam_forward: np.ndarray,
    output_path: Path,
    view_radius_m: float = 25.0,
) -> None:
    """3-panel surface reconstruction figure:
      Panel 1 (top, large): 3D perspective view of reconstructed terrain + camera + intersection points
      Panel 2 (bottom-left): BEV heatmap with contours + coverage overlay
      Panel 3 (bottom-right): Elevation cross-section along camera forward direction
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from matplotlib.colors import Normalize, LightSource
        from matplotlib import cm
    except ImportError:
        return

    cs  = elev.cell_size
    cx_w, cy_w = camera_origin[0], camera_origin[1]
    fwd_xy_unit = cam_forward[:2] / (np.linalg.norm(cam_forward[:2]) + 1e-9)

    # ── Crop grid to view window ─────────────────────────────────────────────
    xl, xr = cx_w - view_radius_m, cx_w + view_radius_m
    yl, yr = cy_w - view_radius_m, cy_w + view_radius_m
    ci_lo = max(0, int((xl - elev.x_min) / cs))
    ci_hi = min(elev.cols, int((xr - elev.x_min) / cs) + 1)
    ri_lo = max(0, int((yl - elev.y_min) / cs))
    ri_hi = min(elev.rows, int((yr - elev.y_min) / cs) + 1)

    xs_crop = elev.x_min + (np.arange(ci_lo, ci_hi) + 0.5) * cs
    ys_crop = elev.y_min + (np.arange(ri_lo, ri_hi) + 0.5) * cs
    z_crop  = elev.z_grid[ri_lo:ri_hi, ci_lo:ci_hi]
    cov_crop = coverage_mask[ri_lo:ri_hi, ci_lo:ci_hi]
    bev_extent = [xs_crop[0] - cs/2, xs_crop[-1] + cs/2,
                  ys_crop[0] - cs/2, ys_crop[-1] + cs/2]

    # Relative elevation
    z_min_real = (elev.z_grid[coverage_mask].min()
                  if coverage_mask.any() else float(z_crop.min()))
    z_rel = z_crop - z_min_real
    cam_z_rel = camera_origin[2] - z_min_real

    # Subsample for 3D surface (stride to keep it fast)
    stride = max(1, min(z_rel.shape) // 60)
    zs = z_rel[::stride, ::stride]
    Xs, Ys = np.meshgrid(xs_crop[::stride], ys_crop[::stride])

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("Surface Reconstruction + Ray Intersection", fontsize=14, y=0.98)

    ax3d  = fig.add_axes([0.03, 0.45, 0.60, 0.50], projection="3d")
    ax_bev = fig.add_axes([0.68, 0.45, 0.30, 0.50])
    ax_prof = fig.add_axes([0.07, 0.06, 0.88, 0.32])

    # ── Panel 1: 3D perspective ───────────────────────────────────────────────
    ls = LightSource(azdeg=225, altdeg=45)
    rgb_surf = ls.shade(zs, cmap=cm.terrain,
                        vmin=0, vmax=float(z_rel.max()),
                        blend_mode="soft")
    ax3d.plot_surface(Xs, Ys, zs, facecolors=rgb_surf,
                      linewidth=0, antialiased=True, alpha=0.85, rcount=60, ccount=60)

    # Real LiDAR points (small scatter on surface)
    cov_s = cov_crop[::stride*2, ::stride*2]
    Xc, Yc = np.meshgrid(xs_crop[::stride*2], ys_crop[::stride*2])
    zc = z_rel[::stride*2, ::stride*2]
    ax3d.scatter(Xc[cov_s], Yc[cov_s], zc[cov_s] + 0.02,
                 c="cyan", s=3, alpha=0.6, zorder=4, label="Real LiDAR pts")

    # Camera position
    ax3d.scatter([cx_w], [cy_w], [cam_z_rel],
                 c="yellow", s=200, marker="^",
                 edgecolors="black", linewidths=1, zorder=8, label="Camera")

    # Sample camera rays + intersections
    if gt_intersections:
        gt_arr  = np.stack(gt_intersections)
        pred_arr = np.stack(pred_intersections)
        errs    = np.asarray(errors_m)
        vmax_e  = max(float(np.percentile(errs, 90)), 0.1)

        # Draw every 5th ray from camera to GT intersection
        for i in range(0, len(gt_arr), max(1, len(gt_arr) // 12)):
            pt = gt_arr[i]
            ax3d.plot([cx_w, pt[0]], [cy_w, pt[1]],
                      [cam_z_rel, pt[2] - z_min_real],
                      color="orange", lw=0.6, alpha=0.5)

        # GT intersections colored by error
        norm = Normalize(vmin=0, vmax=vmax_e)
        sc = ax3d.scatter(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2] - z_min_real,
                          c=errs, cmap="RdYlGn_r", norm=norm,
                          s=40, zorder=6, label="GT boundary")
        ax3d.scatter(pred_arr[:, 0], pred_arr[:, 1], pred_arr[:, 2] - z_min_real,
                     c="deepskyblue", s=20, alpha=0.7, zorder=5, label="Pred boundary")

        cb = fig.colorbar(sc, ax=ax3d, pad=0.02, shrink=0.5, label="3D error (m)")
        cb.ax.tick_params(labelsize=8)

    ax3d.set_xlabel("X (m)", fontsize=8); ax3d.set_ylabel("Y (m)", fontsize=8)
    ax3d.set_zlabel("Rel. elev. (m)", fontsize=8)
    ax3d.set_title("3D surface + rays + boundary intersections", fontsize=10)
    ax3d.tick_params(labelsize=7)
    ax3d.legend(loc="upper left", fontsize=7, markerscale=1.2)
    # Set a good viewing angle (from slightly above, looking forward)
    ax3d.view_init(elev=30, azim=200)

    # ── Panel 2: BEV heatmap ─────────────────────────────────────────────────
    # Full interpolated terrain as continuous color
    im = ax_bev.imshow(z_rel, origin="lower", extent=bev_extent,
                       cmap="terrain", aspect="equal", interpolation="bilinear")
    # Contour lines (real data topology only)
    try:
        cs_obj = ax_bev.contour(xs_crop, ys_crop, z_rel,
                                levels=8, colors="white",
                                linewidths=0.6, alpha=0.5)
        ax_bev.clabel(cs_obj, inline=True, fontsize=6, fmt="%.1fm")
    except Exception:
        pass
    # Real LiDAR cells as dots
    ry, rx = np.where(cov_crop)
    ax_bev.scatter(xs_crop[rx], ys_crop[ry],
                   s=0.5, c="cyan", alpha=0.4, label="Real data")
    # Camera
    ax_bev.scatter([cx_w], [cy_w], c="yellow", s=80, marker="^",
                   edgecolors="black", zorder=5)
    # Forward arrow
    fxy = fwd_xy_unit * view_radius_m * 0.35
    ax_bev.annotate("", xy=(cx_w + fxy[0], cy_w + fxy[1]), xytext=(cx_w, cy_w),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2))
    if gt_intersections:
        ax_bev.scatter(gt_arr[:, 0], gt_arr[:, 1],
                       c=errs, cmap="RdYlGn_r", norm=norm, s=18, zorder=4)
    plt.colorbar(im, ax=ax_bev, label="Rel. elev (m)", fraction=0.046, pad=0.04)
    real_pct = cov_crop.mean() * 100
    ax_bev.set_title(f"BEV heatmap\n(real: {real_pct:.0f}%, interpolated: {100-real_pct:.0f}%)",
                     fontsize=9)
    ax_bev.set_xlabel("X (m)", fontsize=8); ax_bev.set_ylabel("Y (m)", fontsize=8)
    ax_bev.set_xlim(xl, xr); ax_bev.set_ylim(yl, yr)
    ax_bev.tick_params(labelsize=7)

    # ── Panel 3: Elevation cross-section ─────────────────────────────────────
    t_pts = np.linspace(0.5, view_radius_m, 300)
    px_arr = cx_w + t_pts * fwd_xy_unit[0]
    py_arr = cy_w + t_pts * fwd_xy_unit[1]

    # Full profile (interpolated)
    z_prof = np.array([elev.z_at(float(px), float(py)) or np.nan
                       for px, py in zip(px_arr, py_arr)])
    z_prof_rel = z_prof - z_min_real

    # Real-only profile
    def _is_real(px, py):
        ci = int((px - elev.x_min) / cs)
        ri = int((py - elev.y_min) / cs)
        return (0 <= ci < elev.cols and 0 <= ri < elev.rows
                and coverage_mask[ri, ci])

    z_real_only = np.where(
        [_is_real(float(px), float(py)) for px, py in zip(px_arr, py_arr)],
        z_prof_rel, np.nan
    )

    valid = ~np.isnan(z_prof_rel)
    ax_prof.fill_between(t_pts[valid], z_prof_rel[valid],
                         alpha=0.12, color="green", label="_nolegend_")
    ax_prof.plot(t_pts[valid], z_prof_rel[valid],
                 color="gray", lw=1.5, alpha=0.7, label="Surface (interpolated)")
    real_ok = ~np.isnan(z_real_only)
    if real_ok.any():
        ax_prof.plot(t_pts[real_ok], z_real_only[real_ok],
                     color="green", lw=2.5, alpha=0.9, label="Surface (real LiDAR)")

    ax_prof.axhline(cam_z_rel, color="gold", lw=2, linestyle="--",
                    label=f"Camera height = {cam_z_rel:.2f}m above lowest point")

    # Plot intersection points on the profile
    if gt_intersections:
        t_proj_vals, z_proj_vals, err_vals = [], [], []
        for pt, err in zip(gt_intersections, errors_m):
            t_p = float((pt[:2] - camera_origin[:2]) @ fwd_xy_unit)
            if 0 < t_p < view_radius_m:
                t_proj_vals.append(t_p)
                z_proj_vals.append(pt[2] - z_min_real)
                err_vals.append(err)
        if t_proj_vals:
            ax_prof.scatter(t_proj_vals, z_proj_vals,
                            c=err_vals, cmap="RdYlGn_r",
                            norm=Normalize(vmin=0, vmax=vmax_e),
                            s=60, zorder=6, edgecolors="white",
                            linewidths=0.5, label="GT intersections")

    ax_prof.set_xlabel("Distance along camera forward direction (m)", fontsize=10)
    ax_prof.set_ylabel("Relative elevation (m)", fontsize=10)
    ax_prof.set_title("Elevation cross-section — green solid = real LiDAR, grey = interpolated",
                      fontsize=10)
    ax_prof.legend(fontsize=9, loc="upper left")
    ax_prof.set_xlim(0, view_radius_m)
    ax_prof.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def draw_error_histogram(
    errors_m: list[float],
    output_path: Path,
    title: str = "Ray-Surface Boundary Error",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    arr = np.asarray(errors_m)
    if arr.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(arr, bins=min(60, max(10, arr.size // 10)),
            color="steelblue", edgecolor="white", linewidth=0.5)
    for pct, color in ((90, "orange"), (95, "tomato"), (99, "darkred")):
        val = float(np.percentile(arr, pct))
        ax.axvline(val, color=color, linestyle="--", linewidth=1.5,
                   label=f"P{pct}={val:.2f}m")
    ax.set_xlabel("Boundary error (m)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
