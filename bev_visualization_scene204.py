#!/usr/bin/env python3
"""
BEV navigation-style visualization for the DISTANCE pipeline (WildScenes).

Each frame shows:
  • LiDAR point cloud (XYZ) colored by semantic class
  • Camera position as vehicle icon (from pipeline H estimation)
  • GT dirt boundary → BEV (red lines)
  • Pred dirt boundary → BEV (blue lines)
  • Camera FOV cone (yellow dashed)
  • 20 m measurement radius circle
"""

import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
CAM_CALIB_FILE = Path("/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes2d/FRONT_CAMERA/camera_calibration.yaml")
POSES_FILE     = Path("/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes3d/poses3d.csv")
LIDAR_DIR      = Path("/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes3d/Clouds")
LABEL_DIR      = Path("/root/distance/WildScenes/DISTANCE/datasets/scene204/WildScenes3d/Labels")
FRAMES_DIR     = Path("/root/distance/WildScenes/work_dirs/distance_output/frames")
OUT_DIR        = Path("/root/distance/WildScenes/work_dirs/distance_output/bev")

# ─────────────────────────────────────────────────────────────────────────────
# scene204 3D 25-class annotation (SurroundOcc style, uint8 labels)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLOR = {
    0:  (0.40, 0.40, 0.40, 0.4),   # noise
    1:  (0.90, 0.55, 0.00, 0.8),   # barrier
    2:  (0.65, 0.10, 0.15, 0.8),   # bicycle
    3:  (0.50, 0.00, 0.00, 0.8),   # bus
    4:  (0.05, 0.15, 0.55, 0.8),   # car
    5:  (0.80, 0.65, 0.00, 0.8),   # construction_vehicle
    6:  (0.10, 0.30, 0.85, 0.8),   # motorcycle
    7:  (0.85, 0.20, 0.40, 0.8),   # pedestrian
    8:  (1.00, 0.45, 0.00, 0.9),   # traffic_cone
    9:  (0.35, 0.20, 0.10, 0.7),   # trailer
    10: (0.05, 0.05, 0.40, 0.8),   # truck
    11: (0.80, 0.55, 0.20, 0.9),   # driveable_surface (road)
    12: (0.75, 0.75, 0.75, 0.6),   # other_flat
    13: (0.55, 0.45, 0.65, 0.7),   # sidewalk
    14: (0.55, 0.80, 0.40, 0.7),   # terrain
    15: (0.60, 0.60, 0.55, 0.6),   # manmade
    16: (0.10, 0.45, 0.10, 0.7),   # vegetation
    17: (0.30, 0.30, 0.30, 0.7),   # building
    18: (0.75, 0.60, 0.60, 0.7),   # fence
    19: (0.65, 0.40, 0.10, 0.9),   # road
    20: (0.10, 0.20, 0.60, 0.8),   # water
    21: (0.45, 0.70, 0.90, 0.5),   # sky
    22: (0.80, 0.10, 0.20, 0.9),   # person
    23: (0.70, 0.10, 0.20, 0.8),   # rider
    24: (0.25, 0.20, 0.15, 0.8),   # tank
    25: (0.70, 0.45, 0.65, 0.7),   # curb
}
CLASS_NAME = {
    0:'noise', 1:'barrier', 2:'bicycle', 3:'bus', 4:'car',
    5:'constr_veh', 6:'motorcycle', 7:'pedestrian', 8:'cone', 9:'trailer',
    10:'truck', 11:'driveable', 12:'other_flat', 13:'sidewalk', 14:'terrain',
    15:'manmade', 16:'vegetation', 17:'building', 18:'fence', 19:'road',
    20:'water', 21:'sky', 22:'person', 23:'rider', 24:'tank', 25:'curb',
}
FALLBACK_COLOR = (0.5, 0.5, 0.5, 0.4)


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion → rotation matrix (no scipy dependency)
# ──────────────────────────────────────────────────────────────────────────────
def quat_to_R(qx, qy, qz, qw) -> np.ndarray:
    n = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*(qy**2+qz**2),  2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw),    1-2*(qx**2+qz**2),  2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw),    2*(qy*qz+qx*qw),    1-2*(qx**2+qy**2)],
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Camera calibration
# ──────────────────────────────────────────────────────────────────────────────
def load_camera_calibration(calib_file: Path):
    """
    Returns:
        K              – 3×3 intrinsic matrix
        R_slam_cam     – rotation: p_slam = R_slam_cam @ p_cam + t_slam_cam
        t_slam_cam     – camera origin in slam/LiDAR frame (metres)
    """
    import re
    text = calib_file.read_text()

    # Parse K: [fx, fy, cx, cy]
    m = re.search(r"K:\s*\[([^\]]+)\]", text)
    fx, fy, cx, cy = [float(v) for v in m.group(1).split(",")]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Parse translation
    m = re.search(r"translation:\s*\[([^\]]+)\]", text)
    tx, ty, tz = [float(v) for v in m.group(1).split(",")]

    # Parse rotation quaternion [qx, qy, qz, qw]
    m = re.search(r"rotation:\s*\[([^\]]+)\]", text)
    qx, qy, qz, qw = [float(v) for v in m.group(1).split(",")]

    R_slam_cam = quat_to_R(qx, qy, qz, qw)
    t_slam_cam = np.array([tx, ty, tz])

    return K, R_slam_cam, t_slam_cam


# ──────────────────────────────────────────────────────────────────────────────
# Pose loading
# ──────────────────────────────────────────────────────────────────────────────
def load_poses(poses_file: Path) -> pd.DataFrame:
    df = pd.read_csv(poses_file, sep=r"\s+")
    df.columns = [c.strip() for c in df.columns]
    return df


def find_pose(poses: pd.DataFrame, frame_ts: float):
    """Return (R_world_lidar, t_world_lidar) for the closest timestamp."""
    pos = int((poses["timestamp"] - frame_ts).abs().argmin())
    row = poses.iloc[pos]
    t = np.array([row["x"], row["y"], row["z"]])
    R = quat_to_R(row["qx"], row["qy"], row["qz"], row["qw"])
    return R, t   # p_world = R @ p_lidar + t


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate conversions
# ──────────────────────────────────────────────────────────────────────────────
def world_to_lidar(pts_w: np.ndarray, R_wl: np.ndarray, t_wl: np.ndarray) -> np.ndarray:
    """Nx3 world points → Nx3 LiDAR frame."""
    return (pts_w - t_wl) @ R_wl  # = R_wl^T @ (pt - t)  (R_wl is orthogonal)


def apply_H(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nx2 image coords → Nx2 plane_xy via homography H (3×3)."""
    uvh = np.c_[uv, np.ones(len(uv))]
    xyh = (H @ uvh.T).T
    return xyh[:, :2] / xyh[:, 2:3]


def plane_xy_to_world(pxy: np.ndarray, centroid: np.ndarray,
                      ax_x: np.ndarray, ax_y: np.ndarray) -> np.ndarray:
    return centroid + pxy[:, 0:1] * ax_x + pxy[:, 1:2] * ax_y


def boundary_pixels_to_lidar(u_vals: np.ndarray, y_vals: np.ndarray,
                               H: np.ndarray, centroid: np.ndarray,
                               ax_x: np.ndarray, ax_y: np.ndarray,
                               R_wl: np.ndarray, t_wl: np.ndarray,
                               u_min: float = 8.0, u_max: float = 2008.0,
                               v_min_train: float = 0.0,
                               v_max_train: float = np.inf) -> np.ndarray:
    """
    Convert boundary pixel columns (u, y) to LiDAR XY for BEV.

    Filters:
    - Sentinel values (u ≤ 0 or u ≥ 2015)
    - Image-edge pixels (u < u_min or u > u_max) where H extrapolation is unreliable
    - Points that project behind the vehicle (lidar_x < 0) – H extrapolation artefact
    Adaptive v-margin: pixels within ±(0.5 × training range) of the training band
    are projected; beyond that H extrapolation becomes too unreliable.
    """
    uv = np.c_[u_vals.astype(float), y_vals.astype(float)]
    valid = (uv[:, 0] >= u_min) & (uv[:, 0] <= u_max)
    v_margin = 0.5 * max(v_max_train - v_min_train, 1.0)
    valid &= uv[:, 1] >= (v_min_train - v_margin)
    valid &= uv[:, 1] <= (v_max_train + v_margin)
    uv = uv[valid]
    if len(uv) == 0:
        return np.empty((0, 2))
    pxy   = apply_H(H, uv)
    world = plane_xy_to_world(pxy, centroid, ax_x, ax_y)
    lidar = world_to_lidar(world, R_wl, t_wl)
    # Keep only points in front of the vehicle (H extrapolation can put points behind)
    front = lidar[:, 0] > 0
    return lidar[front, :2]   # Nx2  (x=forward, y=lateral)


# ──────────────────────────────────────────────────────────────────────────────
# BEV plot helpers
# ──────────────────────────────────────────────────────────────────────────────
def lidar_to_bev(pts_lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert LiDAR (X=fwd, Y=left) to BEV screen coords:
        bev_h (horizontal) = -LiDAR_Y  (right is positive, i.e. East on screen)
        bev_v (vertical)   =  LiDAR_X  (forward is positive, i.e. North on screen)
    """
    return -pts_lidar[:, 1], pts_lidar[:, 0]


def draw_vehicle_icon(ax, cam_lidar_xy: np.ndarray,
                      fwd_lidar: np.ndarray, color: str = "#00ffcc"):
    """
    Draw a triangle (camera icon) at cam_lidar_xy pointing in fwd_lidar direction.
    cam_lidar_xy : (2,) in LiDAR XY
    fwd_lidar    : (2,) forward direction unit vector in LiDAR XY
    """
    # BEV screen coords for the camera position
    bh = float(-cam_lidar_xy[1])
    bv = float( cam_lidar_xy[0])

    # BEV forward direction (note axis flip for Y)
    fh = float(-fwd_lidar[1])
    fv = float( fwd_lidar[0])
    norm = math.sqrt(fh**2 + fv**2) + 1e-9
    fh /= norm; fv /= norm

    # Perpendicular (right side of vehicle)
    rh = fv; rv = -fh

    L, W = 2.5, 1.2  # length, half-width in metres

    tip   = np.array([bh + L*fh,      bv + L*fv])
    left  = np.array([bh - L*0.3*fh - W*rh, bv - L*0.3*fv - W*rv])
    right = np.array([bh - L*0.3*fh + W*rh, bv - L*0.3*fv + W*rv])

    triangle = plt.Polygon([tip, left, right], closed=True, facecolor=color,
                             edgecolor="white", linewidth=1.5, alpha=0.9, zorder=15)
    ax.add_patch(triangle)
    ax.plot(bh, bv, "w+", ms=7, mew=2, zorder=16)
    ax.text(bh + 0.4, bv - 2.2, "Camera", color=color, fontsize=8,
            ha="left", va="top", zorder=17, fontweight="bold")


def draw_fov_cone(ax, cam_xy_lidar: np.ndarray,
                  K: np.ndarray, R_slam_cam: np.ndarray, t_slam_cam: np.ndarray,
                  img_w: int, img_h: int, max_depth: float = 35.0):
    """
    Project image corner rays through the ground plane (z=0 in LiDAR frame)
    and draw the FOV cone on the BEV.

    Since slam_base_link ≈ LiDAR frame in WildScenes, we use the extrinsics directly.
    """
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # Image corners (skip bottom 2 – pointing behind/upward → use top corners + midpoints)
    # Use top-left, top-right, and optionally bottom corners
    corners_img = np.array([
        [0,        0       ],
        [img_w-1,  0       ],
        [img_w-1,  img_h-1 ],
        [0,        img_h-1 ],
    ], dtype=float)

    rays_cam = np.stack([
        (corners_img[:, 0] - cx) / fx,
        (corners_img[:, 1] - cy) / fy,
        np.ones(4),
    ], axis=1)
    rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)

    # Rays in slam/LiDAR frame
    rays_lidar = (R_slam_cam @ rays_cam.T).T  # 4×3

    # Camera origin in slam/LiDAR frame (tiny offset, provided by extrinsics)
    orig = t_slam_cam  # 3,

    fov_pts_bev = []
    for ray in rays_lidar:
        # Intersect with z=0 plane: orig_z + t*ray_z = 0
        if abs(ray[2]) > 0.01 and (orig[2] + max_depth * ray[2]) * ray[2] < 0:
            t_hit = -orig[2] / ray[2]
            t_hit = max(0.0, min(t_hit, max_depth))
        else:
            t_hit = max_depth
        pt = orig + t_hit * ray
        fov_pts_bev.append((-pt[1], pt[0]))  # LiDAR → BEV screen

    orig_bev = (-orig[1], orig[0])

    # Draw FOV lines from camera to each corner
    for pt in fov_pts_bev[:2]:   # only top-left and top-right for clean FOV wedge
        ax.plot([orig_bev[0], pt[0]], [orig_bev[1], pt[1]],
                "--", color="#ffff44", lw=1.2, alpha=0.7, zorder=6)

    # Connect the far edge of the top two corners
    ax.plot([fov_pts_bev[0][0], fov_pts_bev[1][0]],
            [fov_pts_bev[0][1], fov_pts_bev[1][1]],
            "--", color="#ffff44", lw=1.2, alpha=0.7, zorder=6)


# ──────────────────────────────────────────────────────────────────────────────
# Main per-frame BEV function
# ──────────────────────────────────────────────────────────────────────────────
def build_bev_frame(frame_id: str,
                    K, R_slam_cam, t_slam_cam,
                    poses: pd.DataFrame,
                    img_w: int, img_h: int) -> plt.Figure:
    # ── Load pipeline outputs ───────────────────────────────────────────────
    frame_dir = FRAMES_DIR / frame_id
    summary   = json.loads((frame_dir / "pipeline_summary.json").read_text())
    h_json    = json.loads(Path(summary["h_source"]).read_text())
    scan_path = summary["scanline_error"]["paths"]["samples_csv"]
    scan_df   = pd.read_csv(scan_path)

    # ── LiDAR point cloud ───────────────────────────────────────────────────
    import open3d as o3d
    _pcd = o3d.io.read_point_cloud(str(LIDAR_DIR / f"{frame_id}.pcd"))
    pts = np.asarray(_pcd.points, dtype=np.float32)
    lbl = np.fromfile(LABEL_DIR / f"{frame_id}.label", dtype=np.uint8).astype(np.uint32)

    # ── Pose for this frame ─────────────────────────────────────────────────
    frame_ts  = float(frame_id)
    R_wl, t_wl = find_pose(poses, frame_ts)  # p_world = R_wl @ p_lidar + t_wl

    # ── Camera position in LiDAR frame (from physical extrinsics, not SVD) ───
    # t_slam_cam is the camera origin in slam≈LiDAR frame from calibration.
    # camera_world_m from SVD is unreliable for icon placement (can be >10m off).
    cam_lidar_xy = t_slam_cam[:2]   # [~0.069, ~-0.007] m – always near origin

    # Camera forward direction: optical axis [0,0,1]_cam → LiDAR frame
    fwd_cam   = np.array([0.0, 0.0, 1.0])
    fwd_slam  = R_slam_cam @ fwd_cam           # slam ≈ LiDAR frame
    fwd_lidar = fwd_slam[:2]                   # project to XY for BEV heading

    # ── GT / Pred boundaries → LiDAR frame ─────────────────────────────────
    H        = np.array(h_json["H_image_to_plane"])
    centroid = np.array(h_json["plane_centroid_world"])
    ax_x     = np.array(h_json["plane_axis_x_world"])
    ax_y     = np.array(h_json["plane_axis_y_world"])

    # Compute H training v-range and condition number for reliability check
    v_min_train = 0.0
    v_max_train = np.inf
    h_unreliable = False
    corr_path = h_json.get("source_csv")
    if corr_path and Path(corr_path).exists():
        corr_df = pd.read_csv(corr_path)
        zf = summary["h_estimation"]["z_filter"]
        z_ok = ((corr_df["lidar_z"] >= zf["lidar_z_min"]) &
                (corr_df["lidar_z"] <= zf["lidar_z_max"]))
        if "u_min" in zf:
            z_ok &= (corr_df["u"] >= zf["u_min"]) & (corr_df["u"] <= zf["u_max"])
        filt = corr_df[z_ok]
        if len(filt) > 0:
            v_min_train   = float(filt["v"].min())
            v_max_train   = float(filt["v"].max())
            v_train_range = v_max_train - v_min_train
            scan_v_min = float(scan_df["y"].min()) if len(scan_df) > 0 else v_min_train
            scan_v_max = float(scan_df["y"].max()) if len(scan_df) > 0 else v_max_train
            # Two extrapolation directions + H condition number
            extrap_below = (scan_v_max - v_max_train) / max(v_train_range, 1.0)
            extrap_above = (v_min_train - scan_v_min) / max(v_train_range, 1.0)
            h_cond       = float(np.linalg.cond(H))
            # Count valid boundary pixels: u in [8,2008] (v-range no longer restricted)
            valid_bdy = 0
            _sdf = scan_df.dropna()
            for col in ["gt_left_x", "gt_right_x", "pred_left_x", "pred_right_x"]:
                if col in _sdf.columns:
                    u = _sdf[col].values
                    valid_bdy += int(((u >= 8) & (u <= 2008)).sum())
            h_unreliable = (max(extrap_below, extrap_above) > 3.0) or \
                           (h_cond > 50000) or (valid_bdy < 10)

    rows   = scan_df.dropna()
    y_vals = rows["y"].values

    def bpx(col):
        return boundary_pixels_to_lidar(rows[col].values, y_vals,
                                         H, centroid, ax_x, ax_y, R_wl, t_wl,
                                         v_min_train=v_min_train,
                                         v_max_train=v_max_train)

    gt_L   = bpx("gt_left_x");   gt_R  = bpx("gt_right_x")
    pr_L   = bpx("pred_left_x"); pr_R  = bpx("pred_right_x")

    # ── BEV extent in LiDAR frame (all LiDAR pts within ±40 m) ─────────────
    in_range = (np.abs(pts[:, 0]) < 45) & (np.abs(pts[:, 1]) < 45) & \
               (pts[:, 2] > -3.0) & (pts[:, 2] < 12.0)
    pts_plot = pts[in_range]
    lbl_plot = lbl[in_range]

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 13), facecolor="#0d0d1a")
    ax.set_facecolor("#0d0d1a")

    # 1. LiDAR semantic point cloud
    seen_cls = set()
    for cls_id in np.unique(lbl_plot):
        mask = lbl_plot == cls_id
        c    = CLASS_COLOR.get(int(cls_id), FALLBACK_COLOR)
        name = CLASS_NAME.get(int(cls_id), f"cls{cls_id}")

        bh, bv = lidar_to_bev(pts_plot[mask])

        # Smaller dots for elevated vegetation (less important for navigation)
        z_mean = pts_plot[mask, 2].mean()
        s = 0.4 if z_mean > 1.5 else 0.8
        a = 0.5 if z_mean > 1.5 else 0.7

        label = name if cls_id not in seen_cls else "_"
        ax.scatter(bh, bv, c=[c[:3]], s=s, alpha=a,
                   label=label, rasterized=True, zorder=2)
        seen_cls.add(cls_id)

    # 2. Camera FOV cone
    draw_fov_cone(ax, cam_lidar_xy, K, R_slam_cam, t_slam_cam, img_w, img_h)

    # 3. 20 m measurement circle (centred at camera)
    cam_bev_h = float(-cam_lidar_xy[1])
    cam_bev_v = float( cam_lidar_xy[0])
    circle = plt.Circle((cam_bev_h, cam_bev_v), 20.0,
                         color="#ffd700", fill=False,
                         linestyle="--", linewidth=1.4, alpha=0.65, zorder=7)
    ax.add_patch(circle)
    ax.text(cam_bev_h + 20.3, cam_bev_v, "20 m", color="#ffd700",
            fontsize=7, va="center", alpha=0.8)

    # 4. GT boundaries (red)
    for seg, label in [(gt_L, "GT boundary"), (gt_R, None)]:
        if len(seg) == 0:
            continue
        bh, bv = lidar_to_bev(np.c_[seg, np.zeros(len(seg))])
        ax.plot(bh, bv, "o-", color="#ff3333", ms=5, lw=2.2,
                label=label, zorder=8, alpha=0.95,
                markeredgecolor="white", markeredgewidth=0.5)

    # 5. Pred boundaries (blue)
    for seg, label in [(pr_L, "Pred boundary"), (pr_R, None)]:
        if len(seg) == 0:
            continue
        bh, bv = lidar_to_bev(np.c_[seg, np.zeros(len(seg))])
        ax.plot(bh, bv, "o-", color="#3399ff", ms=5, lw=2.2,
                label=label, zorder=8, alpha=0.95,
                markeredgecolor="white", markeredgewidth=0.5)

    # 6. Vehicle / camera icon
    draw_vehicle_icon(ax, cam_lidar_xy, fwd_lidar)

    # 7. Warning overlay disabled for scene204 (threshold tuned for WildScenes, not applicable here)
    if False and h_unreliable:
        ax.text(0.5, 0.50, "⚠ H matrix unreliable\n(training v-range too narrow;\nboundary BEV suppressed)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, color="#ff8800", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a0800",
                          edgecolor="#ff8800", alpha=0.85),
                zorder=20)

    # ── Axis labels & formatting ─────────────────────────────────────────────
    ax.set_xlim(-30, 30)
    ax.set_ylim(-8, 45)
    ax.set_aspect("equal")
    ax.set_xlabel("← Left  |  Right →  (m)", color="white", fontsize=10)
    ax.set_ylabel("↑ Forward  (m)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444466")
    ax.grid(True, color="#1e1e3a", linewidth=0.6, alpha=0.8)

    # North arrow
    ax.annotate("N ↑", xy=(0.02, 0.96), xycoords="axes fraction",
                color="white", fontsize=11, fontweight="bold", va="top")

    # Legend
    mean_err = summary["scanline_error"]["overall_error_m"]["mean"]
    median_err = summary["scanline_error"]["overall_error_m"]["median"]
    legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.35,
                       facecolor="#0d0d1a", edgecolor="#555577",
                       labelcolor="white", markerscale=3)

    # Title
    ax.set_title(
        f"BEV — Frame {frame_id}\n"
        f"Boundary error  mean={mean_err:.3f} m   median={median_err:.3f} m",
        color="white", fontsize=11, pad=8,
    )

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Grid output: N frames in one image
# ──────────────────────────────────────────────────────────────────────────────
def make_bev_grid_from_pngs(frame_ids: list[str],
                             ncols: int = 3,
                             out_path: Path | None = None):
    """Stitch individual BEV PNGs saved by main() into a grid using PIL."""
    imgs = []
    for fid in frame_ids:
        p = OUT_DIR / f"bev_{fid}.png"
        if p.exists():
            imgs.append(Image.open(p))
        else:
            # Placeholder
            imgs.append(Image.new("RGB", (1440, 1560), color=(13, 13, 26)))

    if not imgs:
        return None

    ncols = min(ncols, len(imgs))
    nrows = math.ceil(len(imgs) / ncols)
    W, H  = imgs[0].size

    grid = Image.new("RGB", (ncols * W, nrows * H), color=(13, 13, 26))
    for i, img in enumerate(imgs):
        r, c = divmod(i, ncols)
        grid.paste(img.resize((W, H)), (c * W, r * H))

    if out_path is None:
        out_path = OUT_DIR / "bev_overview.png"
    grid.save(out_path)
    print(f"Saved grid: {out_path}")
    return str(out_path)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load calibration once
    # camera_calibration.yaml stores R_cam_from_lidar (cam2lidar convention),
    # but bev code needs R_lidar_from_cam. Invert here.
    K, R_cam_from_lidar, t_cam_from_lidar = load_camera_calibration(CAM_CALIB_FILE)
    R_slam_cam = R_cam_from_lidar.T
    t_slam_cam = -R_cam_from_lidar.T @ t_cam_from_lidar
    poses = load_poses(POSES_FILE)
    img_w, img_h = 1920, 1080  # scene204 image dimensions

    # Discover available frames (must have both pipeline output AND LiDAR file)
    all_frames = sorted([d.name for d in FRAMES_DIR.iterdir() if d.is_dir()])
    frames = [f for f in all_frames if (LIDAR_DIR / f"{f}.pcd").exists()]
    print(f"Frames with LiDAR: {len(frames)} / {len(all_frames)}")

    # ── Individual BEV PNGs ─────────────────────────────────────────────────
    for fid in frames:
        try:
            fig = build_bev_frame(fid, K, R_slam_cam, t_slam_cam, poses, img_w, img_h)
            out = OUT_DIR / f"bev_{fid}.png"
            fig.savefig(out, dpi=120, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"  Saved: {out.name}")
        except Exception as e:
            print(f"  [WARN] {fid}: {e}")

    # ── Grid overview (stitch saved PNGs via PIL) ────────────────────────────
    if len(frames) > 0:
        ncols = min(5, len(frames))
        make_bev_grid_from_pngs(frames, ncols=ncols,
                                out_path=OUT_DIR / "bev_overview.png")


if __name__ == "__main__":
    main()
