"""
project_lidar_to_image.py
把单帧激光雷达点云投影到对应相机图像上并可视化，验证标定（内参+畸变+cam2lidar外参）的正确性。
这是给 H 矩阵估计做前置验证的——如果点投影后能跟图像里的真实边缘（路缘石、树干、地面纹理）对齐，
说明标定链路是通的。

关键点：这批图像是**带畸变的原图**（distortion_model: RATIONAL_POLYNOMIAL），
所以直接用 cv2.projectPoints + 完整畸变系数（而不是简单的5参数径向-切向近似）来做投影，
这样畸变（尤其是画面边缘的桶形/枕形形变）才会被正确考虑。

用法：
    python project_lidar_to_image.py \
        --pcd .../centerlidar_scene204_1774428530_785231590.pcd \
        --image .../FRONT_CAMERA_scene204_1774428530_857196032.jpg \
        --calib .../car006.yaml \
        --camera-name FRONT_CAMERA \
        --output .../projected_overlay.png
"""
import argparse

import cv2
import numpy as np
import open3d as o3d
import yaml
from PIL import Image


def load_camera_from_yaml(calib_path: str, camera_name: str):
    with open(calib_path, "r") as f:
        cfg = yaml.safe_load(f)
    cam = next(c for c in cfg["Camera"] if c["name"] == camera_name)

    intrinsics_4x4 = np.array(cam["intrinsics"], dtype=np.float64)
    fx, fy, cx, cy = intrinsics_4x4[0, 0], intrinsics_4x4[1, 1], intrinsics_4x4[0, 2], intrinsics_4x4[1, 2]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    dist = np.array(cam["distortion"]["K"], dtype=np.float64)
    # RATIONAL_POLYNOMIAL 模型标准长度是8: k1,k2,p1,p2,k3,k4,k5,k6。
    # 这批标定文件只给了6个值(最后一个恰好是0)，补零到8给 OpenCV 的 rational model 用。
    if dist.size < 8:
        dist = np.concatenate([dist, np.zeros(8 - dist.size)])

    cam2lidar = np.array(cam["cam2lidar"], dtype=np.float64)  # 4x4: 把相机系的点变换到雷达系
    return camera_matrix, dist, cam2lidar, int(cam["width"]), int(cam["height"])


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    inv = np.eye(4)
    inv[:3, :3] = R.T
    inv[:3, 3] = -R.T @ t
    return inv


def zbuffer_visible_mask(pixels: np.ndarray, depths: np.ndarray) -> np.ndarray:
    n = len(pixels)
    if n == 0:
        return np.zeros(0, dtype=bool)
    u = pixels[:, 0].astype(np.int64)
    v = pixels[:, 1].astype(np.int64)
    mult = int(u.max()) + 1
    key = v * mult + u
    order = np.lexsort((depths, key))
    ks = key[order]
    first = np.ones(n, dtype=bool)
    first[1:] = ks[1:] != ks[:-1]
    visible = np.zeros(n, dtype=bool)
    visible[order[first]] = True
    return visible


def depth_to_color(depths: np.ndarray, dmin: float, dmax: float) -> np.ndarray:
    """近处红、远处蓝，用 OpenCV 的 JET colormap。"""
    norm = np.clip((depths - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8).reshape(-1, 1)
    colored = cv2.applyColorMap(255 - gray, cv2.COLORMAP_JET)  # 反转让近处偏红
    return colored.reshape(-1, 3)[:, ::-1]  # BGR -> RGB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--camera-name", default="FRONT_CAMERA")
    ap.add_argument("--output", required=True)
    ap.add_argument("--point-radius", type=int, default=2)
    ap.add_argument("--direction", choices=["invert", "direct"], default="invert",
                     help="invert: camera_from_lidar = inverse(cam2lidar)（假设cam2lidar是lidar_from_camera）; "
                          "direct: camera_from_lidar = cam2lidar 直接用（假设cam2lidar字面上就是lidar点转到cam系）")
    args = ap.parse_args()

    camera_matrix, dist_coeffs, cam2lidar, cal_w, cal_h = load_camera_from_yaml(args.calib, args.camera_name)
    print(f"相机矩阵:\n{camera_matrix}")
    print(f"畸变系数(8参数rational model): {dist_coeffs}")

    pc = o3d.io.read_point_cloud(args.pcd)
    points_lidar = np.asarray(pc.points, dtype=np.float64)
    print(f"点云点数: {points_lidar.shape[0]}")

    # 点云里有 NaN（无效返回点），投影前先剔除
    valid_pc = ~np.isnan(points_lidar).any(axis=1)
    points_lidar = points_lidar[valid_pc]
    print(f"剔除NaN后点数: {points_lidar.shape[0]}")

    if args.direction == "invert":
        # 假设 cam2lidar 是 lidar_from_camera，取逆得到 camera_from_lidar
        camera_from_lidar = invert_transform(cam2lidar)
    else:
        # 假设 cam2lidar 字面意思就是"雷达点变换到相机系"，直接用
        camera_from_lidar = cam2lidar

    points_h = np.concatenate([points_lidar, np.ones((points_lidar.shape[0], 1))], axis=1)
    points_camera = (camera_from_lidar @ points_h.T).T[:, :3]

    # 只保留相机前方的点
    in_front = points_camera[:, 2] > 0.1
    points_camera = points_camera[in_front]
    print(f"相机前方的点: {points_camera.shape[0]}")

    # 关键过滤：畸变多项式只在有限的归一化半径范围内单调有效。360°雷达里有大量
    # 点虽然 z>0 但其实是从相机光轴极大偏角方向来的（接近90°侧向），x/z,y/z 会
    # 非常大，径向畸变多项式（尤其k1<0时）在这种极端半径下会发散/折返，把本该
    # 落在画面外的点错误地投影回画面内。按图像角点对应的归一化半径算一个安全
    # 上限（留一点余量），投影前先剔除超出这个范围的点。
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    corner_r = np.sqrt(max(cx, cal_w - cx) ** 2 / fx ** 2 + max(cy, cal_h - cy) ** 2 / fy ** 2)
    safe_r = corner_r * 1.2
    x_norm = points_camera[:, 0] / points_camera[:, 2]
    y_norm = points_camera[:, 1] / points_camera[:, 2]
    within_fov = (x_norm ** 2 + y_norm ** 2) < safe_r ** 2
    points_camera = points_camera[within_fov]
    print(f"安全视场半径阈值: {safe_r:.3f}，过滤后剩余点: {points_camera.shape[0]}")

    # cv2.projectPoints 会正确应用完整的 rational polynomial 畸变模型
    rvec = np.zeros(3)
    tvec = np.zeros(3)
    image_points, _ = cv2.projectPoints(
        points_camera.reshape(-1, 1, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    image_points = image_points.reshape(-1, 2)

    with Image.open(args.image) as im:
        img_w, img_h = im.size
        image = np.array(im.convert("RGB"))

    u, v = image_points[:, 0], image_points[:, 1]
    inside = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)
    pixels = np.rint(image_points[inside]).astype(np.int32)
    depths = points_camera[inside, 2]
    print(f"落在图像内的点: {pixels.shape[0]}")

    visible = zbuffer_visible_mask(pixels, depths)
    pixels = pixels[visible]
    depths = depths[visible]
    print(f"z-buffer后可见点(去遮挡): {pixels.shape[0]}")

    colors = depth_to_color(depths, depths.min(), depths.max())

    overlay = image.copy()
    r = args.point_radius
    for (px, py), color in zip(pixels, colors):
        cv2.circle(overlay, (int(px), int(py)), r, tuple(int(c) for c in color), -1)

    Image.fromarray(overlay).save(args.output)
    print(f"深度范围: {depths.min():.2f}m ~ {depths.max():.2f}m (红=近, 蓝=远)")
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
