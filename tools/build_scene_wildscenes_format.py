"""
build_scene_wildscenes_format.py
把自采场景的原始数据（<scene>/parameters/car006.yaml 标定 + point_data/<timestamp>/ 逐帧文件夹）
和标注公司交付的2D标注结果（gtFine/ 目录），整理成模仿WildScenes的目录结构：

WildScenes2d/<CAMERA_NAME>/
    camera_calibration.yaml
    poses2d.csv
    image/<sec>-<nanosec>.jpg
    indexLabel/<sec>-<nanosec>.png   (只有标注了的帧才有)
    color/<sec>-<nanosec>.png        (标注可视化图，非WildScenes标准部分，留作参考)
    polygons/<sec>-<nanosec>.json    (原始多边形标注，非WildScenes标准部分，留作参考)
WildScenes3d/
    poses3d.csv
    Clouds/<sec>.<nanosec>.pcd
    Labels/                          (预留，3D标注还没有时是空文件夹)

关键假设（针对 scene204 这批数据验证过，其它自采场景如果标定文件格式一致应该同样适用）：
- car006.yaml 的 cam2lidar 是直接的 camera_from_lidar（不取逆，已用点云投影验证过，
  见项目memory scene204_cam2lidar_direction）
- lidar2ego 按同样的命名约定，假设也是直接的 ego_from_lidar（未独立验证，是推断延续）
- ego2global 的 rotation=[roll,pitch,yaw] 用 scipy 'xyz' 外旋顺序转旋转矩阵（车辆本体常见约定）
- 标注公司交付的文件命名遵循 <相机名>_<场景名>_<sec>_<nanosec>_{labelIds,color,polygons}.{png,json}，
  跟原始jpg同一份 <sec>_<nanosec> 时间戳前缀对齐

用法:
    python3 build_scene_wildscenes_format.py \
        --raw-root "/path/to/sceneXXX/parameters和point_data所在目录" \
        --scene-name scene204 \
        --annot-dir /path/to/annotation/gtFine \
        --out-root /path/to/output
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CAMERAS = [
    "FRONT_CAMERA", "FRONT_LEFT_CAMERA", "FRONT_RIGHT_CAMERA",
    "BACK_CAMERA", "BACK_LEFT_CAMERA", "BACK_RIGHT_CAMERA",
]


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_quat_wxyz(T: np.ndarray) -> np.ndarray:
    q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
    return np.array([q[3], q[0], q[1], q[2]])


def matrix_to_quat_xyzw(T: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(T[:3, :3]).as_quat()


def ts_underscore_to_dash(ts: str) -> str:
    return ts.replace("_", "-")


def ts_underscore_to_decimal(ts: str) -> str:
    sec, nano = ts.split("_")
    return f"{sec}.{nano}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True, type=Path,
                     help="场景原始数据根目录，下面应有 parameters/car006.yaml 和 point_data/<timestamp>/")
    ap.add_argument("--scene-name", required=True,
                     help="场景名，比如 scene204，用于匹配文件名里的 <相机名>_<场景名>_<timestamp> 模式")
    ap.add_argument("--annot-dir", required=True, type=Path,
                     help="标注公司交付的 gtFine 目录（包含 *_labelIds.png / *_color.png / *_polygons.json）")
    ap.add_argument("--out-root", required=True, type=Path, help="输出目录")
    ap.add_argument("--calib-file", default="car006.yaml", help="标定文件名（默认 car006.yaml）")
    return ap.parse_args()


def main():
    args = parse_args()
    raw_root = args.raw_root
    annot_dir = args.annot_dir
    out_root = args.out_root
    scene_name = args.scene_name

    with open(raw_root / "parameters" / args.calib_file) as f:
        car_cfg = yaml.safe_load(f)

    lidar2ego = np.array(car_cfg["Lidar"]["lidar2ego"], dtype=np.float64)

    cam_calib = {}
    for cam in car_cfg["Camera"]:
        name = cam["name"]
        intrinsics = np.array(cam["intrinsics"], dtype=np.float64)
        fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
        dist = np.array(cam["distortion"]["K"], dtype=np.float64)[:5]  # k1,k2,p1,p2,k3
        cam2lidar = np.array(cam["cam2lidar"], dtype=np.float64)  # camera_from_lidar，直接用
        cam_calib[name] = dict(
            fx=fx, fy=fy, cx=cx, cy=cy, dist=dist, cam2lidar=cam2lidar,
            width=int(cam["width"]), height=int(cam["height"]),
        )

    frame_dirs = sorted(d for d in (raw_root / "point_data").iterdir() if d.is_dir())
    print(f"共 {len(frame_dirs)} 个原始帧文件夹")

    frame_world_from_lidar: dict[str, np.ndarray] = {}
    frame_pcd_path: dict[str, Path] = {}
    # camera -> list of (frame_ts, cam_ts, jpg_path)
    camera_entries: dict[str, list[tuple[str, str, Path]]] = {c: [] for c in CAMERAS}

    for d in frame_dirs:
        frame_ts = d.name
        ego_json = json.loads((d / f"ego2global_{scene_name}_{frame_ts}.json").read_text())
        roll, pitch, yaw = ego_json["rotation"]
        tx, ty, tz = ego_json["translation"]
        R_ego = euler_to_matrix(roll, pitch, yaw)
        world_from_ego = make_transform(R_ego, np.array([tx, ty, tz], dtype=np.float64))
        world_from_lidar = world_from_ego @ lidar2ego
        frame_world_from_lidar[frame_ts] = world_from_lidar

        pcd_files = list(d.glob(f"centerlidar_{scene_name}_*.pcd"))
        assert len(pcd_files) == 1, f"{d} 找到 {len(pcd_files)} 个pcd"
        frame_pcd_path[frame_ts] = pcd_files[0]

        cam_time_json = json.loads((d / f"camera_time_{scene_name}_{frame_ts}.json").read_text())
        cam_time_map = {}
        for entry in cam_time_json:
            cam_time_map.update(entry)

        for cam in CAMERAS:
            cam_ts = cam_time_map.get(cam)
            if cam_ts is None:
                continue
            jpg_files = list(d.glob(f"{cam}_{scene_name}_*.jpg"))
            if not jpg_files:
                continue
            camera_entries[cam].append((frame_ts, cam_ts, jpg_files[0]))

    # ── 3D 部分 ──────────────────────────────────────────────────────
    d3_root = out_root / "WildScenes3d"
    (d3_root / "Clouds").mkdir(parents=True, exist_ok=True)
    (d3_root / "Labels").mkdir(parents=True, exist_ok=True)  # 预留，暂时为空

    with (d3_root / "poses3d.csv").open("w") as f:
        f.write("idx timestamp x y z qw qx qy qz\n")
        for i, frame_ts in enumerate(sorted(frame_world_from_lidar)):
            T = frame_world_from_lidar[frame_ts]
            qw, qx, qy, qz = matrix_to_quat_wxyz(T)
            x, y, z = T[:3, 3]
            ts_decimal = ts_underscore_to_decimal(frame_ts)
            f.write(f"{i} {ts_decimal} {x} {y} {z} {qw} {qx} {qy} {qz}\n")

    for frame_ts, pcd_path in frame_pcd_path.items():
        out_name = ts_underscore_to_decimal(frame_ts) + ".pcd"
        shutil.copy(pcd_path, d3_root / "Clouds" / out_name)

    print(f"3D: poses3d.csv 写了 {len(frame_world_from_lidar)} 行，Clouds/ 拷贝了 {len(frame_pcd_path)} 个pcd")
    print("3D: Labels/ 已创建为空文件夹（预留，等3D标注完成后再放）")

    # ── 2D 部分（每个相机一个伪"序列"）─────────────────────────────────
    total_images = 0
    total_labels = 0
    for cam in CAMERAS:
        entries = camera_entries[cam]
        if not entries:
            print(f"[跳过] {cam} 没有任何帧")
            continue
        cam_root = out_root / "WildScenes2d" / cam
        (cam_root / "image").mkdir(parents=True, exist_ok=True)
        (cam_root / "indexLabel").mkdir(parents=True, exist_ok=True)
        (cam_root / "color").mkdir(parents=True, exist_ok=True)
        (cam_root / "polygons").mkdir(parents=True, exist_ok=True)

        # 同一张相机图片可能被多个LiDAR锚点帧引用（相机采样率低于LiDAR），
        # 按cam_ts去重，只保留第一次出现的那一份，避免poses2d.csv里出现
        # 同一时间戳、不同数值的两行（parse_pose_csv按时间戳做字典查找，
        # 重复key会被后写入的行静默覆盖，产生不确定性）。
        seen_cam_ts: set[str] = set()
        deduped_entries = []
        for frame_ts, cam_ts, jpg_path in sorted(entries, key=lambda e: e[1]):
            if cam_ts in seen_cam_ts:
                continue
            seen_cam_ts.add(cam_ts)
            deduped_entries.append((frame_ts, cam_ts, jpg_path))
        entries = deduped_entries

        calib = cam_calib[cam]
        qx, qy, qz, qw = matrix_to_quat_xyzw(calib["cam2lidar"])
        tx, ty, tz = calib["cam2lidar"][:3, 3]
        calib_text = (
            f"K: [{calib['fx']}, {calib['fy']}, {calib['cx']}, {calib['cy']}]\n"
            f"D: [{', '.join(str(v) for v in calib['dist'])}]\n"
            f"translation: [{tx}, {ty}, {tz}]\n"
            f"rotation: [{qx}, {qy}, {qz}, {qw}]\n"
            f"# 注：translation/rotation 是 cam2lidar(camera_from_lidar) 直接分解得到，\n"
            f"# 当前 distance 包的 extract_correspondences 实际不读取这两个字段\n"
            f"# (相机世界位姿来自 poses2d.csv)，这里填写仅为完整性/以后可能用到。\n"
            f"# width: {calib['width']}, height: {calib['height']}\n"
        )
        (cam_root / "camera_calibration.yaml").write_text(calib_text)

        n_img = 0
        n_lbl = 0
        with (cam_root / "poses2d.csv").open("w") as f:
            f.write("idx timestamp x y z qw qx qy qz\n")
            for i, (frame_ts, cam_ts, jpg_path) in enumerate(sorted(entries, key=lambda e: e[1])):
                world_from_lidar = frame_world_from_lidar[frame_ts]
                # cam2lidar is T_camera_from_lidar; we need T_world_from_camera
                # = T_world_from_lidar @ T_lidar_from_camera
                # = T_world_from_lidar @ inv(cam2lidar)
                world_from_camera = world_from_lidar @ np.linalg.inv(calib["cam2lidar"])
                qw2, qx2, qy2, qz2 = matrix_to_quat_wxyz(world_from_camera)
                x2, y2, z2 = world_from_camera[:3, 3]
                ts_decimal = ts_underscore_to_decimal(cam_ts)
                f.write(f"{i} {ts_decimal} {x2} {y2} {z2} {qw2} {qx2} {qy2} {qz2}\n")

                out_stem = ts_underscore_to_dash(cam_ts)
                shutil.copy(jpg_path, cam_root / "image" / f"{out_stem}.jpg")
                n_img += 1

                label_base = annot_dir / f"{cam}_{scene_name}_{cam_ts}"
                label_png = label_base.with_name(label_base.name + "_labelIds.png")
                color_png = label_base.with_name(label_base.name + "_color.png")
                poly_json = label_base.with_name(label_base.name + "_polygons.json")
                if label_png.exists():
                    shutil.copy(label_png, cam_root / "indexLabel" / f"{out_stem}.png")
                    n_lbl += 1
                if color_png.exists():
                    shutil.copy(color_png, cam_root / "color" / f"{out_stem}.png")
                if poly_json.exists():
                    shutil.copy(poly_json, cam_root / "polygons" / f"{out_stem}.json")

        print(f"{cam}: {n_img} 张图, {n_lbl} 张已标注")
        total_images += n_img
        total_labels += n_lbl

    print(f"\n总计: {total_images} 张图（6路相机），{total_labels} 张已标注")
    print(f"输出目录: {out_root}")


if __name__ == "__main__":
    main()
