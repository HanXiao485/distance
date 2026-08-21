#!/usr/bin/env python3
"""
Compute per-camera optical extrinsics and a ground-plane homography.

This script follows the exact axis convention used in the user's
`front_overlay_no_distortion.py`:

- YAML `cam2lidar` maps camera-body coordinates to lidar coordinates.
- Camera-body frame uses:
    x = front
    y = down
    z = left
- OpenCV optical frame uses:
    x = right
    y = down
    z = front

So we insert a fixed rotation:
    x_opt = -z_cam
    y_opt =  y_cam
    z_opt =  x_cam

The final homography maps plane coordinates [x, y, 1]^T on a chosen frame
(`ego` or `lidar`) at constant z = plane_z to image homogeneous coordinates.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


R_OPT_FROM_CAM_BODY = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def load_yaml_with_fallback(yaml_path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with yaml_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError:
        cmd = [
            "ruby",
            "-e",
            (
                "require 'yaml'; require 'json'; "
                "path = ARGV[0]; "
                "print JSON.generate(YAML.load_file(path))"
            ),
            str(yaml_path),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(proc.stdout)


def as_matrix4(value: Any) -> np.ndarray:
    mat = np.array(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got shape {mat.shape}.")
    return mat


def invert_transform(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    trans = t[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = r.T
    inv[:3, 3] = -(r.T @ trans)
    return inv


def embed_rotation(r: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    return out


def find_camera(data: dict[str, Any], camera_name: str) -> dict[str, Any]:
    cameras = data.get("Camera", [])
    for cam in cameras:
        if cam.get("name") == camera_name:
            return cam
    available = [cam.get("name") for cam in cameras]
    raise ValueError(f"Camera {camera_name!r} not found. Available: {available}")


def build_transforms(data: dict[str, Any], camera_name: str) -> dict[str, np.ndarray | int]:
    cam = find_camera(data, camera_name)

    intrinsics = np.array(cam["intrinsics"], dtype=np.float64)
    camera_matrix = intrinsics[:3, :3]
    t_cam_to_lidar = as_matrix4(cam["cam2lidar"])
    t_lidar_to_cam_body = invert_transform(t_cam_to_lidar)
    t_cam_body_to_opt = embed_rotation(R_OPT_FROM_CAM_BODY)
    t_lidar_to_opt = t_cam_body_to_opt @ t_lidar_to_cam_body

    out: dict[str, np.ndarray | int] = {
        "width": int(cam["width"]),
        "height": int(cam["height"]),
        "K": camera_matrix,
        "T_cam_to_lidar": t_cam_to_lidar,
        "T_lidar_to_cam_body": t_lidar_to_cam_body,
        "T_cam_body_to_opt": t_cam_body_to_opt,
        "T_lidar_to_opt": t_lidar_to_opt,
    }

    lidar_block = data.get("Lidar", {})
    if "lidar2ego" in lidar_block:
        t_lidar_to_ego = as_matrix4(lidar_block["lidar2ego"])
        t_ego_to_lidar = invert_transform(t_lidar_to_ego)
        t_ego_to_opt = t_lidar_to_opt @ t_ego_to_lidar
        out["T_lidar_to_ego"] = t_lidar_to_ego
        out["T_ego_to_lidar"] = t_ego_to_lidar
        out["T_ego_to_opt"] = t_ego_to_opt

    return out


def plane_homography(
    K: np.ndarray,
    T_frame_to_opt: np.ndarray,
    plane_z: float,
) -> np.ndarray:
    r = T_frame_to_opt[:3, :3]
    t = T_frame_to_opt[:3, 3:4]

    basis_x = np.array([[1.0], [0.0], [0.0]], dtype=np.float64)
    basis_y = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    origin = np.array([[0.0], [0.0], [plane_z]], dtype=np.float64)

    h = K @ np.concatenate(
        [
            r @ basis_x,
            r @ basis_y,
            r @ origin + t,
        ],
        axis=1,
    )
    return h


def normalize_h(h: np.ndarray) -> np.ndarray:
    if abs(h[2, 2]) < 1e-12:
        return h
    return h / h[2, 2]


def camera_center_in_source_frame(T_source_to_opt: np.ndarray) -> np.ndarray:
    T_opt_to_source = invert_transform(T_source_to_opt)
    return T_opt_to_source[:3, 3]


def print_matrix(name: str, value: np.ndarray) -> None:
    print(f"{name} =")
    with np.printoptions(precision=6, suppress=True):
        print(value)
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Calibration YAML path")
    parser.add_argument("--camera", default="FRONT_CAMERA", help="Camera name in YAML")
    parser.add_argument(
        "--plane-frame",
        choices=["ego", "lidar"],
        default="ego",
        help="Which frame defines the plane coordinates",
    )
    parser.add_argument(
        "--plane-z",
        type=float,
        default=0.0,
        help="Constant z value of the plane in the selected frame",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save matrices and metadata as JSON",
    )
    args = parser.parse_args()

    yaml_path = Path(args.yaml)
    data = load_yaml_with_fallback(yaml_path)
    transforms = build_transforms(data, args.camera)

    K = transforms["K"]
    T_lidar_to_opt = transforms["T_lidar_to_opt"]
    T_ego_to_opt = transforms.get("T_ego_to_opt")

    if args.plane_frame == "ego":
        if T_ego_to_opt is None:
            raise ValueError("YAML does not contain lidar2ego, so ego-frame homography cannot be built.")
        T_plane_to_opt = T_ego_to_opt
    else:
        T_plane_to_opt = T_lidar_to_opt

    H_plane_to_image = normalize_h(plane_homography(K, T_plane_to_opt, args.plane_z))
    H_image_to_plane = normalize_h(np.linalg.inv(H_plane_to_image))

    print(f"camera = {args.camera}")
    print(f"image_size = {transforms['width']} x {transforms['height']}")
    print("camera_body_axes = {x: front, y: down, z: left}")
    print("optical_axes = {x: right, y: down, z: front}")
    print(f"plane_frame = {args.plane_frame}")
    print(f"plane_definition = z == {args.plane_z} in {args.plane_frame} frame")
    print()

    print_matrix("K", K)
    print_matrix("T_lidar_to_opt", T_lidar_to_opt)
    if T_ego_to_opt is not None:
        print_matrix("T_ego_to_opt", T_ego_to_opt)

    center_lidar = camera_center_in_source_frame(T_lidar_to_opt)
    print_matrix("camera_center_in_lidar", center_lidar.reshape(3, 1))
    if T_ego_to_opt is not None:
        center_ego = camera_center_in_source_frame(T_ego_to_opt)
        print_matrix("camera_center_in_ego", center_ego.reshape(3, 1))

    print_matrix("H_plane_to_image", H_plane_to_image)
    print_matrix("H_image_to_plane", H_image_to_plane)

    if args.output_json:
        payload = {
            "camera": args.camera,
            "image_size": {
                "width": int(transforms["width"]),
                "height": int(transforms["height"]),
            },
            "plane_frame": args.plane_frame,
            "plane_z": args.plane_z,
            "camera_body_axes": {"x": "front", "y": "down", "z": "left"},
            "optical_axes": {"x": "right", "y": "down", "z": "front"},
            "K": K.tolist(),
            "T_lidar_to_opt": T_lidar_to_opt.tolist(),
            "H_plane_to_image": H_plane_to_image.tolist(),
            "H_image_to_plane": H_image_to_plane.tolist(),
        }
        if T_ego_to_opt is not None:
            payload["T_ego_to_opt"] = T_ego_to_opt.tolist()
        Path(args.output_json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved_json = {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
