#!/usr/bin/env python3
"""Create a timestamp-ordered MP4 from dataset images."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def timestamp_from_stem(stem: str) -> float:
    """Parse both ``seconds.nanoseconds`` and ``seconds-nanoseconds`` names."""
    return float(stem.replace("-", ".", 1))


def sorted_image_paths(image_dir: Path) -> list[Path]:
    paths = [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(paths, key=lambda p: (timestamp_from_stem(p.stem), p.name))


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize without distortion and pad to ``(width, height)``."""
    width, height = size
    src_h, src_w = frame.shape[:2]
    scale = min(width / src_w, height / src_h)
    dst_w, dst_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    resized = cv2.resize(frame, (dst_w, dst_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), (13, 13, 26), dtype=frame.dtype)
    x = (width - dst_w) // 2
    y = (height - dst_h) // 2
    canvas[y:y + dst_h, x:x + dst_w] = resized
    return canvas


def transcode_h264(intermediate_path: Path, output_path: Path) -> None:
    """Convert an OpenCV MP4V intermediate to browser-compatible H.264."""
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error",
        "-i", str(intermediate_path), "-an", "-c:v", "libx264",
        "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"H.264 transcoding failed: {result.stderr.strip()}")
    intermediate_path.unlink(missing_ok=True)


def create_video(
    image_paths: Iterable[Path],
    output_path: Path,
    fps: float = 5.0,
    size: tuple[int, int] | None = None,
) -> dict:
    paths = list(image_paths)
    if not paths:
        raise ValueError("No input images were provided")

    first = cv2.imread(str(paths[0]))
    if first is None:
        raise ValueError(f"Cannot read first image: {paths[0]}")
    if size is None:
        size = (first.shape[1], first.shape[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path.with_name(f".{output_path.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(
        str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer: {output_path}")

    written = 0
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(f"[WARN] Skipping unreadable image: {path}")
                continue
            if (frame.shape[1], frame.shape[0]) != size:
                frame = letterbox(frame, size)
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    if written == 0:
        intermediate_path.unlink(missing_ok=True)
        raise RuntimeError("No readable images were written")
    transcode_h264(intermediate_path, output_path)
    return {
        "path": str(output_path),
        "frame_count": written,
        "fps": fps,
        "width": size[0],
        "height": size[1],
        "duration_sec": written / fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    paths = sorted_image_paths(args.image_dir)
    if args.max_frames > 0:
        paths = paths[:args.max_frames]
    metadata = create_video(paths, args.output, args.fps)
    print(metadata)


if __name__ == "__main__":
    main()
