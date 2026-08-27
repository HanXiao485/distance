#!/usr/bin/env python3
"""Run the scene204 segmentation→DISTANCE→FPV→BEV real-time experiment."""
from __future__ import annotations

import argparse
import bisect
import copy
import json
import shutil
import sys
import time
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_input_video import create_video, letterbox, timestamp_from_stem, transcode_h264
from distance.io_utils import resolve_path
from realtime_optimized import (
    install_point_cloud_cache, make_bev_context, process_frame_phase2_in_memory,
    render_bev_opencv, render_fpv_in_memory,
)
from distance.run_sequence import (
    build_frame_config,
    discover_frames,
    find_candidate_images,
    process_frame_phase1,
    read_config,
)


def select_unique_image_frames(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Choose the closest valid point-cloud anchor for every unique input image."""
    anchors, _ = discover_frames(cfg, require_pred_mask=False)
    selected: dict[str, tuple[float, dict[str, Any]]] = {}
    for anchor in anchors:
        rgb = str(Path(anchor["rgb"]).resolve())
        image_ts = timestamp_from_stem(Path(rgb).stem)
        diff = abs(float(anchor["frame_id"]) - image_ts)
        if rgb not in selected or diff < selected[rgb][0]:
            selected[rgb] = (diff, copy.deepcopy(anchor))
    return [item[1] for item in sorted(
        selected.values(), key=lambda item: timestamp_from_stem(Path(item[1]["rgb"]).stem)
    )]


def load_segmentation_model(cfg: dict[str, Any], device: str):
    segmentation = cfg["segmentation"]
    config_path = resolve_path(cfg, segmentation["config"]).resolve()
    checkpoint_path = resolve_path(cfg, segmentation["checkpoint"]).resolve()
    if not config_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model files missing: {config_path}, {checkpoint_path}")

    # Register all custom WildScenes/GOOSE datasets used by current configs.
    try:
        import wildscenes.mmseg_wildscenes.dataset.goose  # noqa: F401
        import wildscenes.mmseg_wildscenes.dataset.goose_category  # noqa: F401
    except ImportError:
        pass
    from mmseg.apis import init_model

    started = time.perf_counter()
    model = init_model(str(config_path), str(checkpoint_path), device=device)
    return model, time.perf_counter() - started


def infer_binary_mask(model, image_path: Path, class_ids: list[int], output_path: Path) -> float:
    from mmseg.apis import inference_model

    started = time.perf_counter()
    result = inference_model(model, str(image_path))
    pred = result.pred_sem_seg.data.squeeze().detach().cpu().numpy().astype(np.int64)
    mask = np.isin(pred, np.asarray(class_ids, dtype=np.int64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output_path)
    return time.perf_counter() - started


def put_label(canvas: np.ndarray, text: str, origin: tuple[int, int],
              color=(255, 255, 255), scale: float = 0.65) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 2, cv2.LINE_AA)


def compose_latency_video(
    records: list[dict[str, Any]], output_path: Path, fps: float,
    show_fpv: bool, show_bev: bool, input_frames: list[np.ndarray],
    visual_frames: dict[int, tuple[np.ndarray | None, np.ndarray | None]],
    size: tuple[int, int] = (1920, 1080), max_duration_sec: float = 0.0,
) -> dict[str, Any]:
    """Render input arrivals against outputs completed by each video time."""
    successful = [r for r in records if not r.get("error")]
    if not records:
        raise ValueError("No timing records available")

    period = 1.0 / fps
    end_time = max(records[-1]["arrival_sec"] + period,
                   max((r["completion_sec"] for r in records), default=0.0) + period)
    truncated = False
    if max_duration_sec > 0 and end_time > max_duration_sec:
        end_time = max_duration_sec
        truncated = True
    frame_count = max(1, int(np.ceil(end_time * fps)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    intermediate_path = output_path.with_name(f".{output_path.stem}.mp4v.mp4")
    writer = cv2.VideoWriter(str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {output_path}")

    width, height = size
    header_h = 72
    half_w = width // 2
    right_height = height - header_h
    arrivals = [r["arrival_sec"] for r in records]
    completions = [r["completion_sec"] for r in successful]

    try:
        for video_index in range(frame_count):
            now = video_index / fps
            input_index = min(max(0, bisect.bisect_right(arrivals, now) - 1), len(records) - 1)
            output_index = bisect.bisect_right(completions, now) - 1
            raw = input_frames[input_index]

            canvas = np.full((height, width, 3), (13, 13, 26), np.uint8)
            canvas[header_h:, :half_w] = letterbox(raw, (half_w, right_height))
            cv2.rectangle(canvas, (half_w, header_h), (width - 1, height - 1), (55, 55, 75), 1)
            displayed = None
            if output_index >= 0:
                displayed = successful[output_index]
                fpv, bev = visual_frames[displayed["index"]]
                if show_fpv and show_bev:
                    top_h = right_height // 2
                    canvas[header_h:header_h + top_h, half_w:] = letterbox(fpv, (width - half_w, top_h))
                    canvas[header_h + top_h:, half_w:] = letterbox(bev, (width - half_w, right_height - top_h))
                    put_label(canvas, "First-person view", (half_w + 18, header_h + 28), scale=0.55)
                    put_label(canvas, "Bird's-eye view", (half_w + 18, header_h + top_h + 28), scale=0.55)
                elif show_fpv:
                    canvas[header_h:, half_w:] = letterbox(fpv, (width - half_w, right_height))
                    put_label(canvas, "First-person view", (half_w + 18, header_h + 28), scale=0.55)
                elif show_bev:
                    canvas[header_h:, half_w:] = letterbox(bev, (width - half_w, right_height))
                    put_label(canvas, "Bird's-eye view", (half_w + 18, header_h + 28), scale=0.55)
                else:
                    put_label(canvas, "Processing complete (visualizations disabled)",
                              (half_w + 85, height // 2), (100, 220, 255), 0.72)
            else:
                put_label(canvas, "WAITING FOR FIRST COMPLETED RESULT...",
                          (half_w + 90, height // 2), (0, 190, 255), 0.75)

            enabled_names = [name for name, enabled in (("FPV", show_fpv), ("BEV", show_bev)) if enabled]
            put_label(canvas, "LIVE INPUT", (22, 46), (80, 255, 120), 0.85)
            put_label(canvas, "COMPLETED OUTPUT: " + (" + ".join(enabled_names) or "TIMING ONLY"),
                      (half_w + 22, 46), (80, 200, 255), 0.8)
            put_label(canvas, f"timeline={now:7.2f}s  input={records[input_index]['image_id']}",
                      (22, height - 20), scale=0.55)
            if displayed:
                lag = max(0.0, records[input_index]["arrival_sec"] - displayed["arrival_sec"])
                latency = displayed["completion_sec"] - displayed["arrival_sec"]
                put_label(canvas, f"result={displayed['image_id']}  E2E={latency:.2f}s  visual lag={lag:.2f}s",
                          (half_w + 22, height - 20), (100, 220, 255), 0.55)
            writer.write(canvas)
    finally:
        writer.release()

    transcode_h264(intermediate_path, output_path)
    return {"path": str(output_path), "fps": fps, "frame_count": frame_count,
            "duration_sec": frame_count / fps, "truncated": truncated}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    success = [r for r in records if not r.get("error")]
    stage_names = ("capture_decode_sec", "segmentation_sec", "error_calculation_sec",
                   "fpv_sec", "bev_sec", "processing_sec")
    stages = {}
    for name in stage_names:
        values = np.asarray([r.get(name, 0.0) for r in success], dtype=float)
        stages[name] = {"count": int(values.size),
                        "mean_ms": float(values.mean() * 1000) if values.size else None,
                        "p50_ms": float(np.percentile(values, 50) * 1000) if values.size else None,
                        "p90_ms": float(np.percentile(values, 90) * 1000) if values.size else None,
                        "max_ms": float(values.max() * 1000) if values.size else None}
    latencies = np.asarray([r["completion_sec"] - r["arrival_sec"] for r in success], dtype=float)
    return {"frames": len(records), "successful_frames": len(success),
            "failed_frames": len(records) - len(success), "stages": stages,
            "end_to_end_latency": {
                "mean_ms": float(latencies.mean() * 1000) if latencies.size else None,
                "p90_ms": float(np.percentile(latencies, 90) * 1000) if latencies.size else None,
                "max_ms": float(latencies.max() * 1000) if latencies.size else None}}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_testing_config(path: Path) -> dict[str, Any]:
    defaults = {
        "distance": {"config": "configs/evaluation.yaml", "task": "scene204_road"},
        "stages": {"segmentation": True, "error_calculation": True,
                   "fpv_visualization": True, "bev_visualization": True,
                   "comparison_video": True},
        "segmentation": {"cached_mask_dir": None},
        "runtime": {"fps": 5.0, "device": "cuda:0", "max_frames": 0,
                    "warmup": 1, "max_output_duration": 0.0},
        "output": {"dir": "real-time-testing/outputs/scene204_road"},
    }
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(document, dict):
        raise ValueError(f"Testing config must contain a mapping: {path}")
    config = deep_merge(defaults, document)
    for name, value in config["stages"].items():
        if not isinstance(value, bool):
            raise TypeError(f"stages.{name} must be true or false")
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-config", type=Path,
                        default=SCRIPT_DIR / "pipeline_config.yaml")
    parser.add_argument("--config", type=Path, default=None, help="Override distance.config")
    parser.add_argument("--task", default=None, help="Override distance.task")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output.dir")
    parser.add_argument("--fps", type=float, default=None, help="Override runtime.fps")
    parser.add_argument("--device", default=None, help="Override runtime.device")
    parser.add_argument("--max-frames", type=int, default=None, help="Override runtime.max_frames")
    parser.add_argument("--warmup", type=int, default=None, help="Override runtime.warmup")
    parser.add_argument("--max-output-duration", type=float, default=None,
                        help="Override runtime.max_output_duration")
    args = parser.parse_args()

    test_cfg = load_testing_config(args.pipeline_config.resolve())
    runtime = test_cfg["runtime"]
    stages = test_cfg["stages"]
    distance_config = (args.config or project_path(test_cfg["distance"]["config"])).resolve()
    task = args.task or test_cfg["distance"]["task"]
    output_dir = (args.output_dir or project_path(test_cfg["output"]["dir"])).resolve()
    fps = args.fps if args.fps is not None else float(runtime["fps"])
    device = args.device or runtime["device"]
    max_frames = args.max_frames if args.max_frames is not None else int(runtime["max_frames"])
    warmup = args.warmup if args.warmup is not None else int(runtime["warmup"])
    max_output_duration = (args.max_output_duration if args.max_output_duration is not None
                           else float(runtime["max_output_duration"]))

    decoded_dir, mask_dir = output_dir / "decoded_frames", output_dir / "pred_masks"
    pipeline_dir = output_dir / "pipeline"
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_config(distance_config, task)
    cfg["project_root"] = str(PROJECT_ROOT)
    model_settings = copy.deepcopy(cfg["segmentation"])
    records = select_unique_image_frames(cfg)
    if max_frames > 0:
        records = records[:max_frames]
    if not records:
        raise RuntimeError("No processable input frames were found")

    source_image_paths = [Path(record["rgb"]).resolve() for record in records]
    input_video_path = output_dir / "input_video.mp4"
    input_metadata = create_video(source_image_paths, input_video_path, fps)
    original_pred_mask_dir = project_path(cfg["dataset"].get("pred_mask_dir", mask_dir))
    cfg["dataset"]["pred_mask_dir"] = str(mask_dir)
    cfg["segmentation"] = {**model_settings, "enabled": False}
    cfg["output_dir"] = str(pipeline_dir)
    # Keep only numerical error results in this benchmark. H candidate search
    # and fitting remain unchanged; diagnostic images, CSV files, histograms,
    # heatmaps and scanline overlays are omitted from the timed error stage.
    cfg.setdefault("h_diagnostics", {})["enabled"] = False
    cfg.setdefault("scanline", {})["save_overlays"] = False

    model, model_load_sec = None, 0.0
    class_ids = [int(value) for value in model_settings.get("class_ids", [])]
    if stages["segmentation"]:
        if not class_ids:
            raise ValueError("The selected task must define segmentation.class_ids")
        model, model_load_sec = load_segmentation_model({**cfg, "segmentation": model_settings}, device)
        for _ in range(max(0, warmup)):
            from mmseg.apis import inference_model
            inference_model(model, str(source_image_paths[0]))

    cached_value = test_cfg["segmentation"].get("cached_mask_dir")
    cached_mask_dir = project_path(cached_value) if cached_value else original_pred_mask_dir
    with Image.open(source_image_paths[0]) as first_image:
        image_size = first_image.size
    install_point_cloud_cache()
    bev_context = make_bev_context(cfg) if stages["bev_visualization"] else None

    # Warm all enabled stages before the timed input timeline starts. No capture
    # frame is consumed here, so video playback and arrival times still start at 0.
    warmup_started = time.perf_counter()
    if warmup > 0:
        with tempfile.TemporaryDirectory(prefix="distance-realtime-warmup-") as tmp_name:
            warm_dir = Path(tmp_name)
            warm_frame = copy.deepcopy(records[0])
            warm_frame["rgb"] = str(source_image_paths[0])
            if stages["segmentation"]:
                warm_mask = warm_dir / "mask.png"
                infer_binary_mask(model, source_image_paths[0], class_ids, warm_mask)
            else:
                warm_mask = cached_mask_dir / f"{source_image_paths[0].stem}.png"
            warm_frame["pred_mask"] = str(warm_mask)
            warm_phase2 = None
            if stages["error_calculation"]:
                warm_candidates = [{"image_id": source_image_paths[0].stem,
                                    "rgb": str(source_image_paths[0]),
                                    "timestamp": timestamp_from_stem(source_image_paths[0].stem)}]
                warm_phase1 = process_frame_phase1(cfg, warm_frame, warm_candidates, warm_dir, False)
                if not warm_phase1.get("error"):
                    warm_phase2 = process_frame_phase2_in_memory(
                        cfg, warm_frame, warm_dir, Path(warm_phase1["h_json"]), False)
            warm_image = cv2.imread(str(source_image_paths[0]))
            if stages["fpv_visualization"] and warm_image is not None:
                render_fpv_in_memory(warm_image, warm_mask)
            if stages["bev_visualization"]:
                usable_phase2 = warm_phase2 if warm_phase2 and not warm_phase2.get("error") else None
                render_bev_opencv(bev_context, warm_frame, usable_phase2)
    warmup_sec = time.perf_counter() - warmup_started

    timeline: list[dict[str, Any]] = []
    input_frames: list[np.ndarray] = []
    visual_frames: dict[int, tuple[np.ndarray | None, np.ndarray | None]] = {}
    available_candidates: list[dict[str, Any]] = []
    last_valid_h_json: Path | None = None
    previous_completion = 0.0
    capture = cv2.VideoCapture(str(input_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open generated input video: {input_video_path}")
    decoded_dir.mkdir(parents=True, exist_ok=True)
    try:
        for index, (frame, source_image_path) in enumerate(zip(records, source_image_paths)):
            arrival = index / fps
            processing_start = max(arrival, previous_completion)
            image_id = source_image_path.stem
            decoded_path = decoded_dir / f"{image_id}.png"
            item = copy.deepcopy(frame)
            item.update({"index": index, "image_id": image_id, "source_rgb": str(source_image_path),
                         "rgb": str(decoded_path), "arrival_sec": arrival,
                         "processing_start_sec": processing_start, "stages": copy.deepcopy(stages)})
            for key in ("capture_decode_sec", "segmentation_sec", "error_calculation_sec", "fpv_sec", "bev_sec"):
                item[key] = 0.0
            print(f"[{index + 1}/{len(records)}] {image_id} -> anchor {frame['frame_id']}")
            total_started = time.perf_counter()
            try:
                decode_started = time.perf_counter()
                ok, decoded_frame = capture.read()
                if not ok or decoded_frame is None:
                    raise RuntimeError(f"Cannot decode input video frame {index}")
                input_frames.append(decoded_frame.copy())
                if not cv2.imwrite(str(decoded_path), decoded_frame):
                    raise RuntimeError(f"Cannot save decoded frame: {decoded_path}")
                item["capture_decode_sec"] = time.perf_counter() - decode_started

                mask_path: Path | None = None
                needs_mask = stages["segmentation"] or stages["error_calculation"] or stages["fpv_visualization"]
                if stages["segmentation"]:
                    mask_path = mask_dir / f"{image_id}.png"
                    item["segmentation_sec"] = infer_binary_mask(model, decoded_path, class_ids, mask_path)
                elif needs_mask:
                    mask_path = cached_mask_dir / f"{image_id}.png"
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"Cached mask required while segmentation is disabled: {mask_path}")
                    item["segmentation_skipped"] = True
                item["pred_mask"] = str(mask_path) if mask_path else ""

                phase2 = None
                if stages["error_calculation"]:
                    error_started = time.perf_counter()
                    available_candidates.append({"image_id": image_id, "rgb": str(decoded_path),
                                                 "timestamp": timestamp_from_stem(image_id)})
                    candidates = find_candidate_images(float(item["frame_id"]), available_candidates,
                                                       float(cfg.get("h_search_window_sec", 15.0)))
                    phase1 = process_frame_phase1(cfg, item, candidates, pipeline_dir, False)
                    if phase1.get("error"):
                        if last_valid_h_json is None:
                            raise RuntimeError(phase1["error"])
                        fallback_h = pipeline_dir / "frames" / item["frame_id"] / "h_estimation" / f"{cfg['class']['name']}_h_estimation.json"
                        fallback_h.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(last_valid_h_json, fallback_h)
                        selected_h_json = fallback_h
                        item["h_fallback"], item["h_fallback_source"] = True, str(last_valid_h_json)
                    else:
                        selected_h_json = Path(phase1["h_json"])
                        last_valid_h_json = selected_h_json
                        item["h_fallback"] = False
                    phase2 = process_frame_phase2_in_memory(cfg, item, pipeline_dir, selected_h_json, False)
                    if phase2.get("error"):
                        raise RuntimeError(phase2["error"])
                    item["error_calculation_sec"] = time.perf_counter() - error_started
                else:
                    item["error_calculation_skipped"] = True

                fpv_frame = None
                if stages["fpv_visualization"]:
                    fpv_frame, item["fpv_sec"] = render_fpv_in_memory(decoded_frame, mask_path)
                    item["fpv_in_memory"] = True
                else:
                    item["fpv_skipped"] = True

                bev_frame = None
                if stages["bev_visualization"]:
                    bev_frame, item["bev_sec"] = render_bev_opencv(bev_context, item, phase2)
                    item["bev_in_memory"] = True
                else:
                    item["bev_skipped"] = True
                visual_frames[index] = (fpv_frame, bev_frame)
            except Exception as exc:
                item["error"] = f"{type(exc).__name__}: {exc}"
                print(f"  [ERROR] {item['error']}")

            item["processing_sec"] = time.perf_counter() - total_started
            item["completion_sec"] = processing_start + item["processing_sec"]
            previous_completion = item["completion_sec"]
            timeline.append(item)
            print("  decode={:.1f}ms seg={:.1f}ms error={:.1f}ms FPV={:.1f}ms BEV={:.1f}ms total={:.1f}ms".format(
                item["capture_decode_sec"] * 1000, item["segmentation_sec"] * 1000,
                item["error_calculation_sec"] * 1000, item["fpv_sec"] * 1000,
                item["bev_sec"] * 1000, item["processing_sec"] * 1000))
    finally:
        capture.release()

    final_video = None
    if stages["comparison_video"]:
        final_video = compose_latency_video(
            timeline, output_dir / "realtime_comparison.mp4", fps,
            show_fpv=stages["fpv_visualization"], show_bev=stages["bev_visualization"],
            input_frames=input_frames, visual_frames=visual_frames,
            max_duration_sec=max_output_duration)

    report = {"task": task, "fps": fps, "pipeline_config": str(args.pipeline_config.resolve()),
              "enabled_stages": stages, "model_load_ms": model_load_sec * 1000,
              "warmup_ms": warmup_sec * 1000, "warmup_excluded_from_timeline": True,
              "input_video": input_metadata, "output_video": final_video,
              "summary": summarize(timeline), "frames": timeline}
    report_path = output_dir / "timing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Timing report: {report_path}")
    if final_video:
        print(f"Comparison video: {final_video['path']}")


if __name__ == "__main__":
    main()
