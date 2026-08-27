"""
run_e2e_pipeline.py — 端到端流水线（一条命令跑完）:

    已转换好的 TRT 引擎
        -> 对数据集全部图片逐张推理 + 二值化可通行 mask（每张图计时）
        -> 自动生成 DISTANCE 配置 yaml
        -> 逐帧调用 distance.run_sequence（H矩阵 + BEV + 边界物理误差，每帧计时）
        -> 汇总打印每张图/每帧耗时 + 总计 + 最终边界误差结果

用法:
    python run_e2e_pipeline.py \
        --engine ../trt_engines/deeplabv3_wildscenes_fp16.trt \
        --model-config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_wildscenes-512x512_standard.py \
        --num-classes 15 \
        --traversable-class 1 \
        --gt-class-id 2 \
        --dataset-root ../data/wildscenes_mini \
        --distance-root . \
        --output-root ./outputs/wildscenes_mini_trt_deeplabv3 \
        --tag deeplabv3_trt
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


# ─────────────────────────── Step 1: TRT mask 推理（逐图计时） ───────────────────────────

def get_norm_params(config_path):
    from mmengine.config import Config
    cfg = Config.fromfile(config_path)
    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32)
    std = np.array(dp.get("std", [58.395, 57.12, 57.375]), dtype=np.float32)
    bgr2rgb = dp.get("bgr_to_rgb", True)
    return mean, std, bgr2rgb


def preprocess(img_path, mean, std, bgr2rgb, inp_dtype, W, H):
    img = np.array(Image.open(img_path).convert("RGB").resize((W, H), Image.BILINEAR),
                   dtype=np.float32)
    if not bgr2rgb:
        img = img[:, :, ::-1]
    img = (img - mean) / std
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis].astype(inp_dtype))


def run_mask_inference(engine_path, model_config, image_dir, output_dir, traversable_class):
    """返回 {frame_id: 推理耗时ms} 以及汇总统计"""
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

    os.makedirs(output_dir, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    input_shape = tuple(engine.get_tensor_shape(inp_name))
    output_shape = tuple(engine.get_tensor_shape(out_name))
    _, _, H, W = input_shape
    inp_np = np.float16 if engine.get_tensor_dtype(inp_name) == trt.DataType.HALF else np.float32
    out_np = np.float16 if engine.get_tensor_dtype(out_name) == trt.DataType.HALF else np.float32

    print(f"[1/3] TRT 引擎: 输入{input_shape}({inp_np.__name__}) 输出{output_shape}({out_np.__name__})")

    d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.dtype(inp_np).itemsize)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(out_np).itemsize)
    h_output = np.empty(output_shape, dtype=out_np)
    stream = cuda.Stream()
    ctx.set_tensor_address(inp_name, int(d_input))
    ctx.set_tensor_address(out_name, int(d_output))

    mean, std, bgr2rgb = get_norm_params(model_config)

    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"未找到图片: {image_dir}")
    print(f"      共 {len(img_paths)} 张图片，开始逐张推理...")

    per_image_ms = {}
    for i, img_path in enumerate(img_paths):
        stem = Path(img_path).stem  # 2D 短横线时间戳格式
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size

        h_input = preprocess(img_path, mean, std, bgr2rgb, inp_np, W, H)

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_image_ms[stem] = elapsed_ms

        logits = h_output[0].astype(np.float32)
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        if (orig_w, orig_h) != (W, H):
            pred = np.array(Image.fromarray(pred).resize((orig_w, orig_h), Image.NEAREST))

        binary_mask = ((pred == traversable_class).astype(np.uint8)) * 255
        Image.fromarray(binary_mask, mode="L").save(
            os.path.join(output_dir, os.path.basename(img_path)))

        print(f"      [{i + 1}/{len(img_paths)}] {stem}: {elapsed_ms:.2f} ms")

    vals = list(per_image_ms.values())
    warm_vals = vals[3:] if len(vals) > 3 else vals
    avg_ms = float(np.mean(warm_vals))
    print(f"      TRT 推理平均延迟（跳过前3张热身）: {avg_ms:.2f} ms  ({1000 / avg_ms:.1f} FPS)")
    return per_image_ms, {"avg_latency_ms": avg_ms, "fps": 1000 / avg_ms, "total_ms": float(np.sum(vals))}


# ─────────────────────────── Step 2: 生成 DISTANCE 配置 ───────────────────────────

def generate_distance_config(dataset_root, mask_dir, output_dir, gt_class_id, config_out_path):
    cfg = {
        "project_root": str(dataset_root),
        "output_dir": str(output_dir),
        "dataset": {
            "dir_2d": str(Path(dataset_root) / "WildScenes2d"),
            "dir_3d": str(Path(dataset_root) / "WildScenes3d"),
            "min_class_pixels": 800000,
            "pred_mask_dir": str(mask_dir),
        },
        "class": {
            "name": "dirt",
            "id_2d": gt_class_id,
            "id_3d_for_h": 1,
        },
        "segmentation": {
            "enabled": False,
        },
        "masks": {
            "pred": {"type": "binary"},
        },
        # profile 显示 H候选搜索里给每个候选都画诊断图（overlay/热力图/直方图）
        # 占了单帧总耗时 70%+（PNG编码开销），且不影响任何数值结果，生产测速时关掉。
        "h_diagnostics": {
            "enabled": False,
        },
        "scanline": {
            "step_px": 10,
            "u_min": 0,
            "u_max": 9999,
            "max_ground_distance_m": 20.0,
            "multi_run_matching": False,
            "gt_edge_margin_px": 0,
            # 同上，scanline_error 里画的两张全分辨率叠加图也是纯可视化产物，
            # profile 显示单帧要占约0.7秒，同样不影响数值结果，关掉提速。
            "save_overlays": False,
        },
    }
    os.makedirs(os.path.dirname(config_out_path), exist_ok=True)
    with open(config_out_path, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"[2/3] DISTANCE 配置已生成: {config_out_path}")
    return config_out_path


# ─────────────────────────── Step 3: 一次性调用 distance.run_sequence（全部帧） ───────────────────────────

def get_frame_ids(dataset_root):
    clouds = sorted(glob.glob(str(Path(dataset_root) / "WildScenes3d" / "Clouds" / "*.bin")))
    return [Path(p).stem for p in clouds]


def run_distance_all_frames(distance_root, config_path, num_frames):
    """
    一次性跑完全部帧（单次进程调用，避免43次子进程启动开销污染计时），
    用 (总耗时 / 帧数) 作为单帧平均处理时间，更接近真实部署时的稳态吞吐。
    """
    print(f"[3/3] 一次性运行 distance.run_sequence（共 {num_frames} 帧，单进程内循环处理）...")
    cmd = [sys.executable, "-m", "distance.run_sequence", "--config", str(config_path)]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(distance_root), capture_output=True, text=True)
    total_ms = (time.perf_counter() - t0) * 1000

    print(result.stdout[-3000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-4000:])
        raise RuntimeError(f"distance.run_sequence 失败，退出码 {result.returncode}")

    avg_per_frame_ms = total_ms / max(num_frames, 1)
    print(f"      总耗时: {total_ms:.1f} ms  ({num_frames} 帧)")
    print(f"      单帧平均: {avg_per_frame_ms:.1f} ms  ({1000 / avg_per_frame_ms:.2f} Hz)")
    return total_ms, avg_per_frame_ms


# ─────────────────────────── 汇总打印 ───────────────────────────

def print_summary(per_image_ms, trt_stats, distance_total_ms, distance_avg_ms, output_dir, tag):
    summary_path = Path(output_dir) / "sequence_summary.json"
    num_frames = len(per_image_ms)
    trt_total_ms = float(np.sum(list(per_image_ms.values())))

    print(f"\n{'=' * 70}")
    print(f"端到端流水线耗时汇总  [{tag}]  (共 {num_frames} 帧)")
    print(f"{'=' * 70}")
    print(f"TRT 推理:      总计 {trt_total_ms:>10.1f} ms   "
          f"平均 {trt_stats['avg_latency_ms']:.2f} ms/张   {trt_stats['fps']:.1f} FPS")
    print(f"DISTANCE 处理: 总计 {distance_total_ms:>10.1f} ms   "
          f"平均 {distance_avg_ms:.1f} ms/帧   {1000 / distance_avg_ms:.2f} Hz")
    grand_total = trt_total_ms + distance_total_ms
    combined_avg = grand_total / max(num_frames, 1)
    print(f"{'-' * 70}")
    print(f"全流程:        总计 {grand_total:>10.1f} ms   "
          f"平均 {combined_avg:.1f} ms/帧   {1000 / combined_avg:.2f} Hz")
    print(f"{'=' * 70}")

    if not summary_path.exists():
        print(f"未找到 {summary_path}")
        return

    with open(summary_path) as f:
        summary = json.load(f)
    agg = summary.get("aggregate", {}).get("scanline", {})
    print(f"\n边界物理误差结果 (米):")
    print(f"  {'':<16}{'mean':>10}{'median':>10}{'P75':>10}{'P90':>10}{'P95':>10}{'frames':>10}")
    for side in ("overall", "left_boundary", "right_boundary"):
        s = agg.get(side, {})
        if s:
            print(f"  {side:<16}"
                  f"{s.get('mean_of_frame_means_m', float('nan')):>10.3f}"
                  f"{s.get('median_of_frame_medians_m', float('nan')):>10.3f}"
                  f"{s.get('mean_of_frame_p75_m', float('nan')):>10.3f}"
                  f"{s.get('mean_of_frame_p90_m', float('nan')):>10.3f}"
                  f"{s.get('mean_of_frame_p95_m', float('nan')):>10.3f}"
                  f"{s.get('valid_frames', 0):>10d}")

    h_agg = summary.get("aggregate", {}).get("h_matrix", {}).get("reproj_error_px", {})
    if h_agg:
        print(f"\nH矩阵重投影误差 (px): mean-of-means={h_agg.get('mean_of_frame_means'):.3f}  "
              f"mean-of-P95={h_agg.get('mean_of_frame_p95'):.3f}")

    print(f"\n完整结果: {summary_path}")
    print(f"{'=' * 70}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--traversable-class", type=int, required=True)
    ap.add_argument("--gt-class-id", type=int, required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--distance-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()

    mask_dir = Path(args.output_root) / "pred_mask"
    config_path = Path(args.output_root) / "config.yaml"

    per_image_ms, trt_stats = run_mask_inference(
        args.engine, args.model_config,
        str(Path(args.dataset_root) / "WildScenes2d" / "image"),
        str(mask_dir), args.traversable_class,
    )

    generate_distance_config(
        args.dataset_root, mask_dir, args.output_root, args.gt_class_id, config_path,
    )

    frame_ids = get_frame_ids(args.dataset_root)
    distance_total_ms, distance_avg_ms = run_distance_all_frames(
        args.distance_root, config_path, len(frame_ids))

    print_summary(per_image_ms, trt_stats, distance_total_ms, distance_avg_ms, args.output_root, args.tag)


if __name__ == "__main__":
    main()
