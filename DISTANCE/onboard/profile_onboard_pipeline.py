"""
profile_onboard_pipeline.py — 实车真实链路耗时测量（不含GT处理、不含H候选搜索）

真实车上应该跑的完整流程：
    TRT推理 -> mask(argmax二值化) -> 清理pred mask(去噪) -> 提取pred边界
    -> 把边界画在原图上(供后端操作员查看) -> [可选]编码成图片准备传输

跟 run_e2e_pipeline.py/distance.run_sequence 的区别：
    - 不读/处理GT mask，不算scanline_error(那是离线验证模型准确度用的)
    - 不做H候选搜索(实车标定是固定的，不用每帧搜索多个候选时间戳配对)
    - 只保留"车上真正要做"的这几步，逐帧计时

用法:
    python profile_onboard_pipeline.py \
        --engine ../trt_engines/deeplabv3_wildscenes_fp16.trt \
        --model-config wildscenes/configs/deeplabv3/deeplabv3_r50-d8_2xb20-80k_wildscenes-512x512_standard.py \
        --traversable-class 1 \
        --image-dir ../data/wildscenes_mini/WildScenes2d/image \
        --output-dir outputs/onboard_test_output \
        --min-component-area-px 5000 \
        --close-iterations 2
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root，让 distance 包可被 import
from distance.pipeline import clean_prediction_mask, extract_external_boundary  # noqa: E402


def get_norm_params(config_path):
    from mmengine.config import Config
    cfg = Config.fromfile(config_path)
    dp = cfg.get("data_preprocessor", {})
    mean = np.array(dp.get("mean", [123.675, 116.28, 103.53]), dtype=np.float32)
    std = np.array(dp.get("std", [58.395, 57.12, 57.375]), dtype=np.float32)
    bgr2rgb = dp.get("bgr_to_rgb", True)
    return mean, std, bgr2rgb


def preprocess(img_rgb_array, mean, std, bgr2rgb, inp_dtype, W, H):
    img = np.array(Image.fromarray(img_rgb_array).resize((W, H), Image.BILINEAR), dtype=np.float32)
    if not bgr2rgb:
        img = img[:, :, ::-1]
    img = (img - mean) / std
    return np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis].astype(inp_dtype))


def draw_boundary_on_image(rgb_image: Image.Image, boundary_mask: np.ndarray, color=(0, 255, 0)) -> Image.Image:
    """把边界像素画在原图上，给后端操作员看的叠加图。"""
    overlay = rgb_image.convert("RGB").copy()
    ys, xs = np.nonzero(boundary_mask)
    draw = ImageDraw.Draw(overlay)
    for x, y in zip(xs.tolist(), ys.tolist()):
        draw.point((x, y), fill=color)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model-config", required=True)
    ap.add_argument("--traversable-class", type=int, required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-component-area-px", type=int, default=5000)
    ap.add_argument("--close-iterations", type=int, default=2)
    ap.add_argument("--save-overlay", action="store_true", help="是否真的保存叠加图（测极限速度时可以不存）")
    args = ap.parse_args()

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa

    os.makedirs(args.output_dir, exist_ok=True)

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    input_shape = tuple(engine.get_tensor_shape(inp_name))
    output_shape = tuple(engine.get_tensor_shape(out_name))
    _, _, H, W = input_shape
    inp_np = np.float16 if engine.get_tensor_dtype(inp_name) == trt.DataType.HALF else np.float32
    out_np = np.float16 if engine.get_tensor_dtype(out_name) == trt.DataType.HALF else np.float32

    print(f"引擎: 输入{input_shape}({inp_np.__name__}) 输出{output_shape}({out_np.__name__})")

    d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.dtype(inp_np).itemsize)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(out_np).itemsize)
    h_output = np.empty(output_shape, dtype=out_np)
    stream = cuda.Stream()
    ctx.set_tensor_address(inp_name, int(d_input))
    ctx.set_tensor_address(out_name, int(d_output))

    mean, std, bgr2rgb = get_norm_params(args.model_config)

    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"未找到图片: {args.image_dir}")
    print(f"共 {len(img_paths)} 张图片，开始逐张跑真实车上链路...")

    stage_times = {"trt推理": [], "mask清理": [], "边界提取": [], "画图": [], "总计": []}

    for i, img_path in enumerate(img_paths):
        img_pil = Image.open(img_path).convert("RGB")
        img_rgb = np.array(img_pil)
        orig_w, orig_h = img_pil.size

        t_start = time.perf_counter()

        # ── 1. TRT 推理 ──────────────────────────────────────────
        t0 = time.perf_counter()
        h_input = preprocess(img_rgb, mean, std, bgr2rgb, inp_np, W, H)
        cuda.memcpy_htod_async(d_input, h_input, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        logits = h_output[0].astype(np.float32)
        pred = np.argmax(logits, axis=0).astype(np.uint8)
        if (orig_w, orig_h) != (W, H):
            pred = np.array(Image.fromarray(pred).resize((orig_w, orig_h), Image.NEAREST))
        binary_mask = pred == args.traversable_class
        t_trt = time.perf_counter() - t0

        # ── 2. 清理 pred mask（只处理pred，不碰GT）──────────────────
        t0 = time.perf_counter()
        cleaned = clean_prediction_mask(binary_mask, args.min_component_area_px, args.close_iterations)
        t_clean = time.perf_counter() - t0

        # ── 3. 提取边界 ──────────────────────────────────────────
        t0 = time.perf_counter()
        boundary = extract_external_boundary(cleaned)
        t_boundary = time.perf_counter() - t0

        # ── 4. 画在原图上（供后端查看）──────────────────────────────
        t0 = time.perf_counter()
        if args.save_overlay:
            overlay = draw_boundary_on_image(img_pil, boundary)
            overlay.save(os.path.join(args.output_dir, os.path.basename(img_path)))
        t_draw = time.perf_counter() - t0

        t_total = time.perf_counter() - t_start

        stage_times["trt推理"].append(t_trt * 1000)
        stage_times["mask清理"].append(t_clean * 1000)
        stage_times["边界提取"].append(t_boundary * 1000)
        stage_times["画图"].append(t_draw * 1000)
        stage_times["总计"].append(t_total * 1000)

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(img_paths)}] 完成")

    print(f"\n{'=' * 60}")
    print(f"实车真实链路耗时统计（跳过前3张热身，{len(img_paths) - 3} 张有效样本）")
    print(f"{'=' * 60}")
    for stage, times in stage_times.items():
        vals = times[3:] if len(times) > 3 else times
        avg = float(np.mean(vals))
        print(f"  {stage:<8}: 平均 {avg:6.2f} ms")

    total_avg = float(np.mean(stage_times["总计"][3:]))
    print(f"{'-' * 60}")
    print(f"端到端: {total_avg:.2f} ms/帧  =  {1000 / total_avg:.2f} Hz")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
