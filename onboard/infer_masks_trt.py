"""
infer_masks_trt.py
用 TRT FP16 引擎对一批图片跑推理，输出可通行区域二值 mask（0/255 单通道 PNG），
文件名与输入图片一致，供 DISTANCE 的 distance/run_sequence.py 直接读取
（masks.pred.type: "binary"）。

用法:
    python infer_masks_trt.py \
        --engine /workspace/WildScenes/trt_engines/ddrnet_fp16.trt \
        --config wildscenes/configs/ddrnet/ddrnet_23-slim_goose_category-512x512.py \
        --image-dir /root/distance/WildScenes/DISTANCE/datasets/wildscenes_mini/WildScenes2d/image \
        --output-dir /root/distance/WildScenes/DISTANCE/datasets/wildscenes_mini/WildScenes2d/ddrnet_trt_pred_mask \
        --traversable-class 3
"""
import argparse
import glob
import os
import time

import numpy as np
from PIL import Image


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
    x = np.ascontiguousarray(img.transpose(2, 0, 1)[np.newaxis].astype(inp_dtype))
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TRT .trt 引擎路径")
    ap.add_argument("--config", required=True, help="mmseg config（读取归一化参数）")
    ap.add_argument("--image-dir", required=True, help="输入图片目录")
    ap.add_argument("--output-dir", required=True, help="mask 输出目录")
    ap.add_argument("--traversable-class", type=int, required=True,
                     help="模型输出里代表可通行区域的类别索引")
    ap.add_argument("--num-classes", type=int, default=12)
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
    input_shape = tuple(engine.get_tensor_shape(inp_name))   # (1, 3, H, W)
    output_shape = tuple(engine.get_tensor_shape(out_name))  # (1, C, H, W)
    _, _, H, W = input_shape
    inp_trt = engine.get_tensor_dtype(inp_name)
    out_trt = engine.get_tensor_dtype(out_name)
    inp_np = np.float16 if inp_trt == trt.DataType.HALF else np.float32
    out_np = np.float16 if out_trt == trt.DataType.HALF else np.float32

    print(f"引擎输入尺寸: {input_shape}  dtype={inp_np.__name__}")
    print(f"引擎输出尺寸: {output_shape}  dtype={out_np.__name__}")

    d_input = cuda.mem_alloc(int(np.prod(input_shape)) * np.dtype(inp_np).itemsize)
    d_output = cuda.mem_alloc(int(np.prod(output_shape)) * np.dtype(out_np).itemsize)
    h_output = np.empty(output_shape, dtype=out_np)
    stream = cuda.Stream()

    ctx.set_tensor_address(inp_name, int(d_input))
    ctx.set_tensor_address(out_name, int(d_output))

    mean, std, bgr2rgb = get_norm_params(args.config)
    print(f"归一化 mean={mean} std={std} bgr2rgb={bgr2rgb}")

    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    if not img_paths:
        raise FileNotFoundError(f"未找到图片: {args.image_dir}")
    print(f"共 {len(img_paths)} 张图片，开始推理...")

    times = []
    for i, img_path in enumerate(img_paths):
        with Image.open(img_path) as im:
            orig_w, orig_h = im.size

        h_input = preprocess(img_path, mean, std, bgr2rgb, inp_np, W, H)

        t0 = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        ctx.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

        logits = h_output[0].astype(np.float32)              # (C, H, W)
        pred = np.argmax(logits, axis=0).astype(np.uint8)    # (H, W) 引擎分辨率

        if (orig_w, orig_h) != (W, H):
            pred = np.array(Image.fromarray(pred).resize((orig_w, orig_h), Image.NEAREST))

        binary_mask = ((pred == args.traversable_class).astype(np.uint8)) * 255

        out_path = os.path.join(args.output_dir, os.path.basename(img_path))
        Image.fromarray(binary_mask, mode="L").save(out_path)

        if (i + 1) % 10 == 0:
            print(f"  [{i + 1}/{len(img_paths)}] 完成")

    avg_ms = float(np.mean(times[3:])) if len(times) > 3 else float(np.mean(times))
    print(f"\n{'=' * 50}")
    print(f"推理完成: {len(img_paths)} 张")
    print(f"平均延迟: {avg_ms:.2f} ms  ({1000 / avg_ms:.1f} FPS)")
    print(f"mask 已保存到: {args.output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
